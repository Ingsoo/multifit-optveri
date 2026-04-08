from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any

# This module contains the actual Gurobi model for the OptVeri formulation.
# If you are reading the code with the paper open, the high-level map is:
# 1. `build_obv_model`: Section 4 base MIQP.
# 2. `_apply_global_valid_inequalities`: globally useful strengthening cuts.
# 3. `_apply_paper_acceleration_constraints`: Section 5 common acceleration cuts.
# 4. `_apply_profile_cardinality_constraints` and below: case/profile-specific cuts.

from multifit_optveri.acceleration import (
    AccelerationCase,
    PAPER_MACHINE_RANGE,
    PAPER_TARGET_RATIO,
)
from multifit_optveri.experiments import ExperimentCase
from multifit_optveri.math_utils import ceil_fraction
from multifit_optveri.models.spec import ObvModelDimensions, derive_obv_dimensions

# Paper/code comparison guide for this file:
# - Section 4 base MIQP lives in `build_obv_model`.
# - Section 5 common acceleration cuts live in `_apply_paper_acceleration_constraints`.
# - Section 5 branch-specific structure lives in `_apply_profile_cardinality_constraints`,
#   `_apply_mtf_base_profile_constraints`, and `_apply_case_profile_constraints`.
# - Some strengthenings below come from the older implementation, not just the prose
#   of the paper. Those are the first places to audit if you want "paper exactness".

if TYPE_CHECKING:
    import gurobipy as gp
    from gurobipy import GRB
    from gurobipy import Model as GurobiModel
    from gurobipy import Var as GurobiVar
    from gurobipy import tupledict as GurobiTupleDict
else:
    GurobiModel = Any
    GurobiVar = Any
    GurobiTupleDict = Any
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:  # pragma: no cover - exercised only when Gurobi is missing
        gp = None
        GRB = None


PVarMap = dict[int, GurobiVar]
TupleVarMap = GurobiTupleDict
# `p` is stored as a plain dict because it is indexed only by job.
# `x`, `q`, and `s` are Gurobi tupledict objects because they are indexed by
# two coordinates and frequently used inside generator expressions.


class GurobiUnavailableError(RuntimeError):
    """Raised when a solve/build action requires gurobipy but it is unavailable."""


OPT_JOB_CARDINALITY_LOWER_BOUND = 3
BOUND_STOP_TOLERANCE = 0


@dataclass
class BuiltObvModel:
    """Bundle the built Gurobi model with a coarse dimension summary."""

    model: GurobiModel
    dimensions: ObvModelDimensions


@dataclass(frozen=True)
class MtfProfileLayout:
    """Expanded positional view of an abstract MTF profile tuple.

    The branch iterators speak in profile counts such as |F1|, |R2|, |F2|, ...
    This dataclass converts those counts into consecutive machine-id blocks and
    index markers (e2/e3/e4, t2/t3/t4) used throughout the Section 5 cuts.
    """

    f1_machines: tuple[int, ...]
    r2_machines: tuple[int, ...]
    f2_machines: tuple[int, ...]
    r3_machines: tuple[int, ...]
    f3_machines: tuple[int, ...]
    r4_machines: tuple[int, ...]
    f4_machines: tuple[int, ...]
    r5_machines: tuple[int, ...]
    e2: int
    e3: int
    e4: int
    e5: int
    t2: int
    t3: int
    t4: int
    t5: int


def _require_gurobi() -> None:
    """Fail fast when model building is requested without gurobipy installed."""

    if gp is None or GRB is None:
        raise GurobiUnavailableError(
            "gurobipy is not available. Install it and activate a Gurobi license to run models."
        )


def _as_float(value: Fraction) -> float:
    """Convert an exact Fraction into the float form expected by Gurobi."""

    return float(value.numerator / value.denominator)


def _common_processing_time_lower_bound(machine_count: int, target_ratio: Fraction) -> Fraction:
    """Return the generic lower bound on p_n used before case-specific cuts."""

    # Common lower bound on p_n used already in the base reasoning and then
    # tightened further by Section 5's case split.
    return (target_ratio - 1) * machine_count / (machine_count - 1)


def _processing_time_lower_bound(case: ExperimentCase) -> Fraction:
    """Return the final lower bound used when creating all p_j variables."""

    # This combines the generic lower bound with the case-specific p_n interval.
    # If the paper/code comparison on variable domains fails, start here.
    lower_bound = _common_processing_time_lower_bound(case.machine_count, case.target_ratio)
    if case.acceleration_case is not AccelerationCase.BASE:
        pn_lower_bound = case.acceleration_case.pn_range.lower
        if pn_lower_bound is not None:
            lower_bound = max(lower_bound, pn_lower_bound)
    return lower_bound


def _opt_job_cardinality_lower_bound(case: ExperimentCase) -> int:
    """Return the minimum OPT machine cardinality already fixed by the branch.

    When an OPT profile is known, we can tighten every p_j upper bound using the
    minimum number of jobs that must share an OPT machine with j.
    """

    if case.opt_profile is None:
        return OPT_JOB_CARDINALITY_LOWER_BOUND

    machine_cardinalities = case.opt_profile.machine_cardinalities
    if not machine_cardinalities:
        return OPT_JOB_CARDINALITY_LOWER_BOUND
    return min(machine_cardinalities)


def _processing_time_upper_bound(case: ExperimentCase, job_index: int, lower_bound: Fraction) -> Fraction:
    """Return the variable upper bound used for p_j.

    This is intentionally stronger than the bare formulation because the old
    implementation used these bounds to improve presolve and tree performance.
    """

    # These bounds are stronger than the pure mathematical formulation and were
    # carried over from the earlier implementation to help presolve. They are an
    # important "implementation choice vs paper text" checkpoint.
    upper_bound = Fraction(1, ((job_index - 1) // case.machine_count) + 1)
    opt_job_cardinality_lower_bound = _opt_job_cardinality_lower_bound(case)
    upper_bound = min(
        upper_bound,
        Fraction(1, 1) - (opt_job_cardinality_lower_bound - 1) * lower_bound,
    )
    if job_index == case.job_count and case.acceleration_case is not AccelerationCase.BASE:
        pn_upper_bound = case.acceleration_case.pn_range.upper
        if pn_upper_bound is not None:
            upper_bound = min(upper_bound, pn_upper_bound)
    return upper_bound


def _tighten_processing_time_bounds_by_split(
    case: ExperimentCase,
    job_index: int,
    lower_bound: Fraction,
    upper_bound: Fraction,
) -> tuple[Fraction, Fraction]:
    """Tighten p_j bounds using the current p_n bounds and the D/D' split.

    The explicit Section 5 constraints remain in the model; this helper only
    adds the constant-domain strengthening implied by them.
    """

    if case.ell is None or job_index == case.job_count:
        return lower_bound, upper_bound

    split_gap = Fraction(3, 17)
    pn_lower_bound = _processing_time_lower_bound(case)
    pn_upper_bound = _processing_time_upper_bound(case, case.job_count, pn_lower_bound)

    if job_index < case.ell:
        lower_bound = max(lower_bound, pn_lower_bound + split_gap)
    else:
        upper_bound = min(upper_bound, pn_upper_bound + split_gap)

    return lower_bound, upper_bound


def _opt_cardinality_upper_bound(lower_bound: Fraction) -> int:
    """Compute the global upper bound on OPT machine cardinality."""

    return ceil_fraction(Fraction(1, 1) / lower_bound) - 1


def _mtf_cardinality_upper_bound(machine_count: int, target_ratio: Fraction) -> int:
    """Compute the global upper bound on MTF machine cardinality."""

    assert target_ratio == PAPER_TARGET_RATIO and machine_count in PAPER_MACHINE_RANGE
    return 4 if machine_count == 8 else 5


def _validate_paper_acceleration_case(case: ExperimentCase) -> None:
    """Guard Section 5 cuts so they are used only in the paper's regime."""

    # Section 5 in the current code is only claimed for the paper setting
    # rho = 20/17 and m in {8, ..., 12}. If you run outside that range, you are
    # no longer checking the same statement as the paper.
    if case.target_ratio != PAPER_TARGET_RATIO:
        target_text = f"{PAPER_TARGET_RATIO.numerator}/" f"{PAPER_TARGET_RATIO.denominator}"
        raise ValueError(
            f"Acceleration case '{case.acceleration_case.value}' " f"is only implemented for target={target_text}."
        )
    if case.machine_count not in PAPER_MACHINE_RANGE:
        machine_range_text = f"{PAPER_MACHINE_RANGE.start}.." f"{PAPER_MACHINE_RANGE.stop - 1}"
        raise ValueError(
            f"Acceleration case '{case.acceleration_case.value}' " f"is only implemented for m in {machine_range_text}."
        )


def _use_exact_mtf(case: ExperimentCase) -> bool:
    """Return whether this branch should use the reduced exact-MTF encoding."""

    return (
        case.acceleration_case in (AccelerationCase.CASE_2, AccelerationCase.CASE_3)
        and case.mtf_profile is not None
        and case.fallback_starts is not None
    )


def _apply_profile_cardinality_constraints(
    model: GurobiModel,
    case: ExperimentCase,
    p: PVarMap,
    x: TupleVarMap,
    z_var: GurobiVar,
    q: TupleVarMap | None,
    jobs: range,
    truncated_jobs: range,
    machines: range,
    target: float,
) -> None:
    """Tie external branch choices (ell / profiles) to actual model constraints."""

    # This function ties the outer branching objects (ell, OPT profile, MTF profile)
    # to actual model constraints. If the iterator families match the paper but the
    # resulting model still differs, this is the next file section to inspect.
    layout: MtfProfileLayout | None = None

    if case.ell is not None:
        # This is the D / D' split from Section 5:
        # jobs before ell belong to D, jobs from ell onward belong to D'.
        for job_index in range(1, case.job_count):
            if job_index < case.ell:
                model.addConstr(
                    p[job_index] >= Fraction(3, 17).numerator / Fraction(3, 17).denominator + p[case.job_count],
                    name=f"processing_time_in_D[{job_index}]",
                )
            else:
                model.addConstr(
                    p[job_index] <= Fraction(3, 17).numerator / Fraction(3, 17).denominator + p[case.job_count],
                    name=f"processing_time_in_D_prime[{job_index}]",
                )

    if case.opt_profile is not None:
        # Fix the number of jobs on each OPT machine according to the chosen
        # coarse OPT profile branch.
        for machine_index, cardinality in enumerate(case.opt_profile.machine_cardinalities, start=1):
            model.addConstr(
                gp.quicksum(x[machine_index, j] for j in jobs) == cardinality,
                name=f"opt_profile_cardinality[{machine_index}]",
            )
        _apply_opt_profile_tail_sum_constraint(model, case, p)

    if case.mtf_profile is not None:
        layout = _build_mtf_profile_layout(case)
        if _use_exact_mtf(case):
            if case.fallback_starts is None:
                raise ValueError("Exact MTF constraints require fallback_starts.")
            _apply_exact_mtf_constraints(
                model,
                case,
                p,
                x,
                z_var,
                machines,
                target,
                layout,
            )
        else:
            if q is None:
                raise ValueError("Generic MTF profile constraints require q variables.")
            # Fix the number of jobs on each MTF machine according to the chosen
            # coarse MTF profile branch, then refine with profile-specific cuts.
            for machine_index, cardinality in enumerate(case.mtf_profile.machine_cardinalities, start=1):
                model.addConstr(
                    gp.quicksum(q[machine_index, j] for j in truncated_jobs) == cardinality,
                    name=f"mtf_profile_cardinality[{machine_index}]",
                )
            _apply_mtf_base_profile_constraints(
                model,
                case,
                p,
                x,
                q,
                truncated_jobs,
                machines,
                target,
                layout,
            )

    if case.opt_profile is not None and layout is not None and case.ell is not None:
        _apply_case_profile_constraints(
            model,
            case,
            p,
            x,
            q,
            jobs,
            truncated_jobs,
            machines,
            target,
            layout,
        )


def _build_mtf_profile_layout(case: ExperimentCase) -> MtfProfileLayout:
    """Expand the abstract MTF profile into concrete machine blocks and indices."""

    if case.mtf_profile is None:
        raise ValueError("MTF profile layout requires case.mtf_profile.")

    # This converts the abstract tuple profile into the paper's consecutive
    # machine blocks F1, R2, F2, R3, F3, R4, F4, R5 and their derived indices
    # e2/e3/e4/e5 and t2/t3/t4/t5. If any of these offsets are wrong, most of the
    # Section 5 structural constraints become shifted.
    machine_ids = list(range(1, case.machine_count + 1))
    profile = case.mtf_profile
    cursor = 0

    def take(count: int) -> tuple[int, ...]:
        # Consume the next consecutive `count` machine ids from left to right.
        nonlocal cursor
        values = tuple(machine_ids[cursor : cursor + count])
        cursor += count
        return values

    f1_machines = take(profile.nF1)
    r2_machines = take(profile.nR2)
    f2_machines = take(profile.nF2)
    r3_machines = take(profile.nR3)
    f3_machines = take(profile.nF3)
    r4_machines = take(profile.nR4)
    f4_machines = take(profile.nF4)
    r5_machines = take(profile.nR5)

    e2 = profile.nF1 + 1
    e3 = e2 + 2 * (profile.nR2 + profile.nF2)
    e4 = e3 + 3 * (profile.nR3 + profile.nF3)
    e5 = e4 + 4 * (profile.nR4 + profile.nF4)
    t2 = e2 + profile.nF1
    t3 = e3 + profile.nF1 + profile.nF2
    t4 = e4 + profile.nF1 + profile.nF2 + profile.nF3
    t5 = e5 + profile.nF1 + profile.nF2 + profile.nF3 + profile.nF4

    return MtfProfileLayout(
        f1_machines=f1_machines,
        r2_machines=r2_machines,
        f2_machines=f2_machines,
        r3_machines=r3_machines,
        f3_machines=f3_machines,
        r4_machines=r4_machines,
        f4_machines=f4_machines,
        r5_machines=r5_machines,
        e2=e2,
        e3=e3,
        e4=e4,
        e5=e5,
        t2=t2,
        t3=t3,
        t4=t4,
        t5=t5,
    )


def _apply_opt_profile_tail_sum_constraint(
    model: GurobiModel,
    case: ExperimentCase,
    p: PVarMap,
) -> None:
    """Add the OPT tail-sum inequality implied by the fixed OPT profile.

    This is the exact profile-aware version of the chapter's additional valid
    condition: the smallest 4|S4| + 5|S5| jobs together fit on the S4/S5
    machines, whose total OPT capacity is |S4| + |S5|.
    """

    if case.opt_profile is None:
        return

    tail_job_count = 4 * case.opt_profile.nS4 + 5 * case.opt_profile.nS5
    if tail_job_count <= 0:
        return

    first_tail_job = case.job_count - tail_job_count + 1
    model.addConstr(
        gp.quicksum(p[job_index] for job_index in range(first_tail_job, case.job_count + 1))
        <= case.opt_profile.nS4 + case.opt_profile.nS5,
        name="opt_profile_tail_sum",
    )


def _apply_mtf_base_profile_constraints(
    model: GurobiModel,
    case: ExperimentCase,
    p: PVarMap,
    x: TupleVarMap,
    q: TupleVarMap,
    truncated_jobs: range,
    machines: range,
    target: float,
    layout: MtfProfileLayout,
) -> None:
    """Add MTF structural cuts shared by all case branches once a profile is fixed."""

    # These are the generic per-profile MTF structural consequences used across
    # cases once a concrete MTF profile is fixed.
    profile = case.mtf_profile
    if profile is None:
        return

    for machine_index in layout.f1_machines:
        # Machines in F1 always receive their diagonal job first.
        model.addConstr(
            q[machine_index, machine_index] == 1,
            name=f"F1_assignment_constr[{machine_index}]",
        )

    if layout.f1_machines:
        # The first non-F1 machine immediately after the F1 block also receives
        # its own diagonal job before the fallback jobs return.
        last_f1 = layout.f1_machines[-1]
        model.addConstr(
            q[last_f1 + 1, last_f1 + 1] == 1,
            name=f"F1_assignment_constr[{last_f1 + 1}]",
        )
        model.addConstr(
            p[last_f1] + p[last_f1 + 1] >= target,
            name=f"F1_valid_constr[{last_f1}]",
        )

    if layout.r2_machines:
        # R2 machines are regular 2-machines: the last such machine plus job n
        # must already be tight enough to explain why n does not fit later.
        model.addConstr(
            p[layout.e2 + 2 * profile.nR2 - 2] + p[layout.e2 + 2 * profile.nR2 - 1] + p[case.job_count] >= target,
            name=f"R2_valid_constr[{layout.r2_machines[-1]}]",
        )
        proc_same_range = range(layout.t2 + 1, layout.e2 + 2 * profile.nR2 - 2)
        for job_index in proc_same_range:
            # Jobs in the regular interior of R2 can be assumed to have equal
            # processing times; symmetry is then broken in OPT accordingly.
            model.addConstr(
                p[job_index] == p[job_index + 1],
                name=f"R2_processing_times[{job_index}]",
            )
            model.addConstrs(
                (
                    gp.quicksum(x[machine_prime, job_index] for machine_prime in range(1, machine_index + 1))
                    >= x[machine_index, job_index + 1]
                    for machine_index in machines
                ),
                name=f"R2_symmetry_break_by_proc[{job_index}]",
            )

    if layout.f2_machines:
        # The tail of the F2 block must already be near-tight in MTF.
        model.addConstr(
            p[layout.e3 - 2] + p[layout.e3 - 1] + p[layout.e3] >= target,
            name=f"F2_valid_constr[{layout.f2_machines[-1]}]",
        )

    if layout.r3_machines:
        # Same story for regular 3-machines: near-tight last machine and equal
        # processing times on the regular interior.
        model.addConstr(
            p[layout.e3 + 3 * profile.nR3 - 3]
            + p[layout.e3 + 3 * profile.nR3 - 2]
            + p[layout.e3 + 3 * profile.nR3 - 1]
            + p[case.job_count]
            >= target,
            name=f"R3_valid_constr[{layout.r3_machines[-1]}]",
        )
        proc_same_range = range(layout.t3 + 1, layout.e3 + 3 * profile.nR3 - 3)
        for job_index in proc_same_range:
            model.addConstr(
                p[job_index] == p[job_index + 1],
                name=f"R3_processing_times[{job_index}]",
            )
            model.addConstrs(
                (
                    gp.quicksum(x[machine_prime, job_index] for machine_prime in range(1, machine_index + 1))
                    >= x[machine_index, job_index + 1]
                    for machine_index in machines
                ),
                name=f"R3_symmetry_break_by_proc[{job_index}]",
            )

    if layout.f3_machines:
        # F3 machines are exactly the place where 4 scheduled jobs appear before n.
        model.addConstr(
            p[layout.e4 - 3] + p[layout.e4 - 2] + p[layout.e4 - 1] + p[layout.e4] >= target,
            name=f"F3_valid_constr[{layout.f3_machines[-1]}]",
        )

    if layout.r4_machines:
        # Regular 4-machines mirror the same pattern as R2/R3.
        model.addConstr(
            p[layout.e4 + 4 * profile.nR4 - 4]
            + p[layout.e4 + 4 * profile.nR4 - 3]
            + p[layout.e4 + 4 * profile.nR4 - 2]
            + p[layout.e4 + 4 * profile.nR4 - 1]
            + p[case.job_count]
            >= target,
            name=f"R4_valid_constr[{layout.r4_machines[-1]}]",
        )
        proc_same_range = range(layout.t4 + 1, layout.e4 + 4 * profile.nR4 - 4)
        for job_index in proc_same_range:
            model.addConstr(
                p[job_index] == p[job_index + 1],
                name=f"R4_processing_times[{job_index}]",
            )
            model.addConstrs(
                (
                    gp.quicksum(x[machine_prime, job_index] for machine_prime in range(1, machine_index + 1))
                    >= x[machine_index, job_index + 1]
                    for machine_index in machines
                ),
                name=f"R4_symmetry_break_by_proc[{job_index}]",
            )

    if layout.f4_machines:
        # F4 machines are the fallback 5-job machines that appear before the
        # regular 5-machine tail.
        model.addConstr(
            p[layout.e5 - 4] + p[layout.e5 - 3] + p[layout.e5 - 2] + p[layout.e5 - 1] + p[layout.e5] >= target,
            name=f"F4_valid_constr[{layout.f4_machines[-1]}]",
        )

    for machine_index in layout.r5_machines:
        # The profile object already says these are regular 5-job MTF machines,
        # so make the exact cardinality explicit at the model level.
        model.addConstr(
            gp.quicksum(q[machine_index, job_index] for job_index in truncated_jobs) == 5,
            name=f"R5_cardinality_constr[{machine_index}]",
        )


def _apply_global_valid_inequalities(
    model: GurobiModel,
    case: ExperimentCase,
    p: PVarMap,
    x: TupleVarMap,
    q: TupleVarMap | None,
    s: TupleVarMap | None,
    jobs: range,
    truncated_jobs: range,
    machines: range,
    target: float,
) -> None:
    """Add global cuts that are useful in all branches.

    These are a mix of paper-level valid inequalities and old-code
    strengthenings. When auditing paper/code fidelity, inspect this block
    separately from the pure Section 4 formulation.
    """

    # These are global strengthenings used regardless of the case split.
    # Some come directly from the paper's valid inequalities, while others were
    # retained from the earlier implementation to speed up solving. If you are
    # checking "paper exact" vs "implementation strengthened", audit this block.
    lower_bound = _processing_time_lower_bound(case)
    opt_cardinality_upper = _opt_cardinality_upper_bound(lower_bound)
    mtf_cardinality_upper = _mtf_cardinality_upper_bound(case.machine_count, case.target_ratio)

    if case.opt_profile is None:
        model.addConstrs(
            (
                gp.quicksum(x[machine_index, job_index] for job_index in jobs) >= OPT_JOB_CARDINALITY_LOWER_BOUND
                for machine_index in machines
            ),
            name="opt_cardinality_lb",
        )
        # Since jobs are sorted, the smallest jobs also satisfy an aggregate upper
        # bound that helps OPT-side pruning.
        model.addConstrs(
            (
                gp.quicksum(x[machine_index, job_index] for job_index in jobs) <= opt_cardinality_upper
                for machine_index in machines
            ),
            name="opt_cardinality_ub",
        )
        model.addConstrs(
            (
                gp.quicksum(x[machine_index, job_index] for job_index in jobs)
                <= gp.quicksum(x[machine_index + 1, job_index] for job_index in jobs)
                for machine_index in range(1, case.machine_count)
            ),
            name="opt_cardinality_order",
        )

        rem = case.job_count % case.machine_count
        if rem != 0:
            quo = case.job_count // case.machine_count
            first_small_job = case.job_count - rem * (quo + 1) + 1
            model.addConstr(
                gp.quicksum(p[job_index] for job_index in range(first_small_job, case.job_count + 1)) <= rem,
                name="opt_smallest_jobs_sum",
            )

    if q is not None:
        model.addConstrs(
            (
                q[machine_index, job_index] == 0
                for machine_index in machines
                for job_index in machines
                if machine_index > job_index
            ),
            name="mtf_init_order",
        )
    if case.mtf_profile is None:
        model.addConstrs(
            (
                2 <= gp.quicksum(q[machine_index, job_index] for job_index in truncated_jobs)
                for machine_index in machines
            ),
            name="mtf_cardinality_lb",
        )
        model.addConstrs(
            (
                gp.quicksum(q[machine_index, job_index] for job_index in truncated_jobs) <= mtf_cardinality_upper
                for machine_index in machines
            ),
            name="mtf_cardinality_ub",
        )
    if s is not None:
        model.addConstrs(
            (
                gp.quicksum(s[machine_index, job_index - 1] for machine_index in machines) + p[job_index]
                == gp.quicksum(s[machine_index, job_index] for machine_index in machines)
                for job_index in range(2, case.job_count)
            ),
            name="mtf_balance",
        )

    if case.solver.legacy_best_bd_stop_at_target and case.enforce_target_lower_bound:
        # Legacy option carried over from the old implementation.
        model.Params.BestBdStop = target + BOUND_STOP_TOLERANCE


def _build_opt_machine_groups(
    case: ExperimentCase, machines: range
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Split OPT machines into consecutive 3-, 4-, and 5-job groups."""

    # Helper used by case-specific Section 5 constraints to interpret the OPT
    # profile as consecutive 3-job / 4-job / 5-job machine groups.
    if case.opt_profile is None:
        return (), (), ()

    machine_ids = tuple(machines)
    s3_end = case.opt_profile.nS3
    s4_end = s3_end + case.opt_profile.nS4
    return (
        machine_ids[:s3_end],
        machine_ids[s3_end:s4_end],
        machine_ids[s4_end : s4_end + case.opt_profile.nS5],
    )


def _machine_after_pair_block(layout: MtfProfileLayout) -> int:
    """Return the first machine after the F1/R2/F2 pairing prefix."""

    if layout.f2_machines:
        return layout.f2_machines[-1] + 1
    if layout.r2_machines:
        return layout.r2_machines[-1] + 1
    if layout.f1_machines:
        return layout.f1_machines[-1] + 1
    return 1


def _apply_r5_constraints(
    model: GurobiModel,
    case: ExperimentCase,
    p: PVarMap,
    x: TupleVarMap,
    machines: range,
    target: float,
    layout: MtfProfileLayout,
) -> None:
    """Add the shared Case 3 constraints for the 5-job MTF tail."""

    # Shared helper for the Case 3 branches when regular 5-job MTF machines are present.
    profile = case.mtf_profile
    if profile is None or not layout.r5_machines:
        return

    model.addConstr(
        p[layout.e5 + 5 * profile.nR5 - 5]
        + p[layout.e5 + 5 * profile.nR5 - 4]
        + p[layout.e5 + 5 * profile.nR5 - 3]
        + p[layout.e5 + 5 * profile.nR5 - 2]
        + p[layout.e5 + 5 * profile.nR5 - 1]
        + p[case.job_count]
        >= target,
        name=f"case34_R5_valid_constr[{layout.r5_machines[-1]}]",
    )

    proc_same_range = range(layout.t5 + 1, layout.e5 + 5 * profile.nR5 - 5)

    for job_index in proc_same_range:
        # As in R2/R3/R4, regular interior jobs can be collapsed to equal sizes.
        model.addConstr(
            p[job_index] == p[job_index + 1],
            name=f"case34_R5_processing_times[{job_index}]",
        )
        model.addConstrs(
            (
                gp.quicksum(x[machine_prime, job_index] for machine_prime in range(1, machine_index + 1))
                >= x[machine_index, job_index + 1]
                for machine_index in machines
            ),
            name=f"case34_R5_symmetry_break_by_proc[{job_index}]",
        )


def _apply_case_profile_constraints(
    model: GurobiModel,
    case: ExperimentCase,
    p: PVarMap,
    x: TupleVarMap,
    q: TupleVarMap,
    jobs: range,
    truncated_jobs: range,
    machines: range,
    target: float,
    layout: MtfProfileLayout,
) -> None:
    """Add the case-specific structural cuts implied by the chosen branch.

    This is the densest part of the file. Read it as:
    1. Case 1 block
    2. Case 2 block
    3. Shared Case 3 prefix
    4. Case 3-1 block
    5. Case 3-2 block
    """

    # This is the main Section 5 branch-specific encoding. Compare its four big
    # branches with the paper's Case 1, Case 2, Case 3-1, and Case 3-2 results.
    # If you suspect a mismatch with the paper, this is usually the first place
    # to inspect after checking the iterators in `branching.py`.
    if case.opt_profile is None or case.mtf_profile is None or case.ell is None:
        return

    profile = case.mtf_profile
    ell = case.ell
    machine_ids = tuple(machines)
    job_ids = tuple(jobs)
    s3_machines, _, s5_machines = _build_opt_machine_groups(case, machines)

    if case.acceleration_case is AccelerationCase.CASE_1:
        # Case 1: p_n >= 11/51.
        # Here the structure is the cleanest: no F1, long jobs are paired in
        # R2/F2, and the first R3 machine is anchored at ell, ell+1, ell+2.
        model.addConstrs(
            (
                x[machine_index, job_index] == 0
                for machine_index in machines
                for job_index in jobs
                if machine_index > job_index
            ),
            name="case1_opt_init",
        )
        model.addConstrs(
            (
                p[job_index] <= _as_float(Fraction(14, 17)) - 2 * p[case.job_count]
                for job_index in range(ell, case.job_count + 1)
            ),
            name="case1_stronger_proc_time_in_D",
        )
        model.addConstrs(
            (x[machine_index, machine_index] == 1 for machine_index in machine_ids[: ell - 1]),
            name="case1_OPT_ell_assignment_constr",
        )
        for machine_index in layout.r2_machines:
            model.addConstr(
                q[machine_index, 2 * machine_index - 1] == 1,
                name=f"case1_R2_consec_1[{machine_index}]",
            )
            model.addConstr(
                q[machine_index, 2 * machine_index] == 1,
                name=f"case1_R2_consec_2[{machine_index}]",
            )
        for machine_index in layout.f2_machines:
            model.addConstr(
                q[machine_index, 2 * machine_index - 1] == 1,
                name=f"case1_F2_consec_1[{machine_index}]",
            )
            model.addConstr(
                q[machine_index, 2 * machine_index] == 1,
                name=f"case1_F2_consec_2[{machine_index}]",
            )
        if layout.f2_machines:
            # The last F2 pair blocks the next short job plus n.
            model.addConstr(
                p[ell - 2] + p[ell - 1] + p[ell + 2] >= target,
                name="case1_F2_valid_constr",
            )
        if layout.r3_machines:
            # The first R3 machine is fixed to the first three short jobs.
            first_r3 = layout.r3_machines[0]
            model.addConstr(q[first_r3, ell] == 1, name=f"case1_R3_consec_1[{first_r3}]")
            model.addConstr(q[first_r3, ell + 1] == 1, name=f"case1_R3_consec_2[{first_r3}]")
            model.addConstr(q[first_r3, ell + 2] == 1, name=f"case1_R3_consec_3[{first_r3}]")
        return

    if case.acceleration_case is AccelerationCase.CASE_2:
        # Case 2: 7/34 <= p_n < 11/51.
        # This largely mirrors Case 1, except the odd/even behavior of ell
        # changes and one OPT machine may contain two long jobs.
        model.addConstrs(
            (
                x[machine_index, job_index] == 0
                for machine_index in machines
                for job_index in jobs
                if machine_index > job_index
            ),
            name="case2_opt_init",
        )
        model.addConstrs(
            (x[machine_index, machine_index] == 1 for machine_index in s3_machines),
            name="case2_OPT_ell_assignment_constr",
        )
        return

    if case.acceleration_case is AccelerationCase.CASE_3:
        s3_prefix_count = 2 * (profile.nF1 + profile.nR2) - 1
        if s3_prefix_count > 0:
            model.addConstrs(
                (
                    gp.quicksum(x[machine_index, job_index] for job_index in jobs) == 3
                    for machine_index in machine_ids[:s3_prefix_count]
                ),
                name="case3_always_S3_constr",
            )
            model.addConstrs(
                (x[machine_index, machine_index] == 1 for machine_index in machine_ids[:s3_prefix_count]),
                name="case3_prefix_diag_constr",
            )
        return

    prefix_end = layout.e4 + 4 * profile.nR4 - 4
    # From here on we are in Case 3-1 or Case 3-2.
    # First add the structural facts shared by both subcases.
    if s5_machines and prefix_end >= 1:
        model.addConstrs(
            (
                x[machine_index, job_index] == 0
                for machine_index in s5_machines
                for job_index in range(1, prefix_end + 1)
            ),
            name="case34_OPT_no5_constr",
        )

    s3_prefix_count = 2 * (profile.nF1 + profile.nR2) - 1
    if s3_prefix_count > 0:
        # Early OPT machines are forced to be 3-job machines by the Case 3 prefix
        # structure proven in the paper.
        model.addConstrs(
            (
                gp.quicksum(x[machine_index, job_index] for job_index in jobs) == 3
                for machine_index in machine_ids[:s3_prefix_count]
            ),
            name="case34_always_S3_constr",
        )

    model.addConstrs(
        (q[machine_index, machine_index + profile.nF1 + 1] == 1 for machine_index in layout.f1_machines),
        name="case34_F1_consec_constrs",
    )
    # After the F1 fallback block, the next diagonal job is also fixed.
    if profile.nF1 + 1 <= case.machine_count and 2 * (profile.nF1 + 1) in truncated_jobs:
        model.addConstr(
            q[profile.nF1 + 1, 2 * (profile.nF1 + 1)] == 1,
            name="case34_F1_assignment_constr",
        )

    for machine_index in layout.r2_machines:
        # Both shared Case 3 subcases keep the consecutive pairing behavior on R2.
        model.addConstr(
            q[machine_index, 2 * machine_index - 1] == 1,
            name=f"case34_R2_consec_1[{machine_index}]",
        )
        model.addConstr(
            q[machine_index, 2 * machine_index] == 1,
            name=f"case34_R2_consec_2[{machine_index}]",
        )

    for machine_index in layout.f2_machines:
        # Both shared Case 3 subcases also keep consecutive pairing on F2.
        model.addConstr(
            q[machine_index, 2 * machine_index - 1] == 1,
            name=f"case34_F2_consec_1[{machine_index}]",
        )
        model.addConstr(
            q[machine_index, 2 * machine_index] == 1,
            name=f"case34_F2_consec_2[{machine_index}]",
        )

    if case.acceleration_case is AccelerationCase.CASE_3_1:
        # Case 3-1: ell is NOT a fallback job on an F2 machine.
        if prefix_end >= 1:
            # Early OPT assignments keep the diagonal/sorted initialization.
            model.addConstrs(
                (
                    x[machine_index, job_index] == 0
                    for machine_index in machines
                    for job_index in range(1, prefix_end + 1)
                    if machine_index > job_index
                ),
                name="case3_valid_ineq_OPT_init_constr",
            )

        diag_count = 2 * (profile.nF1 + profile.nR2 + profile.nF2)
        # The whole prefix up through F2 is diagonally anchored in OPT.
        model.addConstrs(
            (x[machine_index, machine_index] == 1 for machine_index in machine_ids[:diag_count]),
            name="case3_OPT_ell_assignment_constr",
        )

        for machine_index in machine_ids[
            2 * (profile.nF1 + profile.nR2) - 1 : 2 * (profile.nF1 + profile.nR2 + profile.nF2)
        ]:
            # Machines near the F2 boundary cannot exceed 4 jobs in OPT.
            model.addConstr(
                gp.quicksum(x[machine_index, job_index] for job_index in jobs) <= 4,
                name=f"case3_always_less_S4_constr[{machine_index}]",
            )

        for machine_index in machine_ids[2 * (profile.nF1 + profile.nR2 + profile.nF2) :]:
            # Later OPT machines must be 4+ cardinality.
            model.addConstr(
                gp.quicksum(x[machine_index, job_index] for job_index in jobs) >= 4,
                name=f"case3_always_greater_S4_constr[{machine_index}]",
            )

        if ell - 3 == 2 * (profile.nF1 + profile.nR2 + profile.nF2):
            # Special sub-branch where ell starts immediately after the pair block.
            next_machine = _machine_after_pair_block(layout)
            model.addConstr(p[ell] <= case.machine_count / 34, name="case3_ell_proc_time")
            model.addConstr(q[next_machine, ell - 2] == 1, name="case3_ell_consec_1")
            model.addConstr(q[next_machine, ell - 1] == 1, name="case3_ell_consec_2")
            model.addConstr(q[next_machine, ell] == 1, name="case3_ell_consec_3")

        _apply_r5_constraints(
            model,
            case,
            p,
            x,
            machines,
            target,
            layout,
        )
        return

    diag_count = 2 * (profile.nF1 + profile.nR2)
    # Case 3-2: ell IS a fallback job on F2.
    # OPT is anchored only through the F1/R2 prefix; the rest is constrained by
    # consecutive structure and cardinality restrictions around F2.
    model.addConstrs(
        (x[machine_index, machine_index] == 1 for machine_index in machine_ids[:diag_count]),
        name="case4_OPT_ell_assignment_constr",
    )

    if case.machine_count <= diag_count and diag_count >= 1 and diag_count + 1 <= case.job_count:
        model.addConstr(
            x[diag_count, diag_count + 1] == 1,
            name="case4_first_F2_job_with_R2_last_or_next",
        )
    elif profile.nF1 == 0 and profile.nR2 == 0:
        model.addConstr(
            x[1, 1] == 1,
            name="case4_first_F2_job_with_R2_last_or_next",
        )
    elif diag_count >= 1 and diag_count + 1 <= case.machine_count and diag_count + 1 <= case.job_count:
        model.addConstr(
            x[diag_count, diag_count + 1] + x[diag_count + 1, diag_count + 1] == 1,
            name="case4_first_F2_job_with_R2_last_or_next",
        )

    model.addConstrs(
        (
            x[machine_index + 1, job_index + 1] >= x[machine_index, job_index]
            for machine_index in machines
            for job_index in jobs
            if (
                2 * (profile.nF1 + profile.nR2) <= machine_index <= 2 * (profile.nF1 + profile.nR2 + profile.nF2) - 1
                and 2 * (profile.nF1 + profile.nR2) + 1
                <= job_index
                <= 2 * (profile.nF1 + profile.nR2 + profile.nF2) - 1
                and machine_index + 1 in machine_ids
                and job_index + 1 in job_ids
            )
        ),
        name="case4_F2_consecutive_jobs_in_OPT",
    )
    # The monotonicity above forces the consecutive block structure on OPT jobs
    # associated with the F2 region.

    if diag_count >= 1 and diag_count in machine_ids:
        # The last machine before the F2 block cannot still be a 5-job OPT machine.
        model.addConstr(
            gp.quicksum(x[diag_count, job_index] for job_index in jobs) <= 4,
            name="case4_F2_last_job_S3_or_S4",
        )

    for machine_index in machine_ids[diag_count:]:
        # Everything after that point must have OPT cardinality at least 4.
        model.addConstr(
            gp.quicksum(x[machine_index, job_index] for job_index in jobs) >= 4,
            name=f"case4_F2_always_greater_S4_constr[{machine_index}]",
        )

    if layout.f1_machines:
        # Same-processing-time WLOG for the regular and fallback jobs of F1.
        model.addConstr(
            p[profile.nF1] + p[profile.nF1 + 1] >= target,
            name="case4_F1_valid_constr",
        )
        for job_index in range(1, profile.nF1):
            model.addConstr(
                p[job_index] == p[job_index + 1],
                name=f"case4_F1_processing_times[{job_index}]",
            )

    e3 = layout.e2 + 2 * (profile.nR2 + profile.nF2)

    model.addConstr(
        p[e3 - 2] + p[e3 - 1] + p[e3] >= target,
        name="case4_F2_valid_constr",
    )
    # Same-processing-time WLOG for the regular jobs in the F2 region.
    for job_index in range(layout.e2 + 2 * profile.nR2, e3 - 2):
        model.addConstr(
            p[job_index] == p[job_index + 1],
            name=f"case4_F2_processing_times[{job_index}]",
        )

    model.addConstrs(
        (q[layout.f2_machines[offset], ell + offset] == 1 for offset in range(profile.nF2)),
        name="case4_ell_in_F2_consec",
    )
    # In Case 3-2, ell and the following fallback jobs march consecutively
    # through the F2 machines.

    next_machine = layout.f2_machines[-1] + 1
    if ell == 2 * (profile.nF1 + profile.nR2 + profile.nF2) + 2:
        # One possible alignment of ell relative to the first post-F2 machine.
        model.addConstr(q[next_machine, ell - 1] == 1, name="case4_ell_consec_1")
        model.addConstr(q[next_machine, ell + profile.nF2] == 1, name="case4_ell_consec_2")
        model.addConstr(q[next_machine, ell + profile.nF2 + 1] == 1, name="case4_ell_consec_3")
    elif ell == 2 * (profile.nF1 + profile.nR2 + profile.nF2) + 3:
        # The other possible alignment of ell relative to the first post-F2 machine.
        model.addConstr(q[next_machine, ell - 2] == 1, name="case4_ell_consec_1")
        model.addConstr(q[next_machine, ell - 1] == 1, name="case4_ell_consec_2")
        model.addConstr(q[next_machine, ell + profile.nF2] == 1, name="case4_ell_consec_3")

    _apply_r5_constraints(
        model,
        case,
        p,
        x,
        machines,
        target,
        layout,
    )


def build_obv_model(case: ExperimentCase) -> BuiltObvModel:
    """Build the full OBV model for one fully specified experiment case."""

    _require_gurobi()

    # Coarse spec counts used for sanity checks. Useful, but not a substitute
    # for auditing the actual named constraints below.
    dimensions = derive_obv_dimensions(
        case.machine_count,
        case.job_count,
        include_target_lower_bound=case.enforce_target_lower_bound,
        acceleration_case=case.acceleration_case,
        include_profile_cardinality_constraints=(case.mtf_profile is not None or case.opt_profile is not None),
    )

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", case.solver.output_flag)
    env.setParam("LogToConsole", case.solver.output_flag)
    env.setParam("NonConvex", case.solver.non_convex)
    if case.solver.time_limit_seconds is not None:
        env.setParam("TimeLimit", case.solver.time_limit_seconds)
    if case.solver.mip_gap is not None:
        env.setParam("MIPGap", case.solver.mip_gap)
    if case.solver.threads is not None:
        env.setParam("Threads", case.solver.threads)
    if case.solver.presolve is not None:
        env.setParam("Presolve", case.solver.presolve)
    env.start()

    model = gp.Model(f"obv_{case.case_id}", env=env)
    setattr(model, "_multifit_env", env)

    machines = range(1, case.machine_count + 1)
    jobs = range(1, case.job_count + 1)
    truncated_jobs = range(1, case.job_count)
    target = _as_float(case.target_ratio)
    processing_time_lower_bound = _processing_time_lower_bound(case)
    use_exact_mtf = _use_exact_mtf(case)

    # Variable creation: compare these domains with the paper's variable
    # definitions, keeping in mind that the bounds are solver-strengthening
    # choices and may be tighter than the bare formulation.
    p: PVarMap = {}
    for job_index in jobs:
        job_lower_bound = processing_time_lower_bound
        job_upper_bound = _processing_time_upper_bound(case, job_index, processing_time_lower_bound)
        job_lower_bound, job_upper_bound = _tighten_processing_time_bounds_by_split(
            case,
            job_index,
            job_lower_bound,
            job_upper_bound,
        )
        p[job_index] = model.addVar(
            lb=_as_float(job_lower_bound),
            ub=_as_float(job_upper_bound),
            vtype=GRB.CONTINUOUS,
            name=f"p[{job_index}]",
        )
    x = model.addVars(machines, jobs, vtype=GRB.BINARY, name="x")
    q: TupleVarMap | None = None
    s: TupleVarMap | None = None
    if not use_exact_mtf:
        q = model.addVars(machines, truncated_jobs, vtype=GRB.BINARY, name="q")
        s = model.addVars(machines, truncated_jobs, lb=0.0, vtype=GRB.CONTINUOUS, name="s")
    z_var = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="Z")

    # Section 4 base MIQP constraints start here.
    model.addConstrs(
        # Jobs are globally sorted by processing time.
        (p[j] >= p[j + 1] for j in truncated_jobs),
        name="sorting",
    )
    model.addConstrs(
        # Every job is assigned to exactly one OPT machine.
        (gp.quicksum(x[i, j] for i in machines) == 1 for j in jobs),
        name="opt_assign",
    )
    model.addConstrs(
        # OPT makespan is normalized to 1.
        (gp.quicksum(p[j] * x[i, j] for j in jobs) <= 1.0 for i in machines),
        name="opt_makespan",
    )

    if not use_exact_mtf:
        assert q is not None and s is not None
        model.addConstrs(
            # Every job except n is assigned in the simulated MTF schedule.
            (gp.quicksum(q[i, j] for i in machines) == 1 for j in truncated_jobs),
            name="mtf_assign",
        )
        model.addConstrs(
            # Initialization of machine partial sums in MTF.
            (s[i, 1] == q[i, 1] * p[1] for i in machines),
            name="mtf_init",
        )
        model.addConstrs(
            (
                # Incremental update of each machine load under MTF.
                s[i, j] - s[i, j - 1] == q[i, j] * p[j]
                for i in machines
                for j in range(2, case.job_count)
            ),
            name="mtf_contribution",
        )
        model.addConstrs(
            # All truncated MTF loads must stay within the target ratio.
            (s[i, j] <= target for i in machines for j in truncated_jobs),
            name="mtf_feasible",
        )
        model.addConstrs(
            (
                # If job j is not assigned to any machine up to i, then machine i
                # must already be too full to receive it.
                s[i, j - 1] + p[j] >= target * (1.0 - gp.quicksum(q[i_prime, j] for i_prime in range(1, i + 1)))
                for i in machines
                for j in range(2, case.job_count)
            ),
            name="mtf_logic",
        )
        model.addConstrs(
            # Objective value z_var is the minimum post-n overload across machines.
            (s[i, case.job_count - 1] + p[case.job_count] >= z_var for i in machines),
            name="mtf_objective",
        )

    # Global strengthenings used in all runs. Audit separately from the pure
    # base MIQP if you want to know what is paper-essential vs solver-helpful.
    _apply_global_valid_inequalities(
        model,
        case,
        p,
        x,
        q,
        s,
        jobs,
        truncated_jobs,
        machines,
        target,
    )

    if case.enforce_target_lower_bound:
        # In verification mode we only care about candidate counterexamples with
        # objective value at least the claimed target ratio.
        model.addConstr(z_var >= target, name="target_lb")

    # Section 5 common case split conditions.
    if case.acceleration_case is not AccelerationCase.BASE:
        _validate_paper_acceleration_case(case)
        # _apply_paper_acceleration_constraints(model, case, p, x, q, jobs, truncated_jobs, machines)

    # Section 5 profile- and branch-specific structural constraints.
    if case.mtf_profile is not None or case.opt_profile is not None:
        _apply_profile_cardinality_constraints(
            model,
            case,
            p,
            x,
            z_var,
            q,
            jobs,
            truncated_jobs,
            machines,
            target,
        )

    model.setObjective(z_var, GRB.MAXIMIZE)
    # Maximizing z_var asks whether there exists an instance whose final failed
    # MTF placement would exceed the claimed target ratio.
    model.update()
    return BuiltObvModel(model=model, dimensions=dimensions)


def _apply_exact_mtf_assignments(
    model: GurobiModel,
    case: ExperimentCase,
    q: TupleVarMap,
    layout: MtfProfileLayout,
) -> dict[int, tuple[int, ...]]:
    """Fix the full exact MTF assignment once fallback starts are branched."""

    assignment = _build_exact_mtf_assignment(case, layout)
    for machine_index, machine_jobs in assignment.items():
        for offset, job_index in enumerate(machine_jobs, start=1):
            model.addConstr(
                q[machine_index, job_index] == 1,
                name=f"exact_q[{machine_index},{offset}]",
            )
    return assignment


def _build_exact_mtf_assignment(
    case: ExperimentCase,
    layout: MtfProfileLayout,
) -> dict[int, tuple[int, ...]]:
    """Reconstruct the fully determined exact MTF schedule from fallback starts."""

    profile = case.mtf_profile
    starts = case.fallback_starts
    if profile is None or starts is None:
        return {}

    scheduled_job_count = profile.scheduled_job_count
    reserved_jobs = set()

    def fallback_block(start: int | None, count: int) -> list[int]:
        if start is None or count == 0:
            return []
        values = list(range(start, start + count))
        reserved_jobs.update(values)
        return values

    f2_fallback_jobs = fallback_block(starts.s2, profile.nF2)
    f3_fallback_jobs = fallback_block(starts.s3, profile.nF3)
    f4_fallback_jobs = fallback_block(starts.s4, profile.nF4)
    assignment: dict[int, tuple[int, ...]] = {}
    f2_cursor = 0
    f3_cursor = 0
    f4_cursor = 0

    if case.acceleration_case is AccelerationCase.CASE_3:
        for machine_index in layout.f1_machines:
            f1_jobs = (machine_index, profile.nF1 + machine_index + 1)
            assignment[machine_index] = f1_jobs
            reserved_jobs.update(f1_jobs)

    regular_jobs = [job_index for job_index in range(1, scheduled_job_count + 1) if job_index not in reserved_jobs]
    regular_cursor = 0

    def regular_block(regular_count: int) -> tuple[int, ...]:
        nonlocal regular_cursor
        machine_jobs = tuple(regular_jobs[regular_cursor : regular_cursor + regular_count])
        regular_cursor += regular_count
        return machine_jobs

    for machine_index in layout.r2_machines:
        assignment[machine_index] = regular_block(2)

    for machine_index in layout.f2_machines:
        assignment[machine_index] = regular_block(2) + (f2_fallback_jobs[f2_cursor],)
        f2_cursor += 1

    for machine_index in layout.r3_machines:
        assignment[machine_index] = regular_block(3)

    for machine_index in layout.f3_machines:
        assignment[machine_index] = regular_block(3) + (f3_fallback_jobs[f3_cursor],)
        f3_cursor += 1

    for machine_index in layout.r4_machines:
        assignment[machine_index] = regular_block(4)

    for machine_index in layout.f4_machines:
        assignment[machine_index] = regular_block(4) + (f4_fallback_jobs[f4_cursor],)
        f4_cursor += 1

    for machine_index in layout.r5_machines:
        assignment[machine_index] = regular_block(5)

    return assignment


def _apply_exact_mtf_constraints(
    model: GurobiModel,
    case: ExperimentCase,
    p: PVarMap,
    x: TupleVarMap,
    z_var: GurobiVar,
    machines: range,
    target: float,
    layout: MtfProfileLayout,
) -> None:
    """Add exact MTF constraints once fallback starts determine the schedule."""

    profile = case.mtf_profile
    if profile is None or case.fallback_starts is None:
        return

    assignment = _build_exact_mtf_assignment(case, layout)
    fallback_machines = set(layout.f2_machines) | set(layout.f3_machines) | set(layout.f4_machines)

    # Every machine's load + p_n >= target.
    for machine_index, machine_jobs in assignment.items():
        model.addConstr(
            gp.quicksum(p[j] for j in machine_jobs) + p[case.job_count] >= z_var,
            name=f"exact_mtf_objective[{machine_index}]",
        )

    # Fallback machines: regular prefix + next sequential job >= target
    # (explains why the next job didn't fit and a fallback was placed instead).
    for machine_index in fallback_machines:
        machine_jobs = assignment[machine_index]
        regular_jobs = machine_jobs[:-1]
        next_job = regular_jobs[-1] + 1
        model.addConstr(
            gp.quicksum(p[j] for j in regular_jobs) + p[next_job] >= target,
            name=f"exact_mtf_fallback[{machine_index}]",
        )

    def add_rk_equal_processing(
        machine_block: tuple[int, ...],
        *,
        k: int,
        name_prefix: str,
    ) -> None:
        rk_jobs = sorted(job_index for machine_index in machine_block for job_index in assignment[machine_index])
        if len(rk_jobs) <= k - 1:
            return
        equal_jobs = rk_jobs[: len(rk_jobs) - (k - 1)]
        if len(equal_jobs) <= 1:
            return

        segment_start = 0
        while segment_start < len(equal_jobs):
            segment_end = segment_start + 1
            while segment_end < len(equal_jobs) and equal_jobs[segment_end] == equal_jobs[segment_end - 1] + 1:
                segment_end += 1

            segment = equal_jobs[segment_start:segment_end]
            for left_job, right_job in zip(segment, segment[1:]):
                model.addConstr(
                    p[left_job] == p[right_job],
                    name=f"{name_prefix}_processing_times[{left_job}]",
                )
                model.addConstrs(
                    (
                        gp.quicksum(x[machine_prime, left_job] for machine_prime in range(1, machine_index + 1))
                        >= x[machine_index, right_job]
                        for machine_index in machines
                    ),
                    name=f"{name_prefix}_symmetry_break_by_proc[{left_job}]",
                )

            segment_start = segment_end

    def add_fk_regular_equal_processing(
        machine_block: tuple[int, ...],
        *,
        k: int,
        name_prefix: str,
    ) -> None:
        fk_regular_jobs = sorted(
            job_index for machine_index in machine_block for job_index in assignment[machine_index][:k]
        )
        if len(fk_regular_jobs) <= k - 1:
            return
        equal_jobs = fk_regular_jobs[: len(fk_regular_jobs) - (k - 1)]
        if len(equal_jobs) <= 1:
            return

        segment_start = 0
        while segment_start < len(equal_jobs):
            segment_end = segment_start + 1
            while segment_end < len(equal_jobs) and equal_jobs[segment_end] == equal_jobs[segment_end - 1] + 1:
                segment_end += 1

            segment = equal_jobs[segment_start:segment_end]
            for left_job, right_job in zip(segment, segment[1:]):
                model.addConstr(
                    p[left_job] == p[right_job],
                    name=f"{name_prefix}_processing_times[{left_job}]",
                )
                model.addConstrs(
                    (
                        gp.quicksum(x[machine_prime, left_job] for machine_prime in range(1, machine_index + 1))
                        >= x[machine_index, right_job]
                        for machine_index in machines
                    ),
                    name=f"{name_prefix}_symmetry_break_by_proc[{left_job}]",
                )

            segment_start = segment_end

    def add_fallback_block_equal_processing(
        fallback_jobs: tuple[int, ...],
        *,
        name_prefix: str,
    ) -> None:
        for left_job, right_job in zip(fallback_jobs, fallback_jobs[1:]):
            model.addConstr(
                p[left_job] == p[right_job],
                name=f"{name_prefix}_processing_times[{left_job}]",
            )
            model.addConstrs(
                (
                    gp.quicksum(x[machine_prime, left_job] for machine_prime in range(1, machine_index + 1))
                    >= x[machine_index, right_job]
                    for machine_index in machines
                ),
                name=f"{name_prefix}_symmetry_break_by_proc[{left_job}]",
            )

    if case.acceleration_case is AccelerationCase.CASE_3 and layout.f1_machines:
        add_fallback_block_equal_processing(
            tuple(assignment[machine_index][0] for machine_index in layout.f1_machines),
            name_prefix="F1",
        )
        add_fallback_block_equal_processing(
            tuple(assignment[machine_index][1] for machine_index in layout.f1_machines),
            name_prefix="F1_fallback",
        )

    if layout.r2_machines:
        add_rk_equal_processing(layout.r2_machines, k=2, name_prefix="R2")

    if layout.f2_machines:
        add_fk_regular_equal_processing(layout.f2_machines, k=2, name_prefix="F2")
        add_fallback_block_equal_processing(
            tuple(job_index for machine_index in layout.f2_machines for job_index in assignment[machine_index][2:]),
            name_prefix="F2_fallback",
        )

    if layout.r3_machines:
        add_rk_equal_processing(layout.r3_machines, k=3, name_prefix="R3")

    if layout.f3_machines:
        add_fk_regular_equal_processing(layout.f3_machines, k=3, name_prefix="F3")
        add_fallback_block_equal_processing(
            tuple(job_index for machine_index in layout.f3_machines for job_index in assignment[machine_index][3:]),
            name_prefix="F3_fallback",
        )

    if layout.r4_machines:
        add_rk_equal_processing(layout.r4_machines, k=4, name_prefix="R4")

    if layout.f4_machines:
        add_fk_regular_equal_processing(layout.f4_machines, k=4, name_prefix="F4")
        add_fallback_block_equal_processing(
            tuple(job_index for machine_index in layout.f4_machines for job_index in assignment[machine_index][4:]),
            name_prefix="F4_fallback",
        )

    if layout.r5_machines:
        add_rk_equal_processing(layout.r5_machines, k=5, name_prefix="R5")
