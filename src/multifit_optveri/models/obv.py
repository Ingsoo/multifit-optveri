from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from multifit_optveri.acceleration import (
    AccelerationCase,
    PAPER_MACHINE_RANGE,
    PAPER_TARGET_RATIO,
    paper_common_pn_lower_bound,
)
from multifit_optveri.experiments import ExperimentCase
from multifit_optveri.math_utils import ceil_fraction
from multifit_optveri.models.spec import ObvModelDimensions, derive_obv_dimensions

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


class GurobiUnavailableError(RuntimeError):
    """Raised when a solve/build action requires gurobipy but it is unavailable."""


OPT_JOB_CARDINALITY_LOWER_BOUND = 3
BOUND_STOP_TOLERANCE = 1e-6


@dataclass
class BuiltObvModel:
    model: GurobiModel
    dimensions: ObvModelDimensions


@dataclass(frozen=True)
class MtfProfileLayout:
    f1_machines: tuple[int, ...]
    r2_machines: tuple[int, ...]
    f2_machines: tuple[int, ...]
    r3_machines: tuple[int, ...]
    f3_machines: tuple[int, ...]
    r4_machines: tuple[int, ...]
    m5_machines: tuple[int, ...]
    e2: int
    e3: int
    e4: int
    t2: int
    t3: int
    t4: int


def _require_gurobi() -> None:
    if gp is None or GRB is None:
        raise GurobiUnavailableError(
            "gurobipy is not available. Install it and activate a Gurobi license to run models."
        )


def _as_float(value: Fraction) -> float:
    return float(value.numerator / value.denominator)


def _common_processing_time_lower_bound(
    machine_count: int, target_ratio: Fraction
) -> Fraction:
    return (target_ratio - 1) * machine_count / (machine_count - 1)


def _processing_time_lower_bound(case: ExperimentCase) -> Fraction:
    lower_bound = _common_processing_time_lower_bound(
        case.machine_count, case.target_ratio
    )
    if case.acceleration_case is not AccelerationCase.BASE:
        pn_lower_bound = case.acceleration_case.pn_range.lower
        if pn_lower_bound is not None:
            lower_bound = max(lower_bound, pn_lower_bound)
    return lower_bound


def _processing_time_upper_bound(
    case: ExperimentCase, job_index: int, lower_bound: Fraction
) -> Fraction:
    upper_bound = Fraction(1, ((job_index - 1) // case.machine_count) + 1)
    upper_bound = min(
        upper_bound,
        Fraction(1, 1) - (OPT_JOB_CARDINALITY_LOWER_BOUND - 1) * lower_bound,
    )
    if (
        job_index == case.job_count
        and case.acceleration_case is not AccelerationCase.BASE
    ):
        pn_upper_bound = case.acceleration_case.pn_range.upper
        if pn_upper_bound is not None:
            upper_bound = min(upper_bound, pn_upper_bound)
    return upper_bound


def _opt_cardinality_upper_bound(lower_bound: Fraction) -> int:
    return ceil_fraction(Fraction(1, 1) / lower_bound) - 1


def _mtf_cardinality_upper_bound(machine_count: int, target_ratio: Fraction) -> int:
    expression = (Fraction(machine_count, 1) - target_ratio) / (
        machine_count * (target_ratio - 1)
    )
    return ceil_fraction(expression) - 1


def _validate_paper_acceleration_case(case: ExperimentCase) -> None:
    if case.target_ratio != PAPER_TARGET_RATIO:
        raise ValueError(
            f"Acceleration case '{case.acceleration_case.value}' is only implemented for "
            f"target={PAPER_TARGET_RATIO.numerator}/{PAPER_TARGET_RATIO.denominator}."
        )
    if case.machine_count not in PAPER_MACHINE_RANGE:
        raise ValueError(
            f"Acceleration case '{case.acceleration_case.value}' is only implemented for "
            f"m in {PAPER_MACHINE_RANGE.start}..{PAPER_MACHINE_RANGE.stop - 1}."
        )


def _apply_paper_acceleration_constraints(
    model: GurobiModel,
    case: ExperimentCase,
    p: PVarMap,
    x: TupleVarMap,
    q: TupleVarMap,
    jobs: range,
    truncated_jobs: range,
    machines: range,
) -> None:
    common_lb = _as_float(paper_common_pn_lower_bound(case.machine_count))

    model.addConstr(p[case.job_count] >= common_lb, name="pn_common_lb")
    model.addConstrs(
        (gp.quicksum(x[i, j] for j in jobs) <= 5 for i in machines),
        name="opt_cardinality",
    )
    model.addConstrs(
        (
            gp.quicksum(x[i, j] for j in jobs) <= gp.quicksum(x[i + 1, j] for j in jobs)
            for i in range(1, case.machine_count)
        ),
        name="opt_cardinality_order",
    )
    model.addConstrs(
        (gp.quicksum(q[i, j] for j in truncated_jobs) <= 5 for i in machines),
        name="mtf_cardinality",
    )

    pn_range = case.acceleration_case.pn_range
    if pn_range.lower is not None:
        model.addConstr(
            p[case.job_count] >= _as_float(pn_range.lower), name="case_pn_lb"
        )
    if pn_range.upper is not None:
        # The paper uses strict upper bounds, but Gurobi only supports non-strict inequalities.
        model.addConstr(
            p[case.job_count] <= _as_float(pn_range.upper), name="case_pn_ub"
        )


def _apply_profile_cardinality_constraints(
    model: GurobiModel,
    case: ExperimentCase,
    p: PVarMap,
    x: TupleVarMap,
    q: TupleVarMap,
    jobs: range,
    truncated_jobs: range,
    machines: range,
    target: float,
) -> None:
    layout: MtfProfileLayout | None = None

    if case.ell is not None:
        for job_index in range(1, case.job_count):
            if job_index < case.ell:
                model.addConstr(
                    p[job_index]
                    >= Fraction(3, 17).numerator / Fraction(3, 17).denominator
                    + p[case.job_count],
                    name=f"processing_time_in_D[{job_index}]",
                )
            else:
                model.addConstr(
                    p[job_index]
                    <= Fraction(3, 17).numerator / Fraction(3, 17).denominator
                    + p[case.job_count],
                    name=f"processing_time_in_D_prime[{job_index}]",
                )

    if case.opt_profile is not None:
        for machine_index, cardinality in enumerate(
            case.opt_profile.machine_cardinalities, start=1
        ):
            model.addConstr(
                gp.quicksum(x[machine_index, j] for j in jobs) == cardinality,
                name=f"opt_profile_cardinality[{machine_index}]",
            )

    if case.mtf_profile is not None:
        layout = _build_mtf_profile_layout(case)
        for machine_index, cardinality in enumerate(
            case.mtf_profile.machine_cardinalities, start=1
        ):
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
    if case.mtf_profile is None:
        raise ValueError("MTF profile layout requires case.mtf_profile.")

    machine_ids = list(range(1, case.machine_count + 1))
    profile = case.mtf_profile
    cursor = 0

    def take(count: int) -> tuple[int, ...]:
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
    m5_machines = take(profile.nM5)

    e2 = profile.nF1 + 1
    e3 = e2 + 2 * (profile.nR2 + profile.nF2)
    e4 = e3 + 3 * (profile.nR3 + profile.nF3)
    t2 = e2 + profile.nF1
    t3 = e3 + profile.nF1 + profile.nF2
    t4 = e4 + profile.nF1 + profile.nF2 + profile.nF3

    return MtfProfileLayout(
        f1_machines=f1_machines,
        r2_machines=r2_machines,
        f2_machines=f2_machines,
        r3_machines=r3_machines,
        f3_machines=f3_machines,
        r4_machines=r4_machines,
        m5_machines=m5_machines,
        e2=e2,
        e3=e3,
        e4=e4,
        t2=t2,
        t3=t3,
        t4=t4,
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
    profile = case.mtf_profile
    if profile is None:
        return

    for machine_index in layout.f1_machines:
        model.addConstr(
            q[machine_index, machine_index] == 1,
            name=f"F1_assignment_constr[{machine_index}]",
        )

    if layout.f1_machines:
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
        model.addConstr(
            p[layout.e2 + 2 * profile.nR2 - 2]
            + p[layout.e2 + 2 * profile.nR2 - 1]
            + p[case.job_count]
            >= target,
            name=f"R2_valid_constr[{layout.r2_machines[-1]}]",
        )
        if profile.nF1 == 0:
            proc_same_range = range(layout.t2, layout.e2 + 2 * profile.nR2 - 2)
        else:
            proc_same_range = range(layout.t2 + 1, layout.e2 + 2 * profile.nR2 - 2)
        for job_index in proc_same_range:
            model.addConstr(
                p[job_index] == p[job_index + 1],
                name=f"R2_processing_times[{job_index}]",
            )
            model.addConstrs(
                (
                    gp.quicksum(
                        x[machine_prime, job_index]
                        for machine_prime in range(1, machine_index + 1)
                    )
                    >= x[machine_index, job_index + 1]
                    for machine_index in machines
                ),
                name=f"R2_symmetry_break_by_proc[{job_index}]",
            )

    if layout.f2_machines:
        model.addConstr(
            p[layout.e3 - 2] + p[layout.e3 - 1] + p[layout.e3] >= target,
            name=f"F2_valid_constr[{layout.f2_machines[-1]}]",
        )

    if layout.r3_machines:
        model.addConstr(
            p[layout.e3 + 3 * profile.nR3 - 3]
            + p[layout.e3 + 3 * profile.nR3 - 2]
            + p[layout.e3 + 3 * profile.nR3 - 1]
            + p[case.job_count]
            >= target,
            name=f"R3_valid_constr[{layout.r3_machines[-1]}]",
        )
        if profile.nF1 == 0 and profile.nF2 == 0:
            proc_same_range = range(layout.t3, layout.e3 + 3 * profile.nR3 - 3)
        else:
            proc_same_range = range(layout.t3 + 1, layout.e3 + 3 * profile.nR3 - 3)
        for job_index in proc_same_range:
            model.addConstr(
                p[job_index] == p[job_index + 1],
                name=f"R3_processing_times[{job_index}]",
            )
            model.addConstrs(
                (
                    gp.quicksum(
                        x[machine_prime, job_index]
                        for machine_prime in range(1, machine_index + 1)
                    )
                    >= x[machine_index, job_index + 1]
                    for machine_index in machines
                ),
                name=f"R3_symmetry_break_by_proc[{job_index}]",
            )

    if layout.f3_machines:
        model.addConstr(
            p[layout.e4 - 3] + p[layout.e4 - 2] + p[layout.e4 - 1] + p[layout.e4]
            >= target,
            name=f"F3_valid_constr[{layout.f3_machines[-1]}]",
        )

    if layout.r4_machines:
        model.addConstr(
            p[layout.e4 + 4 * profile.nR4 - 4]
            + p[layout.e4 + 4 * profile.nR4 - 3]
            + p[layout.e4 + 4 * profile.nR4 - 2]
            + p[layout.e4 + 4 * profile.nR4 - 1]
            + p[case.job_count]
            >= target,
            name=f"R4_valid_constr[{layout.r4_machines[-1]}]",
        )
        if profile.nF1 == 0 and profile.nF2 == 0 and profile.nF3 == 0:
            proc_same_range = range(layout.t4, layout.e4 + 4 * profile.nR4 - 4)
        else:
            proc_same_range = range(layout.t4 + 1, layout.e4 + 4 * profile.nR4 - 4)
        for job_index in proc_same_range:
            model.addConstr(
                p[job_index] == p[job_index + 1],
                name=f"R4_processing_times[{job_index}]",
            )
            model.addConstrs(
                (
                    gp.quicksum(
                        x[machine_prime, job_index]
                        for machine_prime in range(1, machine_index + 1)
                    )
                    >= x[machine_index, job_index + 1]
                    for machine_index in machines
                ),
                name=f"R4_symmetry_break_by_proc[{job_index}]",
            )

    for machine_index in layout.m5_machines:
        model.addConstr(
            gp.quicksum(q[machine_index, job_index] for job_index in truncated_jobs)
            == 5,
            name=f"M5_cardinality_constr[{machine_index}]",
        )


def _apply_global_valid_inequalities(
    model: GurobiModel,
    case: ExperimentCase,
    p: PVarMap,
    x: TupleVarMap,
    q: TupleVarMap,
    s: TupleVarMap,
    jobs: range,
    truncated_jobs: range,
    machines: range,
    target: float,
) -> None:
    lower_bound = _processing_time_lower_bound(case)
    opt_cardinality_upper = _opt_cardinality_upper_bound(lower_bound)
    mtf_cardinality_upper = _mtf_cardinality_upper_bound(
        case.machine_count, case.target_ratio
    )

    model.addConstrs(
        (
            gp.quicksum(x[machine_index, job_index] for job_index in jobs)
            >= OPT_JOB_CARDINALITY_LOWER_BOUND
            for machine_index in machines
        ),
        name="opt_cardinality_lb",
    )
    model.addConstrs(
        (
            gp.quicksum(x[machine_index, job_index] for job_index in jobs)
            <= opt_cardinality_upper
            for machine_index in machines
        ),
        name="opt_cardinality_ub",
    )

    rem = case.job_count % case.machine_count
    if rem != 0:
        quo = case.job_count // case.machine_count
        first_small_job = case.job_count - rem * (quo + 1) + 1
        model.addConstr(
            gp.quicksum(
                p[job_index] for job_index in range(first_small_job, case.job_count + 1)
            )
            <= rem,
            name="opt_smallest_jobs_sum",
        )

    model.addConstrs(
        (
            q[machine_index, job_index] == 0
            for machine_index in machines
            for job_index in truncated_jobs
            if machine_index > job_index
        ),
        name="mtf_init_order",
    )
    model.addConstr(s[1, 1] == p[1], name="mtf_init_fixed[1]")
    model.addConstrs(
        (
            s[machine_index, 1] == 0
            for machine_index in range(2, case.machine_count + 1)
        ),
        name="mtf_init_fixed",
    )

    model.addConstrs(
        (
            2
            <= gp.quicksum(q[machine_index, job_index] for job_index in truncated_jobs)
            for machine_index in machines
        ),
        name="mtf_cardinality_lb",
    )
    model.addConstrs(
        (
            gp.quicksum(q[machine_index, job_index] for job_index in truncated_jobs)
            <= mtf_cardinality_upper
            for machine_index in machines
        ),
        name="mtf_cardinality_ub",
    )
    model.addConstrs(
        (
            gp.quicksum(s[machine_index, job_index - 1] for machine_index in machines)
            + p[job_index]
            == gp.quicksum(s[machine_index, job_index] for machine_index in machines)
            for job_index in range(2, case.job_count)
        ),
        name="mtf_balance",
    )

    if case.solver.legacy_best_bd_stop_at_target and case.enforce_target_lower_bound:
        model.Params.BestBdStop = target + BOUND_STOP_TOLERANCE


def _build_opt_machine_groups(
    case: ExperimentCase, machines: range
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
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
    *,
    e4: int,
    t4: int,
) -> None:
    profile = case.mtf_profile
    if profile is None or not layout.m5_machines:
        return

    e5 = e4 + 4 * profile.nR4
    t5 = t4 + 4 * profile.nR4
    model.addConstr(
        p[e5 + 5 * profile.nM5 - 5]
        + p[e5 + 5 * profile.nM5 - 4]
        + p[e5 + 5 * profile.nM5 - 3]
        + p[e5 + 5 * profile.nM5 - 2]
        + p[e5 + 5 * profile.nM5 - 1]
        + p[case.job_count]
        >= target,
        name=f"case34_R5_valid_constr[{layout.m5_machines[-1]}]",
    )

    if profile.nF1 == 0 and profile.nF2 == 0 and profile.nF3 == 0:
        proc_same_range = range(t5, e5 + 5 * profile.nM5 - 5)
    else:
        proc_same_range = range(t5 + 1, e5 + 5 * profile.nM5 - 5)

    for job_index in proc_same_range:
        model.addConstr(
            p[job_index] == p[job_index + 1],
            name=f"case34_M5_processing_times[{job_index}]",
        )
        model.addConstrs(
            (
                gp.quicksum(
                    x[machine_prime, job_index]
                    for machine_prime in range(1, machine_index + 1)
                )
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
    if case.opt_profile is None or case.mtf_profile is None or case.ell is None:
        return

    profile = case.mtf_profile
    ell = case.ell
    machine_ids = tuple(machines)
    job_ids = tuple(jobs)
    s3_machines, _, s5_machines = _build_opt_machine_groups(case, machines)

    if case.acceleration_case is AccelerationCase.CASE_1:
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
            (
                x[machine_index, machine_index] == 1
                for machine_index in machine_ids[: ell - 1]
            ),
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
            model.addConstr(
                p[ell - 2] + p[ell - 1] + p[ell + 2] >= target,
                name="case1_F2_valid_constr",
            )
        if layout.r3_machines:
            first_r3 = layout.r3_machines[0]
            model.addConstr(
                q[first_r3, ell] == 1, name=f"case1_R3_consec_1[{first_r3}]"
            )
            model.addConstr(
                q[first_r3, ell + 1] == 1, name=f"case1_R3_consec_2[{first_r3}]"
            )
            model.addConstr(
                q[first_r3, ell + 2] == 1, name=f"case1_R3_consec_3[{first_r3}]"
            )
        return

    if case.acceleration_case is AccelerationCase.CASE_2:
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
        if ell % 2 == 0:
            model.addConstr(
                gp.quicksum(x[machine_index, ell - 1] for machine_index in s3_machines)
                == 1,
                name="case2_even_ell_two_long_machine",
            )
        for machine_index in layout.r2_machines:
            model.addConstr(
                q[machine_index, 2 * machine_index - 1] == 1,
                name=f"case2_R2_consec_1[{machine_index}]",
            )
            model.addConstr(
                q[machine_index, 2 * machine_index] == 1,
                name=f"case2_R2_consec_2[{machine_index}]",
            )
        for machine_index in layout.f2_machines:
            model.addConstr(
                q[machine_index, 2 * machine_index - 1] == 1,
                name=f"case2_F2_consec_1[{machine_index}]",
            )
            model.addConstr(
                q[machine_index, 2 * machine_index] == 1,
                name=f"case2_F2_consec_2[{machine_index}]",
            )
        if layout.f2_machines:
            e3 = case.opt_profile.nS3 + 1
            model.addConstr(
                p[e3 - 1] + p[e3] + p[e3 + 1] >= target,
                name="case2_F2_valid_constr",
            )
        if layout.r3_machines:
            first_r3 = layout.r3_machines[0]
            e3 = case.opt_profile.nS3 + 1
            model.addConstr(q[first_r3, e3] == 1, name=f"case2_R3_consec_1[{first_r3}]")
            model.addConstr(
                q[first_r3, e3 + 1] == 1, name=f"case2_R3_consec_2[{first_r3}]"
            )
            model.addConstr(
                q[first_r3, e3 + 2] == 1, name=f"case2_R3_consec_3[{first_r3}]"
            )
        return

    prefix_end = layout.e4 + 4 * profile.nR4 - 4
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
        model.addConstrs(
            (
                gp.quicksum(x[machine_index, job_index] for job_index in jobs) == 3
                for machine_index in machine_ids[:s3_prefix_count]
            ),
            name="case34_always_S3_constr",
        )

    model.addConstrs(
        (
            q[machine_index, machine_index + profile.nF1 + 1] == 1
            for machine_index in layout.f1_machines
        ),
        name="case34_F1_consec_constrs",
    )
    if (
        profile.nF1 + 1 <= case.machine_count
        and 2 * (profile.nF1 + 1) in truncated_jobs
    ):
        model.addConstr(
            q[profile.nF1 + 1, 2 * (profile.nF1 + 1)] == 1,
            name="case34_F1_assignment_constr",
        )

    for machine_index in layout.r2_machines:
        model.addConstr(
            q[machine_index, 2 * machine_index - 1] == 1,
            name=f"case34_R2_consec_1[{machine_index}]",
        )
        model.addConstr(
            q[machine_index, 2 * machine_index] == 1,
            name=f"case34_R2_consec_2[{machine_index}]",
        )

    for machine_index in layout.f2_machines:
        model.addConstr(
            q[machine_index, 2 * machine_index - 1] == 1,
            name=f"case34_F2_consec_1[{machine_index}]",
        )
        model.addConstr(
            q[machine_index, 2 * machine_index] == 1,
            name=f"case34_F2_consec_2[{machine_index}]",
        )

    if case.acceleration_case is AccelerationCase.CASE_3_1:
        if prefix_end >= 1:
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
        model.addConstrs(
            (
                x[machine_index, machine_index] == 1
                for machine_index in machine_ids[:diag_count]
            ),
            name="case3_OPT_ell_assignment_constr",
        )

        for machine_index in machine_ids[
            2 * (profile.nF1 + profile.nR2)
            - 1 : 2 * (profile.nF1 + profile.nR2 + profile.nF2)
        ]:
            model.addConstr(
                gp.quicksum(x[machine_index, job_index] for job_index in jobs) <= 4,
                name=f"case3_always_less_S4_constr[{machine_index}]",
            )

        for machine_index in machine_ids[
            2 * (profile.nF1 + profile.nR2 + profile.nF2) :
        ]:
            model.addConstr(
                gp.quicksum(x[machine_index, job_index] for job_index in jobs) >= 4,
                name=f"case3_always_greater_S4_constr[{machine_index}]",
            )

        e3 = 1 + 2 * (profile.nF1 + profile.nR2 + profile.nF2)
        e4 = e3 + 3 * (profile.nR3 + profile.nF3)
        t4 = e4 + profile.nF2 + profile.nF3
        if ell - 3 == 2 * (profile.nF1 + profile.nR2 + profile.nF2):
            next_machine = _machine_after_pair_block(layout)
            model.addConstr(
                p[ell] <= case.machine_count / 34, name="case3_ell_proc_time"
            )
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
            e4=e4,
            t4=t4,
        )
        return

    diag_count = 2 * (profile.nF1 + profile.nR2)
    model.addConstrs(
        (
            x[machine_index, machine_index] == 1
            for machine_index in machine_ids[:diag_count]
        ),
        name="case4_OPT_ell_assignment_constr",
    )

    if (
        case.machine_count <= diag_count
        and diag_count >= 1
        and diag_count + 1 <= case.job_count
    ):
        model.addConstr(
            x[diag_count, diag_count + 1] == 1,
            name="case4_first_F2_job_with_R2_last_or_next",
        )
    elif profile.nF1 == 0 and profile.nR2 == 0:
        model.addConstr(
            x[1, 1] == 1,
            name="case4_first_F2_job_with_R2_last_or_next",
        )
    elif (
        diag_count >= 1
        and diag_count + 1 <= case.machine_count
        and diag_count + 1 <= case.job_count
    ):
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
                2 * (profile.nF1 + profile.nR2)
                <= machine_index
                <= 2 * (profile.nF1 + profile.nR2 + profile.nF2) - 1
                and 2 * (profile.nF1 + profile.nR2) + 1
                <= job_index
                <= 2 * (profile.nF1 + profile.nR2 + profile.nF2) - 1
                and machine_index + 1 in machine_ids
                and job_index + 1 in job_ids
            )
        ),
        name="case4_F2_consecutive_jobs_in_OPT",
    )

    if diag_count >= 1 and diag_count in machine_ids:
        model.addConstr(
            gp.quicksum(x[diag_count, job_index] for job_index in jobs) <= 4,
            name="case4_F2_last_job_S3_or_S4",
        )

    for machine_index in machine_ids[diag_count:]:
        model.addConstr(
            gp.quicksum(x[machine_index, job_index] for job_index in jobs) >= 4,
            name=f"case4_F2_always_greater_S4_constr[{machine_index}]",
        )

    if layout.f1_machines:
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
    e4 = e3 + 3 * (profile.nR3 + profile.nF3)
    t4 = e4 + profile.nF3

    model.addConstr(
        p[e3 - 2] + p[e3 - 1] + p[e3] >= target,
        name="case4_F2_valid_constr",
    )
    for job_index in range(layout.e2 + 2 * profile.nR2, e3 - 2):
        model.addConstr(
            p[job_index] == p[job_index + 1],
            name=f"case4_F2_processing_times[{job_index}]",
        )

    model.addConstrs(
        (
            q[layout.f2_machines[offset], ell + offset] == 1
            for offset in range(profile.nF2)
        ),
        name="case4_ell_in_F2_consec",
    )

    next_machine = layout.f2_machines[-1] + 1
    if ell == 2 * (profile.nF1 + profile.nR2 + profile.nF2) + 2:
        model.addConstr(q[next_machine, ell - 1] == 1, name="case4_ell_consec_1")
        model.addConstr(
            q[next_machine, ell + profile.nF2] == 1, name="case4_ell_consec_2"
        )
        model.addConstr(
            q[next_machine, ell + profile.nF2 + 1] == 1, name="case4_ell_consec_3"
        )
    elif ell == 2 * (profile.nF1 + profile.nR2 + profile.nF2) + 3:
        model.addConstr(q[next_machine, ell - 2] == 1, name="case4_ell_consec_1")
        model.addConstr(q[next_machine, ell - 1] == 1, name="case4_ell_consec_2")
        model.addConstr(
            q[next_machine, ell + profile.nF2] == 1, name="case4_ell_consec_3"
        )

    _apply_r5_constraints(
        model,
        case,
        p,
        x,
        machines,
        target,
        layout,
        e4=e4,
        t4=t4,
    )


def build_obv_model(case: ExperimentCase) -> BuiltObvModel:
    _require_gurobi()

    dimensions = derive_obv_dimensions(
        case.machine_count,
        case.job_count,
        include_target_lower_bound=case.enforce_target_lower_bound,
        acceleration_case=case.acceleration_case,
        include_profile_cardinality_constraints=(
            case.mtf_profile is not None or case.opt_profile is not None
        ),
    )

    model = gp.Model(f"obv_{case.case_id}")
    model.Params.NonConvex = case.solver.non_convex
    model.Params.OutputFlag = case.solver.output_flag
    if case.solver.time_limit_seconds is not None:
        model.Params.TimeLimit = case.solver.time_limit_seconds
    if case.solver.mip_gap is not None:
        model.Params.MIPGap = case.solver.mip_gap
    if case.solver.threads is not None:
        model.Params.Threads = case.solver.threads
    if case.solver.presolve is not None:
        model.Params.Presolve = case.solver.presolve

    machines = range(1, case.machine_count + 1)
    jobs = range(1, case.job_count + 1)
    truncated_jobs = range(1, case.job_count)
    target = _as_float(case.target_ratio)
    processing_time_lower_bound = _processing_time_lower_bound(case)

    p = {
        job_index: model.addVar(
            lb=_as_float(processing_time_lower_bound),
            ub=_as_float(
                _processing_time_upper_bound(
                    case, job_index, processing_time_lower_bound
                )
            ),
            vtype=GRB.CONTINUOUS,
            name=f"p[{job_index}]",
        )
        for job_index in jobs
    }
    x = model.addVars(machines, jobs, vtype=GRB.BINARY, name="x")
    q = model.addVars(machines, truncated_jobs, vtype=GRB.BINARY, name="q")
    s = model.addVars(machines, truncated_jobs, lb=0.0, vtype=GRB.CONTINUOUS, name="s")
    z_var = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="Z")

    model.addConstrs(
        (p[j] >= p[j + 1] for j in range(1, case.job_count)),
        name="sorting",
    )
    model.addConstrs(
        (gp.quicksum(x[i, j] for i in machines) == 1 for j in jobs),
        name="opt_assign",
    )
    model.addConstrs(
        (gp.quicksum(p[j] * x[i, j] for j in jobs) <= 1.0 for i in machines),
        name="opt_makespan",
    )

    model.addConstrs(
        (gp.quicksum(q[i, j] for i in machines) == 1 for j in truncated_jobs),
        name="mtf_assign",
    )
    model.addConstrs(
        (s[i, 1] == q[i, 1] * p[1] for i in machines),
        name="mtf_init",
    )
    model.addConstrs(
        (
            s[i, j] - s[i, j - 1] == q[i, j] * p[j]
            for i in machines
            for j in range(2, case.job_count)
        ),
        name="mtf_contribution",
    )
    model.addConstrs(
        (s[i, j] <= target for i in machines for j in truncated_jobs),
        name="mtf_feasible",
    )
    model.addConstrs(
        (
            s[i, j - 1] + p[j]
            >= target
            * (1.0 - gp.quicksum(q[i_prime, j] for i_prime in range(1, i + 1)))
            for i in machines
            for j in range(2, case.job_count)
        ),
        name="mtf_logic",
    )
    model.addConstrs(
        (s[i, case.job_count - 1] + p[case.job_count] >= z_var for i in machines),
        name="mtf_objective",
    )

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
        model.addConstr(z_var >= target, name="target_lb")

    if case.acceleration_case is not AccelerationCase.BASE:
        _validate_paper_acceleration_case(case)
        _apply_paper_acceleration_constraints(
            model, case, p, x, q, jobs, truncated_jobs, machines
        )

    if case.mtf_profile is not None or case.opt_profile is not None:
        _apply_profile_cardinality_constraints(
            model,
            case,
            p,
            x,
            q,
            jobs,
            truncated_jobs,
            machines,
            target,
        )

    model.setObjective(z_var, GRB.MAXIMIZE)
    model.update()
    return BuiltObvModel(model=model, dimensions=dimensions)
