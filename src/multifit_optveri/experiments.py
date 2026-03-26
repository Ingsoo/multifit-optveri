from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from multifit_optveri.acceleration import AccelerationCase
from multifit_optveri.branching import MtfProfile, OptProfile, ell_iterator, iter_mtf_profiles, iter_opt_profiles
from multifit_optveri.config import ExperimentConfig, SolverConfig
from multifit_optveri.math_utils import ceil_fraction, format_ratio

# This file turns the paper-style verification pseudocode into concrete
# `ExperimentCase` objects. If you want to compare the code against the paper's
# outer algorithm, this is the bridge between:
# - abstract loops in the paper (`case`, `m`, `ell`, profiles)
# - concrete model instances later built in `models/obv.py`


@dataclass(frozen=True)
class JobBounds:
    """Admissible interval for n once m and the target ratio are fixed."""

    lower: int
    upper: int


@dataclass(frozen=True)
class ExperimentCase:
    """One fully specified branch of the verification search tree.

    A single instance of this dataclass corresponds to one item in the paper-
    style outer enumeration:
    case -> m -> ell -> MTF profile -> OPT profile -> solve MIQP
    """

    experiment_name: str
    machine_count: int
    job_count: int
    acceleration_case: AccelerationCase
    ell: int | None
    mtf_profile: MtfProfile | None
    opt_profile: OptProfile | None
    target_ratio: Fraction
    output_root: Path
    write_lp: bool
    enforce_target_lower_bound: bool
    solver: SolverConfig
    run_output_root: Path | None = None

    @property
    def target_ratio_text(self) -> str:
        """Human-readable ratio string used in plans and summaries."""

        return format_ratio(self.target_ratio)

    @property
    def instance_id(self) -> str:
        """Base instance identifier ignoring acceleration/profile metadata."""

        return f"m{self.machine_count:02d}_n{self.job_count:03d}"

    @property
    def case_id(self) -> str:
        """Stable id for one fully branched experiment case.

        When comparing with the paper, this id is not mathematically meaningful
        by itself; it is just a compact serialization of the outer branch choice.
        """

        if self.acceleration_case is AccelerationCase.BASE:
            return self.instance_id
        token = self.acceleration_case.value.replace("case_", "c").replace("_", "")
        parts = [token, self.instance_id]
        if self.ell is not None:
            parts.append(f"e{self.ell:02d}")
        if self.mtf_profile is not None:
            parts.append(self.mtf_profile.compact_id)
        if self.opt_profile is not None:
            parts.append(self.opt_profile.compact_id)
        return "_".join(parts)

    @property
    def output_dir(self) -> Path:
        """Directory where artifacts for this specific case are written."""

        if self.run_output_root is not None:
            return self.run_output_root / "cases" / self.case_id
        return self.output_root / self.experiment_name / self.case_id


def derive_job_bounds(machine_count: int, target_ratio: Fraction) -> JobBounds:
    """Return the admissible search range for n for a fixed m and target ratio.

    This is the implementation counterpart of the global finite search bounds
    used to make the verification algorithm computationally finite.
    """

    lower = 3 * machine_count
    ratio_gap = target_ratio - 1
    upper_1 = (
        ceil_fraction(Fraction(machine_count - 1, 1) / (machine_count * ratio_gap)) - 1
    ) * machine_count
    upper_2 = (
        ceil_fraction((Fraction(machine_count, 1) - target_ratio) / (machine_count * ratio_gap))
        - 1
    ) * machine_count + 1
    upper = min(upper_1, upper_2)
    if upper < lower:
        raise ValueError(
            f"Derived upper bound {upper} is smaller than lower bound {lower} for m={machine_count}."
        )
    return JobBounds(lower=lower, upper=upper)


def enumerate_cases(config: ExperimentConfig) -> list[ExperimentCase]:
    """Expand one experiment config into the full list of branched cases.

    Read this function side by side with the paper pseudocode:
    1. iterate over m
    2. derive/choose feasible n values
    3. iterate over acceleration cases
    4. if accelerated: iterate over ell, OPT profile, MTF profile
    5. materialize one `ExperimentCase` per surviving branch
    """

    cases: list[ExperimentCase] = []
    seen_branch_signatures: set[tuple[object, ...]] = set()
    for machine_count in config.machine_values:
        # First restrict the feasible n range for this value of m.
        bounds = derive_job_bounds(machine_count, config.target_ratio)
        allowed_job_counts = (
            set(config.explicit_job_counts)
            if config.explicit_job_counts
            else set(range(bounds.lower, bounds.upper + 1))
        )

        for acceleration_case in config.acceleration_cases:
            if acceleration_case is AccelerationCase.BASE:
                # Base mode corresponds to Section 4 only: no ell/profile
                # branching, just enumerate plain (m, n) instances.
                for job_count in sorted(allowed_job_counts):
                    cases.append(
                        ExperimentCase(
                            experiment_name=config.name,
                            machine_count=machine_count,
                            job_count=job_count,
                            acceleration_case=acceleration_case,
                            ell=None,
                            mtf_profile=None,
                            opt_profile=None,
                            target_ratio=config.target_ratio,
                            output_root=config.output_root,
                            write_lp=config.write_lp,
                            enforce_target_lower_bound=config.enforce_target_lower_bound,
                            solver=config.solver,
                        )
                    )
                continue

            # Acceleration mode corresponds to Section 5:
            # ell -> OPT profile -> MTF profile.
            for ell in ell_iterator(
                machine_count,
                acceleration_case,
                max_job_count=bounds.upper,
            ):
                for opt_profile in iter_opt_profiles(
                    machine_count,
                    ell,
                    acceleration_case,
                ):
                    # OPT profile fixes n immediately, since its machine counts
                    # determine the total number of jobs.
                    job_count = opt_profile.total_job_count
                    if job_count not in allowed_job_counts:
                        continue

                    for mtf_profile in iter_mtf_profiles(
                        machine_count,
                        ell,
                        opt_profile,
                        acceleration_case,
                    ):
                        if mtf_profile.total_job_count != job_count:
                            continue
                        # Materialize the branch into a concrete experiment case.
                        case = ExperimentCase(
                            experiment_name=config.name,
                            machine_count=machine_count,
                            job_count=job_count,
                            acceleration_case=acceleration_case,
                            ell=ell,
                            mtf_profile=mtf_profile,
                            opt_profile=opt_profile,
                            target_ratio=config.target_ratio,
                            output_root=config.output_root,
                            write_lp=config.write_lp,
                            enforce_target_lower_bound=config.enforce_target_lower_bound,
                            solver=config.solver,
                        )
                        signature = _branch_signature(case)
                        if signature in seen_branch_signatures:
                            # Dedupe is an implementation optimization: if two
                            # syntactically different generation paths yield the
                            # same effective branch signature, solve it only once.
                            continue
                        seen_branch_signatures.add(signature)
                        cases.append(case)
    return cases


def render_case_plan(cases: list[ExperimentCase]) -> str:
    """Render the expanded case list as a readable plan string."""

    lines = [f"Total cases: {len(cases)}"]
    for case in cases:
        line = (
            f"- {case.case_id}: m={case.machine_count}, n={case.job_count}, "
            f"acceleration={case.acceleration_case.value}, target={case.target_ratio_text}"
        )
        if case.ell is not None:
            line += f", ell={case.ell}"
        if case.mtf_profile is not None:
            line += f", mtf={case.mtf_profile.compact_id}"
        if case.opt_profile is not None:
            line += f", opt={case.opt_profile.compact_id}"
        lines.append(line)
    return "\n".join(lines)


def _branch_signature(case: ExperimentCase) -> tuple[object, ...]:
    """Return a canonical signature used to remove duplicate branches.

    This is not part of the mathematical statement in the paper; it is purely
    an execution-side optimization so equivalent branches are not solved twice.
    """

    return (
        case.acceleration_case.value,
        case.machine_count,
        case.job_count,
        case.ell,
        (
            (
                case.mtf_profile.nF1,
                case.mtf_profile.nR2,
                case.mtf_profile.nF2,
                case.mtf_profile.nR3,
                case.mtf_profile.nF3,
                case.mtf_profile.nR4,
                case.mtf_profile.nM5,
            )
            if case.mtf_profile is not None
            else None
        ),
        (
            (
                case.opt_profile.nS3,
                case.opt_profile.nS4,
                case.opt_profile.nS5,
                case.opt_profile.pattern,
            )
            if case.opt_profile is not None
            else None
        ),
    )
