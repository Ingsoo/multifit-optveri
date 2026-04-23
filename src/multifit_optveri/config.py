from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import tomllib

from multifit_optveri.acceleration import AccelerationCase, parse_acceleration_case
from multifit_optveri.math_utils import parse_ratio

SUPPORTED_SOLVER_BACKENDS = ("gurobi", "scip")


@dataclass(frozen=True)
class SolverConfig:
    backend: str = "gurobi"
    time_limit_seconds: float | None = None
    mip_gap: float | None = None
    threads: int | None = None
    presolve: int | None = None
    scip_exact: bool = False
    legacy_best_bd_stop_at_target: bool = True
    non_convex: int = 2
    output_flag: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str):
            raise TypeError("solver backend must be a string.")
        normalized_backend = self.backend.strip().lower()
        if normalized_backend not in SUPPORTED_SOLVER_BACKENDS:
            raise ValueError(
                f"Unsupported solver backend {self.backend!r}. "
                f"Expected one of {SUPPORTED_SOLVER_BACKENDS}."
            )
        object.__setattr__(self, "backend", normalized_backend)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    target_ratio: Fraction
    machine_values: tuple[int, ...]
    derive_job_counts: bool
    explicit_job_counts: tuple[int, ...]
    output_root: Path
    write_lp: bool
    enforce_target_lower_bound: bool
    solver: SolverConfig
    write_case_dirs: bool = True
    acceleration_cases: tuple[AccelerationCase, ...] = (AccelerationCase.BASE,)

    def __post_init__(self) -> None:
        if not self.machine_values:
            raise ValueError("machine_values must not be empty.")
        if any(machine <= 0 for machine in self.machine_values):
            raise ValueError("machine_values must be positive.")
        if any(job <= 0 for job in self.explicit_job_counts):
            raise ValueError("explicit_job_counts must be positive when provided.")
        if not self.acceleration_cases:
            raise ValueError("acceleration_cases must not be empty.")
        if len(set(self.acceleration_cases)) != len(self.acceleration_cases):
            raise ValueError("acceleration_cases must not contain duplicates.")
        if not self.derive_job_counts and not self.explicit_job_counts:
            raise ValueError("Either derive_job_counts must be true or explicit_job_counts must be provided.")
        if self.write_lp and not self.write_case_dirs:
            raise ValueError("write_lp requires write_case_dirs to be enabled.")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))

    experiment = raw["experiment"]
    solver = raw.get("solver", {})

    solver_config = SolverConfig(
        backend=solver.get("backend", "gurobi"),
        time_limit_seconds=solver.get("time_limit_seconds"),
        mip_gap=solver.get("mip_gap"),
        threads=solver.get("threads"),
        presolve=solver.get("presolve"),
        scip_exact=bool(solver.get("scip_exact", False)),
        legacy_best_bd_stop_at_target=bool(solver.get("legacy_best_bd_stop_at_target", False)),
        non_convex=solver.get("non_convex", 2),
        output_flag=solver.get("output_flag", 1),
    )

    return ExperimentConfig(
        name=experiment["name"],
        target_ratio=parse_ratio(experiment["target_ratio"]),
        machine_values=tuple(experiment["machine_values"]),
        derive_job_counts=bool(experiment.get("derive_job_counts", True)),
        explicit_job_counts=tuple(experiment.get("explicit_job_counts", [])),
        acceleration_cases=tuple(
            parse_acceleration_case(value)
            for value in experiment.get("acceleration_cases", [AccelerationCase.BASE.value])
        ),
        output_root=Path(experiment.get("output_root", "results")),
        write_lp=bool(experiment.get("write_lp", False)),
        write_case_dirs=bool(experiment.get("write_case_dirs", True)),
        enforce_target_lower_bound=bool(experiment.get("enforce_target_lower_bound", True)),
        solver=solver_config,
    )
