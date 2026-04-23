from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, UTC
import math
from pathlib import Path
import csv
import json

from multifit_optveri.acceleration import AccelerationCase
from multifit_optveri.experiments import ExperimentCase
from multifit_optveri.math_utils import (
    format_ratio,
    format_scaled_rational_values,
    format_sorted_numeric_values,
    parse_ratio,
)
from multifit_optveri.models.obv import (
    _build_exact_mtf_assignment,
    _build_mtf_profile_layout,
)
from multifit_optveri.solver_backends import solve_case_with_backend

try:
    from gurobipy import GRB
except ImportError:  # pragma: no cover - exercised only when Gurobi is missing
    GRB = None

@dataclass(frozen=True)
class SolveResult:
    experiment_name: str
    case_id: str
    acceleration_case: str
    machine_count: int
    job_count: int
    ell: int | None
    mtf_profile: str | None
    fallback_starts: str | None
    opt_profile: str | None
    target_ratio: str
    verification_result: str
    status: str
    objective_value: float | None
    objective_bound: float | None
    runtime_seconds: float | None
    node_count: float | None
    mip_gap: float | None
    optimal_p_values_desc_exact: str | None
    optimal_p_values_desc: str | None
    output_dir: str
    built_at_utc: str


@dataclass(frozen=True)
class RunArtifacts:
    experiment_dir: Path
    run_dir: Path
    cases_dir: Path
    summary_csv_path: Path
    summary_jsonl_path: Path
    overview_json_path: Path
    manifest_json_path: Path
    latest_run_path: Path


def _status_name(status_code: int) -> str:
    if GRB is None:
        return str(status_code)
    return {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }.get(status_code, str(status_code))

def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False))
        handle.write("\n")


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return value if math.isfinite(value) else None


def _write_csv_row(path: Path, fieldnames: list[str], row: dict[str, object], *, write_header: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        if row:
            writer.writerow(row)


def _run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _allocate_run_dir(experiment_dir: Path) -> Path:
    base_name = _run_timestamp()
    candidate = experiment_dir / base_name
    suffix = 1
    while candidate.exists():
        candidate = experiment_dir / f"{base_name}_{suffix:02d}"
        suffix += 1
    return candidate


def _result_csv_fieldnames() -> list[str]:
    return [
        "case_id",
        "machine_count",
        "job_count",
        "acceleration_case",
        "ell",
        "mtf-profile-(f1_r2_f2_r3_f3_r4_f4_r5)",
        "fallback-starts-(s2_s3_s4)",
        "opt-profile-(e3_e4_e5)",
        "verification_result",
        "status",
        "objective_value",
        "objective_bound",
        "runtime_seconds",
        "node_count",
        "mip_gap",
        "optimal-p-values-(desc-exact)",
        "optimal-p-values-(desc-scaled)",
    ]


def _result_summary_payload(result: SolveResult) -> dict[str, object]:
    return {
        "case_id": result.case_id,
        "machine_count": result.machine_count,
        "job_count": result.job_count,
        "acceleration_case": result.acceleration_case,
        "ell": result.ell if result.ell is not None else "",
        "mtf-profile-(f1_r2_f2_r3_f3_r4_f4_r5)": (
            result.mtf_profile if result.mtf_profile is not None else ""
        ),
        "fallback-starts-(s2_s3_s4)": (
            result.fallback_starts if result.fallback_starts is not None else ""
        ),
        "opt-profile-(e3_e4_e5)": (result.opt_profile if result.opt_profile is not None else ""),
        "verification_result": result.verification_result,
        "status": result.status,
        "objective_value": result.objective_value if result.objective_value is not None else "",
        "objective_bound": result.objective_bound if result.objective_bound is not None else "",
        "runtime_seconds": result.runtime_seconds if result.runtime_seconds is not None else "",
        "node_count": result.node_count if result.node_count is not None else "",
        "mip_gap": result.mip_gap if result.mip_gap is not None else "",
        "optimal-p-values-(desc-exact)": (
            result.optimal_p_values_desc_exact if result.optimal_p_values_desc_exact is not None else ""
        ),
        "optimal-p-values-(desc-scaled)": (
            result.optimal_p_values_desc if result.optimal_p_values_desc is not None else ""
        ),
    }


def _result_csv_row(result: SolveResult) -> dict[str, object]:
    return _result_summary_payload(result)


def _verification_result(
    *,
    status: str,
    objective_value: float | None,
    objective_bound: float | None,
    target_ratio: str,
) -> str:
    """Classify one solved branch as verified/not-verified for the paper claim."""

    target = float(parse_ratio(target_ratio))
    if status == "INFEASIBLE":
        return "VERIFIED"
    if objective_value is not None:
        if objective_value <= target:
            return "VERIFIED"
        return "NOT_VERIFIED"
    if status == "USER_OBJ_LIMIT" and objective_bound is not None and objective_bound <= target:
        return "VERIFIED"
    return "UNKNOWN"


def _format_mtf_profile(case: ExperimentCase) -> str | None:
    if case.mtf_profile is None:
        return None
    profile = case.mtf_profile
    return (
        f"({profile.nF1},{profile.nR2},{profile.nF2},{profile.nR3},"
        f"{profile.nF3},{profile.nR4},{profile.nF4},{profile.nR5})"
    )


def _format_fallback_starts(case: ExperimentCase) -> str | None:
    if case.fallback_starts is None:
        return None
    starts = case.fallback_starts
    return f"({starts.s2},{starts.s3},{starts.s4})"


def _format_opt_profile(case: ExperimentCase) -> str | None:
    if case.opt_profile is None:
        return None
    profile = case.opt_profile
    return f"({profile.nS3},{profile.nS4},{profile.nS5})"


def _format_mtf_assignment(case: ExperimentCase) -> dict[str, list[int]] | None:
    if case.mtf_profile is None:
        return None
    if case.acceleration_case not in (AccelerationCase.CASE_2, AccelerationCase.CASE_3):
        return None
    if case.fallback_starts is None:
        return None

    layout = _build_mtf_profile_layout(case)
    assignment = _build_exact_mtf_assignment(case, layout)
    return {
        f"M{machine_index}": list(job_indices)
        for machine_index, job_indices in sorted(assignment.items())
    }


def _extract_optimal_p_values_desc(values: tuple[float, ...] | None) -> str | None:
    if values is None:
        return None
    return format_scaled_rational_values(values)


def _extract_optimal_p_values_desc_exact(values: tuple[float, ...] | None) -> str | None:
    if values is None:
        return None
    return format_sorted_numeric_values(values)


def create_run_artifacts(cases: list[ExperimentCase]) -> RunArtifacts:
    if not cases:
        raise ValueError("Cannot create run artifacts for an empty case list.")

    sample_case = cases[0]
    experiment_dir = sample_case.output_root / sample_case.experiment_name
    run_dir = _allocate_run_dir(experiment_dir)
    cases_dir = run_dir / "cases"
    artifacts = RunArtifacts(
        experiment_dir=experiment_dir,
        run_dir=run_dir,
        cases_dir=cases_dir,
        summary_csv_path=run_dir / "summary.csv",
        summary_jsonl_path=run_dir / "summary.jsonl",
        overview_json_path=run_dir / "overview.json",
        manifest_json_path=run_dir / "manifest.json",
        latest_run_path=experiment_dir / "latest_run.txt",
    )
    if any(case.write_case_dirs for case in cases):
        cases_dir.mkdir(parents=True, exist_ok=True)
    artifacts.latest_run_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.latest_run_path.write_text(str(run_dir), encoding="utf-8")
    return artifacts


def prepare_cases_for_run(cases: list[ExperimentCase], run_dir: Path) -> list[ExperimentCase]:
    return [replace(case, run_output_root=run_dir) for case in cases]


class RunRecorder:
    def __init__(
        self,
        *,
        artifacts: RunArtifacts,
        cases: list[ExperimentCase],
        cli_filters: dict[str, object] | None = None,
    ) -> None:
        self.artifacts = artifacts
        self.cases = cases
        self.cli_filters = cli_filters or {}
        self.results: list[SolveResult] = []

        _write_json(
            self.artifacts.manifest_json_path,
            {
                "experiment_name": cases[0].experiment_name,
                "case_count": len(cases),
                "run_dir": str(self.artifacts.run_dir),
                "created_at_local": datetime.now().isoformat(),
                "cli_filters": self.cli_filters,
            },
        )
        _write_csv_row(
            self.artifacts.summary_csv_path,
            _result_csv_fieldnames(),
            {},
            write_header=True,
        )

    def record(self, result: SolveResult) -> None:
        self.results.append(result)
        _write_csv_row(
            self.artifacts.summary_csv_path,
            _result_csv_fieldnames(),
            _result_csv_row(result),
        )
        _append_jsonl(self.artifacts.summary_jsonl_path, _result_summary_payload(result))
        self._write_overview()

    def finish(self) -> None:
        manifest = json.loads(self.artifacts.manifest_json_path.read_text(encoding="utf-8"))
        manifest["finished_at_local"] = datetime.now().isoformat()
        manifest["completed_case_count"] = len(self.results)
        _write_json(self.artifacts.manifest_json_path, manifest)
        self._write_overview()

    def _write_overview(self) -> None:
        status_counts = Counter(result.status for result in self.results)
        verification_counts = Counter(result.verification_result for result in self.results)
        case_counts = Counter(result.acceleration_case for result in self.results)
        total_runtime = sum(result.runtime_seconds for result in self.results if result.runtime_seconds is not None)
        _write_json(
            self.artifacts.overview_json_path,
            {
                "run_dir": str(self.artifacts.run_dir),
                "completed_case_count": len(self.results),
                "planned_case_count": len(self.cases),
                "status_counts": dict(status_counts),
                "verification_result_counts": dict(verification_counts),
                "acceleration_case_counts": dict(case_counts),
                "total_runtime_seconds": total_runtime,
                "summary_csv": str(self.artifacts.summary_csv_path),
                "summary_jsonl": str(self.artifacts.summary_jsonl_path),
            },
        )


def run_case(case: ExperimentCase) -> SolveResult:
    if case.write_lp and not case.write_case_dirs:
        raise ValueError("write_lp requires write_case_dirs to be enabled for each case.")

    backend_result = solve_case_with_backend(case)
    outcome = backend_result.outcome
    optimal_p_values_desc_exact = _extract_optimal_p_values_desc_exact(outcome.optimal_p_values)
    optimal_p_values_desc = _extract_optimal_p_values_desc(outcome.optimal_p_values)
    result = SolveResult(
        experiment_name=case.experiment_name,
        case_id=case.case_id,
        acceleration_case=case.acceleration_case.value,
        machine_count=case.machine_count,
        job_count=case.job_count,
        ell=case.ell,
        mtf_profile=_format_mtf_profile(case),
        fallback_starts=_format_fallback_starts(case),
        opt_profile=_format_opt_profile(case),
        target_ratio=format_ratio(case.target_ratio),
        verification_result=_verification_result(
            status=outcome.status,
            objective_value=outcome.objective_value,
            objective_bound=outcome.objective_bound,
            target_ratio=format_ratio(case.target_ratio),
        ),
        status=outcome.status,
        objective_value=outcome.objective_value,
        objective_bound=outcome.objective_bound,
        runtime_seconds=outcome.runtime_seconds,
        node_count=outcome.node_count,
        mip_gap=outcome.mip_gap,
        optimal_p_values_desc_exact=optimal_p_values_desc_exact,
        optimal_p_values_desc=optimal_p_values_desc,
        output_dir=str(case.output_dir),
        built_at_utc=datetime.now(UTC).isoformat(),
    )

    summary_payload = {
        "experiment_name": result.experiment_name,
        **_result_summary_payload(result),
        "target_ratio": result.target_ratio,
        "built_at_utc": result.built_at_utc,
        "output_dir": result.output_dir,
        "mtf_assignment": _format_mtf_assignment(case),
    }
    summary_payload["dimensions"] = {
        "total_variables": backend_result.dimensions.total_variables,
        "total_constraints": backend_result.constraint_summary.total_constraints,
        "linear_constraints": backend_result.constraint_summary.linear_constraints,
        "quadratic_constraints": backend_result.constraint_summary.quadratic_constraints,
        "general_constraints": backend_result.constraint_summary.general_constraints,
        "spec_total_variables": backend_result.dimensions.total_variables,
        "spec_total_constraints": backend_result.dimensions.total_constraints,
        "variable_counts": backend_result.dimensions.variable_counts,
        "constraint_counts": backend_result.dimensions.constraint_counts,
    }
    if case.write_case_dirs:
        _write_json(case.output_dir / "summary.json", summary_payload)
    return result


def run_cases(cases: list[ExperimentCase]) -> list[SolveResult]:
    artifacts = create_run_artifacts(cases)
    prepared_cases = prepare_cases_for_run(cases, artifacts.run_dir)
    recorder = RunRecorder(artifacts=artifacts, cases=prepared_cases)
    results: list[SolveResult] = []
    for case in prepared_cases:
        result = run_case(case)
        recorder.record(result)
        results.append(result)
    recorder.finish()
    return results
