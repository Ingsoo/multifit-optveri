from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import tempfile

from multifit_optveri.config import SUPPORTED_SOLVER_BACKENDS
from multifit_optveri.experiments import ExperimentCase
from multifit_optveri.models.obv_core import BuiltObvModel
from multifit_optveri.models import obv_scip
from multifit_optveri.models.obv_gurobi import build_obv_model as build_gurobi_obv_model
from multifit_optveri.models.spec import ObvModelDimensions

try:
    from gurobipy import GRB
except ImportError:  # pragma: no cover - exercised only when gurobipy is missing
    GRB = None


@dataclass(frozen=True)
class ConstraintSummary:
    total_constraints: int
    linear_constraints: int | None
    quadratic_constraints: int | None
    general_constraints: int | None


@dataclass(frozen=True)
class SolverOutcome:
    status: str
    objective_value: float | None
    objective_bound: float | None
    runtime_seconds: float | None
    node_count: float | None
    mip_gap: float | None
    optimal_p_values: tuple[float, ...] | None


@dataclass(frozen=True)
class BackendRunResult:
    dimensions: ObvModelDimensions
    constraint_summary: ConstraintSummary
    outcome: SolverOutcome


def solve_case_with_backend(case: ExperimentCase) -> BackendRunResult:
    if case.solver.backend == "gurobi":
        return _solve_case_with_gurobi(case)
    if case.solver.backend == "scip":
        return _solve_case_with_scip(case)
    raise ValueError(
        f"Unsupported solver backend {case.solver.backend!r}. "
        f"Expected one of {SUPPORTED_SOLVER_BACKENDS}."
    )


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return value if math.isfinite(value) else None


def _gurobi_status_name(status_code: int) -> str:
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


def _extract_gurobi_optimal_p_values(model: object, job_count: int) -> tuple[float, ...] | None:
    if GRB is None or getattr(model, "Status", None) != GRB.OPTIMAL:
        return None

    values: list[float] = []
    for job_index in range(1, job_count + 1):
        variable = model.getVarByName(f"p[{job_index}]")
        if variable is None:
            return None
        values.append(float(variable.X))
    return tuple(values)


def _solve_case_with_gurobi(case: ExperimentCase) -> BackendRunResult:
    built_model: BuiltObvModel = build_gurobi_obv_model(case)
    model = built_model.model

    try:
        if case.write_case_dirs:
            case.output_dir.mkdir(parents=True, exist_ok=True)
        if case.write_lp:
            model.write(str(case.output_dir / "model.lp"))

        model.optimize()

        has_solution = model.SolCount > 0
        outcome = SolverOutcome(
            status=_gurobi_status_name(model.Status),
            objective_value=_finite_or_none(float(model.ObjVal) if has_solution else None),
            objective_bound=_finite_or_none(float(model.ObjBound) if hasattr(model, "ObjBound") else None),
            runtime_seconds=_finite_or_none(float(model.Runtime) if hasattr(model, "Runtime") else None),
            node_count=_finite_or_none(float(model.NodeCount) if hasattr(model, "NodeCount") else None),
            mip_gap=_finite_or_none(float(model.MIPGap) if has_solution and hasattr(model, "MIPGap") else None),
            optimal_p_values=_extract_gurobi_optimal_p_values(model, case.job_count),
        )
        constraint_summary = ConstraintSummary(
            total_constraints=int(model.NumConstrs) + int(model.NumQConstrs) + int(model.NumGenConstrs),
            linear_constraints=int(model.NumConstrs),
            quadratic_constraints=int(model.NumQConstrs),
            general_constraints=int(model.NumGenConstrs),
        )
        return BackendRunResult(
            dimensions=built_model.dimensions,
            constraint_summary=constraint_summary,
            outcome=outcome,
        )
    finally:
        if hasattr(model, "dispose"):
            model.dispose()
        env = getattr(model, "_multifit_env", None)
        if env is not None and hasattr(env, "dispose"):
            env.dispose()


def _solve_case_with_scip(case: ExperimentCase) -> BackendRunResult:
    if case.solver.scip_exact:
        return _solve_case_with_scip_exact(case)

    built_model = obv_scip.build_obv_model(case)
    model = built_model.model

    try:
        if case.write_case_dirs:
            case.output_dir.mkdir(parents=True, exist_ok=True)
        if case.write_lp:
            model.write(str(case.output_dir / "model.lp"))

        model.optimize()

        has_solution = model.SolCount > 0
        outcome = SolverOutcome(
            status=_scip_status_name(model.Status),
            objective_value=_finite_or_none(float(model.ObjVal) if has_solution else None),
            objective_bound=_finite_or_none(float(model.ObjBound) if hasattr(model, "ObjBound") else None),
            runtime_seconds=_finite_or_none(float(model.Runtime) if hasattr(model, "Runtime") else None),
            node_count=_finite_or_none(float(model.NodeCount) if hasattr(model, "NodeCount") else None),
            mip_gap=_finite_or_none(float(model.MIPGap) if has_solution and hasattr(model, "MIPGap") else None),
            optimal_p_values=_extract_scip_optimal_p_values(model, case.job_count),
        )
        constraint_summary = ConstraintSummary(
            total_constraints=int(model.NumConstrs) + int(model.NumQConstrs) + int(model.NumGenConstrs),
            linear_constraints=None,
            quadratic_constraints=None,
            general_constraints=None,
        )
        return BackendRunResult(
            dimensions=built_model.dimensions,
            constraint_summary=constraint_summary,
            outcome=outcome,
        )
    finally:
        if hasattr(model, "dispose"):
            model.dispose()


def _solve_case_with_scip_exact(case: ExperimentCase) -> BackendRunResult:
    export_case = replace(case, solver=replace(case.solver, scip_exact=False))
    built_model = obv_scip.build_obv_model(export_case)
    export_model = built_model.model

    try:
        if case.write_case_dirs:
            case.output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="multifit_scip_exact_") as tmpdir:
            exact_input_path = Path(tmpdir) / f"{case.case_id}.mps"
            export_model.write(str(exact_input_path))

            if case.write_lp:
                persisted_path = case.output_dir / "model.mps"
                persisted_path.write_bytes(exact_input_path.read_bytes())

            model = obv_scip.read_exact_problem(str(exact_input_path), case)
            try:
                model.optimize()

                has_solution = len(model.getSols()) > 0
                status = _scip_status_name(model.getStatus())
                if obv_scip.exact_target_stop_reached(model):
                    status = "USER_OBJ_LIMIT"
                outcome = SolverOutcome(
                    status=status,
                    objective_value=_finite_or_none(float(model.getObjVal()) if has_solution else None),
                    objective_bound=_finite_or_none(float(model.getDualbound())),
                    runtime_seconds=_finite_or_none(float(model.getSolvingTime())),
                    node_count=_finite_or_none(float(model.getNTotalNodes())),
                    mip_gap=_finite_or_none(float(model.getGap()) if has_solution else None),
                    optimal_p_values=None,
                )
                constraint_summary = ConstraintSummary(
                    total_constraints=int(model.getNConss(transformed=False)),
                    linear_constraints=None,
                    quadratic_constraints=None,
                    general_constraints=None,
                )
                return BackendRunResult(
                    dimensions=built_model.dimensions,
                    constraint_summary=constraint_summary,
                    outcome=outcome,
                )
            finally:
                obv_scip.clear_exact_target_stop_state(model)
                model.freeProb()
    finally:
        if hasattr(export_model, "dispose"):
            export_model.dispose()


def _extract_scip_optimal_p_values(model: object, job_count: int) -> tuple[float, ...] | None:
    if _scip_status_name(getattr(model, "Status", "")) != "OPTIMAL":
        return None

    values: list[float] = []
    for job_index in range(1, job_count + 1):
        variable = model.getVarByName(f"p[{job_index}]")
        if variable is None:
            return None
        values.append(float(variable.X))
    return tuple(values)


def _scip_status_name(status: str) -> str:
    normalized = str(status).strip().upper()
    return {
        "OPTIMAL": "OPTIMAL",
        "INFEASIBLE": "INFEASIBLE",
        "TIMELIMIT": "TIME_LIMIT",
        "USERINTERRUPT": "INTERRUPTED",
        "INFORUNBD": "INF_OR_UNBD",
        "UNBOUNDED": "UNBOUNDED",
        "GAPLIMIT": "USER_OBJ_LIMIT",
    }.get(normalized, normalized)
