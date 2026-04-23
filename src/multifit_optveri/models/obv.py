from __future__ import annotations

from multifit_optveri.experiments import ExperimentCase
from multifit_optveri.models import obv_gurobi, obv_scip

BuiltObvModel = obv_gurobi.BuiltObvModel
GurobiUnavailableError = obv_gurobi.GurobiUnavailableError
ScipUnavailableError = obv_scip.ScipUnavailableError
MtfProfileLayout = obv_gurobi.MtfProfileLayout
BOUND_STOP_TOLERANCE = obv_gurobi.BOUND_STOP_TOLERANCE
GRB = obv_gurobi.GRB
gp = obv_gurobi.gp


def build_obv_model(case: ExperimentCase) -> BuiltObvModel:
    if case.solver.backend == "gurobi":
        return obv_gurobi.build_obv_model(case)
    if case.solver.backend == "scip":
        return obv_scip.build_obv_model(case)
    raise ValueError(f"Unsupported solver backend: {case.solver.backend!r}")


def __getattr__(name: str):
    return getattr(obv_gurobi, name)
