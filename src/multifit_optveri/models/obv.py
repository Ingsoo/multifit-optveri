from __future__ import annotations

from multifit_optveri.experiments import ExperimentCase
from multifit_optveri.models import obv_core, obv_gurobi, obv_scip

BuiltObvModel = obv_core.BuiltObvModel
GurobiUnavailableError = obv_gurobi.GurobiUnavailableError
ScipUnavailableError = obv_scip.ScipUnavailableError
MtfProfileLayout = obv_core.MtfProfileLayout
BOUND_STOP_TOLERANCE = obv_core.BOUND_STOP_TOLERANCE
GRB = obv_gurobi.GRB
gp = obv_gurobi.gp
_require_gurobi = obv_gurobi._require_gurobi


def build_obv_model(case: ExperimentCase) -> BuiltObvModel:
    if case.solver.backend == "gurobi":
        return obv_gurobi.build_obv_model(case)
    if case.solver.backend == "scip":
        return obv_scip.build_obv_model(case)
    raise ValueError(f"Unsupported solver backend: {case.solver.backend!r}")


def __getattr__(name: str):
    if hasattr(obv_core, name):
        return getattr(obv_core, name)
    if hasattr(obv_gurobi, name):
        return getattr(obv_gurobi, name)
    return getattr(obv_scip, name)
