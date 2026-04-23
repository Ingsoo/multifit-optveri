from __future__ import annotations

from contextlib import contextmanager

from multifit_optveri.experiments import ExperimentCase
from multifit_optveri.models import obv_core
from multifit_optveri.models.obv_core import BuiltObvModel

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover - exercised only when Gurobi is missing
    gp = None
    GRB = None


class GurobiUnavailableError(RuntimeError):
    """Raised when Gurobi support is requested but gurobipy is unavailable."""


def _require_gurobi() -> None:
    if gp is None or GRB is None:
        raise GurobiUnavailableError(
            "gurobipy is not available. Install it and activate a Gurobi license to run models."
        )


@contextmanager
def _patched_core_backend():
    original_gp = obv_core.gp
    original_grb = obv_core.GRB
    try:
        obv_core.gp = gp
        obv_core.GRB = GRB
        yield
    finally:
        obv_core.gp = original_gp
        obv_core.GRB = original_grb


def build_obv_model(case: ExperimentCase) -> BuiltObvModel:
    _require_gurobi()
    with _patched_core_backend():
        return obv_core.build_obv_model(case)


def __getattr__(name: str):
    return getattr(obv_core, name)
