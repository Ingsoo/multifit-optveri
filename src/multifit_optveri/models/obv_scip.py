from __future__ import annotations

from contextlib import contextmanager
from itertools import product
from typing import Any

from multifit_optveri.experiments import ExperimentCase
from multifit_optveri.models import obv_gurobi
from multifit_optveri.models.obv_gurobi import BuiltObvModel

try:
    from pyscipopt import Model as PyScipModel
    from pyscipopt import quicksum as scip_quicksum
except ImportError:  # pragma: no cover - exercised only when pyscipopt is missing
    PyScipModel = None
    scip_quicksum = None


class ScipUnavailableError(RuntimeError):
    """Raised when SCIP support is requested but PySCIPOpt is unavailable."""


class _ScipVarCompat:
    def __init__(self, model: "_ScipModelCompat", var: Any) -> None:
        self._model = model
        self._var = var

    @property
    def LB(self) -> float:
        return float(self._var.getLbGlobal())

    @LB.setter
    def LB(self, value: float) -> None:
        self._model._scip.chgVarLb(self._var, value)

    @property
    def UB(self) -> float:
        return float(self._var.getUbGlobal())

    @UB.setter
    def UB(self, value: float) -> None:
        self._model._scip.chgVarUb(self._var, value)

    @property
    def X(self) -> float:
        return float(self._model._scip.getVal(self._var))

    def _unwrap(self) -> Any:
        return self._var

    def __add__(self, other: Any) -> Any:
        return self._unwrap() + _unwrap_expr(other)

    def __radd__(self, other: Any) -> Any:
        return _unwrap_expr(other) + self._unwrap()

    def __sub__(self, other: Any) -> Any:
        return self._unwrap() - _unwrap_expr(other)

    def __rsub__(self, other: Any) -> Any:
        return _unwrap_expr(other) - self._unwrap()

    def __mul__(self, other: Any) -> Any:
        return self._unwrap() * _unwrap_expr(other)

    def __rmul__(self, other: Any) -> Any:
        return _unwrap_expr(other) * self._unwrap()

    def __truediv__(self, other: Any) -> Any:
        return self._unwrap() / _unwrap_expr(other)

    def __rtruediv__(self, other: Any) -> Any:
        return _unwrap_expr(other) / self._unwrap()

    def __pow__(self, other: Any) -> Any:
        return self._unwrap() ** _unwrap_expr(other)

    def __neg__(self) -> Any:
        return -self._unwrap()

    def __pos__(self) -> Any:
        return +self._unwrap()

    def __le__(self, other: Any) -> Any:
        return self._unwrap() <= _unwrap_expr(other)

    def __ge__(self, other: Any) -> Any:
        return self._unwrap() >= _unwrap_expr(other)

    def __eq__(self, other: Any) -> Any:  # type: ignore[override]
        return self._unwrap() == _unwrap_expr(other)

    def __repr__(self) -> str:
        return repr(self._var)


def _unwrap_expr(value: Any) -> Any:
    if isinstance(value, _ScipVarCompat):
        return value._unwrap()
    return value


class _ScipTupleDict(dict):
    """Small dict-like stand in for Gurobi tupledict."""


class _ScipParamsCompat:
    def __init__(self, model: "_ScipModelCompat") -> None:
        self._model = model
        self._best_bd_stop: float | None = None

    @property
    def BestBdStop(self) -> float | None:
        return self._best_bd_stop

    @BestBdStop.setter
    def BestBdStop(self, value: float) -> None:
        self._best_bd_stop = value
        self._model._scip.setParam("limits/dual", value)


class _ScipEnvCompat:
    def __init__(self, *, empty: bool = True) -> None:
        self.empty = empty
        self.params: dict[str, object] = {}

    def setParam(self, name: str, value: object) -> None:
        self.params[name] = value

    def start(self) -> None:
        return None

    def dispose(self) -> None:
        return None


class _ScipModelCompat:
    def __init__(self, name: str, env: _ScipEnvCompat | None = None) -> None:
        if PyScipModel is None:
            raise ScipUnavailableError(
                "PySCIPOpt is not available. Install pyscipopt to use the SCIP backend."
            )
        self._name = name
        self._scip = PyScipModel()
        self._vars_by_name: dict[str, _ScipVarCompat] = {}
        self._constraints_by_name: dict[str, Any] = {}
        self.Params = _ScipParamsCompat(self)
        if env is not None:
            for param_name, value in env.params.items():
                self._apply_param(param_name, value)

    def _apply_param(self, name: str, value: object) -> None:
        if name in {"OutputFlag", "LogToConsole"}:
            verbose = 0 if int(value) == 0 else 4
            self._scip.setParam("display/verblevel", verbose)
            return
        if name == "TimeLimit":
            self._scip.setParam("limits/time", value)
            return
        if name == "MIPGap":
            self._scip.setParam("limits/gap", value)
            return
        if name == "Threads":
            self._scip.setParam("parallel/maxnthreads", value)
            return
        if name == "Presolve":
            max_rounds = -1 if int(value) > 0 else 0
            self._scip.setParam("presolving/maxrounds", max_rounds)
            return
        if name == "NonConvex":
            return
        self._scip.setParam(name, value)

    def addVar(
        self,
        *,
        lb: float | None = 0.0,
        ub: float | None = None,
        vtype: str,
        name: str,
    ) -> _ScipVarCompat:
        var = self._scip.addVar(lb=lb, ub=ub, vtype=vtype, name=name)
        wrapped = _ScipVarCompat(self, var)
        self._vars_by_name[name] = wrapped
        return wrapped

    def addVars(
        self,
        *index_sets: range,
        lb: float | None = 0.0,
        ub: float | None = None,
        vtype: str,
        name: str,
    ) -> _ScipTupleDict:
        result = _ScipTupleDict()
        for key_tuple in product(*index_sets):
            rendered_key = ",".join(str(part) for part in key_tuple)
            variable = self.addVar(lb=lb, ub=ub, vtype=vtype, name=f"{name}[{rendered_key}]")
            result[key_tuple if len(key_tuple) > 1 else key_tuple[0]] = variable
        return result

    def addConstr(self, expr: Any, *, name: str) -> Any:
        constraint = self._scip.addCons(_unwrap_expr(expr), name=name)
        self._constraints_by_name[name] = constraint
        return constraint

    def addConstrs(self, generator, *, name: str) -> list[Any]:
        constraints: list[Any] = []
        counter = 0
        while True:
            try:
                expr = next(generator)
            except StopIteration:
                break
            counter += 1
            constraint_name = self._constraint_name_from_generator(generator, name, counter)
            constraints.append(self.addConstr(expr, name=constraint_name))
        return constraints

    def _constraint_name_from_generator(self, generator, prefix: str, counter: int) -> str:
        frame = getattr(generator, "gi_frame", None)
        if frame is None:
            return f"{prefix}[{counter}]"
        indices = [value for key, value in frame.f_locals.items() if not key.startswith(".")]
        if not indices:
            return f"{prefix}[{counter}]"
        rendered = ",".join(str(value) for value in indices)
        return f"{prefix}[{rendered}]"

    def setObjective(self, expr: Any, sense: str) -> None:
        objective_sense = "maximize" if sense == _ScipGrbCompat.MAXIMIZE else "minimize"
        self._scip.setObjective(_unwrap_expr(expr), sense=objective_sense)

    def update(self) -> None:
        return None

    def write(self, path: str) -> None:
        self._scip.writeProblem(filename=path, verbose=False)

    def optimize(self) -> None:
        self._scip.optimize()

    def dispose(self) -> None:
        self._scip.freeProb()

    def getVarByName(self, name: str) -> _ScipVarCompat | None:
        return self._vars_by_name.get(name)

    @property
    def Status(self) -> str:
        return str(self._scip.getStatus()).upper()

    @property
    def SolCount(self) -> int:
        return len(self._scip.getSols())

    @property
    def ObjVal(self) -> float:
        return float(self._scip.getObjVal())

    @property
    def ObjBound(self) -> float:
        return float(self._scip.getDualbound())

    @property
    def Runtime(self) -> float:
        return float(self._scip.getSolvingTime())

    @property
    def NodeCount(self) -> float:
        return float(self._scip.getNTotalNodes())

    @property
    def MIPGap(self) -> float:
        return float(self._scip.getGap())

    @property
    def NumVars(self) -> int:
        return int(self._scip.getNVars(transformed=False))

    @property
    def NumConstrs(self) -> int:
        return int(self._scip.getNConss(transformed=False))

    @property
    def NumQConstrs(self) -> int:
        return 0

    @property
    def NumGenConstrs(self) -> int:
        return 0


class _ScipGrbCompat:
    BINARY = "B"
    CONTINUOUS = "C"
    MAXIMIZE = "maximize"


class _ScipGpCompat:
    Env = _ScipEnvCompat
    Model = _ScipModelCompat
    quicksum = staticmethod(scip_quicksum)


@contextmanager
def _patched_gurobi_module():
    original_gp = obv_gurobi.gp
    original_grb = obv_gurobi.GRB
    try:
        obv_gurobi.gp = _ScipGpCompat()
        obv_gurobi.GRB = _ScipGrbCompat
        yield
    finally:
        obv_gurobi.gp = original_gp
        obv_gurobi.GRB = original_grb


def build_obv_model(case: ExperimentCase) -> BuiltObvModel:
    if PyScipModel is None or scip_quicksum is None:
        raise ScipUnavailableError(
            "PySCIPOpt is not available. Install pyscipopt to use the SCIP backend."
        )

    with _patched_gurobi_module():
        return obv_gurobi.build_obv_model(case)
