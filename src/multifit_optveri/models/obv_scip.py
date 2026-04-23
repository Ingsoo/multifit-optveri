from __future__ import annotations

import ctypes
import ctypes.util
from contextlib import contextmanager
from fractions import Fraction
from functools import lru_cache
import inspect
from itertools import product
import os
from pathlib import Path
from typing import Any

from multifit_optveri.experiments import ExperimentCase
from multifit_optveri.models import obv_core
from multifit_optveri.models.obv_core import BuiltObvModel

try:
    from pyscipopt import Eventhdlr
    from pyscipopt import Model as PyScipModel
    from pyscipopt import SCIP_EVENTTYPE
    from pyscipopt import quicksum as scip_quicksum
except ImportError:  # pragma: no cover - exercised only when pyscipopt is missing
    Eventhdlr = None
    PyScipModel = None
    SCIP_EVENTTYPE = None
    scip_quicksum = None


class ScipUnavailableError(RuntimeError):
    """Raised when SCIP support is requested but PySCIPOpt is unavailable."""


class _ScipRational(ctypes.Structure):
    pass


_SCIP_OKAY = 1
_EXACT_TARGET_STOP_STATE: dict[int, dict[str, Any]] = {}


def _raise_for_scip_error(return_code: int, *, func_name: str) -> None:
    if return_code != _SCIP_OKAY:
        raise ScipUnavailableError(f"SCIP exact helper call {func_name} failed with return code {return_code}.")


@lru_cache(maxsize=1)
def _load_scip_library() -> ctypes.CDLL:
    candidates: list[str] = []
    for discovered in (ctypes.util.find_library("libscip"), ctypes.util.find_library("scip")):
        if discovered:
            candidates.append(discovered)
    scipopt_dir = Path(os.environ["SCIPOPTDIR"]) if "SCIPOPTDIR" in os.environ else None
    if scipopt_dir is not None:
        for relative_path in ("bin/libscip.dll", "lib/libscip.so", "lib/libscip.dylib", "bin/scip.dll"):
            candidate = scipopt_dir / relative_path
            if candidate.exists():
                candidates.append(str(candidate))
    if PyScipModel is not None:
        module_dir = Path(inspect.getfile(PyScipModel)).resolve().parent
        for pattern in ("libscip*.dll", "libscip*.so", "libscip*.dylib", "scip*.dll"):
            for candidate in module_dir.glob(pattern):
                candidates.append(str(candidate))
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    raise ScipUnavailableError(
        "Unable to load the SCIP shared library for exact dual-bound access. "
        "Make sure SCIP is installed with shared-library support."
    )


@lru_cache(maxsize=1)
def _pycapsule_get_pointer() -> Any:
    getter = ctypes.pythonapi.PyCapsule_GetPointer
    getter.argtypes = [ctypes.py_object, ctypes.c_char_p]
    getter.restype = ctypes.c_void_p
    return getter


def _get_scip_pointer(model: Any) -> ctypes.c_void_p:
    if not hasattr(model, "to_ptr"):
        raise ScipUnavailableError("This PySCIPOpt build does not expose Model.to_ptr for exact SCIP access.")
    capsule = model.to_ptr(False)
    pointer = _pycapsule_get_pointer()(capsule, b"scip")
    if not pointer:
        raise ScipUnavailableError("Unable to recover the underlying SCIP pointer from the PySCIPOpt model.")
    return ctypes.c_void_p(pointer)


def _parse_exact_rational_text(text: str) -> Fraction | None:
    normalized = text.strip().lower()
    if normalized in {"-infinity", "-inf"}:
        return None
    if normalized in {"infinity", "+infinity", "inf", "+inf"}:
        raise ScipUnavailableError("Encountered an unexpected positive infinite exact dual bound.")
    return Fraction(text)


def _exact_dual_bound_fraction(model: Any) -> Fraction | None:
    library = _load_scip_library()
    scip_pointer = _get_scip_pointer(model)

    rational = ctypes.POINTER(_ScipRational)()
    library.SCIPrationalCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(_ScipRational))]
    library.SCIPrationalCreate.restype = ctypes.c_int
    _raise_for_scip_error(library.SCIPrationalCreate(ctypes.byref(rational)), func_name="SCIPrationalCreate")

    try:
        library.SCIPgetDualboundExact.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ScipRational)]
        library.SCIPgetDualboundExact.restype = None
        library.SCIPgetDualboundExact(scip_pointer, rational)

        library.SCIPrationalStrLen.argtypes = [ctypes.POINTER(_ScipRational)]
        library.SCIPrationalStrLen.restype = ctypes.c_int
        buffer_size = int(library.SCIPrationalStrLen(rational)) + 1
        text_buffer = ctypes.create_string_buffer(buffer_size)

        library.SCIPrationalToString.argtypes = [ctypes.POINTER(_ScipRational), ctypes.c_char_p, ctypes.c_int]
        library.SCIPrationalToString.restype = ctypes.c_int
        library.SCIPrationalToString(rational, text_buffer, buffer_size)
        rendered = text_buffer.value.decode("ascii").strip()
        return _parse_exact_rational_text(rendered)
    finally:
        library.SCIPrationalFree.argtypes = [ctypes.POINTER(ctypes.POINTER(_ScipRational))]
        library.SCIPrationalFree.restype = None
        library.SCIPrationalFree(ctypes.byref(rational))


_EventhdlrBase = Eventhdlr if Eventhdlr is not None else object


class _ExactDualBoundStopEventhdlr(_EventhdlrBase):  # type: ignore[misc]
    def __init__(self, target_ratio: Fraction) -> None:
        super().__init__()
        self.target_ratio = target_ratio
        self.hit = False

    def eventinit(self) -> None:
        if SCIP_EVENTTYPE is None:
            raise ScipUnavailableError("PySCIPOpt event support is unavailable for exact SCIP stopping.")
        self.model.catchEvent(SCIP_EVENTTYPE.DUALBOUNDIMPROVED, self)

    def eventexit(self) -> None:
        if SCIP_EVENTTYPE is None:
            return
        self.model.dropEvent(SCIP_EVENTTYPE.DUALBOUNDIMPROVED, self)

    def eventexec(self, event: Any) -> None:
        dual_bound = _exact_dual_bound_fraction(self.model)
        if dual_bound is None:
            return
        if dual_bound < self.target_ratio:
            return
        self.hit = True
        state = _EXACT_TARGET_STOP_STATE.setdefault(id(self.model), {})
        state["reached"] = True
        state["bound"] = dual_bound
        self.model.interruptSolve()


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


def _scip_quicksum_compat(values) -> Any:
    if scip_quicksum is None:
        raise ScipUnavailableError("PySCIPOpt is not available. Install pyscipopt to use the SCIP backend.")
    return scip_quicksum(_unwrap_expr(value) for value in values)


def _coerce_int_param(name: str, value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    raise TypeError(f"SCIP parameter {name!r} expected an int-compatible value, got {type(value).__name__}.")


class _ScipTupleDict(dict):
    """Small dict-like stand in for the shared builder's tupledict API."""


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
            raise ScipUnavailableError("PySCIPOpt is not available. Install pyscipopt to use the SCIP backend.")
        self._name = name
        self._scip = PyScipModel()
        self._vars_by_name: dict[str, Any] = {}
        self._constraints_by_name: dict[str, Any] = {}
        self.Params = _ScipParamsCompat(self)
        if env is not None:
            for param_name, value in env.params.items():
                self._apply_param(param_name, value)

    def _apply_param(self, name: str, value: object) -> None:
        _apply_scip_param(self._scip, name, value)

    def addVar(
        self,
        *,
        lb: float | None = 0.0,
        ub: float | None = None,
        vtype: str,
        name: str,
    ) -> Any:
        var = self._scip.addVar(lb=lb, ub=ub, vtype=vtype, name=name)
        self._vars_by_name[name] = var
        return var

    def addVars(
        self,
        *index_sets: range,
        lb: float | None = 0.0,
        ub: float | None = None,
        vtype: str,
        name: str,
    ) -> _ScipTupleDict:
        if not index_sets:
            raise ValueError("addVars requires at least one index set.")
        result = _ScipTupleDict()
        for key_tuple in product(*index_sets):
            rendered_key = ",".join(str(part) for part in key_tuple)
            variable = self.addVar(lb=lb, ub=ub, vtype=vtype, name=f"{name}[{rendered_key}]")
            if len(key_tuple) == 1:
                result[key_tuple[0]] = variable
            else:
                result[key_tuple] = variable
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
        indices = [
            value
            for key, value in frame.f_locals.items()
            if not key.startswith(".") and isinstance(value, (str, int))
        ]
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
        var = self._vars_by_name.get(name)
        if var is None:
            return None
        return _ScipVarCompat(self, var)

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

    @property
    def ExactEnabled(self) -> bool:
        return bool(self._scip.getParam("exact/enable"))


class _ScipGrbCompat:
    BINARY = "B"
    CONTINUOUS = "C"
    MAXIMIZE = "maximize"


class _ScipGpCompat:
    Env = _ScipEnvCompat
    Model = _ScipModelCompat
    quicksum = staticmethod(_scip_quicksum_compat)


def _apply_scip_param(model: Any, name: str, value: object) -> None:
    if name in {"OutputFlag", "LogToConsole"}:
        verbose = 0 if _coerce_int_param(name, value) == 0 else 4
        model.setParam("display/verblevel", verbose)
        return
    if name == "TimeLimit":
        model.setParam("limits/time", value)
        return
    if name == "MIPGap":
        model.setParam("limits/gap", value)
        return
    if name == "Threads":
        model.setParam("parallel/maxnthreads", value)
        return
    if name == "Presolve":
        max_rounds = -1 if _coerce_int_param(name, value) > 0 else 0
        model.setParam("presolving/maxrounds", max_rounds)
        return
    if name == "SCIPExact":
        exact_enabled = bool(value)
        try:
            model.setBoolParam("exact/enable", exact_enabled)
            return
        except KeyError:
            pass

        if hasattr(model, "enableExactSolving"):
            try:
                model.enableExactSolving(exact_enabled)
                return
            except Exception as exc:
                raise ScipUnavailableError(
                    "This SCIP/PySCIPOpt build does not support exact solving. "
                    "Install SCIP 10+ with exact-solving support, or disable solver.scip_exact."
                ) from exc

        raise ScipUnavailableError(
            "This SCIP/PySCIPOpt build does not expose exact solving. "
            "Install SCIP 10+ with exact-solving support, or disable solver.scip_exact."
        )
        return
    if name == "NonConvex":
        return
    model.setParam(name, value)


@contextmanager
def _patched_core_backend():
    original_gp = obv_core.gp
    original_grb = obv_core.GRB
    original_exact_mtf_assignments = obv_core._apply_exact_mtf_assignments
    try:
        obv_core.gp = _ScipGpCompat()
        obv_core.GRB = _ScipGrbCompat
        obv_core._apply_exact_mtf_assignments = _apply_exact_mtf_assignments_scip
        yield
    finally:
        obv_core.gp = original_gp
        obv_core.GRB = original_grb
        obv_core._apply_exact_mtf_assignments = original_exact_mtf_assignments


def _apply_exact_mtf_assignments_scip(
    model: _ScipModelCompat,
    case: ExperimentCase,
    q: _ScipTupleDict,
    layout: obv_core.MtfProfileLayout,
) -> dict[int, tuple[int, ...]]:
    assignment = obv_core._build_exact_mtf_assignment(case, layout)
    assigned_machine_by_job = {
        job_index: machine_index for machine_index, machine_jobs in assignment.items() for job_index in machine_jobs
    }
    if len(assigned_machine_by_job) != case.job_count - 1:
        raise ValueError("Exact MTF assignment must cover each truncated job exactly once.")
    for job_index in range(1, case.job_count):
        assigned_machine = assigned_machine_by_job[job_index]
        for machine_index in range(1, case.machine_count + 1):
            value = 1.0 if machine_index == assigned_machine else 0.0
            model._scip.chgVarLb(q[machine_index, job_index], value)
            model._scip.chgVarUb(q[machine_index, job_index], value)
    return assignment


def build_obv_model(case: ExperimentCase) -> BuiltObvModel:
    if PyScipModel is None or scip_quicksum is None:
        raise ScipUnavailableError("PySCIPOpt is not available. Install pyscipopt to use the SCIP backend.")

    with _patched_core_backend():
        return obv_core.build_obv_model(case)


def install_exact_target_stop_handler(model: Any, case: ExperimentCase) -> None:
    if Eventhdlr is None or SCIP_EVENTTYPE is None:
        raise ScipUnavailableError("PySCIPOpt event support is unavailable for exact SCIP stopping.")
    if not case.solver.legacy_best_bd_stop_at_target or not case.enforce_target_lower_bound:
        return
    _load_scip_library()
    _get_scip_pointer(model)

    handler = _ExactDualBoundStopEventhdlr(case.target_ratio)
    model.includeEventhdlr(
        handler,
        "multifit_exact_target_stop",
        "Interrupt exact SCIP once the exact dual bound reaches the target ratio.",
    )
    _EXACT_TARGET_STOP_STATE[id(model)] = {
        "handler": handler,
        "reached": False,
        "bound": None,
    }


def exact_target_stop_reached(model: Any) -> bool:
    state = _EXACT_TARGET_STOP_STATE.get(id(model))
    if state is not None:
        return bool(state.get("reached", False))
    return bool(getattr(model, "_multifit_exact_target_reached", False))


def clear_exact_target_stop_state(model: Any) -> None:
    _EXACT_TARGET_STOP_STATE.pop(id(model), None)


def read_exact_problem(problem_path: str, case: ExperimentCase) -> Any:
    if PyScipModel is None:
        raise ScipUnavailableError("PySCIPOpt is not available. Install pyscipopt to use the SCIP backend.")

    model = PyScipModel()
    _apply_scip_param(model, "SCIPExact", True)
    _apply_scip_param(model, "OutputFlag", case.solver.output_flag)
    _apply_scip_param(model, "LogToConsole", case.solver.output_flag)
    if case.solver.time_limit_seconds is not None:
        _apply_scip_param(model, "TimeLimit", case.solver.time_limit_seconds)
    if case.solver.mip_gap is not None:
        _apply_scip_param(model, "MIPGap", case.solver.mip_gap)
    if case.solver.threads is not None:
        _apply_scip_param(model, "Threads", case.solver.threads)
    if case.solver.presolve is not None:
        _apply_scip_param(model, "Presolve", case.solver.presolve)
    model.readProblem(problem_path)
    install_exact_target_stop_handler(model, case)
    return model
