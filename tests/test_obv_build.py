from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast
import unittest

from multifit_optveri.acceleration import AccelerationCase
from multifit_optveri.branching import FallbackStarts, MtfProfile, OptProfile
from multifit_optveri.config import SolverConfig
from multifit_optveri.experiments import ExperimentCase
from multifit_optveri.models import obv
from multifit_optveri.models.obv import build_obv_model
from multifit_optveri.models.spec import derive_obv_dimensions

if TYPE_CHECKING:
    from gurobipy import Model as GurobiModel
    from gurobipy import Var as GurobiVar
else:
    GurobiModel = Any
    GurobiVar = Any


def _linear_row_var_names(model: GurobiModel, constr_name: str) -> list[str]:
    constr = model.getConstrByName(constr_name)
    if constr is None:
        return []
    row = model.getRow(constr)
    return [row.getVar(index).VarName for index in range(row.size())]


class _ConstraintCountModel(Protocol):
    NumConstrs: int
    NumQConstrs: int
    NumGenConstrs: int


def _constraint_total(model: _ConstraintCountModel) -> int:
    return int(model.NumConstrs) + int(model.NumQConstrs) + int(model.NumGenConstrs)


def _case(
    *,
    acceleration_case: AccelerationCase,
    machine_count: int = 8,
    job_count: int = 24,
    ell: int | None = None,
    mtf_profile: MtfProfile | None = None,
    opt_profile: OptProfile | None = None,
    fallback_starts: FallbackStarts | None = None,
) -> ExperimentCase:
    return ExperimentCase(
        experiment_name="demo",
        machine_count=machine_count,
        job_count=job_count,
        acceleration_case=acceleration_case,
        ell=ell,
        mtf_profile=mtf_profile,
        opt_profile=opt_profile,
        target_ratio=Fraction(20, 17),
        output_root=Path("results"),
        write_lp=False,
        enforce_target_lower_bound=True,
        solver=SolverConfig(output_flag=0),
        fallback_starts=fallback_starts,
    )


@unittest.skipIf(obv.gp is None or obv.GRB is None, "gurobipy is unavailable")
class ObvBuildTests(unittest.TestCase):
    def test_base_model_matches_spec_dimensions(self) -> None:
        case = _case(acceleration_case=AccelerationCase.BASE)
        built = build_obv_model(case)
        expected = derive_obv_dimensions(
            8,
            24,
            include_target_lower_bound=True,
            acceleration_case=AccelerationCase.BASE,
            include_profile_cardinality_constraints=False,
        )

        try:
            model: GurobiModel = built.model
            self.assertEqual(built.dimensions, expected)
            self.assertEqual(model.NumVars, expected.total_variables)
            self.assertEqual(_constraint_total(model), expected.total_constraints)
            self.assertEqual(model.ModelSense, obv.GRB.MAXIMIZE)
        finally:
            built.model.dispose()

    def test_profiled_acceleration_case_builds_with_extra_constraints(self) -> None:
        case = _case(
            acceleration_case=AccelerationCase.CASE_2,
            ell=10,
            job_count=24,
            mtf_profile=MtfProfile(0, 4, 0, 1, 0, 3, 0, 0),
            opt_profile=OptProfile(8, 0, 0, pattern="two_long"),
            fallback_starts=FallbackStarts(None, None, None),
        )
        built = build_obv_model(case)
        base_expected = derive_obv_dimensions(
            8,
            24,
            include_target_lower_bound=True,
            acceleration_case=AccelerationCase.CASE_2,
            include_profile_cardinality_constraints=True,
        )

        try:
            model: GurobiModel = built.model
            self.assertEqual(model.NumVars, base_expected.total_variables)
            self.assertGreater(_constraint_total(model), base_expected.total_constraints)
            pn_var = cast(GurobiVar | None, model.getVarByName("p[24]"))
            self.assertIsNotNone(pn_var)
            self.assertAlmostEqual(pn_var.LB, float(Fraction(7, 34)))
            self.assertAlmostEqual(pn_var.UB, float(Fraction(11, 51)))
        finally:
            built.model.dispose()

    def test_split_constraints_also_tighten_variable_bounds(self) -> None:
        case = _case(
            acceleration_case=AccelerationCase.CASE_2,
            ell=10,
            job_count=24,
            mtf_profile=MtfProfile(0, 4, 0, 1, 0, 3, 0, 0),
            opt_profile=OptProfile(8, 0, 0, pattern="two_long"),
            fallback_starts=FallbackStarts(None, None, None),
        )
        built = build_obv_model(case)

        try:
            model: GurobiModel = built.model
            p1_var = cast(GurobiVar | None, model.getVarByName("p[1]"))
            p10_var = cast(GurobiVar | None, model.getVarByName("p[10]"))
            self.assertIsNotNone(p1_var)
            self.assertIsNotNone(p10_var)
            self.assertAlmostEqual(p1_var.LB, float(Fraction(13, 34)))
            self.assertAlmostEqual(p10_var.UB, float(Fraction(20, 51)))
        finally:
            built.model.dispose()

    def test_profiled_opt_tail_sum_constraint_is_added(self) -> None:
        case = _case(
            acceleration_case=AccelerationCase.CASE_1,
            ell=9,
            job_count=24,
            mtf_profile=MtfProfile(0, 4, 0, 2, 0, 1, 0, 1),
            opt_profile=OptProfile(8, 0, 0, pattern="case1"),
        )
        built = build_obv_model(case)

        try:
            model: GurobiModel = built.model
            self.assertIsNotNone(model.getConstrByName("opt_profile_tail_sum"))
        finally:
            built.model.dispose()

    def test_opt_profile_without_three_job_machines_tightens_p_upper_bounds(self) -> None:
        case = _case(
            acceleration_case=AccelerationCase.BASE,
            machine_count=8,
            job_count=32,
            opt_profile=OptProfile(0, 8, 0),
        )
        built = build_obv_model(case)

        try:
            model: GurobiModel = built.model
            p1_var = cast(GurobiVar | None, model.getVarByName("p[1]"))
            self.assertIsNotNone(p1_var)
            self.assertAlmostEqual(p1_var.UB, float(Fraction(47, 119)))
        finally:
            built.model.dispose()

    def test_case_2_exact_assignment_constraints_are_added(self) -> None:
        case = _case(
            acceleration_case=AccelerationCase.CASE_2,
            machine_count=8,
            job_count=30,
            ell=9,
            mtf_profile=MtfProfile(0, 0, 1, 4, 0, 1, 0, 2),
            opt_profile=OptProfile(7, 1, 0, pattern="regular"),
            fallback_starts=FallbackStarts(6, None, None),
        )
        built = build_obv_model(case)

        try:
            model: GurobiModel = built.model
            self.assertIsNotNone(model.getConstrByName("case2_exact_q[1,1]"))
            self.assertIsNotNone(model.getConstrByName("case2_exact_q[1,3]"))
            self.assertIsNotNone(model.getConstrByName("R2_valid_constr[1]"))
            self.assertIsNotNone(model.getConstrByName("R3_valid_constr[5]"))
            self.assertIsNotNone(model.getConstrByName("R4_valid_constr[6]"))
            self.assertIsNotNone(model.getConstrByName("R5_valid_constr[8]"))
            self.assertIsNotNone(model.getConstrByName("R3_processing_times[3]"))
            self.assertIsNotNone(model.getConstrByName("R3_processing_times[4]"))
            self.assertIsNotNone(model.getConstrByName("R3_processing_times[7]"))
            self.assertIsNotNone(model.getConstrByName("R3_processing_times[12]"))
            self.assertIsNone(model.getConstrByName("R3_processing_times[5]"))
            self.assertIsNone(model.getConstrByName("F2_processing_times[3]"))
            self.assertIsNone(model.getConstrByName("opt_cardinality_lb[1]"))
            self.assertIsNone(model.getConstrByName("opt_cardinality_ub[1]"))
            self.assertIsNone(model.getConstrByName("opt_cardinality_order[1]"))
            self.assertIsNone(model.getConstrByName("opt_smallest_jobs_sum"))
            self.assertIsNone(model.getConstrByName("mtf_cardinality_lb[1]"))
            self.assertIsNone(model.getConstrByName("mtf_cardinality_ub[1]"))
            self.assertIsNone(model.getConstrByName("mtf_init_fixed[1]"))
            self.assertIsNone(model.getConstrByName("mtf_profile_cardinality[1]"))
        finally:
            built.model.dispose()

    def test_case_2_exact_f2_regular_and_fallback_processing_equalities_are_added(self) -> None:
        case = _case(
            acceleration_case=AccelerationCase.CASE_2,
            machine_count=12,
            job_count=42,
            ell=8,
            mtf_profile=MtfProfile(0, 0, 3, 6, 1, 0, 0, 2),
            opt_profile=OptProfile(6, 6, 0, pattern="two_long"),
            fallback_starts=FallbackStarts(10, 32, None),
        )
        built = build_obv_model(case)

        try:
            model: GurobiModel = built.model
            self.assertIsNotNone(model.getConstrByName("F2_processing_times[1]"))
            self.assertIsNotNone(model.getConstrByName("F2_processing_times[2]"))
            self.assertIsNotNone(model.getConstrByName("F2_processing_times[3]"))
            self.assertIsNotNone(model.getConstrByName("F2_processing_times[4]"))
            self.assertIsNone(model.getConstrByName("F2_processing_times[5]"))
            self.assertIsNotNone(model.getConstrByName("F2_fallback_processing_times[10]"))
            self.assertIsNotNone(model.getConstrByName("F2_fallback_processing_times[11]"))
            self.assertIsNone(model.getConstrByName("F2_fallback_processing_times[12]"))
            self.assertEqual(
                _linear_row_var_names(model, "F2_valid_constr[3]"),
                ["p[5]", "p[6]", "p[7]"],
            )
        finally:
            built.model.dispose()


if __name__ == "__main__":
    unittest.main()
