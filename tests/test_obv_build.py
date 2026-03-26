from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import unittest

from multifit_optveri.acceleration import AccelerationCase
from multifit_optveri.branching import MtfProfile, OptProfile
from multifit_optveri.config import SolverConfig
from multifit_optveri.experiments import ExperimentCase
from multifit_optveri.models import obv
from multifit_optveri.models.obv import build_obv_model
from multifit_optveri.models.spec import derive_obv_dimensions


def _case(
    *,
    acceleration_case: AccelerationCase,
    machine_count: int = 8,
    job_count: int = 24,
    ell: int | None = None,
    mtf_profile: MtfProfile | None = None,
    opt_profile: OptProfile | None = None,
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
            self.assertEqual(built.dimensions, expected)
            self.assertEqual(built.model.NumVars, expected.total_variables)
            self.assertEqual(built.model.NumConstrs, expected.total_constraints)
            self.assertEqual(built.model.ModelSense, obv.GRB.MAXIMIZE)
        finally:
            built.model.dispose()

    def test_profiled_acceleration_case_builds_with_extra_constraints(self) -> None:
        case = _case(
            acceleration_case=AccelerationCase.CASE_2,
            ell=10,
            job_count=24,
            mtf_profile=MtfProfile(0, 4, 0, 1, 0, 3, 0),
            opt_profile=OptProfile(8, 0, 0, pattern="two_long"),
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
            self.assertEqual(built.model.NumVars, base_expected.total_variables)
            self.assertGreater(built.model.NumConstrs, base_expected.total_constraints)
            pn_var = built.model.getVarByName("p[24]")
            self.assertIsNotNone(pn_var)
            self.assertAlmostEqual(pn_var.LB, float(Fraction(7, 34)))
            self.assertAlmostEqual(pn_var.UB, float(Fraction(11, 51)))
        finally:
            built.model.dispose()


if __name__ == "__main__":
    unittest.main()
