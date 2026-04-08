from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from unittest.mock import patch
import unittest

from multifit_optveri.acceleration import AccelerationCase
from multifit_optveri.branching import FallbackStarts, MtfProfile, OptProfile
from multifit_optveri.config import SolverConfig, load_experiment_config
from multifit_optveri.experiments import ExperimentCase, enumerate_cases
from multifit_optveri.models import obv
from multifit_optveri.models.obv import (
    GurobiUnavailableError,
    _as_float,
    _build_mtf_profile_layout,
    _build_opt_machine_groups,
    _build_exact_mtf_assignment,
    _common_processing_time_lower_bound,
    _machine_after_pair_block,
    _mtf_cardinality_upper_bound,
    _opt_cardinality_upper_bound,
    _processing_time_lower_bound,
    _processing_time_upper_bound,
    _require_gurobi,
    _validate_paper_acceleration_case,
    build_obv_model,
)


def _sample_case(
    *,
    acceleration_case: AccelerationCase = AccelerationCase.CASE_1,
    machine_count: int = 8,
    job_count: int = 24,
    ell: int = 9,
    mtf_profile: MtfProfile | None = None,
    opt_profile: OptProfile | None = None,
    fallback_starts: FallbackStarts | None = None,
    target_ratio: Fraction = Fraction(20, 17),
) -> ExperimentCase:
    return ExperimentCase(
        experiment_name="demo",
        machine_count=machine_count,
        job_count=job_count,
        acceleration_case=acceleration_case,
        ell=ell,
        mtf_profile=mtf_profile,
        opt_profile=opt_profile,
        target_ratio=target_ratio,
        output_root=Path("results"),
        write_lp=False,
        enforce_target_lower_bound=True,
        solver=SolverConfig(),
        fallback_starts=fallback_starts,
    )


class ObvHelperTests(unittest.TestCase):
    def test_as_float_and_processing_time_bounds(self) -> None:
        case = _sample_case()

        self.assertAlmostEqual(_as_float(Fraction(3, 2)), 1.5)
        self.assertEqual(_common_processing_time_lower_bound(8, Fraction(20, 17)), Fraction(24, 119))
        self.assertEqual(_processing_time_lower_bound(case), Fraction(11, 51))
        self.assertEqual(_processing_time_upper_bound(case, 1, Fraction(11, 51)), Fraction(29, 51))
        self.assertEqual(_processing_time_upper_bound(case, 9, Fraction(11, 51)), Fraction(1, 2))
        self.assertEqual(_processing_time_upper_bound(case, 24, Fraction(11, 51)), Fraction(1, 3))

    def test_cardinality_upper_bounds_match_old_strengthening(self) -> None:
        self.assertEqual(_opt_cardinality_upper_bound(Fraction(11, 51)), 4)
        self.assertEqual(_mtf_cardinality_upper_bound(8, Fraction(20, 17)), 4)

    def test_validate_paper_acceleration_case(self) -> None:
        _validate_paper_acceleration_case(_sample_case())

        with self.assertRaises(ValueError):
            _validate_paper_acceleration_case(_sample_case(target_ratio=Fraction(6, 5)))
        with self.assertRaises(ValueError):
            _validate_paper_acceleration_case(_sample_case(machine_count=7))

    def test_layout_helpers(self) -> None:
        case = _sample_case(
            acceleration_case=AccelerationCase.CASE_3_2,
            job_count=27,
            ell=6,
            mtf_profile=MtfProfile(1, 2, 1, 1, 1, 1, 0, 1),
            opt_profile=OptProfile(3, 4, 1, pattern="generic"),
        )

        layout = _build_mtf_profile_layout(case)
        s3_machines, s4_machines, s5_machines = _build_opt_machine_groups(case, range(1, 9))

        self.assertEqual(layout.f1_machines, (1,))
        self.assertEqual(layout.r2_machines, (2, 3))
        self.assertEqual(layout.f2_machines, (4,))
        self.assertEqual(layout.r3_machines, (5,))
        self.assertEqual(layout.f3_machines, (6,))
        self.assertEqual(layout.r4_machines, (7,))
        self.assertEqual(layout.f4_machines, ())
        self.assertEqual(layout.r5_machines, (8,))
        self.assertEqual((layout.e2, layout.e3, layout.e4, layout.e5), (2, 8, 14, 18))
        self.assertEqual((layout.t2, layout.t3, layout.t4, layout.t5), (3, 10, 17, 21))
        self.assertEqual(s3_machines, (1, 2, 3))
        self.assertEqual(s4_machines, (4, 5, 6, 7))
        self.assertEqual(s5_machines, (8,))
        self.assertEqual(_machine_after_pair_block(layout), 5)

    def test_case_2_exact_assignment_reconstruction(self) -> None:
        case = _sample_case(
            acceleration_case=AccelerationCase.CASE_2,
            machine_count=8,
            job_count=30,
            ell=9,
            mtf_profile=MtfProfile(0, 0, 1, 4, 0, 1, 0, 2),
            opt_profile=OptProfile(7, 1, 0, pattern="regular"),
            fallback_starts=FallbackStarts(6, None, None),
        )

        layout = _build_mtf_profile_layout(case)
        assignment = _build_exact_mtf_assignment(case, layout)

        self.assertEqual(
            assignment,
            {
                1: (1, 2, 6),
                2: (3, 4, 5),
                3: (7, 8, 9),
                4: (10, 11, 12),
                5: (13, 14, 15),
                6: (16, 17, 18, 19),
                7: (20, 21, 22, 23, 24),
                8: (25, 26, 27, 28, 29),
            },
        )

    def test_case_3_exact_assignment_reconstruction(self) -> None:
        case = _sample_case(
            acceleration_case=AccelerationCase.CASE_3,
            machine_count=8,
            job_count=25,
            ell=3,
            mtf_profile=MtfProfile(3, 1, 0, 0, 0, 4, 0, 0),
            opt_profile=OptProfile(7, 1, 0, pattern="generic"),
            fallback_starts=FallbackStarts(None, None, None),
        )

        layout = _build_mtf_profile_layout(case)
        assignment = _build_exact_mtf_assignment(case, layout)

        self.assertEqual(
            assignment,
            {
                1: (1, 5),
                2: (2, 6),
                3: (3, 7),
                4: (4, 8),
                5: (9, 10, 11, 12),
                6: (13, 14, 15, 16),
                7: (17, 18, 19, 20),
                8: (21, 22, 23, 24),
            },
        )

    def test_gurobi_guards_raise_without_solver(self) -> None:
        with patch.object(obv, "gp", None), patch.object(obv, "GRB", None):
            with self.assertRaises(GurobiUnavailableError):
                _require_gurobi()
            with self.assertRaises(GurobiUnavailableError):
                build_obv_model(_sample_case())

    def test_processing_time_bounds_are_consistent_for_all_m8_branches(self) -> None:
        config = load_experiment_config("configs/experiments/paper_base.toml")

        for case in enumerate_cases(config):
            if case.machine_count != 8:
                continue
            lower_bound = _processing_time_lower_bound(case)
            for job_index in range(1, case.job_count + 1):
                upper_bound = _processing_time_upper_bound(case, job_index, lower_bound)
                self.assertGreaterEqual(upper_bound, lower_bound, msg=f"{case.case_id} job {job_index}")


if __name__ == "__main__":
    unittest.main()
