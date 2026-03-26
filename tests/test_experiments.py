from __future__ import annotations

from pathlib import Path
import unittest

from multifit_optveri.acceleration import AccelerationCase
from multifit_optveri.config import ExperimentConfig, SolverConfig
from multifit_optveri.experiments import ExperimentCase, derive_job_bounds, enumerate_cases
from multifit_optveri.math_utils import parse_ratio


class ExperimentTests(unittest.TestCase):
    def test_derived_job_bounds_match_paper_range(self) -> None:
        target = parse_ratio("20/17")
        expected = {
            8: (24, 32),
            9: (27, 37),
            10: (30, 41),
            11: (33, 55),
            12: (36, 60),
        }
        actual = {
            machine_count: (
                derive_job_bounds(machine_count, target).lower,
                derive_job_bounds(machine_count, target).upper,
            )
            for machine_count in expected
        }
        self.assertEqual(actual, expected)

    def test_case_enumeration_follows_case_mtfo_opto_branching(self) -> None:
        config = ExperimentConfig(
            name="paper_base",
            target_ratio=parse_ratio("20/17"),
            machine_values=(8,),
            derive_job_counts=False,
            explicit_job_counts=(24,),
            acceleration_cases=(AccelerationCase.CASE_1,),
            output_root=Path("results"),
            write_lp=False,
            enforce_target_lower_bound=True,
            solver=SolverConfig(),
        )

        cases = enumerate_cases(config)

        self.assertGreater(len(cases), 0)
        self.assertTrue(all(case.ell == 9 for case in cases))
        self.assertTrue(all(case.job_count == 24 for case in cases))
        self.assertTrue(all(case.mtf_profile is not None for case in cases))
        self.assertTrue(all(case.opt_profile is not None for case in cases))
        self.assertTrue(all(case.case_id.startswith("c1_m08_n024_e09_") for case in cases))

    def test_case_output_dir_uses_run_root_when_present(self) -> None:
        case = ExperimentCase(
            experiment_name="paper_base",
            machine_count=8,
            job_count=24,
            acceleration_case=AccelerationCase.CASE_1,
            ell=9,
            mtf_profile=None,
            opt_profile=None,
            target_ratio=parse_ratio("20/17"),
            output_root=Path("results"),
            write_lp=False,
            enforce_target_lower_bound=True,
            solver=SolverConfig(),
            run_output_root=Path("results/paper_base/20260326_120000"),
        )

        self.assertEqual(
            case.output_dir,
            Path("results/paper_base/20260326_120000/cases") / case.case_id,
        )


if __name__ == "__main__":
    unittest.main()
