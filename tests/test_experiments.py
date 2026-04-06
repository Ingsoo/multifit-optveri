from __future__ import annotations

from pathlib import Path
import unittest

from multifit_optveri.acceleration import AccelerationCase
from multifit_optveri.config import ExperimentConfig, SolverConfig
from multifit_optveri.experiments import (
    ExperimentCase,
    _upper_2_machine_block_count,
    derive_job_bounds,
    enumerate_cases,
)
from multifit_optveri.math_utils import parse_ratio


class ExperimentTests(unittest.TestCase):
    def test_upper_2_machine_block_count_matches_paper_values(self) -> None:
        target = parse_ratio("20/17")
        actual = {machine_count: _upper_2_machine_block_count(machine_count, target) for machine_count in range(8, 13)}
        self.assertEqual(actual, {8: 4, 9: 5, 10: 5, 11: 5, 12: 5})

    def test_derived_job_bounds_match_paper_range(self) -> None:
        target = parse_ratio("20/17")
        expected = {
            8: (24, 32),
            9: (27, 45),
            10: (30, 50),
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

    def test_case_enumeration_follows_case_mtf_then_opt_branching(self) -> None:
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

    def test_case_enumeration_prefilters_machine_job_and_acceleration_case(self) -> None:
        config = ExperimentConfig(
            name="paper_base",
            target_ratio=parse_ratio("20/17"),
            machine_values=(8, 9),
            derive_job_counts=True,
            explicit_job_counts=(),
            acceleration_cases=(AccelerationCase.CASE_1, AccelerationCase.CASE_2),
            output_root=Path("results"),
            write_lp=False,
            enforce_target_lower_bound=True,
            solver=SolverConfig(),
        )

        cases = enumerate_cases(
            config,
            machine=8,
            job=24,
            acceleration_case=AccelerationCase.CASE_1,
            limit=5,
        )

        self.assertEqual(len(cases), 5)
        self.assertTrue(all(case.machine_count == 8 for case in cases))
        self.assertTrue(all(case.job_count == 24 for case in cases))
        self.assertTrue(
            all(case.acceleration_case is AccelerationCase.CASE_1 for case in cases)
        )

    def test_case_2_enumeration_materializes_fallback_starts(self) -> None:
        config = ExperimentConfig(
            name="paper_base",
            target_ratio=parse_ratio("20/17"),
            machine_values=(12,),
            derive_job_counts=False,
            explicit_job_counts=(42,),
            acceleration_cases=(AccelerationCase.CASE_2,),
            output_root=Path("results"),
            write_lp=False,
            enforce_target_lower_bound=True,
            solver=SolverConfig(),
        )

        cases = enumerate_cases(config, limit=10)

        self.assertTrue(cases)
        self.assertTrue(all(case.fallback_starts is not None for case in cases))
        self.assertTrue(any("fs" in case.case_id for case in cases))

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
