from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from multifit_optveri.acceleration import AccelerationCase
from multifit_optveri.config import SolverConfig
from multifit_optveri.experiments import ExperimentCase
from multifit_optveri.math_utils import parse_ratio
from multifit_optveri.runner import create_run_artifacts, prepare_cases_for_run


class RunnerLayoutTests(unittest.TestCase):
    def test_create_run_artifacts_and_prepare_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            case = ExperimentCase(
                experiment_name="demo",
                machine_count=8,
                job_count=24,
                acceleration_case=AccelerationCase.CASE_1,
                ell=9,
                mtf_profile=None,
                opt_profile=None,
                target_ratio=parse_ratio("20/17"),
                output_root=Path(tmpdir),
                write_lp=False,
                enforce_target_lower_bound=True,
                solver=SolverConfig(),
            )

            artifacts = create_run_artifacts([case])
            prepared_case = prepare_cases_for_run([case], artifacts.run_dir)[0]

            self.assertTrue(artifacts.run_dir.exists())
            self.assertTrue(artifacts.cases_dir.exists())
            self.assertEqual(
                artifacts.latest_run_path.read_text(encoding="utf-8"),
                str(artifacts.run_dir),
            )
            self.assertEqual(
                prepared_case.output_dir,
                artifacts.run_dir / "cases" / prepared_case.case_id,
            )

    def test_create_run_artifacts_skips_cases_dir_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            case = ExperimentCase(
                experiment_name="demo",
                machine_count=8,
                job_count=24,
                acceleration_case=AccelerationCase.CASE_1,
                ell=9,
                mtf_profile=None,
                opt_profile=None,
                target_ratio=parse_ratio("20/17"),
                output_root=Path(tmpdir),
                write_lp=False,
                enforce_target_lower_bound=True,
                solver=SolverConfig(),
                write_case_dirs=False,
            )

            artifacts = create_run_artifacts([case])

            self.assertFalse(artifacts.cases_dir.exists())


if __name__ == "__main__":
    unittest.main()
