from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from multifit_optveri.acceleration import AccelerationCase
from multifit_optveri.config import ExperimentConfig, SolverConfig, load_experiment_config
from multifit_optveri.math_utils import parse_ratio


class ConfigTests(unittest.TestCase):
    def test_load_experiment_config(self) -> None:
        path = Path("tests/fixtures/demo_config.toml")
        config = load_experiment_config(path)

        self.assertEqual(config.name, "demo")
        self.assertEqual(config.machine_values, (8, 9))
        self.assertEqual(str(config.target_ratio), "20/17")
        self.assertTrue(config.derive_job_counts)
        self.assertEqual(
            config.acceleration_cases,
            (
                AccelerationCase.CASE_1,
                AccelerationCase.CASE_2,
                AccelerationCase.CASE_3_1,
                AccelerationCase.CASE_3_2,
            ),
        )
        self.assertEqual(config.solver.threads, 2)
        self.assertEqual(config.output_root, Path("results"))
        self.assertTrue(config.write_case_dirs)

    def test_load_experiment_config_can_disable_case_dirs(self) -> None:
        config_text = """[experiment]
name = "demo"
target_ratio = "20/17"
machine_values = [8]
derive_job_counts = true
explicit_job_counts = []
acceleration_cases = ["case_1"]
output_root = "results"
write_lp = false
write_case_dirs = false
enforce_target_lower_bound = true
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.toml"
            path.write_text(config_text, encoding="utf-8")
            config = load_experiment_config(path)

        self.assertFalse(config.write_case_dirs)

    def test_experiment_config_rejects_lp_without_case_dirs(self) -> None:
        with self.assertRaises(ValueError):
            ExperimentConfig(
                name="demo",
                target_ratio=parse_ratio("20/17"),
                machine_values=(8,),
                derive_job_counts=True,
                explicit_job_counts=(),
                acceleration_cases=(AccelerationCase.CASE_1,),
                output_root=Path("results"),
                write_lp=True,
                enforce_target_lower_bound=True,
                solver=SolverConfig(),
                write_case_dirs=False,
            )


if __name__ == "__main__":
    unittest.main()
