from __future__ import annotations

from pathlib import Path
import unittest

from multifit_optveri.acceleration import AccelerationCase
from multifit_optveri.config import load_experiment_config


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


if __name__ == "__main__":
    unittest.main()
