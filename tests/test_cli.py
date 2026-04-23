from __future__ import annotations

from fractions import Fraction
from io import StringIO
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import contextlib
import tempfile
import unittest

from multifit_optveri.acceleration import AccelerationCase
from multifit_optveri.cli import _build_parser, _filter_cases, _load_case_ids, main
from multifit_optveri.config import ExperimentConfig, SolverConfig
from multifit_optveri.experiments import ExperimentCase


def _sample_case(case_id_suffix: str) -> ExperimentCase:
    return ExperimentCase(
        experiment_name="demo",
        machine_count=8,
        job_count=24,
        acceleration_case=AccelerationCase.CASE_1,
        ell=9,
        mtf_profile=None,
        opt_profile=None,
        target_ratio=Fraction(20, 17),
        output_root=Path("results"),
        write_lp=False,
        enforce_target_lower_bound=True,
        solver=SolverConfig(),
        run_output_root=Path("results/demo/20260326_120000"),
    )


class CliTests(unittest.TestCase):
    def test_build_parser_requires_subcommand(self) -> None:
        parser = _build_parser()
        with contextlib.redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args([])

    def test_filter_cases_applies_all_filters(self) -> None:
        cases = [_sample_case("a"), _sample_case("b")]
        filtered = _filter_cases(cases, machine=8, job=24, acceleration_case="case_1", limit=1)
        self.assertEqual(len(filtered), 1)

    def test_load_case_ids_ignores_comments_and_blanks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            case_list = Path(tmpdir) / "cases.txt"
            case_list.write_text("# comment\ncase_a\n\n case_b \n", encoding="utf-8")

            self.assertEqual(_load_case_ids(case_list), {"case_a", "case_b"})

    def test_main_plan_prints_case_plan(self) -> None:
        output = StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(["plan", "--config", "tests/fixtures/demo_config.toml"])
        self.assertEqual(exit_code, 0)
        self.assertIn("Enumerating cases...", output.getvalue())
        self.assertIn("Total cases:", output.getvalue())

    def test_main_plan_passes_filters_into_enumeration(self) -> None:
        case_a = _sample_case("a")
        output = StringIO()

        with patch(
            "multifit_optveri.cli.load_experiment_config",
            return_value=SimpleNamespace(),
        ), patch(
            "multifit_optveri.cli.enumerate_cases",
            return_value=[case_a],
        ) as enumerate_cases_mock, contextlib.redirect_stdout(output):
            exit_code = main(["plan", "--config", "dummy.toml", "--machine", "8", "--limit", "1"])

        self.assertEqual(exit_code, 0)
        enumerate_cases_mock.assert_called_once_with(
            unittest.mock.ANY,
            machine=8,
            job=None,
            acceleration_case=None,
            limit=None,
        )
        self.assertIn("Total cases: 1", output.getvalue())
        self.assertIn("m=8", output.getvalue())

    def test_main_plan_filters_with_case_list(self) -> None:
        case_a = _sample_case("a")
        case_b = replace(case_a, job_count=25)
        output = StringIO()

        with tempfile.TemporaryDirectory() as tmpdir:
            case_list = Path(tmpdir) / "cases.txt"
            case_list.write_text(f"{case_a.case_id}\n", encoding="utf-8")
            with patch(
                "multifit_optveri.cli.load_experiment_config",
                return_value=SimpleNamespace(),
            ), patch(
                "multifit_optveri.cli.enumerate_cases",
                return_value=[case_a, case_b],
            ), contextlib.redirect_stdout(output):
                exit_code = main(["plan", "--config", "dummy.toml", "--case-list", str(case_list)])

        self.assertEqual(exit_code, 0)
        rendered = output.getvalue()
        self.assertIn(case_a.case_id, rendered)
        self.assertNotIn(case_b.case_id, rendered)

    def test_main_run_uses_recorder_flow(self) -> None:
        config = ExperimentConfig(
            name="demo",
            target_ratio=Fraction(20, 17),
            machine_values=(8,),
            derive_job_counts=False,
            explicit_job_counts=(24,),
            acceleration_cases=(AccelerationCase.CASE_1,),
            output_root=Path("results"),
            write_lp=False,
            enforce_target_lower_bound=True,
            solver=SolverConfig(),
        )
        case = _sample_case("x")

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = SimpleNamespace(
                run_dir=Path(tmpdir) / "demo" / "20260326_120000",
                summary_csv_path=Path(tmpdir) / "demo" / "20260326_120000" / "summary.csv",
                overview_json_path=Path(tmpdir) / "demo" / "20260326_120000" / "overview.json",
            )
            recorder = SimpleNamespace(record=lambda result: None, finish=lambda: None)
            result = SimpleNamespace(
                case_id=case.case_id,
                acceleration_case=case.acceleration_case.value,
                status="INFEASIBLE",
                objective_value=None,
                runtime_seconds=0.1,
            )
            output = StringIO()
            with patch("multifit_optveri.cli.load_experiment_config", return_value=config), patch(
                "multifit_optveri.cli.enumerate_cases", return_value=[case]
            ), patch(
                "multifit_optveri.cli.create_run_artifacts", return_value=artifacts
            ), patch(
                "multifit_optveri.cli.prepare_cases_for_run", return_value=[case]
            ), patch(
                "multifit_optveri.cli.RunRecorder", return_value=recorder
            ), patch(
                "multifit_optveri.cli.run_case", return_value=result
            ), contextlib.redirect_stdout(output):
                exit_code = main(["run", "--config", "dummy.toml", "--machine", "8"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Enumerated 1 case(s) matching CLI filters.", output.getvalue())
        self.assertIn("Running 1 case(s)...", output.getvalue())
        self.assertIn("Run directory:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
