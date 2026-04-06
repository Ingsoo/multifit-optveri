from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest

from multifit_optveri.acceleration import AccelerationCase
from multifit_optveri.branching import MtfProfile, OptProfile
from multifit_optveri.config import SolverConfig
from multifit_optveri.experiments import ExperimentCase
from multifit_optveri.math_utils import parse_ratio
from multifit_optveri.runner import (
    RunRecorder,
    SolveResult,
    _allocate_run_dir,
    _extract_optimal_p_values_desc,
    _format_mtf_profile,
    _format_opt_profile,
    _result_csv_fieldnames,
    _result_csv_row,
    _status_name,
    _verification_result,
    create_run_artifacts,
    run_cases,
)


def _sample_case(output_root: Path) -> ExperimentCase:
    return ExperimentCase(
        experiment_name="demo",
        machine_count=8,
        job_count=24,
        acceleration_case=AccelerationCase.CASE_1,
        ell=9,
        mtf_profile=None,
        opt_profile=None,
        target_ratio=parse_ratio("20/17"),
        output_root=output_root,
        write_lp=False,
        enforce_target_lower_bound=True,
        solver=SolverConfig(),
    )


def _sample_result(output_dir: Path, *, case_id: str = "case_a") -> SolveResult:
    return SolveResult(
        experiment_name="demo",
        case_id=case_id,
        acceleration_case="case_1",
        machine_count=8,
        job_count=24,
        ell=9,
        mtf_profile="(0,4,0,1,0,3,0,0)",
        opt_profile="(8,0,0)",
        target_ratio="20/17",
        verification_result="VERIFIED",
        status="INFEASIBLE",
        objective_value=None,
        objective_bound=1.17647,
        runtime_seconds=0.25,
        node_count=0.0,
        mip_gap=None,
        optimal_p_values_desc=None,
        output_dir=str(output_dir),
        built_at_utc="2026-03-26T00:00:00+00:00",
    )


class RunnerTests(unittest.TestCase):
    def test_status_name_uses_grb_mapping(self) -> None:
        fake_grb = SimpleNamespace(
            OPTIMAL=2,
            INFEASIBLE=3,
            TIME_LIMIT=9,
            INTERRUPTED=11,
            INF_OR_UNBD=4,
            UNBOUNDED=5,
            USER_OBJ_LIMIT=15,
        )
        with patch("multifit_optveri.runner.GRB", fake_grb):
            self.assertEqual(_status_name(2), "OPTIMAL")
            self.assertEqual(_status_name(15), "USER_OBJ_LIMIT")
            self.assertEqual(_status_name(999), "999")

    def test_result_csv_helpers(self) -> None:
        row = _result_csv_row(_sample_result(Path("results/demo/case_a")))

        self.assertIn("case_id", _result_csv_fieldnames())
        self.assertIn("mtf-profile-(f1_r2_f2_r3_f3_r4_f4_r5)", _result_csv_fieldnames())
        self.assertIn("opt-profile-(e3_e4_e5)", _result_csv_fieldnames())
        self.assertIn("verification_result", _result_csv_fieldnames())
        self.assertIn("optimal-p-values-(desc-scaled)", _result_csv_fieldnames())
        self.assertEqual(row["case_id"], "case_a")
        self.assertEqual(row["objective_value"], "")
        self.assertEqual(row["status"], "INFEASIBLE")
        self.assertEqual(row["verification_result"], "VERIFIED")
        self.assertEqual(row["mtf-profile-(f1_r2_f2_r3_f3_r4_f4_r5)"], "(0,4,0,1,0,3,0,0)")
        self.assertEqual(row["opt-profile-(e3_e4_e5)"], "(8,0,0)")
        self.assertEqual(row["optimal-p-values-(desc-scaled)"], "")

    def test_verification_result_matches_algorithm_rule(self) -> None:
        self.assertEqual(
            _verification_result(status="INFEASIBLE", objective_value=None, target_ratio="20/17"),
            "VERIFIED",
        )
        self.assertEqual(
            _verification_result(status="OPTIMAL", objective_value=20 / 17, target_ratio="20/17"),
            "VERIFIED",
        )
        self.assertEqual(
            _verification_result(status="OPTIMAL", objective_value=1.2, target_ratio="20/17"),
            "NOT_VERIFIED",
        )
        self.assertEqual(
            _verification_result(status="TIME_LIMIT", objective_value=None, target_ratio="20/17"),
            "UNKNOWN",
        )

    def test_profile_format_helpers(self) -> None:
        case = ExperimentCase(
            experiment_name="demo",
            machine_count=8,
            job_count=24,
            acceleration_case=AccelerationCase.CASE_2,
            ell=10,
            mtf_profile=None,
            opt_profile=None,
            target_ratio=parse_ratio("20/17"),
            output_root=Path("results"),
            write_lp=False,
            enforce_target_lower_bound=True,
            solver=SolverConfig(),
        )
        self.assertIsNone(_format_mtf_profile(case))
        self.assertIsNone(_format_opt_profile(case))

        profiled_case = ExperimentCase(
            experiment_name="demo",
            machine_count=8,
            job_count=24,
            acceleration_case=AccelerationCase.CASE_2,
            ell=10,
            mtf_profile=MtfProfile(0, 4, 0, 1, 0, 3, 0, 0),
            opt_profile=OptProfile(8, 0, 0),
            target_ratio=parse_ratio("20/17"),
            output_root=Path("results"),
            write_lp=False,
            enforce_target_lower_bound=True,
            solver=SolverConfig(),
        )
        self.assertEqual(_format_mtf_profile(profiled_case), "(0,4,0,1,0,3,0,0)")
        self.assertEqual(_format_opt_profile(profiled_case), "(8,0,0)")

    def test_extract_optimal_p_values_desc_only_for_optimal_status(self) -> None:
        fake_grb = SimpleNamespace(OPTIMAL=2)

        class _Var:
            def __init__(self, value: float) -> None:
                self.X = value

        class _Model:
            Status = 2

            def getVarByName(self, name: str):
                values = {
                    "p[1]": _Var(0.499999999999),
                    "p[2]": _Var(0.333333333333),
                    "p[3]": _Var(0.2),
                }
                return values.get(name)

        class _NonOptimalModel(_Model):
            Status = 3

        with patch("multifit_optveri.runner.GRB", fake_grb):
            self.assertEqual(_extract_optimal_p_values_desc(_Model(), 3), "[15, 10, 6]")
            self.assertIsNone(_extract_optimal_p_values_desc(_NonOptimalModel(), 3))

    def test_allocate_run_dir_adds_suffix_when_needed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "demo"
            first_dir = experiment_dir / "20260326_120000"
            first_dir.mkdir(parents=True)
            with patch("multifit_optveri.runner._run_timestamp", return_value="20260326_120000"):
                self.assertEqual(
                    _allocate_run_dir(experiment_dir),
                    experiment_dir / "20260326_120000_01",
                )

    def test_create_run_artifacts_rejects_empty_case_list(self) -> None:
        with self.assertRaises(ValueError):
            create_run_artifacts([])

    def test_run_recorder_writes_summary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            case = _sample_case(Path(tmpdir))
            artifacts = create_run_artifacts([case])
            recorder = RunRecorder(artifacts=artifacts, cases=[case], cli_filters={"machine": 8})
            result = _sample_result(artifacts.cases_dir / "case_a")

            recorder.record(result)
            recorder.finish()

            self.assertTrue(artifacts.summary_csv_path.exists())
            self.assertTrue(artifacts.summary_jsonl_path.exists())
            self.assertTrue(artifacts.overview_json_path.exists())
            self.assertTrue(artifacts.manifest_json_path.exists())
            summary_csv = artifacts.summary_csv_path.read_text(encoding="utf-8")
            summary_jsonl = artifacts.summary_jsonl_path.read_text(encoding="utf-8")
            self.assertIn("case_a", summary_csv)
            self.assertIn("mtf-profile-(f1_r2_f2_r3_f3_r4_f4_r5)", summary_csv)
            self.assertIn("verification_result", summary_csv)
            self.assertNotIn("output_dir", summary_csv)
            self.assertIn("\"verification_result\": \"VERIFIED\"", summary_jsonl)
            self.assertIn("\"optimal-p-values-(desc-scaled)\": \"\"", summary_jsonl)
            self.assertNotIn("\"output_dir\":", summary_jsonl)
            self.assertIn("\"completed_case_count\": 1", artifacts.overview_json_path.read_text(encoding="utf-8"))
            self.assertIn("\"verification_result_counts\": {", artifacts.overview_json_path.read_text(encoding="utf-8"))

    def test_run_cases_uses_run_recorder_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            cases = [_sample_case(output_root), _sample_case(output_root)]

            def fake_run_case(case: ExperimentCase) -> SolveResult:
                return _sample_result(case.output_dir, case_id=case.case_id)

            with patch("multifit_optveri.runner.run_case", side_effect=fake_run_case):
                results = run_cases(cases)

            self.assertEqual(len(results), 2)
            experiment_dir = output_root / "demo"
            latest_run = Path((experiment_dir / "latest_run.txt").read_text(encoding="utf-8"))
            self.assertTrue((latest_run / "summary.csv").exists())
            self.assertTrue((latest_run / "overview.json").exists())


if __name__ == "__main__":
    unittest.main()
