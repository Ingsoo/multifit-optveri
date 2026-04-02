from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_schedules.py"
_SPEC = importlib.util.spec_from_file_location("plot_schedules_script", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
plot_schedules_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(plot_schedules_script)


class PlotSchedulesScriptTests(unittest.TestCase):
    def test_resolve_jobs_file_prefers_inputs_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            jobs_path = root / "inputs" / "schedules" / "jobs_26.txt"
            jobs_path.parent.mkdir(parents=True, exist_ok=True)
            jobs_path.write_text("4\n3\n2\n1\n", encoding="utf-8")

            resolved = plot_schedules_script._resolve_jobs_file(root, Path("jobs_26.txt"))

            self.assertEqual(resolved, jobs_path)

    def test_load_jobs_argument_supports_instance_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            jobs_path = root / "inputs" / "schedules" / "demo.txt"
            jobs_path.parent.mkdir(parents=True, exist_ok=True)
            jobs_path.write_text("9/17\n7/17\n", encoding="utf-8")

            namespace = type(
                "Args",
                (),
                {"jobs": None, "jobs_file": None, "instance": "demo", "instances": None, "batch": None},
            )()
            jobs_text, instance_label = plot_schedules_script._load_jobs_argument(root, namespace)

            self.assertEqual(jobs_text, "9/17\n7/17\n")
            self.assertEqual(instance_label, "demo")

    def test_load_jobs_argument_rejects_batch_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            namespace = type(
                "Args",
                (),
                {
                    "jobs": None,
                    "jobs_file": None,
                    "instance": None,
                    "instances": ["demo_01", "demo_02"],
                    "batch": None,
                },
            )()

            with self.assertRaises(ValueError):
                plot_schedules_script._load_jobs_argument(root, namespace)

    def test_parse_instance_text_supports_machine_header(self) -> None:
        machine_count, jobs_text = plot_schedules_script._parse_instance_text("machines=8\n9\n8\n7\n")

        self.assertEqual(machine_count, 8)
        self.assertEqual(jobs_text, "9\n8\n7")

    def test_resolve_machine_count_prefers_file_metadata(self) -> None:
        namespace = type("Args", (), {"machines": None})()

        machine_count = plot_schedules_script._resolve_machine_count(namespace, 8, "demo")

        self.assertEqual(machine_count, 8)

    def test_resolve_batch_paths_supports_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inputs_dir = root / "inputs" / "schedules"
            inputs_dir.mkdir(parents=True, exist_ok=True)
            (inputs_dir / "demo_01.txt").write_text("machines=8\n9\n", encoding="utf-8")
            (inputs_dir / "demo_02.txt").write_text("machines=8\n8\n", encoding="utf-8")
            (inputs_dir / "other.txt").write_text("machines=8\n7\n", encoding="utf-8")

            paths = plot_schedules_script._resolve_batch_paths(root, "demo_")

            self.assertEqual([path.name for path in paths], ["demo_01.txt", "demo_02.txt"])

    def test_resolve_output_path_defaults_to_artifacts_schedules(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            namespace = type("Args", (), {"output": None})()

            output_path = plot_schedules_script._resolve_output_path(root, namespace, "jobs_26")

            self.assertEqual(output_path, root / "artifacts" / "schedules" / "jobs_26.png")


if __name__ == "__main__":
    unittest.main()
