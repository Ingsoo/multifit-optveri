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
                {"jobs": None, "jobs_file": None, "instance": "demo"},
            )()
            jobs_text, instance_label = plot_schedules_script._load_jobs_argument(root, namespace)

            self.assertEqual(jobs_text, "9/17\n7/17\n")
            self.assertEqual(instance_label, "demo")

    def test_resolve_output_path_defaults_to_artifacts_schedules(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            namespace = type("Args", (), {"output": None})()

            output_path = plot_schedules_script._resolve_output_path(root, namespace, "jobs_26")

            self.assertEqual(output_path, root / "artifacts" / "schedules" / "jobs_26.png")


if __name__ == "__main__":
    unittest.main()
