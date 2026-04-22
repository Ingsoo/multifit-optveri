from __future__ import annotations

from fractions import Fraction
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from multifit_optveri import schedules
from multifit_optveri.schedules import (
    SchedulingUnavailableError,
    first_fit_schedule,
    multifit_schedule,
    parse_processing_times,
    plot_schedule_comparison,
    plot_multifit_history,
    render_schedule_text,
    solve_opt_schedule,
)


class ScheduleTests(unittest.TestCase):
    def test_parse_processing_times_accepts_fractions(self) -> None:
        self.assertEqual(
            parse_processing_times("9/17, 7/17, 6/17, 5/17"),
            (
                Fraction(9, 17),
                Fraction(7, 17),
                Fraction(6, 17),
                Fraction(5, 17),
            ),
        )

    def test_parse_processing_times_accepts_newlines_and_spaces(self) -> None:
        self.assertEqual(
            parse_processing_times("9/17\n7/17 6/17;5/17"),
            (
                Fraction(9, 17),
                Fraction(7, 17),
                Fraction(6, 17),
                Fraction(5, 17),
            ),
        )

    def test_first_fit_schedule_returns_none_when_capacity_is_too_small(self) -> None:
        result = first_fit_schedule((Fraction(4, 1), Fraction(3, 1), Fraction(2, 1)), 2, Fraction(4, 1))
        self.assertIsNone(result)

    def test_first_fit_schedule_assigns_jobs_in_decreasing_order(self) -> None:
        result = first_fit_schedule((Fraction(4, 1), Fraction(3, 1), Fraction(2, 1), Fraction(1, 1)), 2, Fraction(5, 1))

        assert result is not None
        self.assertEqual(result.makespan, Fraction(5, 1))
        self.assertEqual([job.job_id for job in result.machines[0].jobs], [1, 4])
        self.assertEqual([job.job_id for job in result.machines[1].jobs], [2, 3])
        self.assertEqual([job.is_fallback for job in result.machines[0].jobs], [False, True])
        self.assertEqual([job.is_fallback for job in result.machines[1].jobs], [False, False])

    def test_multifit_schedule_finds_feasible_schedule(self) -> None:
        result = multifit_schedule((Fraction(4, 1), Fraction(3, 1), Fraction(2, 1), Fraction(1, 1)), 2, iterations=12)

        self.assertLessEqual(result.makespan, Fraction(5, 1))
        self.assertEqual(result.machine_count, 2)
        self.assertIsNotNone(result.feasibility_capacity)
        self.assertTrue(result.attempts)
        self.assertTrue(all(attempt.capacity.denominator == 1 for attempt in result.attempts))
        self.assertTrue(all(attempt.schedule is not None for attempt in result.attempts))
        self.assertIn("MULTIFIT-FFD", render_schedule_text(result))
        self.assertIn("(F)", render_schedule_text(result))
        capacities = [attempt.capacity for attempt in result.attempts]
        self.assertEqual(capacities, [Fraction(10, 1), Fraction(7, 1), Fraction(6, 1), Fraction(5, 1)])

    def test_multifit_schedule_invokes_attempt_callback(self) -> None:
        seen_attempts = []

        result = multifit_schedule(
            (Fraction(4, 1), Fraction(3, 1), Fraction(2, 1), Fraction(1, 1)),
            2,
            iterations=4,
            attempt_callback=seen_attempts.append,
        )

        self.assertEqual(len(seen_attempts), len(result.attempts))
        self.assertEqual(
            [attempt.iteration for attempt in seen_attempts],
            list(range(1, len(seen_attempts) + 1)),
        )

    @unittest.skipIf(schedules.GRB is None, "gurobipy is unavailable")
    def test_opt_schedule_solves_minmax_assignment(self) -> None:
        try:
            result = solve_opt_schedule(
                (Fraction(4, 1), Fraction(3, 1), Fraction(2, 1), Fraction(1, 1)),
                2,
            )
        except Exception as exc:
            if exc.__class__.__name__ == "GurobiError":
                self.skipTest(f"gurobi runtime is unavailable: {exc}")
            raise

        self.assertEqual(result.machine_count, 2)
        self.assertEqual(result.makespan, Fraction(5, 1))
        self.assertEqual(sum(machine.load for machine in result.machines), Fraction(10, 1))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "matplotlib is unavailable")
    def test_plot_schedule_comparison_writes_png_when_matplotlib_is_available(self) -> None:
        multifit = first_fit_schedule(
            (Fraction(4, 1), Fraction(3, 1), Fraction(2, 1), Fraction(1, 1)),
            2,
            Fraction(5, 1),
        )
        self.assertIsNotNone(multifit)

        optimum = schedules.ScheduleResult(
            algorithm="OPT",
            machine_count=2,
            machines=multifit.machines,  # Reuse a known feasible layout for the plotting smoke test.
            makespan=Fraction(5, 1),
            feasibility_capacity=None,
            sorted_job_ids=multifit.sorted_job_ids,
            sorted_processing_times=multifit.sorted_processing_times,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "comparison.png"
            try:
                written_path = plot_schedule_comparison(multifit, optimum, output_path)
            except SchedulingUnavailableError:
                self.skipTest("matplotlib is unavailable")
                return

            self.assertEqual(written_path, output_path)
            self.assertTrue(output_path.exists())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "matplotlib is unavailable")
    def test_plot_schedule_comparison_uses_requested_labels(self) -> None:
        multifit = first_fit_schedule(
            (Fraction(4, 1), Fraction(3, 1), Fraction(2, 1), Fraction(1, 1)),
            2,
            Fraction(5, 1),
        )
        self.assertIsNotNone(multifit)

        optimum = schedules.ScheduleResult(
            algorithm="OPT",
            machine_count=2,
            machines=multifit.machines,
            makespan=Fraction(5, 1),
            feasibility_capacity=None,
            sorted_job_ids=multifit.sorted_job_ids,
            sorted_processing_times=multifit.sorted_processing_times,
        )

        captured: dict[str, object] = {}

        def _capture_savefig(self, *args, **kwargs) -> None:
            captured["suptitle"] = self._suptitle.get_text() if self._suptitle is not None else None
            captured["axis_titles"] = [axis.get_title() for axis in self.axes]
            captured["yticklabels"] = [tick.get_text() for tick in self.axes[0].get_yticklabels()]
            captured["xticklabels"] = [tick.get_text() for tick in self.axes[0].get_xticklabels()]

        output_path = Path("comparison.png")
        with patch("matplotlib.figure.Figure.savefig", autospec=True, side_effect=_capture_savefig):
            written_path = plot_schedule_comparison(
                multifit,
                optimum,
                output_path,
                title="MTF(I) = 11 vs OPT(I) = 11",
            )

        self.assertEqual(written_path, output_path)
        self.assertEqual(captured["suptitle"], "MTF(I) = 11 vs OPT(I) = 11")
        self.assertEqual(captured["axis_titles"], ["MULTIFIT (Cmax=5)\nFFD capa = 5", "OPT (Cmax=5)"])
        self.assertEqual(captured["yticklabels"], ["1", "2"])
        self.assertEqual(captured["xticklabels"][:7], ["0", "1", "2", "3", "4", "5", "6"])

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "matplotlib is unavailable")
    def test_plot_schedule_comparison_shows_all_integer_xticks_in_range(self) -> None:
        multifit = first_fit_schedule(
            (Fraction(12, 1), Fraction(10, 1), Fraction(10, 1), Fraction(9, 1), Fraction(8, 1), Fraction(8, 1)),
            3,
            Fraction(17, 1),
        )
        self.assertIsNotNone(multifit)

        optimum = schedules.ScheduleResult(
            algorithm="OPT",
            machine_count=3,
            machines=multifit.machines,
            makespan=Fraction(17, 1),
            feasibility_capacity=None,
            sorted_job_ids=multifit.sorted_job_ids,
            sorted_processing_times=multifit.sorted_processing_times,
        )

        captured: dict[str, object] = {}

        def _capture_savefig(self, *args, **kwargs) -> None:
            captured["xticklabels"] = [tick.get_text() for tick in self.axes[0].get_xticklabels()]

        with patch("matplotlib.figure.Figure.savefig", autospec=True, side_effect=_capture_savefig):
            plot_schedule_comparison(multifit, optimum, Path("comparison.png"))

        xticklabels = captured["xticklabels"]
        self.assertIn("15", xticklabels)
        self.assertIn("16", xticklabels)
        self.assertIn("17", xticklabels)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "matplotlib is unavailable")
    def test_plot_schedule_comparison_supports_different_machine_counts(self) -> None:
        multifit = schedules.first_fit_overflow_schedule(
            (Fraction(4, 1), Fraction(3, 1), Fraction(2, 1)),
            2,
            Fraction(4, 1),
        )
        optimum = schedules.ScheduleResult(
            algorithm="OPT",
            machine_count=2,
            machines=multifit.machines[:2],
            makespan=Fraction(4, 1),
            feasibility_capacity=None,
            sorted_job_ids=multifit.sorted_job_ids,
            sorted_processing_times=multifit.sorted_processing_times,
        )

        captured: dict[str, object] = {}

        def _capture_savefig(self, *args, **kwargs) -> None:
            captured["left_yticklabels"] = [tick.get_text() for tick in self.axes[0].get_yticklabels()]
            captured["right_yticklabels"] = [tick.get_text() for tick in self.axes[1].get_yticklabels()]

        with patch("matplotlib.figure.Figure.savefig", autospec=True, side_effect=_capture_savefig):
            plot_schedule_comparison(multifit, optimum, Path("comparison.png"))

        self.assertEqual(captured["left_yticklabels"], ["1", "2", "3"])
        self.assertEqual(captured["right_yticklabels"], [])

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "matplotlib is unavailable")
    def test_plot_schedule_comparison_reuses_colors_for_equal_processing_times(self) -> None:
        multifit = first_fit_schedule(
            (Fraction(4, 1), Fraction(4, 1), Fraction(2, 1), Fraction(2, 1)),
            2,
            Fraction(6, 1),
        )
        self.assertIsNotNone(multifit)

        optimum = schedules.ScheduleResult(
            algorithm="OPT",
            machine_count=2,
            machines=multifit.machines,
            makespan=Fraction(6, 1),
            feasibility_capacity=None,
            sorted_job_ids=multifit.sorted_job_ids,
            sorted_processing_times=multifit.sorted_processing_times,
        )

        captured: dict[str, object] = {}

        def _capture_savefig(self, *args, **kwargs) -> None:
            patches = self.axes[0].patches
            captured["facecolors"] = [patch.get_facecolor() for patch in patches]

        with patch("matplotlib.figure.Figure.savefig", autospec=True, side_effect=_capture_savefig):
            plot_schedule_comparison(multifit, optimum, Path("comparison.png"))

        facecolors = captured["facecolors"]
        self.assertEqual(len(facecolors), 4)
        self.assertEqual(facecolors[0], facecolors[2])
        self.assertEqual(facecolors[1], facecolors[3])
        self.assertNotEqual(facecolors[0], facecolors[1])

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "matplotlib is unavailable")
    def test_plot_schedule_comparison_keeps_labels_for_unit_width_jobs(self) -> None:
        multifit = first_fit_schedule(
            (Fraction(12, 1), Fraction(10, 1), Fraction(10, 1), Fraction(9, 1), Fraction(8, 1), Fraction(8, 1), Fraction(1, 1)),
            3,
            Fraction(18, 1),
        )
        self.assertIsNotNone(multifit)

        optimum = schedules.ScheduleResult(
            algorithm="OPT",
            machine_count=3,
            machines=multifit.machines,
            makespan=Fraction(18, 1),
            feasibility_capacity=None,
            sorted_job_ids=multifit.sorted_job_ids,
            sorted_processing_times=multifit.sorted_processing_times,
        )

        captured: dict[str, object] = {}

        def _capture_savefig(self, *args, **kwargs) -> None:
            captured["texts"] = [text.get_text() for text in self.axes[0].texts]
            captured["rotations"] = [text.get_rotation() for text in self.axes[0].texts]

        with patch("matplotlib.figure.Figure.savefig", autospec=True, side_effect=_capture_savefig):
            plot_schedule_comparison(multifit, optimum, Path("comparison.png"))

        self.assertIn("j7\n1", captured["texts"])
        unit_label_index = captured["texts"].index("j7\n1")
        self.assertEqual(captured["rotations"][unit_label_index], 0.0)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "matplotlib is unavailable")
    def test_plot_multifit_history_writes_attempt_pngs_when_matplotlib_is_available(self) -> None:
        multifit = multifit_schedule(
            (Fraction(4, 1), Fraction(3, 1), Fraction(2, 1), Fraction(1, 1)),
            2,
            iterations=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "history"
            try:
                written_paths = plot_multifit_history(multifit, output_dir, title_prefix="demo")
            except SchedulingUnavailableError:
                self.skipTest("matplotlib is unavailable")
                return

            self.assertEqual(len(written_paths), len(multifit.attempts))
            self.assertTrue(all(path.exists() for path in written_paths))


if __name__ == "__main__":
    unittest.main()
