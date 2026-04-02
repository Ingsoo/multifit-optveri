from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

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
        result = solve_opt_schedule(
            (Fraction(4, 1), Fraction(3, 1), Fraction(2, 1), Fraction(1, 1)),
            2,
        )

        self.assertEqual(result.machine_count, 2)
        self.assertEqual(result.makespan, Fraction(5, 1))
        self.assertEqual(sum(machine.load for machine in result.machines), Fraction(10, 1))

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
