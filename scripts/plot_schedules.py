from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot MULTIFIT and OPT schedules for a given job list.")
    parser.add_argument("--machines", type=int, required=True, help="Number of identical machines.")
    jobs_group = parser.add_mutually_exclusive_group(required=True)
    jobs_group.add_argument(
        "--jobs",
        help="Comma-separated processing times, e.g. '9/17,7/17,6/17,5/17'.",
    )
    jobs_group.add_argument(
        "--jobs-file",
        type=Path,
        help="Text file containing processing times separated by commas, spaces, semicolons, or newlines.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "schedule_comparison.png",
        help="Where to save the rendered schedule comparison PNG.",
    )
    parser.add_argument(
        "--multifit-iterations",
        type=int,
        default=30,
        help="Number of MULTIFIT binary-search iterations.",
    )
    return parser


def _load_jobs_argument(args: argparse.Namespace) -> str:
    if args.jobs is not None:
        return args.jobs
    if args.jobs_file is None:
        raise ValueError("Either --jobs or --jobs-file must be provided.")
    return args.jobs_file.read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from multifit_optveri.math_utils import format_ratio
    from multifit_optveri.schedules import (
        multifit_schedule,
        parse_processing_times,
        plot_schedule_comparison,
        render_schedule_text,
        solve_opt_schedule,
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    processing_times = parse_processing_times(_load_jobs_argument(args))
    multifit = multifit_schedule(
        processing_times,
        args.machines,
        iterations=args.multifit_iterations,
    )
    optimum = solve_opt_schedule(processing_times, args.machines)
    title = (
        f"{len(processing_times)} jobs on {args.machines} machines | "
        f"MULTIFIT={format_ratio(multifit.makespan)} vs OPT={format_ratio(optimum.makespan)}"
    )
    output_path = plot_schedule_comparison(multifit, optimum, args.output, title=title)

    print(render_schedule_text(multifit))
    print()
    print(render_schedule_text(optimum))
    print()
    print(f"Figure saved to: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        from multifit_optveri.schedules import SchedulingUnavailableError
    except ModuleNotFoundError:
        root = Path(__file__).resolve().parents[1]
        src = root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from multifit_optveri.schedules import SchedulingUnavailableError

    try:
        raise SystemExit(main())
    except SchedulingUnavailableError as exc:
        raise SystemExit(str(exc)) from exc
