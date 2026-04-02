from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_inputs_dir(root: Path) -> Path:
    return root / "inputs" / "schedules"


def _default_artifacts_dir(root: Path) -> Path:
    return root / "artifacts" / "schedules"


def _default_history_dir(root: Path, instance_label: str) -> Path:
    return _default_artifacts_dir(root) / f"{instance_label}_history"


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
    jobs_group.add_argument(
        "--instance",
        help="Instance name resolved as inputs/schedules/<name>.txt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the rendered schedule comparison PNG. Defaults to artifacts/schedules/<instance>.png.",
    )
    parser.add_argument(
        "--multifit-iterations",
        type=int,
        default=30,
        help="Number of MULTIFIT binary-search iterations.",
    )
    parser.add_argument(
        "--save-multifit-history",
        action="store_true",
        help="Save one PNG per MULTIFIT capacity attempt under artifacts/schedules/<instance>_history/.",
    )
    return parser


def _resolve_jobs_file(root: Path, jobs_file: Path) -> Path:
    if jobs_file.is_absolute():
        return jobs_file
    if jobs_file.exists():
        return jobs_file
    candidate = _default_inputs_dir(root) / jobs_file
    return candidate


def _load_jobs_argument(root: Path, args: argparse.Namespace) -> tuple[str, str]:
    if args.jobs is not None:
        return args.jobs, "inline_jobs"
    if args.instance is not None:
        instance_path = _default_inputs_dir(root) / f"{args.instance}.txt"
        return instance_path.read_text(encoding="utf-8"), args.instance
    if args.jobs_file is None:
        raise ValueError("One of --jobs, --jobs-file, or --instance must be provided.")
    jobs_path = _resolve_jobs_file(root, args.jobs_file)
    return jobs_path.read_text(encoding="utf-8"), jobs_path.stem


def _resolve_output_path(root: Path, args: argparse.Namespace, instance_label: str) -> Path:
    if args.output is not None:
        return args.output
    return _default_artifacts_dir(root) / f"{instance_label}.png"


def _log(message: str) -> None:
    print(f"[plot_schedules] {message}")


def main(argv: list[str] | None = None) -> int:
    root = _repo_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from multifit_optveri.math_utils import format_ratio
    from multifit_optveri.schedules import (
        multifit_schedule,
        parse_processing_times,
        plot_schedule_comparison,
        plot_multifit_history,
        render_schedule_text,
        solve_opt_schedule,
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    jobs_text, instance_label = _load_jobs_argument(root, args)
    _log(f"Loading instance '{instance_label}'")
    processing_times = parse_processing_times(jobs_text)
    _log(f"Parsed {len(processing_times)} jobs for {args.machines} machines")

    def log_multifit_attempt(attempt) -> None:
        status_text = "feasible" if attempt.feasible else "needs extra machine"
        _log(
            f"MULTIFIT attempt {attempt.iteration:02d}: "
            f"capacity={format_ratio(attempt.capacity)} -> {status_text}"
        )

    _log("Running MULTIFIT...")
    multifit = multifit_schedule(
        processing_times,
        args.machines,
        iterations=args.multifit_iterations,
        attempt_callback=log_multifit_attempt,
    )
    _log(f"MULTIFIT finished with Cmax={format_ratio(multifit.makespan)}")
    _log("Running OPT...")
    optimum = solve_opt_schedule(processing_times, args.machines)
    _log(f"OPT finished with Cmax={format_ratio(optimum.makespan)}")
    title = (
        f"{len(processing_times)} jobs on {args.machines} machines | "
        f"MULTIFIT={format_ratio(multifit.makespan)} vs OPT={format_ratio(optimum.makespan)}"
    )
    output_path = plot_schedule_comparison(
        multifit,
        optimum,
        _resolve_output_path(root, args, instance_label),
        title=title,
    )
    history_paths: tuple[Path, ...] = ()
    if args.save_multifit_history:
        _log("Saving MULTIFIT history figures...")
        history_paths = plot_multifit_history(
            multifit,
            _default_history_dir(root, instance_label),
            title_prefix=f"{instance_label} on {args.machines} machines",
        )

    print(render_schedule_text(multifit))
    print()
    print(render_schedule_text(optimum))
    print()
    print(f"Figure saved to: {output_path}")
    if history_paths:
        print(f"MULTIFIT history saved to: {_default_history_dir(root, instance_label)}")
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
