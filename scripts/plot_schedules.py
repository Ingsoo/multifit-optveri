from __future__ import annotations

import argparse
from pathlib import Path
import re
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
    parser.add_argument(
        "--machines",
        type=int,
        default=None,
        help="Number of identical machines. Optional when the instance file contains 'machines=<int>'.",
    )
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
    jobs_group.add_argument(
        "--instances",
        nargs="+",
        help="Batch mode: resolve multiple instance names from inputs/schedules/<name>.txt.",
    )
    jobs_group.add_argument(
        "--batch",
        help="Batch mode by pattern or prefix, e.g. 'demo', 'demo_*', or 'jobs_32_mce_*'.",
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
        "--capa",
        default=None,
        help="Fixed FFD capacity to plot instead of running MULTIFIT, e.g. '21' or '20/17'.",
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


def _resolve_batch_paths(root: Path, batch: str) -> tuple[Path, ...]:
    inputs_dir = _default_inputs_dir(root)
    pattern = batch
    if not any(token in batch for token in "*?[]"):
        pattern = f"{batch}*.txt"
    elif not batch.endswith(".txt"):
        pattern = f"{batch}.txt"
    paths = tuple(sorted(inputs_dir.glob(pattern)))
    if not paths:
        raise FileNotFoundError(f"No instance files matched batch pattern '{batch}' in {inputs_dir}.")
    return paths


def _parse_instance_text(raw_text: str) -> tuple[int | None, str]:
    lines = raw_text.splitlines()
    machine_count: int | None = None
    body_lines: list[str] = []

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        match = re.fullmatch(r"(?:machines|machine|m)\s*[:=]\s*(\d+)", stripped, flags=re.IGNORECASE)
        if index == 0 and match is not None:
            machine_count = int(match.group(1))
            continue
        body_lines.append(line)

    return machine_count, "\n".join(body_lines)


def _resolve_machine_count(
    args: argparse.Namespace,
    file_machine_count: int | None,
    instance_label: str,
) -> int:
    if file_machine_count is not None and args.machines is not None and file_machine_count != args.machines:
        raise ValueError(
            f"Instance '{instance_label}' declares machines={file_machine_count}, "
            f"but CLI requested --machines {args.machines}."
        )
    if file_machine_count is not None:
        return file_machine_count
    if args.machines is not None:
        return args.machines
    raise ValueError(
        f"Instance '{instance_label}' does not declare a machine count and --machines was not provided."
    )


def _load_jobs_argument(root: Path, args: argparse.Namespace) -> tuple[str, str]:
    if args.jobs is not None:
        return args.jobs, "inline_jobs"
    if args.instance is not None:
        instance_path = _default_inputs_dir(root) / f"{args.instance}.txt"
        return instance_path.read_text(encoding="utf-8"), args.instance
    if args.instances is not None:
        raise ValueError("Use batch mode helpers for --instances.")
    if args.batch is not None:
        raise ValueError("Use batch mode helpers for --batch.")
    if args.jobs_file is None:
        raise ValueError("One of --jobs, --jobs-file, --instance, --instances, or --batch must be provided.")
    jobs_path = _resolve_jobs_file(root, args.jobs_file)
    return jobs_path.read_text(encoding="utf-8"), jobs_path.stem


def _resolve_output_path(root: Path, args: argparse.Namespace, instance_label: str) -> Path:
    if args.output is not None:
        return args.output
    return _default_artifacts_dir(root) / f"{instance_label}.png"


def _log(message: str) -> None:
    print(f"[plot_schedules] {message}")


def _build_fixed_ffd_schedule(
    processing_times,
    machine_count: int,
    capacity_text: str,
):
    from multifit_optveri.math_utils import parse_ratio
    from multifit_optveri.schedules import first_fit_overflow_schedule, first_fit_schedule

    capacity = parse_ratio(capacity_text)
    ffd = first_fit_schedule(processing_times, machine_count, capacity)
    if ffd is None:
        ffd = first_fit_overflow_schedule(processing_times, machine_count, capacity)
    return ffd.__class__(
        algorithm="FFD",
        machine_count=ffd.machine_count,
        machines=ffd.machines,
        makespan=ffd.makespan,
        sorted_job_ids=ffd.sorted_job_ids,
        sorted_processing_times=ffd.sorted_processing_times,
        feasibility_capacity=ffd.feasibility_capacity,
        attempts=ffd.attempts,
    )


def _run_instance(
    root: Path,
    args: argparse.Namespace,
    *,
    jobs_text: str,
    instance_label: str,
) -> int:
    from multifit_optveri.math_utils import format_ratio
    from multifit_optveri.schedules import (
        multifit_schedule,
        parse_processing_times,
        plot_schedule_comparison,
        plot_multifit_history,
        render_schedule_text,
        solve_opt_schedule,
    )

    file_machine_count, parsed_jobs_text = _parse_instance_text(jobs_text)
    machine_count = _resolve_machine_count(args, file_machine_count, instance_label)
    _log(f"Loading instance '{instance_label}'")
    processing_times = parse_processing_times(parsed_jobs_text)
    _log(f"Parsed {len(processing_times)} jobs for {machine_count} machines")

    history_paths: tuple[Path, ...] = ()
    def log_multifit_attempt(attempt) -> None:
        status_text = "feasible" if attempt.feasible else "needs extra machine"
        _log(
            f"MULTIFIT attempt {attempt.iteration:02d}: "
            f"capacity={format_ratio(attempt.capacity)} -> {status_text}"
        )

    if args.capa is not None:
        if args.save_multifit_history:
            raise ValueError("--save-multifit-history cannot be used together with --capa.")
        _log(f"Running fixed-capacity FFD with capa={args.capa}...")
        left_schedule = _build_fixed_ffd_schedule(processing_times, machine_count, args.capa)
        if left_schedule.machine_count > machine_count:
            _log(
                f"FFD needed {left_schedule.machine_count} machines, so the figure includes "
                f"{left_schedule.machine_count - machine_count} overflow machine(s)."
            )
        _log(f"FFD finished with Cmax={format_ratio(left_schedule.makespan)}")
    else:
        _log("Running MULTIFIT...")
        left_schedule = multifit_schedule(
            processing_times,
            machine_count,
            iterations=args.multifit_iterations,
            attempt_callback=log_multifit_attempt,
        )
        _log(f"MULTIFIT finished with Cmax={format_ratio(left_schedule.makespan)}")

    _log("Running OPT...")
    optimum = solve_opt_schedule(processing_times, machine_count)
    _log(f"OPT finished with Cmax={format_ratio(optimum.makespan)}")
    title_prefix = "FFD" if args.capa is not None else "MTF"
    title = f"{title_prefix}(I) = {format_ratio(left_schedule.makespan)} vs OPT(I) = {format_ratio(optimum.makespan)}"
    output_path = plot_schedule_comparison(
        left_schedule,
        optimum,
        _resolve_output_path(root, args, instance_label),
        title=title,
    )
    if args.save_multifit_history:
        _log("Saving MULTIFIT history figures...")
        history_paths = plot_multifit_history(
            left_schedule,
            _default_history_dir(root, instance_label),
            title_prefix=f"{instance_label} on {machine_count} machines",
        )

    print(render_schedule_text(left_schedule))
    print()
    print(render_schedule_text(optimum))
    print()
    print(f"Figure saved to: {output_path}")
    if history_paths:
        print(f"MULTIFIT history saved to: {_default_history_dir(root, instance_label)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    root = _repo_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.instances is not None:
        if args.output is not None:
            raise SystemExit("--output cannot be used with --instances. Use per-instance default outputs instead.")
        for instance_label in args.instances:
            instance_path = _default_inputs_dir(root) / f"{instance_label}.txt"
            jobs_text = instance_path.read_text(encoding="utf-8")
            _run_instance(root, args, jobs_text=jobs_text, instance_label=instance_label)
            print()
        return 0

    if args.batch is not None:
        if args.output is not None:
            raise SystemExit("--output cannot be used with --batch. Use per-instance default outputs instead.")
        for instance_path in _resolve_batch_paths(root, args.batch):
            _run_instance(
                root,
                args,
                jobs_text=instance_path.read_text(encoding="utf-8"),
                instance_label=instance_path.stem,
            )
            print()
        return 0

    jobs_text, instance_label = _load_jobs_argument(root, args)
    return _run_instance(root, args, jobs_text=jobs_text, instance_label=instance_label)


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
