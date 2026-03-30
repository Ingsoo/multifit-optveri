from __future__ import annotations

import argparse
from pathlib import Path

from multifit_optveri.acceleration import AccelerationCase
from multifit_optveri.config import load_experiment_config
from multifit_optveri.experiments import enumerate_cases, render_case_plan
from multifit_optveri.runner import RunRecorder, create_run_artifacts, prepare_cases_for_run, run_case


def _add_case_filter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--machine", type=int, help="Restrict to a single machine count.")
    parser.add_argument("--job", type=int, help="Restrict to a single job count.")
    parser.add_argument(
        "--acceleration-case",
        choices=[case.value for case in AccelerationCase],
        help="Restrict to a single acceleration case.",
    )
    parser.add_argument("--limit", type=int, help="Use only the first N matching cases.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="multifit-optveri")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Print the experiment plan.")
    plan_parser.add_argument("--config", required=True, type=Path)
    _add_case_filter_arguments(plan_parser)

    run_parser = subparsers.add_parser("run", help="Build and solve experiment cases.")
    run_parser.add_argument("--config", required=True, type=Path)
    _add_case_filter_arguments(run_parser)
    return parser


def _filter_cases(
    cases,
    machine: int | None,
    job: int | None,
    acceleration_case: str | None,
    limit: int | None,
):
    filtered = [
        case
        for case in cases
        if (machine is None or case.machine_count == machine)
        and (job is None or case.job_count == job)
        and (acceleration_case is None or case.acceleration_case.value == acceleration_case)
    ]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = load_experiment_config(args.config)
    cases = enumerate_cases(config)
    filtered_cases = _filter_cases(
        cases,
        args.machine,
        args.job,
        args.acceleration_case,
        args.limit,
    )

    if args.command == "plan":
        print(render_case_plan(filtered_cases))
        return 0

    if not filtered_cases:
        raise SystemExit("No experiment cases matched the supplied filters.")

    artifacts = create_run_artifacts(filtered_cases)
    prepared_cases = prepare_cases_for_run(filtered_cases, artifacts.run_dir)
    recorder = RunRecorder(
        artifacts=artifacts,
        cases=prepared_cases,
        cli_filters={
            "machine": args.machine,
            "job": args.job,
            "acceleration_case": args.acceleration_case,
            "limit": args.limit,
            "config": str(args.config),
        },
    )

    print(f"Running {len(prepared_cases)} case(s)...", flush=True)
    print(f"Run directory: {artifacts.run_dir}", flush=True)
    print(f"Summary CSV: {artifacts.summary_csv_path}", flush=True)
    for index, case in enumerate(prepared_cases, start=1):
        print(
            f"[{index}/{len(prepared_cases)}] Starting {case.case_id} "
            f"(m={case.machine_count}, n={case.job_count}, "
            f"acceleration={case.acceleration_case.value}, ell={case.ell})",
            flush=True,
        )
        result = run_case(case)
        recorder.record(result)
        print(
            f"{result.case_id}: acceleration={result.acceleration_case}, "
            f"status={result.status}, objective={result.objective_value}, "
            f"time={result.runtime_seconds}",
            flush=True,
        )
    recorder.finish()
    print(f"Finished. Overview: {artifacts.overview_json_path}", flush=True)
    return 0
