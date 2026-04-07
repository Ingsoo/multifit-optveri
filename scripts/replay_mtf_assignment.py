from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_bootstrap_src()

from multifit_optveri.branching import FallbackStarts, MtfProfile
from multifit_optveri.math_utils import format_ratio, parse_ratio
from multifit_optveri.schedules import first_fit_schedule, render_schedule_text


@dataclass(frozen=True)
class _Layout:
    f1_machines: tuple[int, ...]
    r2_machines: tuple[int, ...]
    f2_machines: tuple[int, ...]
    r3_machines: tuple[int, ...]
    f3_machines: tuple[int, ...]
    r4_machines: tuple[int, ...]
    f4_machines: tuple[int, ...]
    r5_machines: tuple[int, ...]


def _parse_mtf_profile(text: str) -> MtfProfile:
    values = [int(part.strip()) for part in text.strip().strip("()").split(",")]
    if len(values) != 8:
        raise ValueError(f"Expected 8 MTF profile entries, got {len(values)} from {text!r}.")
    return MtfProfile(*values)


def _parse_fallback_starts(text: str) -> FallbackStarts:
    def parse_one(part: str) -> int | None:
        stripped = part.strip()
        if stripped in {"", "None", "none", "null"}:
            return None
        return int(stripped)

    values = [parse_one(part) for part in text.strip().strip("()").split(",")]
    if len(values) != 3:
        raise ValueError(f"Expected 3 fallback-start entries, got {len(values)} from {text!r}.")
    return FallbackStarts(values[0], values[1], values[2])


def _parse_processing_times(text: str) -> tuple[Fraction, ...]:
    raw = text.strip().strip("[]")
    if not raw:
        return ()
    return tuple(parse_ratio(part.strip()) for part in raw.split(",") if part.strip())


def _build_layout(machine_count: int, profile: MtfProfile) -> _Layout:
    machine_ids = list(range(1, machine_count + 1))
    cursor = 0

    def take(count: int) -> tuple[int, ...]:
        nonlocal cursor
        values = tuple(machine_ids[cursor : cursor + count])
        cursor += count
        return values

    return _Layout(
        f1_machines=take(profile.nF1),
        r2_machines=take(profile.nR2),
        f2_machines=take(profile.nF2),
        r3_machines=take(profile.nR3),
        f3_machines=take(profile.nF3),
        r4_machines=take(profile.nR4),
        f4_machines=take(profile.nF4),
        r5_machines=take(profile.nR5),
    )


def _expected_case2_assignment(
    machine_count: int,
    profile: MtfProfile,
    starts: FallbackStarts,
) -> dict[int, tuple[int, ...]]:
    layout = _build_layout(machine_count, profile)
    scheduled_job_count = profile.scheduled_job_count
    fallback_jobs: set[int] = set()

    def fallback_block(start: int | None, count: int) -> list[int]:
        if start is None or count == 0:
            return []
        values = list(range(start, start + count))
        fallback_jobs.update(values)
        return values

    f2_fallback_jobs = fallback_block(starts.s2, profile.nF2)
    f3_fallback_jobs = fallback_block(starts.s3, profile.nF3)
    f4_fallback_jobs = fallback_block(starts.s4, profile.nF4)
    regular_jobs = [job_index for job_index in range(1, scheduled_job_count + 1) if job_index not in fallback_jobs]

    assignment: dict[int, tuple[int, ...]] = {}
    regular_cursor = 0
    f2_cursor = 0
    f3_cursor = 0
    f4_cursor = 0

    def regular_block(regular_count: int) -> tuple[int, ...]:
        nonlocal regular_cursor
        jobs = tuple(regular_jobs[regular_cursor : regular_cursor + regular_count])
        regular_cursor += regular_count
        return jobs

    for machine_index in layout.f1_machines + layout.r2_machines:
        assignment[machine_index] = regular_block(2)

    for machine_index in layout.f2_machines:
        assignment[machine_index] = regular_block(2) + (f2_fallback_jobs[f2_cursor],)
        f2_cursor += 1

    for machine_index in layout.r3_machines:
        assignment[machine_index] = regular_block(3)

    for machine_index in layout.f3_machines:
        assignment[machine_index] = regular_block(3) + (f3_fallback_jobs[f3_cursor],)
        f3_cursor += 1

    for machine_index in layout.r4_machines:
        assignment[machine_index] = regular_block(4)

    for machine_index in layout.f4_machines:
        assignment[machine_index] = regular_block(4) + (f4_fallback_jobs[f4_cursor],)
        f4_cursor += 1

    for machine_index in layout.r5_machines:
        assignment[machine_index] = regular_block(5)

    return assignment


def _actual_assignment_text(assignment: dict[int, tuple[int, ...]]) -> str:
    lines = []
    for machine_index in sorted(assignment):
        jobs = ", ".join(str(job_index) for job_index in assignment[machine_index])
        lines.append(f"M{machine_index}: [{jobs}]")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay MTF from one case summary and compare with the modeled assignment."
    )
    parser.add_argument("summary_json", help="Path to case summary.json")
    args = parser.parse_args()

    summary_path = Path(args.summary_json).resolve()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    machine_count = int(payload["machine_count"])
    target_ratio = parse_ratio(str(payload["target_ratio"]))
    acceleration_case = str(payload["acceleration_case"])
    processing_times_text = str(payload.get("optimal-p-values-(desc-exact)", ""))
    if not processing_times_text:
        raise ValueError("summary.json is missing 'optimal-p-values-(desc-exact)'. Re-run after the new output fields.")

    processing_times = _parse_processing_times(processing_times_text)
    schedule = first_fit_schedule(processing_times, machine_count, target_ratio)
    report_lines = [
        f"summary: {summary_path}",
        f"case: {acceleration_case}",
        f"machines: {machine_count}",
        f"capacity: {format_ratio(target_ratio)}",
    ]

    if schedule is None:
        report_lines.append("replay_status: INFEASIBLE")
    else:
        report_lines.append("replay_status: FEASIBLE")
        report_lines.append("")
        report_lines.append(render_schedule_text(schedule))

    mtf_profile_text = str(payload.get("mtf-profile-(f1_r2_f2_r3_f3_r4_f4_r5)", ""))
    fallback_starts_text = str(payload.get("fallback-starts-(s2_s3_s4)", ""))

    if acceleration_case == "case_2" and mtf_profile_text and fallback_starts_text and schedule is not None:
        profile = _parse_mtf_profile(mtf_profile_text)
        starts = _parse_fallback_starts(fallback_starts_text)
        expected = _expected_case2_assignment(machine_count, profile, starts)
        actual = {machine.machine_id: tuple(job.job_id for job in machine.jobs) for machine in schedule.machines}
        matches = expected == actual
        report_lines.extend(
            [
                "",
                f"assignment_match: {matches}",
                "",
                "expected_assignment:",
                _actual_assignment_text(expected),
                "",
                "actual_assignment:",
                _actual_assignment_text(actual),
            ]
        )
    else:
        report_lines.extend(
            [
                "",
                "assignment_match: SKIPPED",
                "reason: requires case_2 + fallback starts + feasible replay schedule",
            ]
        )

    report_text = "\n".join(report_lines) + "\n"
    output_path = summary_path.with_name("mtf_replay_check.txt")
    output_path.write_text(report_text, encoding="utf-8")
    print(report_text, end="")
    print(f"written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
