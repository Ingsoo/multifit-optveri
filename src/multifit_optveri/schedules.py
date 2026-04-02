from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from multifit_optveri.math_utils import format_ratio, parse_ratio

if TYPE_CHECKING:
    import gurobipy as gp
    from gurobipy import GRB
    from gurobipy import Model as GurobiModel
else:
    GurobiModel = Any
    try:
        from gurobipy import GRB
        import gurobipy as gp
    except ImportError:  # pragma: no cover - exercised only when Gurobi is missing
        gp = None
        GRB = None


class SchedulingUnavailableError(RuntimeError):
    """Raised when an optional scheduling dependency is unavailable."""


@dataclass(frozen=True)
class ScheduledJob:
    job_id: int
    processing_time: Fraction
    assignment_step: int | None = None
    is_fallback: bool = False


@dataclass(frozen=True)
class MachineSchedule:
    machine_id: int
    jobs: tuple[ScheduledJob, ...]
    load: Fraction


@dataclass(frozen=True)
class ScheduleResult:
    algorithm: str
    machine_count: int
    machines: tuple[MachineSchedule, ...]
    makespan: Fraction
    sorted_job_ids: tuple[int, ...]
    sorted_processing_times: tuple[Fraction, ...]
    feasibility_capacity: Fraction | None = None
    attempts: tuple[MultifitAttempt, ...] = ()


@dataclass(frozen=True)
class MultifitAttempt:
    iteration: int
    capacity: Fraction
    feasible: bool
    schedule: ScheduleResult | None = None


def _floor_fraction(value: Fraction) -> Fraction:
    return Fraction(value.numerator // value.denominator, 1)


def parse_processing_times(values: str | Iterable[str | int | float | Fraction]) -> tuple[Fraction, ...]:
    """Parse processing times from a comma-separated string or iterable."""

    if isinstance(values, str):
        raw_items = [item.strip() for item in re.split(r"[\s,;]+", values)]
        items = [item for item in raw_items if item]
    else:
        items = list(values)

    if not items:
        raise ValueError("At least one processing time is required.")

    parsed = tuple(parse_ratio(item) for item in items)
    if any(value <= 0 for value in parsed):
        raise ValueError("All processing times must be positive.")
    return parsed


def _sorted_jobs(processing_times: Sequence[Fraction]) -> list[ScheduledJob]:
    return sorted(
        (ScheduledJob(job_id=index + 1, processing_time=value) for index, value in enumerate(processing_times)),
        key=lambda job: (-job.processing_time, job.job_id),
    )


def _build_schedule_result(
    *,
    algorithm: str,
    jobs: Sequence[ScheduledJob],
    machine_jobs: Sequence[list[ScheduledJob]],
    machine_loads: Sequence[Fraction],
    capacity: Fraction | None,
) -> ScheduleResult:
    machines = tuple(
        MachineSchedule(
            machine_id=machine_index + 1,
            jobs=tuple(machine_jobs[machine_index]),
            load=machine_loads[machine_index],
        )
        for machine_index in range(len(machine_jobs))
    )
    return ScheduleResult(
        algorithm=algorithm,
        machine_count=len(machine_jobs),
        machines=machines,
        makespan=max(machine_loads, default=Fraction(0, 1)),
        feasibility_capacity=capacity,
        sorted_job_ids=tuple(job.job_id for job in jobs),
        sorted_processing_times=tuple(job.processing_time for job in jobs),
    )


def first_fit_overflow_schedule(
    processing_times: Sequence[Fraction],
    machine_count: int,
    capacity: Fraction,
) -> ScheduleResult:
    """Run FFD while allowing extra machines beyond the requested machine count."""

    if machine_count <= 0:
        raise ValueError("machine_count must be positive.")
    if capacity <= 0:
        raise ValueError("capacity must be positive.")

    jobs = _sorted_jobs(processing_times)
    machine_jobs: list[list[ScheduledJob]] = [[] for _ in range(machine_count)]
    machine_loads: list[Fraction] = [Fraction(0, 1) for _ in range(machine_count)]
    machine_opened: list[bool] = [False for _ in range(machine_count)]

    for assignment_step, job in enumerate(jobs, start=1):
        assigned = False
        for machine_index in range(len(machine_jobs)):
            if machine_loads[machine_index] + job.processing_time <= capacity:
                is_fallback = bool(machine_jobs[machine_index]) and any(
                    machine_opened[later_machine_index]
                    for later_machine_index in range(machine_index + 1, len(machine_jobs))
                )
                machine_jobs[machine_index].append(
                    ScheduledJob(
                        job_id=job.job_id,
                        processing_time=job.processing_time,
                        assignment_step=assignment_step,
                        is_fallback=is_fallback,
                    )
                )
                machine_loads[machine_index] += job.processing_time
                machine_opened[machine_index] = True
                assigned = True
                break
        if not assigned:
            machine_jobs.append(
                [
                    ScheduledJob(
                        job_id=job.job_id,
                        processing_time=job.processing_time,
                        assignment_step=assignment_step,
                        is_fallback=False,
                    )
                ]
            )
            machine_loads.append(job.processing_time)
            machine_opened.append(True)

    return _build_schedule_result(
        algorithm="MULTIFIT-FFD",
        jobs=jobs,
        machine_jobs=machine_jobs,
        machine_loads=machine_loads,
        capacity=capacity,
    )


def first_fit_schedule(
    processing_times: Sequence[Fraction],
    machine_count: int,
    capacity: Fraction,
) -> ScheduleResult | None:
    """Run first-fit decreasing for a fixed capacity."""

    if machine_count <= 0:
        raise ValueError("machine_count must be positive.")
    if capacity <= 0:
        raise ValueError("capacity must be positive.")

    jobs = _sorted_jobs(processing_times)
    machine_jobs: list[list[ScheduledJob]] = [[] for _ in range(machine_count)]
    machine_loads: list[Fraction] = [Fraction(0, 1) for _ in range(machine_count)]
    machine_opened: list[bool] = [False for _ in range(machine_count)]

    for assignment_step, job in enumerate(jobs, start=1):
        assigned = False
        for machine_index in range(machine_count):
            if machine_loads[machine_index] + job.processing_time <= capacity:
                is_fallback = bool(machine_jobs[machine_index]) and any(
                    machine_opened[later_machine_index]
                    for later_machine_index in range(machine_index + 1, machine_count)
                )
                machine_jobs[machine_index].append(
                    ScheduledJob(
                        job_id=job.job_id,
                        processing_time=job.processing_time,
                        assignment_step=assignment_step,
                        is_fallback=is_fallback,
                    )
                )
                machine_loads[machine_index] += job.processing_time
                machine_opened[machine_index] = True
                assigned = True
                break
        if not assigned:
            return None

    return _build_schedule_result(
        algorithm="MULTIFIT-FFD",
        jobs=jobs,
        machine_jobs=machine_jobs,
        machine_loads=machine_loads,
        capacity=capacity,
    )


def multifit_schedule(
    processing_times: Sequence[Fraction],
    machine_count: int,
    *,
    iterations: int = 30,
) -> ScheduleResult:
    """Run the classic MULTIFIT binary search with FFD feasibility checks."""

    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if machine_count <= 0:
        raise ValueError("machine_count must be positive.")

    times = tuple(processing_times)
    if not times:
        raise ValueError("At least one processing time is required.")

    total = sum(times, start=Fraction(0, 1))
    lower = max(max(times), total / machine_count)
    upper = max(2 * total / machine_count, max(times))
    attempts: list[MultifitAttempt] = []

    upper = _floor_fraction(upper)
    best_schedule = first_fit_schedule(times, machine_count, upper)
    while best_schedule is None:
        attempts.append(
            MultifitAttempt(
                iteration=len(attempts) + 1,
                capacity=upper,
                feasible=False,
                schedule=first_fit_overflow_schedule(times, machine_count, upper),
            )
        )
        upper *= 2
        best_schedule = first_fit_schedule(times, machine_count, upper)
    attempts.append(
        MultifitAttempt(
            iteration=len(attempts) + 1,
            capacity=upper,
            feasible=True,
            schedule=best_schedule,
        )
    )

    for _ in range(iterations):
        mid = _floor_fraction((lower + upper) / 2)
        candidate = first_fit_schedule(times, machine_count, mid)
        if candidate is None:
            attempts.append(
                MultifitAttempt(
                    iteration=len(attempts) + 1,
                    capacity=mid,
                    feasible=False,
                    schedule=first_fit_overflow_schedule(times, machine_count, mid),
                )
            )
            lower = mid
            continue
        attempts.append(
            MultifitAttempt(
                iteration=len(attempts) + 1,
                capacity=mid,
                feasible=True,
                schedule=candidate,
            )
        )
        upper = mid
        best_schedule = candidate

    upper = _floor_fraction(upper)
    final_schedule = first_fit_schedule(times, machine_count, upper)
    if final_schedule is None:
        raise RuntimeError("MULTIFIT failed to produce a feasible schedule at the final upper bound.")
    attempts.append(
        MultifitAttempt(
            iteration=len(attempts) + 1,
            capacity=upper,
            feasible=True,
            schedule=final_schedule,
        )
    )
    return ScheduleResult(
        algorithm=final_schedule.algorithm,
        machine_count=final_schedule.machine_count,
        machines=final_schedule.machines,
        makespan=final_schedule.makespan,
        sorted_job_ids=final_schedule.sorted_job_ids,
        sorted_processing_times=final_schedule.sorted_processing_times,
        feasibility_capacity=final_schedule.feasibility_capacity,
        attempts=tuple(attempts),
    )


def _require_gurobi() -> None:
    if gp is None or GRB is None:
        raise SchedulingUnavailableError(
            "gurobipy is required for the exact OPT schedule. Install gurobipy and activate a license."
        )


def solve_opt_schedule(
    processing_times: Sequence[Fraction],
    machine_count: int,
) -> ScheduleResult:
    """Solve the exact min-max identical-machine scheduling model."""

    _require_gurobi()
    if machine_count <= 0:
        raise ValueError("machine_count must be positive.")

    times = tuple(processing_times)
    if not times:
        raise ValueError("At least one processing time is required.")

    model: GurobiModel = gp.Model("opt_schedule")
    model.Params.OutputFlag = 0

    jobs = range(len(times))
    machines = range(machine_count)
    x = model.addVars(machines, jobs, vtype=GRB.BINARY, name="x")
    cmax = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="Cmax")

    model.addConstrs(
        (gp.quicksum(x[machine_index, job_index] for machine_index in machines) == 1 for job_index in jobs),
        name="assign",
    )
    model.addConstrs(
        (
            gp.quicksum(float(times[job_index]) * x[machine_index, job_index] for job_index in jobs) <= cmax
            for machine_index in machines
        ),
        name="load_ub",
    )
    model.setObjective(cmax, GRB.MINIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"OPT schedule model did not solve to optimality. Status={model.Status}")

    machine_jobs: list[list[ScheduledJob]] = [[] for _ in machines]
    for machine_index in machines:
        assigned_jobs = [
            ScheduledJob(job_id=job_index + 1, processing_time=times[job_index])
            for job_index in jobs
            if x[machine_index, job_index].X > 0.5
        ]
        assigned_jobs.sort(key=lambda job: (-job.processing_time, job.job_id))
        machine_jobs[machine_index] = assigned_jobs

    machine_loads = [
        sum((job.processing_time for job in assigned_jobs), start=Fraction(0, 1)) for assigned_jobs in machine_jobs
    ]
    sorted_jobs = _sorted_jobs(times)
    return ScheduleResult(
        algorithm="OPT",
        machine_count=machine_count,
        machines=tuple(
            MachineSchedule(
                machine_id=machine_index + 1,
                jobs=tuple(machine_jobs[machine_index]),
                load=machine_loads[machine_index],
            )
            for machine_index in machines
        ),
        makespan=Fraction(str(float(model.ObjVal))),
        feasibility_capacity=None,
        sorted_job_ids=tuple(job.job_id for job in sorted_jobs),
        sorted_processing_times=tuple(job.processing_time for job in sorted_jobs),
    )


def _schedule_color_map(*schedules: ScheduleResult):
    import matplotlib.pyplot as plt

    job_ids = sorted({job.job_id for schedule in schedules for machine in schedule.machines for job in machine.jobs})
    cmap = plt.get_cmap("tab20")
    return {job_id: cmap((job_id - 1) % cmap.N) for job_id in job_ids}


def _draw_schedule_axis(ax, schedule: ScheduleResult, x_limit: float, color_map: dict[int, object]) -> bool:
    machine_count = schedule.machine_count
    y_positions = list(range(machine_count, 0, -1))
    has_fallback = False
    for y_value, machine in zip(y_positions, schedule.machines):
        left = 0.0
        for job in machine.jobs:
            width = float(job.processing_time)
            edge_color = "crimson" if job.is_fallback else "black"
            hatch = "//" if job.is_fallback else ""
            has_fallback = has_fallback or job.is_fallback
            ax.barh(
                y_value,
                width,
                left=left,
                height=0.72,
                color=color_map[job.job_id],
                edgecolor=edge_color,
                linewidth=1.5 if job.is_fallback else 0.8,
                hatch=hatch,
            )
            if width >= 0.06 * x_limit:
                label = f"j{job.job_id}\n{format_ratio(job.processing_time)}"
                if job.is_fallback:
                    label += "\nF"
                ax.text(
                    left + width / 2,
                    y_value,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
            left += width
        ax.text(left + 0.01 * x_limit, y_value, format_ratio(machine.load), va="center", fontsize=9)

    ax.set_xlim(0, x_limit)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"M{i}" for i in range(1, machine_count + 1)])
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    subtitle = f"{schedule.algorithm} (Cmax={format_ratio(schedule.makespan)})"
    if schedule.feasibility_capacity is not None:
        subtitle += f"\nFFD cap={format_ratio(schedule.feasibility_capacity)}"
        ax.axvline(float(schedule.feasibility_capacity), color="crimson", linestyle=":", linewidth=1.5)
    ax.set_title(subtitle)
    ax.set_xlabel("Load")
    return has_fallback


def _fallback_legend_handles():
    import matplotlib.pyplot as plt

    return [
        plt.Rectangle((0, 0), 1, 1, facecolor="lightgray", edgecolor="black", linewidth=0.8, label="Regular"),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="lightgray",
            edgecolor="crimson",
            linewidth=1.5,
            hatch="//",
            label="Fallback",
        ),
    ]


def plot_schedule_comparison(
    multifit: ScheduleResult,
    optimum: ScheduleResult,
    output_path: Path,
    *,
    title: str | None = None,
) -> Path:
    """Render a side-by-side schedule comparison figure."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise SchedulingUnavailableError("matplotlib is required to render schedule figures.") from exc

    if multifit.machine_count != optimum.machine_count:
        raise ValueError("Schedule figures require the same machine count for both schedules.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    machine_count = multifit.machine_count
    x_limit = 1.05 * float(max(multifit.makespan, optimum.makespan))
    color_map = _schedule_color_map(multifit, optimum)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, max(4, 0.7 * machine_count)),
        sharey=True,
        constrained_layout=True,
    )

    has_fallback = _draw_schedule_axis(axes[0], multifit, x_limit, color_map)
    _draw_schedule_axis(axes[1], optimum, x_limit, color_map)
    axes[0].set_ylabel("Machine")
    if has_fallback:
        axes[0].legend(handles=_fallback_legend_handles(), loc="lower right", frameon=False)

    if title is not None:
        fig.suptitle(title, fontsize=13)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_multifit_history(
    multifit: ScheduleResult,
    output_dir: Path,
    *,
    title_prefix: str | None = None,
) -> tuple[Path, ...]:
    """Render one PNG per MULTIFIT attempt."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise SchedulingUnavailableError("matplotlib is required to render schedule figures.") from exc

    if not multifit.attempts:
        raise ValueError("MULTIFIT history is unavailable for this schedule.")

    output_dir.mkdir(parents=True, exist_ok=True)
    feasible_attempts = [attempt.schedule for attempt in multifit.attempts if attempt.schedule is not None]
    x_limit = 1.05 * float(
        max(
            [multifit.makespan]
            + [attempt.schedule.makespan for attempt in multifit.attempts if attempt.schedule is not None]
            + [attempt.capacity for attempt in multifit.attempts]
        )
    )
    color_map = _schedule_color_map(multifit, *feasible_attempts)
    written_paths: list[Path] = []

    for attempt in multifit.attempts:
        status_text = "feasible" if attempt.feasible else "infeasible"
        output_path = output_dir / f"attempt_{attempt.iteration:02d}_{status_text}.png"
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(8, max(4, 0.7 * multifit.machine_count)),
            constrained_layout=True,
        )
        if attempt.schedule is None:
            raise RuntimeError("MULTIFIT history attempt is missing its schedule trace.")
        has_fallback = _draw_schedule_axis(ax, attempt.schedule, x_limit, color_map)
        if not attempt.feasible:
            ax.text(
                0.99,
                0.98,
                "needs extra machine(s)",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                color="crimson",
            )
        if has_fallback:
            ax.legend(handles=_fallback_legend_handles(), loc="lower right", frameon=False)
        if title_prefix is not None:
            fig.suptitle(f"{title_prefix} | Attempt {attempt.iteration}", fontsize=13)
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        written_paths.append(output_path)

    return tuple(written_paths)


def render_schedule_text(schedule: ScheduleResult) -> str:
    """Return a compact textual summary for terminal use."""

    lines = [f"{schedule.algorithm}: Cmax={format_ratio(schedule.makespan)}"]
    if schedule.feasibility_capacity is not None:
        lines[0] += f", cap={format_ratio(schedule.feasibility_capacity)}"
    for machine in schedule.machines:
        jobs_text = ", ".join(
            f"j{job.job_id}:{format_ratio(job.processing_time)}"
            f"({'F' if job.is_fallback else 'R'})"
            for job in machine.jobs
        )
        lines.append(f"  M{machine.machine_id} [{format_ratio(machine.load)}]: {jobs_text}")
    return "\n".join(lines)
