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

    machines = tuple(
        MachineSchedule(
            machine_id=machine_index + 1,
            jobs=tuple(machine_jobs[machine_index]),
            load=machine_loads[machine_index],
        )
        for machine_index in range(machine_count)
    )
    return ScheduleResult(
        algorithm="MULTIFIT-FFD",
        machine_count=machine_count,
        machines=machines,
        makespan=max(machine_loads, default=Fraction(0, 1)),
        feasibility_capacity=capacity,
        sorted_job_ids=tuple(job.job_id for job in jobs),
        sorted_processing_times=tuple(job.processing_time for job in jobs),
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

    best_schedule = first_fit_schedule(times, machine_count, upper)
    while best_schedule is None:
        upper *= 2
        best_schedule = first_fit_schedule(times, machine_count, upper)

    for _ in range(iterations):
        mid = (lower + upper) / 2
        candidate = first_fit_schedule(times, machine_count, mid)
        if candidate is None:
            lower = mid
            continue
        upper = mid
        best_schedule = candidate

    final_schedule = first_fit_schedule(times, machine_count, upper)
    if final_schedule is None:
        raise RuntimeError("MULTIFIT failed to produce a feasible schedule at the final upper bound.")
    return final_schedule


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
    job_ids = sorted({job.job_id for machine in multifit.machines + optimum.machines for job in machine.jobs})
    cmap = plt.get_cmap("tab20")
    color_map = {job_id: cmap((job_id - 1) % cmap.N) for job_id in job_ids}
    legend_handles = None

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, max(4, 0.7 * machine_count)),
        sharey=True,
        constrained_layout=True,
    )

    def draw(ax, schedule: ScheduleResult) -> None:
        nonlocal legend_handles
        y_positions = list(range(machine_count, 0, -1))
        for y_value, machine in zip(y_positions, schedule.machines):
            left = 0.0
            for job in machine.jobs:
                width = float(job.processing_time)
                edge_color = "crimson" if job.is_fallback else "black"
                hatch = "//" if job.is_fallback else ""
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
                    ax.text(
                        left + width / 2,
                        y_value,
                        f"j{job.job_id}\n{format_ratio(job.processing_time)}",
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
        if schedule.algorithm.startswith("MULTIFIT") and legend_handles is None:
            legend_handles = [
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

    draw(axes[0], multifit)
    draw(axes[1], optimum)
    axes[0].set_ylabel("Machine")
    if legend_handles is not None:
        fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False)

    if title is not None:
        fig.suptitle(title, fontsize=13)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


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
