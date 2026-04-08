"""Solve ~100 cases without target LB, extract processing times, run FFD with target as capacity."""
from __future__ import annotations

import json
import os
import random
import sys
from fractions import Fraction
from pathlib import Path

def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

_bootstrap_src()

from multifit_optveri.acceleration import AccelerationCase, PAPER_TARGET_RATIO
from multifit_optveri.branching import FallbackStarts, MtfProfile, OptProfile
from multifit_optveri.config import SolverConfig
from multifit_optveri.experiments import ExperimentCase
from multifit_optveri.math_utils import parse_ratio
from multifit_optveri.models.obv import build_obv_model
from multifit_optveri.schedules import first_fit_schedule


def _parse_optional_int(value):
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "None", "none", "null"}:
        return None
    return int(text)


def _parse_mtf_profile(text):
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    values = [int(part.strip()) for part in raw.strip("()").split(",")]
    if len(values) != 8:
        return None
    return MtfProfile(*values)


def _parse_opt_profile(text, acceleration_case, ell):
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    values = [int(part.strip()) for part in raw.strip("()").split(",")]
    if len(values) != 3:
        return None
    if acceleration_case is AccelerationCase.CASE_1:
        pattern = "case1"
    elif acceleration_case is AccelerationCase.CASE_2:
        pattern = "two_long" if (ell is not None and ell % 2 == 0) else "regular"
    else:
        pattern = "generic"
    return OptProfile(values[0], values[1], values[2], pattern=pattern)


def _parse_fallback_starts(text):
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    parts = [part.strip() for part in raw.strip("()").split(",")]
    if len(parts) != 3:
        return None
    return FallbackStarts(
        s2=_parse_optional_int(parts[0]),
        s3=_parse_optional_int(parts[1]),
        s4=_parse_optional_int(parts[2]),
    )


def case_from_summary(summary_path: Path) -> ExperimentCase:
    raw_text = summary_path.read_text(encoding="utf-8")
    sanitized = raw_text.replace(": infinity", ": null").replace(": -infinity", ": null")
    d = json.loads(sanitized)

    acc = AccelerationCase(d["acceleration_case"])
    ell = _parse_optional_int(d.get("ell"))
    opt_profile = _parse_opt_profile(d.get("opt-profile-(e3_e4_e5)"), acc, ell)

    output_dir = Path(d["output_dir"])
    run_output_root = output_dir.parents[1]

    return ExperimentCase(
        experiment_name=str(d["experiment_name"]),
        machine_count=int(d["machine_count"]),
        job_count=int(d["job_count"]),
        acceleration_case=acc,
        ell=ell,
        mtf_profile=_parse_mtf_profile(d.get("mtf-profile-(f1_r2_f2_r3_f3_r4_f4_r5)")),
        opt_profile=opt_profile,
        target_ratio=parse_ratio(str(d["target_ratio"])),
        output_root=Path("results"),
        write_lp=False,
        enforce_target_lower_bound=False,  # no target LB
        solver=SolverConfig(output_flag=0, non_convex=2, time_limit_seconds=30),
        fallback_starts=_parse_fallback_starts(d.get("fallback-starts-(s2_s3_s4)")),
        run_output_root=run_output_root,
    )


def main():
    base = Path("results/paper_base/20260407_163035/cases")
    if not base.exists():
        print(f"Directory not found: {base}")
        return 1

    # Only pick cases that were feasible (USER_OBJ_LIMIT) in the original run
    feasible_summaries = []
    for summary_path in sorted(base.glob("*/summary.json")):
        raw = summary_path.read_text(encoding="utf-8").replace(": infinity", ": null").replace(": -infinity", ": null")
        data = json.loads(raw)
        if data.get("status") == "USER_OBJ_LIMIT":
            feasible_summaries.append(summary_path)

    print(f"Found {len(feasible_summaries)} originally feasible cases")
    random.seed(42)
    sample = random.sample(feasible_summaries, min(100, len(feasible_summaries)))

    results = []
    for i, summary_path in enumerate(sample):
        case_name = summary_path.parent.name
        print(f"[{i+1}/{len(sample)}] {case_name} ... ", end="", flush=True)

        try:
            case = case_from_summary(summary_path)
            built = build_obv_model(case)
            model = built.model
            model.Params.OutputFlag = 0
            model.optimize()

            if model.Status not in (2, 13):  # OPTIMAL or SUBOPTIMAL
                print(f"status={model.Status}, skipped")
                built.model.dispose()
                continue

            n = case.job_count
            m = case.machine_count
            p_vals = {}
            for v in model.getVars():
                if v.VarName.startswith("p["):
                    idx = int(v.VarName.split("[")[1].rstrip("]"))
                    p_vals[idx] = v.X

            built.model.dispose()

            if len(p_vals) < n:
                print("missing p vars, skipped")
                continue

            processing_times = [Fraction(p_vals[j]).limit_denominator(10000) for j in range(1, n + 1)]
            target = case.target_ratio
            capacity = target  # use target as capacity

            schedule = first_fit_schedule(processing_times, m, capacity)
            if schedule is None:
                feasible = False
                makespan_str = "INFEASIBLE"
            else:
                feasible = True
                makespan_str = f"{float(schedule.makespan):.6f}"

            obj_val = model.ObjVal if hasattr(model, "ObjVal") else None
            results.append({
                "case": case_name,
                "m": m,
                "n": n,
                "obj": f"{p_vals.get(n, 0):.6f}" if n in p_vals else "?",
                "z_obj": f"{obj_val:.6f}" if obj_val is not None else "?",
                "ffd_feasible": feasible,
                "ffd_makespan": makespan_str,
                "target": f"{float(target):.6f}",
                "processing_times": [f"{float(pt):.6f}" for pt in processing_times],
            })
            print(f"Z={obj_val:.4f}, FFD={'OK' if feasible else 'INFEASIBLE'} makespan={makespan_str}")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Write summary
    out_path = Path("results/sample_mtf_schedules.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} results to {out_path}")

    # Print summary table
    feasible_count = sum(1 for r in results if r["ffd_feasible"])
    infeasible_count = sum(1 for r in results if not r["ffd_feasible"])
    print(f"\nSummary: {feasible_count} feasible, {infeasible_count} infeasible out of {len(results)} solved")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
