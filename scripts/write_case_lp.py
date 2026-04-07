from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_bootstrap_src()

from multifit_optveri.acceleration import AccelerationCase
from multifit_optveri.branching import FallbackStarts, MtfProfile, OptProfile
from multifit_optveri.config import SolverConfig
from multifit_optveri.experiments import ExperimentCase
from multifit_optveri.math_utils import parse_ratio
from multifit_optveri.models.obv import build_obv_model


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if text in {"", "None", "none", "null"}:
        return None
    return int(text)


def _parse_mtf_profile(text: str | None) -> MtfProfile | None:
    if text is None:
        return None
    raw = text.strip()
    if not raw:
        return None
    values = [int(part.strip()) for part in raw.strip("()").split(",")]
    if len(values) != 8:
        raise ValueError(f"Expected 8 MTF profile entries, got {len(values)} from {text!r}.")
    return MtfProfile(*values)


def _parse_opt_profile(text: str | None) -> OptProfile | None:
    if text is None:
        return None
    raw = text.strip()
    if not raw:
        return None
    values = [int(part.strip()) for part in raw.strip("()").split(",")]
    if len(values) != 3:
        raise ValueError(f"Expected 3 OPT profile entries, got {len(values)} from {text!r}.")
    return OptProfile(values[0], values[1], values[2], pattern="generic")


def _infer_opt_pattern(acceleration_case: AccelerationCase, ell: int | None) -> str:
    if acceleration_case is AccelerationCase.CASE_1:
        return "case1"
    if acceleration_case is AccelerationCase.CASE_2:
        if ell is not None and ell % 2 == 0:
            return "two_long"
        return "regular"
    return "generic"


def _parse_fallback_starts(text: str | None) -> FallbackStarts | None:
    if text is None:
        return None
    raw = text.strip()
    if not raw:
        return None
    parts = [part.strip() for part in raw.strip("()").split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 fallback start entries, got {len(parts)} from {text!r}.")
    return FallbackStarts(
        s2=_parse_optional_int(parts[0]),
        s3=_parse_optional_int(parts[1]),
        s4=_parse_optional_int(parts[2]),
    )


def _case_from_summary(summary_path: Path) -> ExperimentCase:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    output_dir = Path(payload["output_dir"])
    run_output_root = output_dir.parents[1]
    acceleration_case = AccelerationCase(payload["acceleration_case"])
    ell = _parse_optional_int(str(payload.get("ell", "")))
    opt_profile = _parse_opt_profile(payload.get("opt-profile-(e3_e4_e5)"))
    if opt_profile is not None:
        opt_profile = OptProfile(
            opt_profile.m3,
            opt_profile.m4,
            opt_profile.m5,
            pattern=_infer_opt_pattern(acceleration_case, ell),
        )

    return ExperimentCase(
        experiment_name=str(payload["experiment_name"]),
        machine_count=int(payload["machine_count"]),
        job_count=int(payload["job_count"]),
        acceleration_case=acceleration_case,
        ell=ell,
        mtf_profile=_parse_mtf_profile(payload.get("mtf-profile-(f1_r2_f2_r3_f3_r4_f4_r5)")),
        opt_profile=opt_profile,
        target_ratio=parse_ratio(str(payload["target_ratio"])),
        output_root=Path("results"),
        write_lp=True,
        enforce_target_lower_bound=True,
        solver=SolverConfig(output_flag=0, non_convex=2),
        fallback_starts=_parse_fallback_starts(payload.get("fallback-starts-(s2_s3_s4)")),
        run_output_root=run_output_root,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild one case from summary.json and write its model.lp.")
    parser.add_argument("summary_json", type=Path, help="Path to a case summary.json file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional explicit LP output path. Defaults to <case output dir>/model.lp.",
    )
    args = parser.parse_args(argv)

    summary_path = args.summary_json.resolve()
    case = _case_from_summary(summary_path)
    built = build_obv_model(case)
    try:
        output_path = args.output.resolve() if args.output is not None else case.output_dir / "model.lp"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        built.model.write(str(output_path))
        print(f"Wrote LP: {output_path}")
        print(f"Case: {case.case_id}")
    finally:
        built.model.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
