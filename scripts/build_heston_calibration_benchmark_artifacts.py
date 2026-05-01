from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from option_pricing.diagnostics.heston import (
    run_heston_calibration_benchmark_diagnostics,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Heston calibration analytic-vs-finite-difference Jacobian "
            "benchmark diagnostic artifacts."
        )
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ROOT / "benchmarks" / "artifacts" / "heston_calibration",
        help="Directory for Heston calibration benchmark artifacts.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Measured repeat count per Jacobian mode.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup count per Jacobian mode.",
    )
    parser.add_argument(
        "--noise-vol-bps",
        type=float,
        default=0.0,
        help="Absolute Black-vol quote noise in basis points.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used only when --noise-vol-bps is positive.",
    )
    return parser.parse_args(argv)


def _write_frame(
    frame: pd.DataFrame,
    *,
    stem: str,
    artifacts_dir: Path,
) -> dict[str, str]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    csv_path = artifacts_dir / f"{stem}.csv"
    json_path = artifacts_dir / f"{stem}.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")
    return {
        "csv": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
        "json": str(json_path.relative_to(ROOT)).replace("\\", "/"),
    }


def _write_json(
    payload: dict[str, Any],
    *,
    name: str,
    artifacts_dir: Path,
) -> str:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path.relative_to(ROOT)).replace("\\", "/")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    artifacts_dir = args.artifacts_dir.resolve()

    diagnostics = run_heston_calibration_benchmark_diagnostics(
        repeat=args.repeat,
        warmup=args.warmup,
        noise_vol_bps=args.noise_vol_bps,
        random_seed=args.random_seed,
    )

    outputs = {
        "runs": _write_frame(
            diagnostics.tables["runs"],
            stem="heston_calibration_jacobian_runs",
            artifacts_dir=artifacts_dir,
        ),
        "summary": _write_frame(
            diagnostics.tables["summary"],
            stem="heston_calibration_jacobian_summary",
            artifacts_dir=artifacts_dir,
        ),
        "parameter_recovery": _write_frame(
            diagnostics.tables["parameter_recovery"],
            stem="heston_calibration_jacobian_parameter_recovery",
            artifacts_dir=artifacts_dir,
        ),
        "residuals": _write_frame(
            diagnostics.tables["residuals"],
            stem="heston_calibration_jacobian_residuals",
            artifacts_dir=artifacts_dir,
        ),
        "meta": _write_json(
            diagnostics.meta,
            name="heston_calibration_jacobian_meta.json",
            artifacts_dir=artifacts_dir,
        ),
    }
    # REVIEW: Timing artifact files are generated on demand because their
    # numeric contents are environment-dependent.
    index_path = _write_json(
        outputs,
        name="heston_calibration_jacobian_artifacts.json",
        artifacts_dir=artifacts_dir,
    )
    print(index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
