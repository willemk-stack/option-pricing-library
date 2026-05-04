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

import numpy as np
import pandas as pd

from option_pricing.diagnostics.heston import (
    run_heston_calibration_benchmark_diagnostics,
)
from option_pricing.numerics.quadrature import QuadratureConfig

PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "smoke": {
        "scenario": "Small deterministic synthetic grid for smoke and wiring coverage.",
        "repeat": 1,
        "warmup": 0,
        "random_seed": 42,
        "noise_vol_bps": 0.0,
        "expiries": (0.5, 1.0),
        "log_moneyness": (-0.08, 0.0, 0.08),
        "quad_cfg": QuadratureConfig(u_max=50.0, n_panels=6, nodes_per_panel=6),
        "notes": [
            "NOTE: Smoke benchmark keeps a small synthetic quote grid and cheap quadrature so analytic-vs-finite-difference coverage stays fast enough for regression workflows.",
            "NOTE: Smoke timings are directional only and should not be treated as release-grade speed estimates.",
        ],
    },
    "release": {
        "scenario": "Release benchmark with a larger deterministic quote grid, warmup, repeated measurements, and more production-like Gauss-Legendre quadrature.",
        "repeat": 5,
        "warmup": 1,
        "random_seed": 42,
        "noise_vol_bps": 0.0,
        "expiries": (0.25, 0.5, 1.0, 2.0),
        "log_moneyness": (-0.12, -0.06, 0.0, 0.06, 0.12),
        "quad_cfg": QuadratureConfig(u_max=120.0, n_panels=16, nodes_per_panel=16),
        "notes": [
            "NOTE: Release benchmark is stronger than smoke for this configured scenario, but it still measures one deterministic synthetic setup rather than universal analytic-Jacobian performance.",
            "NOTE: Release timings remain environment-dependent and should be read as directional evidence for this optimizer, objective, and quote-grid configuration.",
        ],
    },
}


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
        "--profile",
        choices=("all", "smoke", "release"),
        default="all",
        help="Benchmark profile or profiles to build.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="Override measured repeat count for the selected profile(s).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Override warmup count for the selected profile(s).",
    )
    parser.add_argument(
        "--noise-vol-bps",
        type=float,
        default=None,
        help="Override absolute Black-vol quote noise in basis points.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Override the deterministic random seed for the selected profile(s).",
    )
    return parser.parse_args(argv)


def _selected_profiles(profile: str) -> tuple[str, ...]:
    if profile == "smoke":
        return ("smoke",)
    if profile == "release":
        return ("release",)
    return ("smoke", "release")


def _resolved_profile(profile_name: str, args: argparse.Namespace) -> dict[str, Any]:
    defaults = PROFILE_DEFAULTS[profile_name]
    return {
        **defaults,
        "label": profile_name,
        "repeat": defaults["repeat"] if args.repeat is None else int(args.repeat),
        "warmup": defaults["warmup"] if args.warmup is None else int(args.warmup),
        "noise_vol_bps": (
            defaults["noise_vol_bps"]
            if args.noise_vol_bps is None
            else float(args.noise_vol_bps)
        ),
        "random_seed": (
            defaults["random_seed"]
            if args.random_seed is None
            else int(args.random_seed)
        ),
    }


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


def _profile_stem(profile_name: str, table_name: str) -> str:
    return f"heston_calibration_jacobian_{profile_name}_{table_name}"


def _profile_quad_cfg_payload(quad_cfg: QuadratureConfig) -> dict[str, float | int]:
    return {
        "u_max": float(quad_cfg.u_max),
        "n_panels": int(quad_cfg.n_panels),
        "nodes_per_panel": int(quad_cfg.nodes_per_panel),
    }


def _apply_profile_metadata(
    diagnostics: Any,
    *,
    profile: dict[str, Any],
) -> None:
    profile_name = str(profile["label"])
    diagnostics.meta.update(
        {
            "benchmark_label": profile_name,
            "scenario": str(profile["scenario"]),
            "quote_grid": {
                "expiries": list(profile["expiries"]),
                "log_moneyness": list(profile["log_moneyness"]),
            },
            "quadrature": _profile_quad_cfg_payload(profile["quad_cfg"]),
            "notes": [
                *profile["notes"],
                "NOTE: Optimizer method and loss choices determine benchmark comparability.",
                "NOTE: Residuals use the repo's existing vega-scaled price objective.",
                "NOTE: Timing and speedup ratios are environment-dependent.",
            ],
        }
    )

    summary = diagnostics.tables["summary"].copy()
    summary["benchmark_label"] = profile_name
    summary.loc[summary["mode"] == "finite_difference", "notes"] = (
        f"{profile_name} finite-difference baseline"
    )
    summary.loc[summary["mode"] == "analytic", "notes"] = (
        f"NOTE: {profile_name} analytic-vs-finite-difference speedup ratios are directional and environment-dependent."
    )
    diagnostics.tables["summary"] = summary

    for table_name in ("runs", "parameter_recovery", "residuals"):
        table = diagnostics.tables[table_name].copy()
        table["benchmark_label"] = profile_name
        diagnostics.tables[table_name] = table


def _build_profile_outputs(
    *,
    profile: dict[str, Any],
    artifacts_dir: Path,
) -> dict[str, Any]:
    diagnostics = run_heston_calibration_benchmark_diagnostics(
        expiries=np.asarray(profile["expiries"], dtype=np.float64),
        log_moneyness=np.asarray(profile["log_moneyness"], dtype=np.float64),
        quad_cfg=profile["quad_cfg"],
        repeat=int(profile["repeat"]),
        warmup=int(profile["warmup"]),
        noise_vol_bps=float(profile["noise_vol_bps"]),
        random_seed=int(profile["random_seed"]),
    )
    _apply_profile_metadata(diagnostics, profile=profile)

    profile_name = str(profile["label"])
    return {
        "benchmark_label": profile_name,
        "scenario": str(profile["scenario"]),
        "runs": _write_frame(
            diagnostics.tables["runs"],
            stem=_profile_stem(profile_name, "runs"),
            artifacts_dir=artifacts_dir,
        ),
        "summary": _write_frame(
            diagnostics.tables["summary"],
            stem=_profile_stem(profile_name, "summary"),
            artifacts_dir=artifacts_dir,
        ),
        "parameter_recovery": _write_frame(
            diagnostics.tables["parameter_recovery"],
            stem=_profile_stem(profile_name, "parameter_recovery"),
            artifacts_dir=artifacts_dir,
        ),
        "residuals": _write_frame(
            diagnostics.tables["residuals"],
            stem=_profile_stem(profile_name, "residuals"),
            artifacts_dir=artifacts_dir,
        ),
        "meta": _write_json(
            diagnostics.meta,
            name=f"heston_calibration_jacobian_{profile_name}_meta.json",
            artifacts_dir=artifacts_dir,
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    artifacts_dir = args.artifacts_dir.resolve()

    outputs = {
        profile_name: _build_profile_outputs(
            profile=_resolved_profile(profile_name, args),
            artifacts_dir=artifacts_dir,
        )
        for profile_name in _selected_profiles(args.profile)
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
