"""Export CSVs for the local-vol end-to-end workflow.

This script runs the library's flagship Capstone 2 pipeline:

    surface quotes -> (SVI) implied surface -> local vol -> PDE repricing grid

and can export either:

1) a *single* diagnostics table to a CSV (the original behavior), or
2) a *poster bundle* of CSVs that directly support the "mountain" style plots
   shown in the capstone/poster figures:

   - Gatheral vs Dupire local-vol differences (grid, long-form)
   - PDE convergence sweep (single-option)
   - Round-trip repricing residual mountain (IV/price error vs (T, K))
   - Local-vol "calibration struggles" (worst points + reason counts)
   - SVI fit / repair diagnostics (optional but useful)

Typical usage (from repo root):

    # Single table (default):
    python scripts/generate_localvol_full_csv.py --out out/repricing_grid.csv

    # Poster bundle (recommended for plotting):
    python scripts/generate_localvol_full_csv.py --poster

Notes
-----
- This script relies on pandas because the diagnostics tables are pandas
  DataFrames.
- If you haven't installed dev dependencies, use:
    python -m pip install -e '.[dev]'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _ensure_src_on_path() -> None:
    """Ensure repo/src is importable when running from a source checkout."""

    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _load_overrides(path_or_json: str) -> dict[str, Any]:
    """Load overrides from a JSON string or a path to a JSON file."""

    p = Path(path_or_json)
    if p.exists() and p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(path_or_json)


def _get_run_capstone2() -> Any:
    """Import run_capstone2, adding repo/src to sys.path as a fallback."""

    try:
        from option_pricing.demos.capstone2 import run_capstone2  # type: ignore

        return run_capstone2
    except Exception:  # noqa: BLE001
        _ensure_src_on_path()
        from option_pricing.demos.capstone2 import run_capstone2  # type: ignore

        return run_capstone2


def _write_df(df: Any, path: Path) -> None:
    """Write a DataFrame to CSV, ensuring its parent directory exists."""

    # Pandas is a dev dependency; import here so the error points to install step.
    import pandas as pd  # noqa: PLC0415

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _reasons_to_str(mask: int) -> str:
    """Decode LVInvalidReason bitmask to a stable, plot-friendly string."""

    try:
        from option_pricing.vol.local_vol_types import LVInvalidReason  # type: ignore
    except Exception:  # noqa: BLE001
        _ensure_src_on_path()
        from option_pricing.vol.local_vol_types import LVInvalidReason  # type: ignore

    if mask == 0:
        return ""
    parts: list[str] = []
    for r in LVInvalidReason:
        if int(r) != 0 and (mask & int(r)):
            if r.name is not None:
                parts.append(r.name)
    return "|".join(parts)


def _build_lv_compare_grid_df(lv_cmp: Any) -> Any:
    """Long-form (T, K) grid for Gatheral-vs-Dupire comparison."""

    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    Ts = np.asarray(lv_cmp.expiries, dtype=float).reshape(-1)
    Ks = np.asarray(lv_cmp.strikes, dtype=float).reshape(-1)
    Fs = np.asarray(lv_cmp.forwards, dtype=float).reshape(-1)

    y = np.asarray(lv_cmp.y, dtype=float)
    g_sig = np.asarray(lv_cmp.g_sigma, dtype=float)
    g_den = np.asarray(lv_cmp.g_denom, dtype=float)
    g_inv = np.asarray(lv_cmp.g_invalid, dtype=bool)
    g_reason = np.asarray(lv_cmp.g_reason, dtype=np.uint32)

    d_sig = np.asarray(lv_cmp.dupire.sigma, dtype=float)
    d_den = np.asarray(lv_cmp.dupire.denom, dtype=float)
    d_inv = np.asarray(lv_cmp.dupire.invalid, dtype=bool)
    d_reason = np.asarray(lv_cmp.dupire.reason, dtype=np.uint32)

    diff_sig = np.asarray(lv_cmp.diff_sigma, dtype=float)
    inv_union = np.asarray(lv_cmp.invalid_union, dtype=bool)

    rows: list[dict[str, Any]] = []
    for i, T in enumerate(Ts):
        for j, K in enumerate(Ks):
            rows.append(
                {
                    "T": float(T),
                    "K": float(K),
                    "F": float(Fs[i]) if i < len(Fs) else float("nan"),
                    "y": float(y[i, j]) if y.ndim == 2 else float("nan"),
                    "sigma_gatheral": float(g_sig[i, j]),
                    "sigma_dupire": float(d_sig[i, j]),
                    "diff_sigma": float(diff_sig[i, j]),
                    "denom_gatheral": float(g_den[i, j]),
                    "denom_dupire": float(d_den[i, j]),
                    "invalid_gatheral": bool(g_inv[i, j]),
                    "invalid_dupire": bool(d_inv[i, j]),
                    "invalid_union": bool(inv_union[i, j]),
                    "reasons_gatheral": _reasons_to_str(int(g_reason[i, j])),
                    "reasons_dupire": _reasons_to_str(int(d_reason[i, j])),
                }
            )

    return pd.DataFrame(rows)


def _build_repricing_mountain_df(artifacts: Any) -> Any:
    """Long-form (T, K) repricing residuals, ready for a heatmap/mountain plot."""

    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    if "repricing_grid" not in artifacts.tables:
        return pd.DataFrame()

    grid = artifacts.tables["repricing_grid"].copy()
    if grid.empty:
        return grid

    fwd = None
    try:
        fwd = artifacts.synthetic.get("forward", None)
    except Exception:  # noqa: BLE001
        fwd = None

    if callable(fwd) and "T" in grid.columns:
        Ts = grid["T"].astype(float).to_numpy()
        F = np.array([float(fwd(float(t))) for t in Ts], dtype=float)
        grid["F"] = F
        if "K" in grid.columns:
            K = grid["K"].astype(float).to_numpy()
            grid["moneyness"] = K / F
            grid["y"] = np.log(K / F)

    # Signed errors are often more useful for heatmaps than absolute error.
    if "pde_price" in grid.columns and "target_price" in grid.columns:
        grid["price_error"] = grid["pde_price"].astype(float) - grid[
            "target_price"
        ].astype(float)

    if "pde_iv" in grid.columns and "target_iv" in grid.columns:
        grid["iv_error_bp"] = 1e4 * (
            grid["pde_iv"].astype(float) - grid["target_iv"].astype(float)
        )
        # Keep a consistent name for absolute error in bp.
        if "abs_iv_error_bp" not in grid.columns:
            grid["abs_iv_error_bp"] = np.abs(grid["iv_error_bp"].astype(float))

    return grid


def _build_svi_residual_mountain_df(artifacts: Any) -> Any:
    """Long-form (T, K) residuals for SVI calibration vs observed quotes."""

    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    quotes = artifacts.tables.get("quotes_df", None)
    if quotes is None or quotes.empty:
        return pd.DataFrame()

    surface = artifacts.surfaces.get("svi_repaired", None)
    if surface is None:
        return pd.DataFrame()

    df = quotes.copy()
    if not {"T", "K", "iv_obs"}.issubset(set(df.columns)):
        return pd.DataFrame()

    iv_svi = np.full((len(df),), np.nan, dtype=float)
    for T, idx in df.groupby("T").groups.items():
        ii = np.asarray(list(idx), dtype=int)
        Ks = df.loc[ii, "K"].astype(float).to_numpy()
        try:
            iv = np.asarray(surface.iv(Ks, float(T)), dtype=float).reshape(-1)
            if iv.shape == Ks.shape:
                iv_svi[ii] = iv
        except Exception:  # noqa: BLE001
            # leave NaNs
            pass

    df["iv_svi"] = iv_svi
    df["iv_resid_bp"] = 1e4 * (df["iv_svi"].astype(float) - df["iv_obs"].astype(float))
    df["abs_iv_resid_bp"] = np.abs(df["iv_resid_bp"].astype(float))

    # Useful for mapping the mountain in log-moneyness coordinates.
    if "F" in df.columns and "y" not in df.columns:
        F = df["F"].astype(float).to_numpy()
        K = df["K"].astype(float).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            df["y"] = np.log(K / F)

    return df


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end local-vol workflow and export a table to CSV."
    )

    parser.add_argument(
        "--poster",
        action="store_true",
        help=(
            "Export a poster-friendly bundle of CSVs (mountains, convergence, "
            "calibration struggles) into an output folder."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help=(
            "Output folder for --poster mode. If omitted, defaults to "
            "out/capstone2_poster_data/profile_<profile>_seed_<seed>."
        ),
    )

    parser.add_argument(
        "--out",
        type=str,
        default="out/localvol_repricing_grid.csv",
        help="Output CSV path (default: out/localvol_repricing_grid.csv).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="full",
        choices=("quick", "full"),
        help="Workflow profile (default: full).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for synthetic surface generation (default: 7).",
    )
    parser.add_argument(
        "--table",
        type=str,
        default="repricing_grid",
        help=(
            "Which capstone2 table to export (default: repricing_grid). "
            "Examples: repricing_summary, quotes_df, svi_fit_compare, "
            "localvol_worst_points."
        ),
    )
    parser.add_argument(
        "--overrides",
        type=str,
        default="",
        help=(
            "JSON overrides (string) or path to a JSON file. "
            "Use this to toggle RUN_* flags or tweak configs."
        ),
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default="",
        help=("Optional path to also write repricing_summary as CSV (if present)."),
    )
    parser.add_argument(
        "--meta-out",
        type=str,
        default="",
        help="Optional path to write meta/config as JSON.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    overrides: dict[str, Any] | None = None
    if args.overrides:
        overrides = _load_overrides(args.overrides)

    run_capstone2 = _get_run_capstone2()
    artifacts = run_capstone2(profile=args.profile, seed=args.seed, overrides=overrides)

    # Poster bundle export (folder of CSVs)
    if bool(args.poster):
        out_dir = (
            Path(args.out_dir)
            if str(args.out_dir).strip()
            else Path(
                f"out/capstone2_poster_data/profile_{args.profile}_seed_{args.seed}"
            )
        )

        # 1) SVI calibration residual "mountain" (observed quotes vs fitted SVI)
        svi_mtn = _build_svi_residual_mountain_df(artifacts)
        if not svi_mtn.empty:
            p = out_dir / "mountain" / "svi_calibration_residual_mountain.csv"
            _write_df(svi_mtn, p)
            print(f"Wrote SVI calibration residual mountain CSV to {p}")
        else:
            print(
                "[poster] quotes_df or svi_repaired missing; skipping SVI residual mountain"
            )

        # 2) Round-trip repricing residual "mountain" (uses repricing_grid)
        pde_mtn = _build_repricing_mountain_df(artifacts)
        if not pde_mtn.empty:
            p = out_dir / "mountain" / "pde_roundtrip_residual_mountain.csv"
            _write_df(pde_mtn, p)
            print(f"Wrote PDE round-trip residual mountain CSV to {p}")
        else:
            print(
                "[poster] repricing_grid missing/empty; skipping PDE residual mountain"
            )

        # 3) Gatheral vs Dupire differences (full grid, long-form)
        lv_cmp = artifacts.reports.get("lv_compare", None)
        if lv_cmp is not None:
            lv_cmp_df = _build_lv_compare_grid_df(lv_cmp)
            p = out_dir / "mountain" / "gatheral_vs_dupire_diff_grid.csv"
            _write_df(lv_cmp_df, p)
            print(f"Wrote Gatheral vs Dupire diff grid CSV to {p}")
        else:
            print("[poster] lv_compare not present; skipping Gatheral vs Dupire grid")

        # 4) PDE convergence sweep (single option)
        conv = artifacts.tables.get("convergence_grid", None)
        if conv is not None:
            p = out_dir / "convergence" / "localvol_pde_convergence.csv"
            _write_df(conv, p)
            print(f"Wrote PDE convergence sweep CSV to {p}")
        else:
            print("[poster] convergence_grid not present; skipping convergence CSV")

        # 5) Local-vol "calibration struggles"
        worst = artifacts.tables.get("localvol_worst_points", None)
        if worst is not None:
            p = out_dir / "calibration" / "localvol_worst_points.csv"
            _write_df(worst, p)
            print(f"Wrote local-vol worst points CSV to {p}")

        reasons = artifacts.tables.get("localvol_reason_counts", None)
        if reasons is not None:
            p = out_dir / "calibration" / "localvol_invalid_reason_counts.csv"
            _write_df(reasons, p)
            print(f"Wrote local-vol reason counts CSV to {p}")

        # 6) SVI calibration health (optional but often plotted alongside)
        svi_fit = artifacts.tables.get("svi_fit_compare", None)
        if svi_fit is not None:
            _write_df(svi_fit, out_dir / "calibration" / "svi_fit_compare.csv")

        svi_attempts = artifacts.tables.get("svi_repair_attempts", None)
        if svi_attempts is not None and not svi_attempts.empty:
            _write_df(
                svi_attempts,
                out_dir / "calibration" / "svi_repair_attempts.csv",
            )

        svi_fail = artifacts.tables.get("svi_repair_failure_summary", None)
        if svi_fail is not None and not svi_fail.empty:
            _write_df(
                svi_fail,
                out_dir / "calibration" / "svi_repair_failure_summary.csv",
            )

        # Also keep raw repricing summary around if present.
        rep_grid = artifacts.tables.get("repricing_grid", None)
        if rep_grid is not None:
            _write_df(rep_grid, out_dir / "repricing" / "repricing_grid.csv")

        rep_sum = artifacts.tables.get("repricing_summary", None)
        if rep_sum is not None:
            _write_df(rep_sum, out_dir / "repricing" / "repricing_summary.csv")

        # PDE baseline digital sweep (optional; used in the stability poster).
        for key in ("digital_errors", "digital_grouped", "digital_frontier"):
            df = artifacts.tables.get(key, None)
            if df is not None:
                _write_df(df, out_dir / "convergence" / f"{key}.csv")

        # Meta/config snapshot for reproducibility.
        meta = {
            "profile": str(args.profile),
            "seed": int(args.seed),
            "cfg": dict(getattr(artifacts, "cfg", {})),
            "flags": dict(getattr(artifacts, "flags", {})),
            "meta": dict(getattr(artifacts, "meta", {})),
        }
        (out_dir / "meta").mkdir(parents=True, exist_ok=True)
        (out_dir / "meta" / "run_meta.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )

        # A tiny index file so it's obvious what's what when you open the folder.
        index_txt = "Capstone 2 poster CSV bundle\n\n"
        index_txt += "mountain/\n"
        index_txt += "  - svi_calibration_residual_mountain.csv            (observed quotes vs fitted SVI; iv_resid_bp, y)\n"
        index_txt += "  - pde_roundtrip_residual_mountain.csv              (repricing residuals; iv_error_bp, y, price_error)\n"
        index_txt += "  - gatheral_vs_dupire_diff_grid.csv                 (T,K diff_sigma etc; long-form)\n\n"
        index_txt += "convergence/\n"
        index_txt += "  - localvol_pde_convergence.csv                     (Nx,Nt sweep for a single option)\n"
        index_txt += "  - digital_errors.csv / digital_grouped.csv / digital_frontier.csv (optional baseline)\n\n"
        index_txt += "calibration/\n"
        index_txt += "  - localvol_worst_points.csv                        (where Gatheral denom is small / invalid)\n"
        index_txt += "  - localvol_invalid_reason_counts.csv               (counts by LVInvalidReason flag)\n"
        index_txt += "  - svi_fit_compare.csv                              (SVI no-repair vs repaired fit quality)\n"
        index_txt += "  - svi_repair_attempts.csv / svi_repair_failure_summary.csv (optional repair diagnostics)\n\n"
        index_txt += "repricing/\n"
        index_txt += "  - repricing_grid.csv                               (raw repricing grid)\n"
        index_txt += "  - repricing_summary.csv                            (aggregate repricing errors)\n\n"
        index_txt += "meta/\n"
        index_txt += "  - run_meta.json                                    (cfg + flags snapshot)\n"
        (out_dir / "README.txt").write_text(index_txt, encoding="utf-8")

        print(f"\n[poster] Wrote poster bundle to: {out_dir}")
        return 0

    if args.table not in artifacts.tables:
        available = ", ".join(sorted(artifacts.tables.keys()))
        raise SystemExit(f"Unknown table '{args.table}'. Available tables: {available}")

    table = artifacts.tables[args.table]

    out_path = Path(args.out)
    _write_df(table, out_path)
    print(f"Wrote {len(table):,} rows to {out_path}")

    if args.summary_out and "repricing_summary" in artifacts.tables:
        summary = artifacts.tables["repricing_summary"]
        try:
            p2 = Path(args.summary_out)
            _write_df(summary, p2)
            print(f"Wrote repricing_summary to {p2}")
        except TypeError:
            pass

    if args.meta_out:
        meta = {
            "profile": str(args.profile),
            "seed": int(args.seed),
            "cfg": dict(getattr(artifacts, "cfg", {})),
            "flags": dict(getattr(artifacts, "flags", {})),
            "meta": dict(getattr(artifacts, "meta", {})),
            "exported_table": str(args.table),
        }
        p3 = Path(args.meta_out)
        p3.parent.mkdir(parents=True, exist_ok=True)
        p3.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        print(f"Wrote meta JSON to {p3}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
