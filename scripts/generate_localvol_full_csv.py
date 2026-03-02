"""Export CSVs for the local-vol end-to-end workflow.

This script runs the library's flagship Capstone 2 pipeline:

    surface quotes -> (SVI) implied surface -> local vol -> PDE repricing grid

and can export either:

1) a *single* diagnostics table to a CSV (the original behavior), or
2) a *poster bundle* of CSVs that directly support the "mountain" style plots
   shown in the capstone/poster figures.

Poster bundle outputs (when available)
-------------------------------------
**mountain/**
- gatheral_vs_dupire_diff_grid.csv
    Long-form (T, K) grid: Gatheral vs Dupire local vol + denominators + invalid
    masks + reason strings + diff_sigma
- pde_roundtrip_residual_mountain.csv
    Long-form repricing residuals (T, K) with price_error and iv_error_bp
- svi_calibration_residual_mountain.csv
    Long-form residuals of fitted SVI vs observed quote IVs (bp), plus y if possible

**convergence/**
- localvol_pde_convergence.csv
    Single-option convergence sweep (expected columns depend on capstone2 implementation)
- digital_errors.csv / digital_grouped.csv / digital_frontier.csv
    Optional baseline/stability tables if capstone provides them

**calibration/**
- localvol_worst_points.csv
    Worst LV points (where denominators are small/invalid), for “struggles” plots
- localvol_invalid_reason_counts.csv
    Counts by LVInvalidReason flag (bar plots)
- svi_fit_compare.csv, svi_repair_attempts.csv, svi_repair_failure_summary.csv
    Optional but useful SVI health diagnostics

**repricing/**
- repricing_grid.csv
- repricing_summary.csv

**surface/**  (NEW: for hero + consistent coordinate systems)
- quotes_df.csv
    Raw quotes table (useful for scatter overlays + debugging)
- forwards_by_T.csv
    Forward curve sampled at the capstone expiries (or inferred from repricing grid)
- iv_surface_grid.csv
    Regular (T, K) grid of implied vols from the fitted/repaired SVI surface,
    with F(T), y=log(K/F) when possible. Great for hero heatmaps/3D surfaces.

**meta/**
- run_meta.json
    cfg/flags/meta snapshot for reproducibility

Typical usage (from repo root):

    # Single table (default):
    python scripts/generate_localvol_full_csv.py --out out/repricing_grid.csv

    # Poster bundle (recommended for plotting):
    python scripts/generate_localvol_full_csv.py --poster

Notes
-----
- This script relies on pandas because the diagnostics tables are pandas DataFrames.
- If you haven't installed dev dependencies, use:
    python -m pip install -e '.[dev]'
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
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


def _build_forward_curve_df(artifacts: Any) -> Any:
    """Build a forwards_by_T table (T, F) from whatever the capstone provides."""
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    # 1) Best: lv_compare forwards at expiries
    lv_cmp = artifacts.reports.get("lv_compare", None)
    if lv_cmp is not None:
        Ts = np.asarray(getattr(lv_cmp, "expiries", []), dtype=float).reshape(-1)
        Fs = np.asarray(getattr(lv_cmp, "forwards", []), dtype=float).reshape(-1)
        if Ts.size > 0 and Fs.size == Ts.size:
            return pd.DataFrame({"T": Ts.astype(float), "F": Fs.astype(float)})

    # 2) Next: repricing grid already has F
    grid = artifacts.tables.get("repricing_grid", None)
    if grid is not None and not grid.empty and {"T", "F"}.issubset(set(grid.columns)):
        df = grid[["T", "F"]].copy()
        df["T"] = df["T"].astype(float)
        df["F"] = df["F"].astype(float)
        # Aggregate to a single forward per expiry (median is robust)
        df = df.groupby("T", as_index=False)["F"].median()
        return df.sort_values("T").reset_index(drop=True)

    # 3) Fallback: callable forward function + expiries from any table
    fwd = None
    try:
        fwd = artifacts.synthetic.get("forward", None)
    except Exception:  # noqa: BLE001
        fwd = None

    if callable(fwd):
        Ts: list[float] = []
        # try repricing grid Ts first, else quotes_df Ts
        q1 = artifacts.tables.get("repricing_grid", None)
        if q1 is not None and not q1.empty and "T" in q1.columns:
            Ts = sorted({float(t) for t in q1["T"].astype(float).to_list()})
        else:
            q2 = artifacts.tables.get("quotes_df", None)
            if q2 is not None and not q2.empty and "T" in q2.columns:
                Ts = sorted({float(t) for t in q2["T"].astype(float).to_list()})

        if Ts:
            F = np.array([float(fwd(float(t))) for t in Ts], dtype=float)
            return pd.DataFrame({"T": np.asarray(Ts, dtype=float), "F": F})

    return pd.DataFrame()


def _make_forward_lookup(artifacts: Any) -> Callable[[float], float] | None:
    """Return a forward lookup f(T) via interpolation on forwards_by_T if possible."""
    import numpy as np  # noqa: PLC0415

    fwd = None
    try:
        fwd = artifacts.synthetic.get("forward", None)
    except Exception:  # noqa: BLE001
        fwd = None
    if callable(fwd):
        return lambda t: float(fwd(float(t)))

    fwd_df = _build_forward_curve_df(artifacts)
    if fwd_df is None or fwd_df.empty or not {"T", "F"}.issubset(set(fwd_df.columns)):
        return None

    Ts = fwd_df["T"].astype(float).to_numpy()
    Fs = fwd_df["F"].astype(float).to_numpy()
    if Ts.size == 0:
        return None

    # Ensure sorted for interpolation
    order = np.argsort(Ts)
    Ts = Ts[order]
    Fs = Fs[order]

    def _interp(t: float) -> float:
        tt = float(t)
        if tt <= float(Ts[0]):
            return float(Fs[0])
        if tt >= float(Ts[-1]):
            return float(Fs[-1])
        return float(np.interp(tt, Ts, Fs))

    return _interp


def _add_forward_and_moneyness_cols(df: Any, artifacts: Any) -> Any:
    """Ensure df has F, moneyness, y if possible (non-destructive)."""
    import numpy as np  # noqa: PLC0415

    if df is None or df.empty:
        return df

    out = df.copy()

    if "T" not in out.columns or "K" not in out.columns:
        return out

    # 1) If F already present, use it
    if "F" in out.columns:
        try:
            F = out["F"].astype(float).to_numpy()
            K = out["K"].astype(float).to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                if "moneyness" not in out.columns:
                    out["moneyness"] = K / F
                if "y" not in out.columns:
                    out["y"] = np.log(K / F)
            return out
        except Exception:  # noqa: BLE001
            pass

    # 2) Try to compute F from a forward lookup (callable or interpolated)
    f = _make_forward_lookup(artifacts)
    if f is None:
        return out

    Ts = out["T"].astype(float).to_numpy()
    F = np.array([float(f(float(t))) for t in Ts], dtype=float)
    out["F"] = F

    K = out["K"].astype(float).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["moneyness"] = K / F
        out["y"] = np.log(K / F)

    return out


def _build_repricing_mountain_df(artifacts: Any) -> Any:
    """Long-form (T, K) repricing residuals, ready for a heatmap/mountain plot."""
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    if "repricing_grid" not in artifacts.tables:
        return pd.DataFrame()

    grid = artifacts.tables["repricing_grid"].copy()
    if grid.empty:
        return grid

    # Ensure F/moneyness/y if possible (robust to missing artifacts.synthetic['forward'])
    grid = _add_forward_and_moneyness_cols(grid, artifacts)

    # Signed errors are often more useful for heatmaps than absolute error.
    if "pde_price" in grid.columns and "target_price" in grid.columns:
        grid["price_error"] = grid["pde_price"].astype(float) - grid[
            "target_price"
        ].astype(float)

    if "pde_iv" in grid.columns and "target_iv" in grid.columns:
        grid["iv_error_bp"] = 1e4 * (
            grid["pde_iv"].astype(float) - grid["target_iv"].astype(float)
        )
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

    # Prefer repaired surface, fallback to any plausible SVI-like surface key.
    surface = artifacts.surfaces.get("svi_repaired", None)
    if surface is None:
        for key in ("svi", "svi_surface", "svi_raw", "implied_svi"):
            surface = artifacts.surfaces.get(key, None)
            if surface is not None:
                break
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
            pass

    df["iv_svi"] = iv_svi
    df["iv_resid_bp"] = 1e4 * (df["iv_svi"].astype(float) - df["iv_obs"].astype(float))
    df["abs_iv_resid_bp"] = np.abs(df["iv_resid_bp"].astype(float))

    # Ensure forward/moneyness/y if possible (non-destructive)
    df = _add_forward_and_moneyness_cols(df, artifacts)

    return df


def _build_iv_surface_grid_df(artifacts: Any) -> Any:
    """Regular (T, K) grid of implied vols from fitted/repaired SVI, for hero plots."""
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    # Find a surface
    surface = artifacts.surfaces.get("svi_repaired", None)
    if surface is None:
        for key in ("svi", "svi_surface", "svi_raw", "implied_svi"):
            surface = artifacts.surfaces.get(key, None)
            if surface is not None:
                break
    if surface is None:
        return pd.DataFrame()

    Ts: list[float] = []
    Ks: list[float] = []

    # Prefer lv_compare grid axes if available (gives a nice rectangular mesh)
    lv_cmp = artifacts.reports.get("lv_compare", None)
    if lv_cmp is not None:
        try:
            Ts = [float(t) for t in list(lv_cmp.expiries)]
            Ks = [float(k) for k in list(lv_cmp.strikes)]
        except Exception:  # noqa: BLE001
            Ts, Ks = [], []

    # Else use repricing grid unique axes, else quotes unique axes
    if not Ts or not Ks:
        g = artifacts.tables.get("repricing_grid", None)
        if g is not None and not g.empty and {"T", "K"}.issubset(set(g.columns)):
            Ts = sorted({float(t) for t in g["T"].astype(float).to_list()})
            Ks = sorted({float(k) for k in g["K"].astype(float).to_list()})

    if not Ts or not Ks:
        q = artifacts.tables.get("quotes_df", None)
        if q is not None and not q.empty and {"T", "K"}.issubset(set(q.columns)):
            Ts = sorted({float(t) for t in q["T"].astype(float).to_list()})
            Ks = sorted({float(k) for k in q["K"].astype(float).to_list()})

    if not Ts or not Ks:
        return pd.DataFrame()

    f = _make_forward_lookup(artifacts)

    rows: list[dict[str, Any]] = []
    Ks_arr = np.asarray(Ks, dtype=float)
    for T in Ts:
        try:
            iv = np.asarray(surface.iv(Ks_arr, float(T)), dtype=float).reshape(-1)
        except Exception:  # noqa: BLE001
            iv = np.full((len(Ks_arr),), np.nan, dtype=float)

        F = float("nan")
        if f is not None:
            try:
                F = float(f(float(T)))
            except Exception:  # noqa: BLE001
                F = float("nan")

        for K, v in zip(Ks_arr, iv, strict=False):
            y = float("nan")
            m = float("nan")
            if np.isfinite(F) and F > 0:
                m = float(K / F)
                with np.errstate(divide="ignore", invalid="ignore"):
                    y = float(np.log(K / F))
            rows.append(
                {
                    "T": float(T),
                    "K": float(K),
                    "F": float(F),
                    "moneyness": float(m),
                    "y": float(y),
                    "iv_svi": float(v) if np.isfinite(v) else float("nan"),
                }
            )

    return pd.DataFrame(rows)


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

        # 0) Surface helpers (NEW)
        quotes = artifacts.tables.get("quotes_df", None)
        if quotes is not None:
            _write_df(quotes, out_dir / "surface" / "quotes_df.csv")

        fwd_df = _build_forward_curve_df(artifacts)
        if fwd_df is not None and not fwd_df.empty:
            _write_df(fwd_df, out_dir / "surface" / "forwards_by_T.csv")

        iv_grid = _build_iv_surface_grid_df(artifacts)
        if iv_grid is not None and not iv_grid.empty:
            _write_df(iv_grid, out_dir / "surface" / "iv_surface_grid.csv")
        else:
            print("[poster] SVI surface missing; skipping iv_surface_grid.csv")

        # 1) SVI calibration residual "mountain" (observed quotes vs fitted SVI)
        svi_mtn = _build_svi_residual_mountain_df(artifacts)
        if not svi_mtn.empty:
            p = out_dir / "mountain" / "svi_calibration_residual_mountain.csv"
            _write_df(svi_mtn, p)
            print(f"Wrote SVI calibration residual mountain CSV to {p}")
        else:
            print(
                "[poster] quotes_df or SVI surface missing; skipping SVI residual mountain"
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
            _write_df(svi_attempts, out_dir / "calibration" / "svi_repair_attempts.csv")

        svi_fail = artifacts.tables.get("svi_repair_failure_summary", None)
        if svi_fail is not None and not svi_fail.empty:
            _write_df(
                svi_fail, out_dir / "calibration" / "svi_repair_failure_summary.csv"
            )

        # Raw repricing tables
        rep_grid = artifacts.tables.get("repricing_grid", None)
        if rep_grid is not None:
            _write_df(rep_grid, out_dir / "repricing" / "repricing_grid.csv")

        rep_sum = artifacts.tables.get("repricing_summary", None)
        if rep_sum is not None:
            _write_df(rep_sum, out_dir / "repricing" / "repricing_summary.csv")

        # PDE baseline digital sweep (optional; used in stability posters).
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
        index_txt += "surface/\n"
        index_txt += "  - quotes_df.csv                                    (raw quotes, scatter overlays)\n"
        index_txt += "  - forwards_by_T.csv                                 (forward curve sampled at expiries)\n"
        index_txt += "  - iv_surface_grid.csv                               (SVI implied vol on a regular (T,K) grid)\n\n"
        index_txt += "mountain/\n"
        index_txt += (
            "  - svi_calibration_residual_mountain.csv            (iv_resid_bp, y)\n"
        )
        index_txt += "  - pde_roundtrip_residual_mountain.csv              (iv_error_bp, y, price_error)\n"
        index_txt += "  - gatheral_vs_dupire_diff_grid.csv                 (diff_sigma, denom, invalid, reasons; long-form)\n\n"
        index_txt += "convergence/\n"
        index_txt += "  - localvol_pde_convergence.csv                     (Nx/Nt sweep for a single option)\n"
        index_txt += "  - digital_errors.csv / digital_grouped.csv / digital_frontier.csv (optional baseline)\n\n"
        index_txt += "calibration/\n"
        index_txt += (
            "  - localvol_worst_points.csv                        (worst LV points)\n"
        )
        index_txt += "  - localvol_invalid_reason_counts.csv               (counts by LVInvalidReason)\n"
        index_txt += (
            "  - svi_fit_compare.csv                              (SVI fit quality)\n"
        )
        index_txt += "  - svi_repair_attempts.csv / svi_repair_failure_summary.csv (optional repair diagnostics)\n\n"
        index_txt += "repricing/\n"
        index_txt += "  - repricing_grid.csv                               (raw repricing grid)\n"
        index_txt += "  - repricing_summary.csv                            (aggregate repricing errors)\n\n"
        index_txt += "meta/\n"
        index_txt += "  - run_meta.json                                    (cfg + flags snapshot)\n"
        (out_dir / "README.txt").write_text(index_txt, encoding="utf-8")

        print(f"\n[poster] Wrote poster bundle to: {out_dir}")
        return 0

    # Single-table export mode (original behavior)
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
