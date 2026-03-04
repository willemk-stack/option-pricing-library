from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------
# Utilities
# -----------------------------


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _ensure_cols(df: pd.DataFrame, cols: list[str], *, name: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[warn] {name}: missing columns {missing} (have={list(df.columns)})")
        return False
    return True


def _finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def _unique_sorted(vals: pd.Series, *, round_decimals: int = 12) -> np.ndarray:
    a = pd.to_numeric(vals, errors="coerce").astype(float)
    a = a[np.isfinite(a)]
    if round_decimals is not None:
        a = a.round(round_decimals)
    return np.array(sorted(set(a.to_list())), dtype=float)


def _pivot_grid(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    value: str,
    round_decimals: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_vals, y_vals, Z[y_i, x_j]) for a rectangular grid."""
    x_vals = _unique_sorted(df[x], round_decimals=round_decimals)
    y_vals = _unique_sorted(df[y], round_decimals=round_decimals)

    d = df.copy()
    d[x] = pd.to_numeric(d[x], errors="coerce").astype(float).round(round_decimals)
    d[y] = pd.to_numeric(d[y], errors="coerce").astype(float).round(round_decimals)
    d[value] = pd.to_numeric(d[value], errors="coerce").astype(float)

    table = d.pivot(index=y, columns=x, values=value).reindex(
        index=y_vals, columns=x_vals
    )
    Z = table.to_numpy(dtype=float)
    return x_vals, y_vals, Z


def _write_png(fig: Any, out: Path, *, dpi: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"[ok] wrote {out}")


# -----------------------------
# IV surface helpers
# -----------------------------

IVXMode = Literal["moneyness", "y", "K"]


def _iv_surface_grid(
    iv_grid: pd.DataFrame,
    *,
    x_mode: IVXMode,
    nK: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Build a *rectangular* IV surface grid.

    The poster bundle's iv_surface_grid.csv is uniform in y/moneyness.
    Pivoting on strike K produces a sparse (T,K) matrix (diagonal stripes).

    Returns (x_vals, T_vals, IV[T_i, x_j], x_label).
    """
    if not _ensure_cols(iv_grid, ["T", "iv_svi"], name="iv_surface_grid"):
        return np.array([]), np.array([]), np.empty((0, 0)), ""

    d = iv_grid.copy()
    d["T"] = pd.to_numeric(d["T"], errors="coerce").astype(float)
    d["iv_svi"] = pd.to_numeric(d["iv_svi"], errors="coerce").astype(float)

    # Prefer true grids when possible.
    if x_mode == "moneyness" and "moneyness" in d.columns:
        x, T, Z = _pivot_grid(d, x="moneyness", y="T", value="iv_svi")
        if np.isfinite(Z).all():
            return x, T, Z, "Moneyness K/F"

    if x_mode == "y" and "y" in d.columns:
        x, T, Z = _pivot_grid(d, x="y", y="T", value="iv_svi")
        if np.isfinite(Z).all():
            return x, T, Z, "Log-moneyness y = log(K/F)"

    # Strike-based grid: regrid each T-slice onto a common strike axis.
    if not _ensure_cols(d, ["K"], name="iv_surface_grid"):
        return np.array([]), np.array([]), np.empty((0, 0)), ""

    d["K"] = pd.to_numeric(d["K"], errors="coerce").astype(float)
    T_vals = _unique_sorted(d["T"])
    if T_vals.size == 0:
        return np.array([]), np.array([]), np.empty((0, 0)), ""

    K_min = float(np.nanmin(d["K"].to_numpy(dtype=float)))
    K_max = float(np.nanmax(d["K"].to_numpy(dtype=float)))
    if not (np.isfinite(K_min) and np.isfinite(K_max) and K_max > K_min):
        return np.array([]), np.array([]), np.empty((0, 0)), ""

    K_common = np.linspace(K_min, K_max, int(nK), dtype=float)
    IV = np.full((T_vals.size, K_common.size), np.nan, dtype=float)

    for i, T in enumerate(T_vals):
        g = d.loc[np.isclose(d["T"], T)].copy()
        g = g[np.isfinite(g["K"]) & np.isfinite(g["iv_svi"])].sort_values("K")
        if len(g) < 2:
            continue
        k = g["K"].to_numpy(dtype=float)
        v = g["iv_svi"].to_numpy(dtype=float)
        IV[i, :] = np.interp(K_common, k, v)

    return K_common, T_vals, IV, "Strike K"


def _choose_smile_maturities(T_vals: np.ndarray, max_curves: int = 6) -> np.ndarray:
    """Pick <=max_curves maturities that *exist* in the grid (evenly spaced)."""
    T_vals = np.array(
        sorted({float(t) for t in T_vals if np.isfinite(float(t))}), dtype=float
    )
    if T_vals.size <= max_curves:
        return T_vals

    idx = np.linspace(0, T_vals.size - 1, int(max_curves)).round().astype(int)
    picked = T_vals[idx]
    picked = np.unique(np.concatenate([[T_vals[0]], picked, [T_vals[-1]]]))
    if picked.size > max_curves:
        idx2 = np.linspace(0, picked.size - 1, int(max_curves)).round().astype(int)
        picked = picked[idx2]
    return picked


# -----------------------------
# Convergence helpers
# -----------------------------


def _self_convergence(conv: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (grid_points, |pde_price - pde_price(finest)|) excluding the finest point itself."""
    if conv.empty or not _ensure_cols(
        conv, ["grid_points", "pde_price"], name="convergence_grid"
    ):
        return np.array([]), np.array([])

    d = conv.copy()
    d["grid_points"] = pd.to_numeric(d["grid_points"], errors="coerce").astype(float)
    d["pde_price"] = pd.to_numeric(d["pde_price"], errors="coerce").astype(float)
    d = d[np.isfinite(d["grid_points"]) & np.isfinite(d["pde_price"])].sort_values(
        "grid_points"
    )
    if len(d) < 2:
        return np.array([]), np.array([])

    ref = float(d["pde_price"].iloc[-1])
    d2 = d.iloc[:-1].copy()  # drop finest -> avoids a zero point on log scale
    err = np.abs(d2["pde_price"] - ref).to_numpy(dtype=float)
    gp = d2["grid_points"].to_numpy(dtype=float)
    ok = np.isfinite(gp) & np.isfinite(err) & (err > 0)
    return gp[ok], err[ok]


# -----------------------------
# Plotters (matplotlib imported lazily)
# -----------------------------


def plot_hero_iv_heatmap(
    iv_grid: pd.DataFrame, out: Path, *, dpi: int, banner: bool = True
) -> None:
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    x, T, IV, xlab = _iv_surface_grid(iv_grid, x_mode="moneyness")
    if IV.size == 0:
        return

    figsize = (14, 3.6) if banner else (8, 5)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        IV,
        origin="lower",
        aspect="auto",
        extent=[float(x.min()), float(x.max()), float(T.min()), float(T.max())],
        interpolation="nearest",
    )
    ax.set_title("Implied Vol Surface", pad=8)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Maturity T (years)")

    fig.tight_layout()
    _write_png(fig, out, dpi=dpi)
    plt.close(fig)


def plot_hero_iv_3d(iv_grid: pd.DataFrame, out: Path, *, dpi: int) -> None:
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    x, T, IV, xlab = _iv_surface_grid(iv_grid, x_mode="moneyness")
    if IV.size == 0:
        return

    XX, TT = np.meshgrid(x, T)
    fig = plt.figure(figsize=(14, 4.2))
    ax = fig.add_subplot(111, projection="3d")

    IVm = np.ma.masked_invalid(IV)
    ax.plot_surface(XX, TT, IVm, linewidth=0, antialiased=True)

    ax.set_title("Implied Vol Surface (3D)", pad=10)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Maturity T")
    ax.set_zlabel("IV")

    ax.view_init(elev=25, azim=-60)
    fig.tight_layout()
    _write_png(fig, out, dpi=dpi)
    plt.close(fig)


def plot_iv_smiles(iv_grid: pd.DataFrame, out: Path, *, dpi: int) -> None:
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not _ensure_cols(iv_grid, ["T", "iv_svi"], name="iv_surface_grid"):
        return

    d = iv_grid.copy()
    d["T"] = pd.to_numeric(d["T"], errors="coerce").astype(float)
    d["iv_svi"] = pd.to_numeric(d["iv_svi"], errors="coerce").astype(float)

    x_col = "moneyness" if "moneyness" in d.columns else "K"
    if x_col not in d.columns:
        print("[warn] iv_surface_grid: no moneyness/K column for smiles")
        return

    d[x_col] = pd.to_numeric(d[x_col], errors="coerce").astype(float)

    Ts_sel = _choose_smile_maturities(_unique_sorted(d["T"]), max_curves=6)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for T in Ts_sel:
        g = d.loc[np.isclose(d["T"], T)].copy()
        g = g[np.isfinite(g[x_col]) & np.isfinite(g["iv_svi"])].sort_values(x_col)
        if g.empty:
            continue
        ax.plot(g[x_col].to_numpy(), g["iv_svi"].to_numpy(), label=f"T={float(T):g}")

    ax.set_title("Implied Vol Smiles (selected maturities)")
    ax.set_xlabel("Moneyness K/F" if x_col == "moneyness" else "Strike K")
    ax.set_ylabel("Implied vol")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    _write_png(fig, out, dpi=dpi)
    plt.close(fig)


def plot_pde_vs_bs_scatter(repricing: pd.DataFrame, out: Path, *, dpi: int) -> None:
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not _ensure_cols(
        repricing, ["target_price", "pde_price"], name="repricing_grid"
    ):
        return

    x, y = _finite_xy(
        repricing["target_price"].to_numpy(), repricing["pde_price"].to_numpy()
    )
    if x.size == 0:
        print("[warn] repricing_grid: no finite points for scatter")
        return

    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    denom = float(np.sum((x - float(np.mean(x))) ** 2))
    r2 = float(1.0 - np.sum((y - x) ** 2) / denom) if denom > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(x, y, s=10, alpha=0.8)

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi])

    title = f"PDE vs Black-76 (RMSE={rmse:.3g}" + (
        f", R²={r2:.3f})" if np.isfinite(r2) else ")"
    )
    ax.set_title(title)
    ax.set_xlabel("Black-76 target price")
    ax.set_ylabel("Local-vol PDE price")
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    _write_png(fig, out, dpi=dpi)
    plt.close(fig)


def plot_pde_convergence(conv: pd.DataFrame, out: Path, *, dpi: int) -> None:
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # Many people expect the error to a fixed "target" to go to zero as the grid is refined.
    # In this project the exported CSV includes both:
    #   - abs_error  : |pde_price - target_price|
    #   - discretization (grid) error, best seen as self-convergence vs the finest grid.
    # Plot both so the curve doesn't look "weird" if the target isn't the PDE limit.

    if conv.empty or not _ensure_cols(
        conv, ["grid_points", "pde_price"], name="convergence_grid"
    ):
        print("[warn] convergence_grid missing/empty; skipping convergence plot")
        return

    d = conv.copy()
    d["grid_points"] = pd.to_numeric(d["grid_points"], errors="coerce").astype(float)
    d["pde_price"] = pd.to_numeric(d["pde_price"], errors="coerce").astype(float)
    if "target_price" in d.columns:
        d["target_price"] = pd.to_numeric(d["target_price"], errors="coerce").astype(
            float
        )

    d = d[np.isfinite(d["grid_points"]) & np.isfinite(d["pde_price"])].sort_values(
        "grid_points"
    )
    if len(d) < 2:
        print("[warn] convergence_grid: too few points")
        return

    # Self-convergence (exclude finest -> avoids a zero point on log scale)
    gp_sc, err_sc = _self_convergence(d)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    if gp_sc.size:
        idx = np.argsort(gp_sc)
        ax.loglog(gp_sc[idx], err_sc[idx], marker="o", label="|p - p(finest)|")

    # Error to target (if present)
    if "target_price" in d.columns and np.isfinite(d["target_price"]).any():
        tgt = float(d["target_price"].dropna().iloc[0])
        err_t = np.abs(d["pde_price"].to_numpy(dtype=float) - tgt)
        gp_t = d["grid_points"].to_numpy(dtype=float)
        ok = np.isfinite(gp_t) & np.isfinite(err_t) & (err_t > 0)
        if ok.any():
            ax.loglog(gp_t[ok], err_t[ok], marker="o", label="|p - target|")

    ax.set_title("PDE Convergence")
    ax.set_xlabel("Grid points (Nx × Nt)")
    ax.set_ylabel("Absolute error")
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    _write_png(fig, out, dpi=dpi)
    plt.close(fig)


def plot_localvol_heatmap_and_mask(
    lv_cmp: pd.DataFrame,
    out_lv: Path,
    out_mask: Path,
    *,
    dpi: int,
) -> None:
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if lv_cmp.empty:
        print("[warn] lv_compare grid missing/empty; skipping local-vol plots")
        return

    if not _ensure_cols(
        lv_cmp, ["T", "K", "sigma_gatheral"], name="gatheral_vs_dupire_diff_grid"
    ):
        return

    K, T, SIG = _pivot_grid(lv_cmp, x="K", y="T", value="sigma_gatheral")

    # Local vol heatmap
    fig1, ax1 = plt.subplots(figsize=(8, 5.2))
    im1 = ax1.imshow(
        SIG,
        origin="lower",
        aspect="auto",
        extent=[float(K.min()), float(K.max()), float(T.min()), float(T.max())],
        interpolation="nearest",
    )
    ax1.set_title("Local Vol Surface (Gatheral)")
    ax1.set_xlabel("Strike K")
    ax1.set_ylabel("Maturity T (years)")
    fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Local vol")
    fig1.tight_layout()
    _write_png(fig1, out_lv, dpi=dpi)
    plt.close(fig1)

    # Invalid region mask: prefer Gatheral if it ever triggers; else show union.
    mask_col = ""
    title = ""
    if "invalid_gatheral" in lv_cmp.columns and bool(
        pd.to_numeric(lv_cmp["invalid_gatheral"], errors="coerce")
        .fillna(0)
        .astype(int)
        .any()
    ):
        mask_col = "invalid_gatheral"
        title = "Local Vol Diagnostics: Invalid Mask (Gatheral)"
    elif "invalid_union" in lv_cmp.columns:
        mask_col = "invalid_union"
        title = "Local Vol Diagnostics: Invalid Mask (Union)"
    elif "invalid_dupire" in lv_cmp.columns:
        mask_col = "invalid_dupire"
        title = "Local Vol Diagnostics: Invalid Mask (Dupire)"

    if not mask_col:
        print(
            "[warn] lv_compare grid has no invalid mask column; skipping invalid mask plot"
        )
        return

    _, _, MASK = _pivot_grid(lv_cmp, x="K", y="T", value=mask_col)
    MASK = np.asarray(MASK, dtype=float)

    fig2, ax2 = plt.subplots(figsize=(8, 5.2))
    im2 = ax2.imshow(
        MASK,
        origin="lower",
        aspect="auto",
        extent=[float(K.min()), float(K.max()), float(T.min()), float(T.max())],
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax2.set_title(title)
    ax2.set_xlabel("Strike K")
    ax2.set_ylabel("Maturity T (years)")
    fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Invalid (1=yes)")
    fig2.tight_layout()
    _write_png(fig2, out_mask, dpi=dpi)
    plt.close(fig2)


def plot_pricing_error_heatmap(repricing: pd.DataFrame, out: Path, *, dpi: int) -> None:
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if repricing.empty:
        print("[warn] repricing_grid missing/empty; skipping pricing error heatmap")
        return

    if not _ensure_cols(
        repricing, ["T", "K", "pde_price", "target_price"], name="repricing_grid"
    ):
        return

    tmp = repricing.copy()
    tmp["price_error"] = pd.to_numeric(tmp["pde_price"], errors="coerce").astype(
        float
    ) - pd.to_numeric(tmp["target_price"], errors="coerce").astype(float)

    K, T, ERR = _pivot_grid(tmp, x="K", y="T", value="price_error")
    vmax = float(np.nanmax(np.abs(ERR))) if np.isfinite(ERR).any() else 1.0

    fig, ax = plt.subplots(figsize=(8, 5.2))
    im = ax.imshow(
        ERR,
        origin="lower",
        aspect="auto",
        extent=[float(K.min()), float(K.max()), float(T.min()), float(T.max())],
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title("Pricing Error Heatmap (PDE - Black-76)")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Maturity T (years)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Price error")

    fig.tight_layout()
    _write_png(fig, out, dpi=dpi)
    plt.close(fig)


def plot_pricing_error_surface_3d(repricing: pd.DataFrame, ax: Any) -> None:
    """3D 'mountain' surface of (PDE - target) on (K,T). Draws into an existing Axes3D."""
    if repricing.empty or not _ensure_cols(
        repricing, ["T", "K", "pde_price", "target_price"], name="repricing_grid"
    ):
        ax.set_title("PDE - Target")
        return

    tmp = repricing.copy()
    tmp["T"] = pd.to_numeric(tmp["T"], errors="coerce").astype(float)
    tmp["K"] = pd.to_numeric(tmp["K"], errors="coerce").astype(float)
    tmp["pde_price"] = pd.to_numeric(tmp["pde_price"], errors="coerce").astype(float)
    tmp["target_price"] = pd.to_numeric(tmp["target_price"], errors="coerce").astype(
        float
    )
    tmp = tmp[
        np.isfinite(tmp["T"])
        & np.isfinite(tmp["K"])
        & np.isfinite(tmp["pde_price"])
        & np.isfinite(tmp["target_price"])
    ].copy()
    if tmp.empty:
        ax.set_title("PDE - Target")
        return

    tmp["price_error"] = tmp["pde_price"] - tmp["target_price"]

    K, T, ERR = _pivot_grid(tmp, x="K", y="T", value="price_error")
    KK, TT = np.meshgrid(K, T)

    ERRm = np.ma.masked_invalid(ERR)
    ax.plot_surface(KK, TT, ERRm, linewidth=0, antialiased=True)

    ax.set_title("PDE - Target")
    ax.set_xlabel("K")
    ax.set_ylabel("T")
    ax.set_zlabel("Δ price")
    ax.view_init(elev=28, azim=-55)


def plot_readme_gallery_strip(
    iv_grid: pd.DataFrame, repricing: pd.DataFrame, out: Path, *, dpi: int
) -> None:
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))

    # Tile 1: IV heatmap (moneyness)
    x, T, IV, _ = _iv_surface_grid(iv_grid, x_mode="moneyness")
    if IV.size:
        axs[0].imshow(
            IV,
            origin="lower",
            aspect="auto",
            extent=[float(x.min()), float(x.max()), float(T.min()), float(T.max())],
            interpolation="nearest",
        )
        axs[0].set_title("IV Surface")
        axs[0].set_xlabel("K/F")
        axs[0].set_ylabel("T")

    # Tile 2: smiles (selected maturities)
    if _ensure_cols(iv_grid, ["T", "iv_svi"], name="iv_surface_grid"):
        d = iv_grid.copy()
        d["T"] = pd.to_numeric(d["T"], errors="coerce").astype(float)
        d["iv_svi"] = pd.to_numeric(d["iv_svi"], errors="coerce").astype(float)
        x_col = "moneyness" if "moneyness" in d.columns else "K"
        if x_col in d.columns:
            d[x_col] = pd.to_numeric(d[x_col], errors="coerce").astype(float)
            Ts_sel = _choose_smile_maturities(_unique_sorted(d["T"]), max_curves=4)
            for Tval in Ts_sel:
                g = d.loc[np.isclose(d["T"], Tval)].copy()
                g = g[np.isfinite(g[x_col]) & np.isfinite(g["iv_svi"])].sort_values(
                    x_col
                )
                if not g.empty:
                    axs[1].plot(g[x_col].to_numpy(), g["iv_svi"].to_numpy())
            axs[1].set_title("Smile Slices")
            axs[1].set_xlabel("K/F" if x_col == "moneyness" else "K")
            axs[1].set_ylabel("IV")

    # Tile 3: validation scatter
    if _ensure_cols(repricing, ["target_price", "pde_price"], name="repricing_grid"):
        x_sc, y_sc = _finite_xy(
            repricing["target_price"].to_numpy(), repricing["pde_price"].to_numpy()
        )
        if x_sc.size:
            axs[2].scatter(x_sc, y_sc, s=8, alpha=0.8)
            lo = float(min(x_sc.min(), y_sc.min()))
            hi = float(max(x_sc.max(), y_sc.max()))
            axs[2].plot([lo, hi], [lo, hi])
        axs[2].set_title("PDE vs Black-76")
        axs[2].set_xlabel("Target")
        axs[2].set_ylabel("PDE")

    fig.tight_layout()
    _write_png(fig, out, dpi=dpi)
    plt.close(fig)


def plot_capstone_posters(
    iv_grid: pd.DataFrame,
    lv_cmp: pd.DataFrame,
    repricing: pd.DataFrame,
    conv: pd.DataFrame,
    out_poster: Path,
    out_proofblock: Path,
    *,
    dpi: int,
    poster_layout: Literal["grid", "wide"] = "grid",
) -> None:
    """Render the README poster + proof block.

    Changes vs prior version:
      - Uses a 2x2 poster layout by default (less wide).
      - Replaces the 'PDE-Target' heatmap with a 3D 'mountain' surface.
    """
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # -----------------
    # Poster
    # -----------------
    if poster_layout == "wide":
        # Legacy 1x4, but still switch the error tile to 3D.
        fig, axs = plt.subplots(1, 4, figsize=(14, 4.6), constrained_layout=True)

        # IV surface (strike grid, regridded)
        x, T, IV, _ = _iv_surface_grid(iv_grid, x_mode="K")
        if IV.size:
            axs[0].imshow(
                IV,
                origin="lower",
                aspect="auto",
                extent=[float(x.min()), float(x.max()), float(T.min()), float(T.max())],
                interpolation="nearest",
            )
            axs[0].set_title("IV Surface")
            axs[0].set_xlabel("K")
            axs[0].set_ylabel("T")

        # Local vol
        if not lv_cmp.empty and _ensure_cols(
            lv_cmp, ["T", "K", "sigma_gatheral"], name="lv_compare"
        ):
            K2, T2, SIG = _pivot_grid(lv_cmp, x="K", y="T", value="sigma_gatheral")
            axs[1].imshow(
                SIG,
                origin="lower",
                aspect="auto",
                extent=[
                    float(K2.min()),
                    float(K2.max()),
                    float(T2.min()),
                    float(T2.max()),
                ],
                interpolation="nearest",
            )
            axs[1].set_title("Local Vol")
            axs[1].set_xlabel("K")
            axs[1].set_ylabel("T")

        # Replace axis 2 with a 3D axis for mountain
        axs[2].remove()
        ax3d = fig.add_subplot(1, 4, 3, projection="3d")
        plot_pricing_error_surface_3d(repricing, ax3d)

        # Validation
        if _ensure_cols(
            repricing, ["target_price", "pde_price"], name="repricing_grid"
        ):
            x_sc, y_sc = _finite_xy(
                repricing["target_price"].to_numpy(), repricing["pde_price"].to_numpy()
            )
            if x_sc.size:
                axs[3].scatter(x_sc, y_sc, s=8, alpha=0.8)
                lo = float(min(x_sc.min(), y_sc.min()))
                hi = float(max(x_sc.max(), y_sc.max()))
                axs[3].plot([lo, hi], [lo, hi])
            axs[3].set_title("Validation")
            axs[3].set_xlabel("Target")
            axs[3].set_ylabel("PDE")

        # No figure-level title: avoids overlapping subplot titles in constrained layouts.
        _write_png(fig, out_poster, dpi=dpi)
        plt.close(fig)

    else:
        # Recommended 2x2 grid (less wide + fewer heatmaps adjacent)
        fig = plt.figure(figsize=(12.0, 8.2), constrained_layout=True)
        # Reserve headroom for a figure-level title.
        gs = fig.add_gridspec(2, 2, top=0.86)

        # (0,0) IV surface (strike grid, regridded; keeps axes consistent with the other panels)
        ax00 = fig.add_subplot(gs[0, 0])
        x, T, IV, _ = _iv_surface_grid(iv_grid, x_mode="K")
        if IV.size:
            ax00.imshow(
                IV,
                origin="lower",
                aspect="auto",
                extent=[float(x.min()), float(x.max()), float(T.min()), float(T.max())],
                interpolation="nearest",
            )
        ax00.set_title("IV Surface")
        ax00.set_xlabel("K")
        ax00.set_ylabel("T")

        # (0,1) Local vol heatmap
        ax01 = fig.add_subplot(gs[0, 1])
        if not lv_cmp.empty and _ensure_cols(
            lv_cmp, ["T", "K", "sigma_gatheral"], name="lv_compare"
        ):
            K2, T2, SIG = _pivot_grid(lv_cmp, x="K", y="T", value="sigma_gatheral")
            ax01.imshow(
                SIG,
                origin="lower",
                aspect="auto",
                extent=[
                    float(K2.min()),
                    float(K2.max()),
                    float(T2.min()),
                    float(T2.max()),
                ],
                interpolation="nearest",
            )
        ax01.set_title("Local Vol")
        ax01.set_xlabel("K")
        ax01.set_ylabel("T")

        # (1,0) Mountain surface of PDE-target
        ax10 = fig.add_subplot(gs[1, 0], projection="3d")
        plot_pricing_error_surface_3d(repricing, ax10)

        # (1,1) Validation scatter
        ax11 = fig.add_subplot(gs[1, 1])
        if _ensure_cols(
            repricing, ["target_price", "pde_price"], name="repricing_grid"
        ):
            x_sc, y_sc = _finite_xy(
                repricing["target_price"].to_numpy(), repricing["pde_price"].to_numpy()
            )
            if x_sc.size:
                ax11.scatter(x_sc, y_sc, s=8, alpha=0.8)
                lo = float(min(x_sc.min(), y_sc.min()))
                hi = float(max(x_sc.max(), y_sc.max()))
                ax11.plot([lo, hi], [lo, hi])
        ax11.set_title("Validation")
        ax11.set_xlabel("Target")
        ax11.set_ylabel("PDE")

        # No figure-level title: the four panel titles are already self-explanatory.
        _write_png(fig, out_poster, dpi=dpi)
        plt.close(fig)

    # -----------------
    # Proof block (2x2)
    # -----------------
    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 9))

    # (0,0) local vol heatmap
    if not lv_cmp.empty and _ensure_cols(
        lv_cmp, ["T", "K", "sigma_gatheral"], name="lv_compare"
    ):
        K2, T2, SIG = _pivot_grid(lv_cmp, x="K", y="T", value="sigma_gatheral")
        im = axs2[0, 0].imshow(
            SIG,
            origin="lower",
            aspect="auto",
            extent=[float(K2.min()), float(K2.max()), float(T2.min()), float(T2.max())],
            interpolation="nearest",
        )
        axs2[0, 0].set_title("Local Vol")
        axs2[0, 0].set_xlabel("K")
        axs2[0, 0].set_ylabel("T")
        fig2.colorbar(im, ax=axs2[0, 0], fraction=0.046, pad=0.04)

    # (0,1) invalid mask
    mask_col = ""
    title = "Invalid Mask"
    if "invalid_gatheral" in lv_cmp.columns and bool(
        pd.to_numeric(lv_cmp["invalid_gatheral"], errors="coerce")
        .fillna(0)
        .astype(int)
        .any()
    ):
        mask_col = "invalid_gatheral"
        title = "Invalid Mask (Gatheral)"
    elif "invalid_union" in lv_cmp.columns:
        mask_col = "invalid_union"
        title = "Invalid Mask (Union)"
    elif "invalid_dupire" in lv_cmp.columns:
        mask_col = "invalid_dupire"
        title = "Invalid Mask (Dupire)"

    if mask_col:
        K4, T4, MASK = _pivot_grid(lv_cmp, x="K", y="T", value=mask_col)
        im = axs2[0, 1].imshow(
            np.asarray(MASK, dtype=float),
            origin="lower",
            aspect="auto",
            extent=[float(K4.min()), float(K4.max()), float(T4.min()), float(T4.max())],
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axs2[0, 1].set_title(title)
        axs2[0, 1].set_xlabel("K")
        axs2[0, 1].set_ylabel("T")
        fig2.colorbar(im, ax=axs2[0, 1], fraction=0.046, pad=0.04)

    # (1,0) convergence (self-convergence, excluding finest)
    gp, err = _self_convergence(conv)
    if gp.size:
        idx = np.argsort(gp)
        gp, err = gp[idx], err[idx]
        axs2[1, 0].loglog(gp, err, marker="o")
    axs2[1, 0].set_title("Convergence")
    axs2[1, 0].set_xlabel("Nx×Nt")
    axs2[1, 0].set_ylabel("|Δ price|")

    # (1,1) scatter validation
    if _ensure_cols(repricing, ["target_price", "pde_price"], name="repricing_grid"):
        x_sc, y_sc = _finite_xy(
            repricing["target_price"].to_numpy(), repricing["pde_price"].to_numpy()
        )
        if x_sc.size:
            axs2[1, 1].scatter(x_sc, y_sc, s=8, alpha=0.8)
            lo = float(min(x_sc.min(), y_sc.min()))
            hi = float(max(x_sc.max(), y_sc.max()))
            axs2[1, 1].plot([lo, hi], [lo, hi])
    axs2[1, 1].set_title("PDE vs Black-76")
    axs2[1, 1].set_xlabel("Target")
    axs2[1, 1].set_ylabel("PDE")

    fig2.suptitle("Capstone Proof Block", y=1.02)
    fig2.tight_layout()
    _write_png(fig2, out_proofblock, dpi=dpi)
    plt.close(fig2)


# -----------------------------
# Data generation orchestration
# -----------------------------


def maybe_generate_poster_data(args: argparse.Namespace, data_dir: Path) -> None:
    if args.no_data_gen:
        return

    if data_dir.exists() and not args.regen_data:
        return

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "generate_localvol_full_csv.py"),
        "--poster",
        "--profile",
        str(args.profile),
        "--seed",
        str(int(args.seed)),
        "--out-dir",
        str(data_dir),
    ]
    if args.overrides:
        cmd += ["--overrides", args.overrides]

    print("[run] generating poster CSV bundle via:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate README images for Slot A/B/C (hero, mini gallery strip, capstone poster/proof block) "
            "from the Capstone2 poster CSV bundle."
        )
    )

    p.add_argument("--profile", default="full", choices=("quick", "full"))
    p.add_argument("--seed", type=int, default=7)

    p.add_argument(
        "--data-dir",
        type=str,
        default="",
        help=(
            "Folder containing the poster CSV bundle (surface/, repricing/, mountain/, convergence/, meta/). "
            "If omitted, uses out/capstone2_poster_data/profile_<profile>_seed_<seed>."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(Path("docs") / "assets" / "readme"),
        help="Where to write README images (default: docs/assets/readme).",
    )

    p.add_argument("--dpi", type=int, default=200)

    p.add_argument(
        "--regen-data",
        action="store_true",
        help="Force re-generation of the poster CSV bundle before plotting.",
    )
    p.add_argument(
        "--no-data-gen",
        action="store_true",
        help="Do not attempt to generate data; only plot from existing CSVs.",
    )
    p.add_argument(
        "--overrides",
        type=str,
        default="",
        help="Overrides JSON string or path, forwarded to generate_localvol_full_csv.py.",
    )

    p.add_argument(
        "--skip-3d",
        action="store_true",
        help="Skip the 3D hero render (matplotlib 3D can be slow on some CI).",
    )

    p.add_argument(
        "--poster-layout",
        choices=("grid", "wide"),
        default="grid",
        help="Poster layout: 'grid' (2x2, recommended) or 'wide' (1x4 legacy).",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = (
            ROOT
            / "out"
            / "capstone2_poster_data"
            / f"profile_{args.profile}_seed_{int(args.seed)}"
        )

    out_dir = ROOT / Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Ensure data exists (optional)
    try:
        maybe_generate_poster_data(args, data_dir)
    except subprocess.CalledProcessError as e:
        print(f"[err] data generation failed: {e}")
        return 2

    # 2) Load poster bundle CSVs
    iv_grid = _read_csv(data_dir / "surface" / "iv_surface_grid.csv")
    repricing = _read_csv(data_dir / "repricing" / "repricing_grid.csv")
    lv_cmp = _read_csv(data_dir / "mountain" / "gatheral_vs_dupire_diff_grid.csv")
    conv = _read_csv(data_dir / "convergence" / "localvol_pde_convergence.csv")

    # 3) Render individual plots
    plot_hero_iv_heatmap(
        iv_grid, out_dir / "hero_iv_heatmap_banner.png", dpi=args.dpi, banner=True
    )
    if not args.skip_3d:
        plot_hero_iv_3d(iv_grid, out_dir / "hero_iv_3d_banner.png", dpi=args.dpi)

    plot_iv_smiles(iv_grid, out_dir / "iv_smiles.png", dpi=args.dpi)
    plot_pde_vs_bs_scatter(repricing, out_dir / "pde_vs_bs_scatter.png", dpi=args.dpi)
    plot_pde_convergence(conv, out_dir / "pde_convergence.png", dpi=args.dpi)
    plot_pricing_error_heatmap(
        repricing, out_dir / "pricing_error_heatmap.png", dpi=args.dpi
    )
    plot_localvol_heatmap_and_mask(
        lv_cmp,
        out_lv=out_dir / "local_vol_heatmap.png",
        out_mask=out_dir / "invalid_mask_heatmap.png",
        dpi=args.dpi,
    )

    # 4) Render composites for README slots
    plot_readme_gallery_strip(
        iv_grid, repricing, out_dir / "readme_gallery_strip.png", dpi=args.dpi
    )

    # Keep the existing filename for README compatibility (even though layout is now 2x2 by default).
    plot_capstone_posters(
        iv_grid,
        lv_cmp,
        repricing,
        conv,
        out_poster=out_dir / "poster_capstone_wide.png",
        out_proofblock=out_dir / "poster_proofblock_2x2.png",
        dpi=args.dpi,
        poster_layout=args.poster_layout,
    )

    # 5) Manifest
    manifest = {
        "slot_A_hero": "hero_iv_heatmap_banner.png",
        "slot_B_gallery_strip": "readme_gallery_strip.png",
        "slot_C_capstone_poster": "poster_capstone_wide.png",
        "slot_C_proof_block": "poster_proofblock_2x2.png",
        "individual": [
            "iv_smiles.png",
            "pde_vs_bs_scatter.png",
            "pde_convergence.png",
            "local_vol_heatmap.png",
            "invalid_mask_heatmap.png",
            "pricing_error_heatmap.png",
        ],
    }
    (out_dir / "README_IMAGES_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[ok] wrote {out_dir / 'README_IMAGES_MANIFEST.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
