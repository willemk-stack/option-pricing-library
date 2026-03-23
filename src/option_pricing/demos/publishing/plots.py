from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from option_pricing.viz.publishing import (
    PUBLISHING_THEMES,
    SVG_TEXT_FONT_STACK,
    copy_light_variant,
    file_to_data_uri,
    publishing_palette,
    publishing_style,
    save_figure,
    style_colorbar,
    themed_asset_paths,
)

from .bundle import load_bundle_dataframe
from .config import get_visual_build_config
from .types import BundleManifest, PlotSpec

ROOT = Path(__file__).resolve().parents[4]

PRESET_SPECS: dict[str, tuple[PlotSpec, ...]] = {
    "static": (
        PlotSpec(
            preset="static",
            filename="svi_repaired_surface_heatmap.png",
            renderer="surface_heatmap",
            datasets=("surface/svi_repaired_grid",),
            title="SVI-Repaired Surface",
            kwargs={"value_col": "iv", "x_col": "y", "cmap": "viridis"},
        ),
        PlotSpec(
            preset="static",
            filename="svi_smile_slices.png",
            renderer="smile_slices",
            datasets=("surface/svi_repaired_grid",),
            title="SVI Smile Slices",
            kwargs={},
        ),
        PlotSpec(
            preset="static",
            filename="quote_surface_compare.png",
            renderer="quote_compare",
            datasets=("surface/quote_surface_compare",),
            title="Quote Fit Comparison",
            kwargs={},
        ),
    ),
    "dupire": (
        PlotSpec(
            preset="dupire",
            filename="essvi_smoothed_surface_heatmap.png",
            renderer="surface_heatmap",
            datasets=("surface/essvi_smoothed_grid",),
            title="Smoothed eSSVI Surface",
            kwargs={"value_col": "iv", "x_col": "y", "cmap": "viridis"},
        ),
        PlotSpec(
            preset="dupire",
            filename="localvol_gatheral_heatmap.png",
            renderer="localvol_heatmap",
            datasets=("localvol/gatheral_grid",),
            title="Gatheral Local Vol",
            kwargs={"value_col": "sigma_loc", "cmap": "cividis"},
        ),
        PlotSpec(
            preset="dupire",
            filename="gatheral_vs_dupire_diff_heatmap.png",
            renderer="diff_heatmap",
            datasets=("localvol/gatheral_vs_dupire_grid",),
            title="Gatheral vs Dupire",
            kwargs={"value_col": "diff_sigma_loc", "cmap": "coolwarm"},
        ),
    ),
    "poster": (
        PlotSpec(
            preset="poster",
            filename="hero_essvi_surface.png",
            renderer="surface_heatmap",
            datasets=("surface/essvi_smoothed_grid",),
            title="Smoothed eSSVI Surface",
            kwargs={
                "value_col": "iv",
                "x_col": "y",
                "cmap": "viridis",
                "banner": True,
            },
        ),
        PlotSpec(
            preset="poster",
            filename="poster_essvi_localvol_pde.png",
            renderer="poster_composite",
            datasets=(
                "surface/essvi_smoothed_grid",
                "localvol/gatheral_grid",
                "repricing/pde_roundtrip_grid",
                "repricing/convergence_grid",
            ),
            title="eSSVI -> Local Vol -> PDE",
            kwargs={},
        ),
    ),
    "docs": (
        PlotSpec(
            preset="docs",
            filename="docs_surface_story_triptych.png",
            renderer="story_triptych",
            datasets=(
                "surface/svi_repaired_grid",
                "surface/essvi_smoothed_grid",
                "localvol/gatheral_grid",
            ),
            title="Surface Story",
            kwargs={},
        ),
        PlotSpec(
            preset="docs",
            filename="docs_essvi_smoothness.png",
            renderer="smoothness_compare",
            datasets=("calibration/essvi_time_smoothness",),
            title="Time Smoothness",
            kwargs={},
        ),
    ),
    "numerics": (
        PlotSpec(
            preset="numerics",
            filename="pde_roundtrip_scatter.png",
            renderer="repricing_scatter",
            datasets=("repricing/pde_roundtrip_grid",),
            title="PDE Round-Trip Validation",
            kwargs={},
        ),
        PlotSpec(
            preset="numerics",
            filename="pde_convergence.png",
            renderer="convergence",
            datasets=("repricing/convergence_grid",),
            title="PDE Convergence",
            kwargs={},
        ),
        PlotSpec(
            preset="numerics",
            filename="pde_price_error_heatmap.png",
            renderer="price_error_heatmap",
            datasets=("repricing/pde_roundtrip_grid",),
            title="PDE Price Error",
            kwargs={"cmap": "coolwarm"},
        ),
    ),
    "showcase": (
        PlotSpec(
            preset="showcase",
            filename="readme_proof_card.svg",
            renderer="readme_proof_card",
            datasets=(),
            title="README proof card",
            kwargs={},
        ),
        PlotSpec(
            preset="showcase",
            filename="reviewer_proof_panel.svg",
            renderer="reviewer_proof_panel",
            datasets=(),
            title="Reviewer proof panel",
            kwargs={},
        ),
    ),
}


def build_plot_specs(*, preset: str) -> list[PlotSpec]:
    key = str(preset).strip().lower()
    if key not in PRESET_SPECS:
        raise ValueError(f"Unknown plot preset: {preset}")
    return list(PRESET_SPECS[key])


def _as_manifest(manifest_or_path: BundleManifest | str | Path) -> BundleManifest:
    if isinstance(manifest_or_path, BundleManifest):
        return manifest_or_path
    return BundleManifest.load(Path(manifest_or_path))


def _unique_sorted(series: pd.Series) -> np.ndarray:
    return np.array(
        sorted(set(pd.to_numeric(series, errors="coerce").dropna())), dtype=float
    )


def _pivot_grid(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    value: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_vals = _unique_sorted(df[x])
    y_vals = _unique_sorted(df[y])
    table = (
        df.assign(
            **{
                x: pd.to_numeric(df[x], errors="coerce"),
                y: pd.to_numeric(df[y], errors="coerce"),
                value: pd.to_numeric(df[value], errors="coerce"),
            }
        )
        .pivot(index=y, columns=x, values=value)
        .reindex(index=y_vals, columns=x_vals)
    )
    return x_vals, y_vals, table.to_numpy(dtype=float)


def _choose_smile_maturities(T_vals: np.ndarray, *, max_curves: int = 6) -> np.ndarray:
    Ts = np.asarray(sorted(set(np.asarray(T_vals, dtype=float))), dtype=float)
    if Ts.size <= max_curves:
        return Ts
    idx = np.linspace(0, Ts.size - 1, max_curves).round().astype(int)
    return np.asarray(np.unique(Ts[idx]), dtype=float)


def _surface_heatmap(
    df: pd.DataFrame,
    *,
    spec: PlotSpec,
    out_path: Path,
    dpi: int,
    theme: str,
) -> Path:
    x_col = str(spec.kwargs.get("x_col", "y"))
    value_col = str(spec.kwargs.get("value_col", "iv"))
    cmap = str(spec.kwargs.get("cmap", "viridis"))
    banner = bool(spec.kwargs.get("banner", False))
    x_vals, T_vals, Z = _pivot_grid(df, x=x_col, y="T", value=value_col)
    if Z.size == 0:
        raise ValueError(f"{spec.filename}: empty grid")

    figsize = (14, 3.8) if banner else (8.4, 5.2)
    with publishing_style(theme=theme) as plt:
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(
            Z,
            origin="lower",
            aspect="auto",
            extent=[
                float(x_vals.min()),
                float(x_vals.max()),
                float(T_vals.min()),
                float(T_vals.max()),
            ],
            cmap=cmap,
            interpolation="nearest",
        )
        ax.set_title(spec.title)
        ax.set_xlabel("Log-moneyness y" if x_col == "y" else "Strike K")
        ax.set_ylabel("Maturity T")
        colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=value_col)
        style_colorbar(colorbar, theme=theme)
        return save_figure(fig, out_path, dpi=dpi)


def _smile_slices(
    df: pd.DataFrame,
    *,
    spec: PlotSpec,
    out_path: Path,
    dpi: int,
    theme: str,
) -> Path:
    with publishing_style(theme=theme) as plt:
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        for T in _choose_smile_maturities(df["T"].to_numpy(dtype=float)):
            g = df.loc[np.isclose(df["T"], T)].sort_values("y")
            ax.plot(g["y"].to_numpy(), g["iv"].to_numpy(), label=f"T={float(T):g}")
        ax.set_title(spec.title)
        ax.set_xlabel("Log-moneyness y")
        ax.set_ylabel("Implied vol")
        ax.legend(loc="best", fontsize=8)
        return save_figure(fig, out_path, dpi=dpi)


def _quote_compare(
    df: pd.DataFrame,
    *,
    spec: PlotSpec,
    out_path: Path,
    dpi: int,
    theme: str,
) -> Path:
    grouped = (
        df.groupby("T", as_index=False)
        .agg(
            svi_mae_bp=("iv_resid_svi_bp", lambda s: float(np.mean(np.abs(s)))),
            nodal_mae_bp=(
                "iv_resid_essvi_nodal_bp",
                lambda s: float(np.mean(np.abs(s))),
            ),
            smoothed_mae_bp=(
                "iv_resid_essvi_smoothed_bp",
                lambda s: float(np.mean(np.abs(s))),
            ),
        )
        .sort_values("T")
    )
    with publishing_style(theme=theme) as plt:
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        ax.plot(grouped["T"], grouped["svi_mae_bp"], marker="o", label="SVI repaired")
        ax.plot(grouped["T"], grouped["nodal_mae_bp"], marker="o", label="eSSVI nodal")
        ax.plot(
            grouped["T"],
            grouped["smoothed_mae_bp"],
            marker="o",
            label="eSSVI smoothed",
        )
        ax.set_title(spec.title)
        ax.set_xlabel("Maturity T")
        ax.set_ylabel("Mean abs IV error (bp)")
        ax.legend(loc="best")
        return save_figure(fig, out_path, dpi=dpi)


def _localvol_heatmap(
    df: pd.DataFrame,
    *,
    spec: PlotSpec,
    out_path: Path,
    dpi: int,
    theme: str,
) -> Path:
    value_col = str(spec.kwargs.get("value_col", "sigma_loc"))
    cmap = str(spec.kwargs.get("cmap", "cividis"))
    clean = df.copy()
    clean.loc[clean["invalid"].astype(bool), value_col] = np.nan
    y_vals, T_vals, Z = _pivot_grid(clean, x="y", y="T", value=value_col)
    with publishing_style(theme=theme) as plt:
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        im = ax.imshow(
            Z,
            origin="lower",
            aspect="auto",
            extent=[
                float(y_vals.min()),
                float(y_vals.max()),
                float(T_vals.min()),
                float(T_vals.max()),
            ],
            cmap=cmap,
            interpolation="nearest",
        )
        ax.set_title(spec.title)
        ax.set_xlabel("Log-moneyness y")
        ax.set_ylabel("Maturity T")
        colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=value_col)
        style_colorbar(colorbar, theme=theme)
        return save_figure(fig, out_path, dpi=dpi)


def _diff_heatmap(
    df: pd.DataFrame,
    *,
    spec: PlotSpec,
    out_path: Path,
    dpi: int,
    theme: str,
) -> Path:
    value_col = str(spec.kwargs.get("value_col", "diff_sigma_loc"))
    cmap = str(spec.kwargs.get("cmap", "coolwarm"))
    clean = df.copy()
    clean.loc[clean["invalid_union"].astype(bool), value_col] = np.nan
    K_vals, T_vals, Z = _pivot_grid(clean, x="K", y="T", value=value_col)
    vmax = float(np.nanmax(np.abs(Z))) if np.isfinite(Z).any() else 1.0
    with publishing_style(theme=theme) as plt:
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        im = ax.imshow(
            Z,
            origin="lower",
            aspect="auto",
            extent=[
                float(K_vals.min()),
                float(K_vals.max()),
                float(T_vals.min()),
                float(T_vals.max()),
            ],
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(spec.title)
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Maturity T")
        colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=value_col)
        style_colorbar(colorbar, theme=theme)
        return save_figure(fig, out_path, dpi=dpi)


def _repricing_scatter(
    df: pd.DataFrame,
    *,
    spec: PlotSpec,
    out_path: Path,
    dpi: int,
    theme: str,
) -> Path:
    x = df["target_price"].to_numpy(dtype=float)
    y = df["pde_price"].to_numpy(dtype=float)
    palette = publishing_palette(theme)
    with publishing_style(theme=theme) as plt:
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        ax.scatter(x, y, s=12, alpha=0.85)
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        ax.plot([lo, hi], [lo, hi], color=palette["reference"], linewidth=1.1)
        ax.set_title(spec.title)
        ax.set_xlabel("Target price")
        ax.set_ylabel("PDE price")
        ax.set_aspect("equal", adjustable="box")
        return save_figure(fig, out_path, dpi=dpi)


def _convergence(
    df: pd.DataFrame,
    *,
    spec: PlotSpec,
    out_path: Path,
    dpi: int,
    theme: str,
) -> Path:
    data = df.sort_values("grid_points").copy()
    gp = data["grid_points"].to_numpy(dtype=float)
    pde = data["pde_price"].to_numpy(dtype=float)
    ref = float(pde[-1])
    err_self = np.abs(pde[:-1] - ref)
    gp_self = gp[:-1]
    with publishing_style(theme=theme) as plt:
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ok = err_self > 0.0
        if ok.any():
            ax.loglog(gp_self[ok], err_self[ok], marker="o", label="|p - p(finest)|")
        if "target_price" in data.columns:
            tgt = data["target_price"].astype(float).to_numpy()
            err_tgt = np.abs(pde - tgt)
            ok_tgt = err_tgt > 0.0
            if ok_tgt.any():
                ax.loglog(gp[ok_tgt], err_tgt[ok_tgt], marker="o", label="|p - target|")
        ax.set_title(spec.title)
        ax.set_xlabel("Grid points")
        ax.set_ylabel("Absolute error")
        ax.legend(loc="best", fontsize=8)
        return save_figure(fig, out_path, dpi=dpi)


def _price_error_heatmap(
    df: pd.DataFrame,
    *,
    spec: PlotSpec,
    out_path: Path,
    dpi: int,
    theme: str,
) -> Path:
    K_vals, T_vals, Z = _pivot_grid(df, x="K", y="T", value="price_error")
    vmax = float(np.nanmax(np.abs(Z))) if np.isfinite(Z).any() else 1.0
    with publishing_style(theme=theme) as plt:
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        im = ax.imshow(
            Z,
            origin="lower",
            aspect="auto",
            extent=[
                float(K_vals.min()),
                float(K_vals.max()),
                float(T_vals.min()),
                float(T_vals.max()),
            ],
            cmap=str(spec.kwargs.get("cmap", "coolwarm")),
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(spec.title)
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Maturity T")
        colorbar = fig.colorbar(
            im, ax=ax, fraction=0.046, pad=0.04, label="price_error"
        )
        style_colorbar(colorbar, theme=theme)
        return save_figure(fig, out_path, dpi=dpi)


def _poster_composite(
    datasets: dict[str, pd.DataFrame],
    *,
    spec: PlotSpec,
    out_path: Path,
    dpi: int,
    theme: str,
) -> Path:
    surf = datasets["surface/essvi_smoothed_grid"]
    localvol = datasets["localvol/gatheral_grid"]
    repricing = datasets["repricing/pde_roundtrip_grid"]
    x_s, T_s, Z_s = _pivot_grid(surf, x="y", y="T", value="iv")
    y_lv, T_lv, Z_lv = _pivot_grid(
        localvol.loc[~localvol["invalid"].astype(bool)], x="y", y="T", value="sigma_loc"
    )
    K_err, T_err, Z_err = _pivot_grid(repricing, x="K", y="T", value="price_error")
    vmax = float(np.nanmax(np.abs(Z_err))) if np.isfinite(Z_err).any() else 1.0

    palette = publishing_palette(theme)
    with publishing_style(theme=theme) as plt:
        fig, axs = plt.subplots(2, 2, figsize=(12.2, 8.6), constrained_layout=True)

        im0 = axs[0, 0].imshow(
            Z_s,
            origin="lower",
            aspect="auto",
            extent=[
                float(x_s.min()),
                float(x_s.max()),
                float(T_s.min()),
                float(T_s.max()),
            ],
            cmap="viridis",
            interpolation="nearest",
        )
        axs[0, 0].set_title("Smoothed eSSVI Surface")
        axs[0, 0].set_xlabel("Log-moneyness y")
        axs[0, 0].set_ylabel("Maturity T")
        colorbar0 = fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)
        style_colorbar(colorbar0, theme=theme)

        im1 = axs[0, 1].imshow(
            Z_lv,
            origin="lower",
            aspect="auto",
            extent=[
                float(y_lv.min()),
                float(y_lv.max()),
                float(T_lv.min()),
                float(T_lv.max()),
            ],
            cmap="cividis",
            interpolation="nearest",
        )
        axs[0, 1].set_title("Gatheral Local Vol")
        axs[0, 1].set_xlabel("Log-moneyness y")
        axs[0, 1].set_ylabel("Maturity T")
        colorbar1 = fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)
        style_colorbar(colorbar1, theme=theme)

        im2 = axs[1, 0].imshow(
            Z_err,
            origin="lower",
            aspect="auto",
            extent=[
                float(K_err.min()),
                float(K_err.max()),
                float(T_err.min()),
                float(T_err.max()),
            ],
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        axs[1, 0].set_title("PDE Price Error")
        axs[1, 0].set_xlabel("Strike K")
        axs[1, 0].set_ylabel("Maturity T")
        colorbar2 = fig.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)
        style_colorbar(colorbar2, theme=theme)

        x = repricing["target_price"].to_numpy(dtype=float)
        y = repricing["pde_price"].to_numpy(dtype=float)
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        axs[1, 1].scatter(x, y, s=10, alpha=0.85)
        axs[1, 1].plot(
            [lo, hi],
            [lo, hi],
            color=palette["reference"],
            linewidth=1.1,
        )
        axs[1, 1].set_title("PDE Round-Trip Validation")
        axs[1, 1].set_xlabel("Target price")
        axs[1, 1].set_ylabel("PDE price")

        return save_figure(fig, out_path, dpi=dpi)


def _story_triptych(
    datasets: dict[str, pd.DataFrame],
    *,
    spec: PlotSpec,
    out_path: Path,
    dpi: int,
    theme: str,
) -> Path:
    svi = datasets["surface/svi_repaired_grid"]
    essvi = datasets["surface/essvi_smoothed_grid"]
    lv = datasets["localvol/gatheral_grid"]
    x0, T0, Z0 = _pivot_grid(svi, x="y", y="T", value="iv")
    x1, T1, Z1 = _pivot_grid(essvi, x="y", y="T", value="iv")
    x2, T2, Z2 = _pivot_grid(
        lv.loc[~lv["invalid"].astype(bool)], x="y", y="T", value="sigma_loc"
    )

    with publishing_style(theme=theme) as plt:
        fig, axs = plt.subplots(1, 3, figsize=(15.2, 4.6), constrained_layout=True)
        for ax, Z, x_vals, T_vals, title, cmap in [
            (axs[0], Z0, x0, T0, "SVI repaired", "viridis"),
            (axs[1], Z1, x1, T1, "eSSVI smoothed", "viridis"),
            (axs[2], Z2, x2, T2, "Local vol", "cividis"),
        ]:
            im = ax.imshow(
                Z,
                origin="lower",
                aspect="auto",
                extent=[
                    float(x_vals.min()),
                    float(x_vals.max()),
                    float(T_vals.min()),
                    float(T_vals.max()),
                ],
                cmap=cmap,
                interpolation="nearest",
            )
            ax.set_title(title)
            ax.set_xlabel("Log-moneyness y")
            ax.set_ylabel("Maturity T")
            colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            style_colorbar(colorbar, theme=theme)
        return save_figure(fig, out_path, dpi=dpi)


def _smoothness_compare(
    df: pd.DataFrame,
    *,
    spec: PlotSpec,
    out_path: Path,
    dpi: int,
    theme: str,
) -> Path:
    with publishing_style(theme=theme) as plt:
        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        ax.plot(
            df["T_knot"], df["max_abs_wT_jump_svi"], marker="o", label="SVI repaired"
        )
        ax.plot(
            df["T_knot"],
            df["max_abs_wT_jump_smoothed"],
            marker="o",
            label="eSSVI smoothed",
        )
        ax.set_title(spec.title)
        ax.set_xlabel("Knot maturity")
        ax.set_ylabel("Max |jump in w_T|")
        ax.legend(loc="best")
        return save_figure(fig, out_path, dpi=dpi)


def _require_match(pattern: str, text: str, *, label: str) -> re.Match[str]:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Could not extract {label} from docs source")
    return match


def _load_reviewer_proof_panel_metrics() -> dict[str, str]:
    essvi_text = (ROOT / "docs" / "user_guides" / "essvi_smooth_handoff.md").read_text(
        encoding="utf-8"
    )
    localvol_text = (
        ROOT / "docs" / "user_guides" / "localvol_pde_validation.md"
    ).read_text(encoding="utf-8")

    seam_match = _require_match(
        r"\| `T = 0\.15` \| `([^`]+)` \| `([^`]+)` \|",
        essvi_text,
        label="published seam-jump pair",
    )
    projection_match = _require_match(
        r"\| Projection summary \| `price_rmse = ([^`]+)` \| `max_abs_price_error = ([^`]+)` \| `projection_dupire_invalid_count = ([^`]+)` \|",
        essvi_text,
        label="projection summary",
    )
    repriced_options_match = _require_match(
        r"\| Repriced options \| `([^`]+)` \|",
        localvol_text,
        label="repriced options",
    )
    mean_abs_price_error_match = _require_match(
        r"\| Mean abs price error \| `([^`]+)` \|",
        localvol_text,
        label="mean absolute price error",
    )
    max_abs_iv_error_match = _require_match(
        r"\| Max abs IV error \| `([^`]+)` \|",
        localvol_text,
        label="max absolute IV error",
    )

    return {
        "seam_svi": seam_match.group(1),
        "seam_smoothed": seam_match.group(2),
        "projection_invalid_count": projection_match.group(3),
        "repriced_options": repriced_options_match.group(1),
        "mean_abs_price_error": mean_abs_price_error_match.group(1),
        "max_abs_iv_error": max_abs_iv_error_match.group(1),
    }


def _fmt_speedup(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}x"
    return f"{value:.1f}x"


def _fmt_compact_sci(value: float, *, decimals: int) -> str:
    mantissa, exponent = f"{value:.{decimals}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _fmt_bp(value: float) -> str:
    return f"{value:.1f} bp"


def _load_readme_proof_card_metrics() -> dict[str, str]:
    metrics = _load_reviewer_proof_panel_metrics()
    iv_speedup_rows = pd.read_csv(ROOT / "benchmarks" / "artifacts" / "iv_speedup.csv")
    iv_last = iv_speedup_rows.sort_values("n_strikes").iloc[-1]
    metrics["seam_svi"] = _fmt_compact_sci(float(metrics["seam_svi"]), decimals=2)
    metrics["seam_smoothed"] = _fmt_compact_sci(
        float(metrics["seam_smoothed"]), decimals=1
    )
    metrics["mean_abs_price_error"] = _fmt_compact_sci(
        float(metrics["mean_abs_price_error"]), decimals=1
    )
    metrics["max_abs_iv_error"] = _fmt_bp(
        float(metrics["max_abs_iv_error"].removesuffix(" bp"))
    )
    metrics["iv_speedup"] = _fmt_speedup(float(iv_last["vectorized_speedup_x"]))
    metrics["iv_strikes"] = str(int(iv_last["n_strikes"]))
    return metrics


def _readme_proof_card(
    datasets: dict[str, pd.DataFrame],
    *,
    spec: PlotSpec,
    out_path: Path,
    dpi: int,
    theme: str,
) -> Path:
    del datasets, spec, dpi

    metrics = _load_readme_proof_card_metrics()
    palette = publishing_palette(theme)
    font_stack = SVG_TEXT_FONT_STACK

    if theme == "dark":
        colors = {
            "page_bg": "#08101F",
            "panel_bg": "#0F172A",
            "panel_stroke": "#334155",
            "card_bg": "#111C2F",
            "card_stroke": "#334155",
            "accent": "#8CC9FF",
            "success": "#91E0D7",
            "text": palette["text"],
            "muted": palette["muted_text"],
            "pill_bg": "#16243A",
            "pill_text": "#BEE3FF",
        }
    else:
        colors = {
            "page_bg": "#EEF3F8",
            "panel_bg": "#FFFFFF",
            "panel_stroke": "#D7E0EA",
            "card_bg": "#F8FBFD",
            "card_stroke": "#D7E0EA",
            "accent": "#0B5CAB",
            "success": "#0F766E",
            "text": palette["text"],
            "muted": "#4A6277",
            "pill_bg": "#EAF1F8",
            "pill_text": "#0B5CAB",
        }

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="760" viewBox="0 0 1280 760" role="img" aria-labelledby="title desc">
  <title id="title">README proof card for the option pricing library</title>
  <desc id="desc">Four readable proof tiles for README width covering surface repair, smooth Dupire handoff, local-vol and PDE validation, and benchmark plus delivery evidence. Metrics are loaded from the published proof pages and committed performance artifacts.</desc>

  <rect width="1280" height="760" fill="{colors["page_bg"]}" />
  <rect x="24" y="24" width="1232" height="712" rx="30" ry="30" fill="{colors["panel_bg"]}" stroke="{colors["panel_stroke"]}" stroke-width="2" />

  <rect x="64" y="66" width="170" height="34" rx="17" ry="17" fill="{colors["pill_bg"]}" />
  <text x="86" y="89" font-family="{font_stack}" font-size="16" font-weight="700" letter-spacing="1" fill="{colors["pill_text"]}">PROOF AT A GLANCE</text>
  <text x="64" y="148" font-family="{font_stack}" font-size="42" font-weight="700" fill="{colors["text"]}">What is already proven in this repo</text>
  <text x="64" y="184" font-family="{font_stack}" font-size="20" fill="{colors["muted"]}">Readable at GitHub README width, generated from the same published proof pages and benchmark artifacts as the docs.</text>

  <rect x="64" y="228" width="552" height="214" rx="24" ry="24" fill="{colors["card_bg"]}" stroke="{colors["card_stroke"]}" stroke-width="2" />
  <text x="96" y="276" font-family="{font_stack}" font-size="28" font-weight="700" fill="{colors["text"]}">Surface repair</text>
  <text x="96" y="316" font-family="{font_stack}" font-size="20" fill="{colors["muted"]}">Quoted versus repaired surfaces stay visible.</text>
  <text x="96" y="350" font-family="{font_stack}" font-size="20" fill="{colors["muted"]}">No-arbitrage checks and per-expiry SVI residuals stay reviewable.</text>
  <text x="96" y="398" font-family="{font_stack}" font-size="18" font-weight="700" fill="{colors["accent"]}">Open: Surface workflow and decision guide</text>

  <rect x="664" y="228" width="552" height="214" rx="24" ry="24" fill="{colors["card_bg"]}" stroke="{colors["card_stroke"]}" stroke-width="2" />
  <text x="696" y="276" font-family="{font_stack}" font-size="28" font-weight="700" fill="{colors["text"]}">Smooth Dupire handoff</text>
  <text x="696" y="322" font-family="{font_stack}" font-size="36" font-weight="700" fill="{colors["accent"]}">{metrics["seam_svi"]} -&gt; {metrics["seam_smoothed"]}</text>
  <text x="696" y="356" font-family="{font_stack}" font-size="20" fill="{colors["muted"]}">Worst published seam jump after smoothing.</text>
  <text x="696" y="396" font-family="{font_stack}" font-size="20" fill="{colors["muted"]}">Dupire invalid-count check stays at <tspan font-weight="700" fill="{colors["success"]}">{metrics["projection_invalid_count"]}</tspan>.</text>

  <rect x="64" y="474" width="552" height="214" rx="24" ry="24" fill="{colors["card_bg"]}" stroke="{colors["card_stroke"]}" stroke-width="2" />
  <text x="96" y="522" font-family="{font_stack}" font-size="28" font-weight="700" fill="{colors["text"]}">Local-vol and PDE validation</text>
  <text x="96" y="568" font-family="{font_stack}" font-size="36" font-weight="700" fill="{colors["accent"]}">{metrics["repriced_options"]} repricings</text>
  <text x="96" y="602" font-family="{font_stack}" font-size="20" fill="{colors["muted"]}">Mean abs price error {metrics["mean_abs_price_error"]}.</text>
  <text x="96" y="636" font-family="{font_stack}" font-size="20" fill="{colors["muted"]}">Max abs IV error {metrics["max_abs_iv_error"]} on the published sweep.</text>

  <rect x="664" y="474" width="552" height="214" rx="24" ry="24" fill="{colors["card_bg"]}" stroke="{colors["card_stroke"]}" stroke-width="2" />
  <text x="696" y="522" font-family="{font_stack}" font-size="28" font-weight="700" fill="{colors["text"]}">Benchmarks and delivery</text>
  <text x="696" y="568" font-family="{font_stack}" font-size="36" font-weight="700" fill="{colors["accent"]}">{metrics["iv_speedup"]} IV slice speedup</text>
  <text x="696" y="602" font-family="{font_stack}" font-size="20" fill="{colors["muted"]}">Published at {metrics["iv_strikes"]} strikes from committed benchmark artifacts.</text>
  <text x="696" y="636" font-family="{font_stack}" font-size="20" fill="{colors["muted"]}">README, docs visuals, and proof pages are regenerated and checked in CI.</text>
</svg>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    return out_path


def _reviewer_proof_panel(
    datasets: dict[str, pd.DataFrame],
    *,
    spec: PlotSpec,
    out_path: Path,
    dpi: int,
    theme: str,
) -> Path:
    del datasets, spec, dpi

    metrics = _load_reviewer_proof_panel_metrics()
    palette = publishing_palette(theme)
    image_suffix = "dark" if theme == "dark" else "light"
    font_stack = SVG_TEXT_FONT_STACK
    quote_compare_uri = file_to_data_uri(
        ROOT
        / "docs"
        / "assets"
        / "generated"
        / "static"
        / f"quote_surface_compare.{image_suffix}.png"
    )
    smooth_surface_uri = file_to_data_uri(
        ROOT
        / "docs"
        / "assets"
        / "generated"
        / "dupire"
        / f"essvi_smoothed_surface_heatmap.{image_suffix}.png"
    )
    poster_uri = file_to_data_uri(
        ROOT
        / "docs"
        / "assets"
        / "generated"
        / "poster"
        / f"poster_essvi_localvol_pde.{image_suffix}.png"
    )

    if theme == "dark":
        colors = {
            "page_bg": "#08101F",
            "card_bg": "#0F172A",
            "card_stroke": "#314056",
            "header_start": "#18263C",
            "header_end": "#101B2D",
            "accent": "#8CC9FF",
            "success": "#91E0D7",
            "text": palette["text"],
            "muted": palette["muted_text"],
            "surface_fill": "#23344A",
            "chip_fill": "#020617",
            "chip_text": "#FFFFFF",
        }
    else:
        colors = {
            "page_bg": "#EEF3F8",
            "card_bg": "#FFFFFF",
            "card_stroke": "#D7E0EA",
            "header_start": "#F4F8FC",
            "header_end": "#EAF1F7",
            "accent": "#0B5CAB",
            "success": "#0F766E",
            "text": palette["text"],
            "muted": "#486074",
            "surface_fill": "#DCE7F1",
            "chip_fill": "#0B1F33",
            "chip_text": "#FFFFFF",
        }

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900" role="img" aria-labelledby="title desc">
  <title id="title">Reviewer proof panel for the option pricing library</title>
  <desc id="desc">Workflow strip from static surface to eSSVI smoothing to local vol to PDE repricing, four published proof metrics, and three inlined thumbnails reused from tracked docs assets.</desc>
  <defs>
    <linearGradient id="headerBg" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="{colors["header_start"]}" />
      <stop offset="100%" stop-color="{colors["header_end"]}" />
    </linearGradient>
    <clipPath id="thumbA">
      <rect x="760" y="450" width="340" height="150" rx="20" ry="20" />
    </clipPath>
    <clipPath id="thumbB">
      <rect x="1154" y="450" width="340" height="150" rx="20" ry="20" />
    </clipPath>
    <clipPath id="thumbC">
      <rect x="760" y="628" width="734" height="178" rx="20" ry="20" />
    </clipPath>
  </defs>

  <rect width="1600" height="900" fill="{colors["page_bg"]}" />
  <rect x="28" y="28" width="1544" height="844" rx="34" ry="34" fill="{colors["card_bg"]}" stroke="{colors["card_stroke"]}" stroke-width="2" />

  <text x="72" y="108" font-family="{font_stack}" font-size="54" font-weight="700" fill="{colors["text"]}">Reviewer proof panel</text>
  <text x="72" y="148" font-family="{font_stack}" font-size="24" fill="{colors["muted"]}">Fast scan of the repo's strongest surface, smoothing, local-vol, and PDE evidence.</text>
  <text x="72" y="182" font-family="{font_stack}" font-size="18" fill="{colors["muted"]}">Metrics are copied from the published eSSVI bridge and local vol + PDE proof pages.</text>

  <rect x="72" y="222" width="1456" height="122" rx="28" ry="28" fill="url(#headerBg)" stroke="{colors["card_stroke"]}" stroke-width="2" />
  <text x="102" y="262" font-family="{font_stack}" font-size="18" font-weight="700" letter-spacing="1" fill="{colors["accent"]}">WORKFLOW</text>

  <rect x="102" y="278" width="258" height="44" rx="18" ry="18" fill="{colors["card_bg"]}" stroke="{colors["card_stroke"]}" />
  <text x="126" y="307" font-family="{font_stack}" font-size="24" font-weight="700" fill="{colors["text"]}">Static surface</text>
  <text x="130" y="335" font-family="{font_stack}" font-size="16" fill="{colors["muted"]}">Fit and repair noisy quotes</text>

  <text x="382" y="308" font-family="{font_stack}" font-size="28" font-weight="700" fill="{colors["muted"]}">-&gt;</text>

  <rect x="430" y="278" width="270" height="44" rx="18" ry="18" fill="{colors["card_bg"]}" stroke="{colors["card_stroke"]}" />
  <text x="454" y="307" font-family="{font_stack}" font-size="24" font-weight="700" fill="{colors["text"]}">eSSVI smoothing</text>
  <text x="458" y="335" font-family="{font_stack}" font-size="16" fill="{colors["muted"]}">Make w_T usable for Dupire</text>

  <text x="722" y="308" font-family="{font_stack}" font-size="28" font-weight="700" fill="{colors["muted"]}">-&gt;</text>

  <rect x="770" y="278" width="214" height="44" rx="18" ry="18" fill="{colors["card_bg"]}" stroke="{colors["card_stroke"]}" />
  <text x="794" y="307" font-family="{font_stack}" font-size="24" font-weight="700" fill="{colors["text"]}">Local vol</text>
  <text x="798" y="335" font-family="{font_stack}" font-size="16" fill="{colors["muted"]}">Inspect invalid masks early</text>

  <text x="1006" y="308" font-family="{font_stack}" font-size="28" font-weight="700" fill="{colors["muted"]}">-&gt;</text>

  <rect x="1054" y="278" width="294" height="44" rx="18" ry="18" fill="{colors["card_bg"]}" stroke="{colors["card_stroke"]}" />
  <text x="1078" y="307" font-family="{font_stack}" font-size="24" font-weight="700" fill="{colors["text"]}">PDE repricing</text>
  <text x="1082" y="335" font-family="{font_stack}" font-size="16" fill="{colors["muted"]}">Check round-trip error and convergence</text>

  <text x="72" y="396" font-family="{font_stack}" font-size="18" font-weight="700" letter-spacing="1" fill="{colors["accent"]}">PUBLISHED PROOF METRICS</text>

  <rect x="72" y="418" width="292" height="144" rx="24" ry="24" fill="{colors["card_bg"]}" stroke="{colors["card_stroke"]}" stroke-width="2" />
  <text x="96" y="452" font-family="{font_stack}" font-size="18" font-weight="700" fill="{colors["text"]}">Seam jump</text>
  <text x="96" y="500" font-family="{font_stack}" font-size="34" font-weight="700" fill="{colors["accent"]}">{metrics["seam_svi"]} -&gt; {metrics["seam_smoothed"]}</text>
  <text x="96" y="536" font-family="{font_stack}" font-size="16" fill="{colors["muted"]}">Largest published w_T seam jump</text>

  <rect x="388" y="418" width="292" height="144" rx="24" ry="24" fill="{colors["card_bg"]}" stroke="{colors["card_stroke"]}" stroke-width="2" />
  <text x="412" y="452" font-family="{font_stack}" font-size="18" font-weight="700" fill="{colors["text"]}">Dupire invalids</text>
  <text x="412" y="512" font-family="{font_stack}" font-size="52" font-weight="700" fill="{colors["success"]}">{metrics["projection_invalid_count"]}</text>
  <text x="412" y="536" font-family="{font_stack}" font-size="16" fill="{colors["muted"]}">projection_dupire_invalid_count</text>

  <rect x="72" y="586" width="292" height="144" rx="24" ry="24" fill="{colors["card_bg"]}" stroke="{colors["card_stroke"]}" stroke-width="2" />
  <text x="96" y="620" font-family="{font_stack}" font-size="18" font-weight="700" fill="{colors["text"]}">Repriced options</text>
  <text x="96" y="680" font-family="{font_stack}" font-size="52" font-weight="700" fill="{colors["accent"]}">{metrics["repriced_options"]}</text>
  <text x="96" y="704" font-family="{font_stack}" font-size="16" fill="{colors["muted"]}">Published local-vol/PDE sweep size</text>

  <rect x="388" y="586" width="292" height="144" rx="24" ry="24" fill="{colors["card_bg"]}" stroke="{colors["card_stroke"]}" stroke-width="2" />
  <text x="412" y="620" font-family="{font_stack}" font-size="18" font-weight="700" fill="{colors["text"]}">Mean abs price error</text>
  <text x="412" y="680" font-family="{font_stack}" font-size="42" font-weight="700" fill="{colors["accent"]}">{metrics["mean_abs_price_error"]}</text>
  <text x="412" y="704" font-family="{font_stack}" font-size="16" fill="{colors["muted"]}">Published PDE repricing summary</text>

  <text x="72" y="780" font-family="{font_stack}" font-size="18" font-weight="700" letter-spacing="1" fill="{colors["accent"]}">WHAT TO OPEN NEXT</text>
  <text x="72" y="812" font-family="{font_stack}" font-size="18" fill="{colors["muted"]}">Use the split proof pages for the full tables, notebook links, and diagnostics.</text>

  <rect x="726" y="380" width="802" height="448" rx="28" ry="28" fill="{colors["card_bg"]}" stroke="{colors["card_stroke"]}" stroke-width="2" />
  <text x="760" y="416" font-family="{font_stack}" font-size="18" font-weight="700" letter-spacing="1" fill="{colors["accent"]}">SUPPORTING VISUALS REUSED FROM TRACKED ASSETS</text>

  <rect x="760" y="450" width="340" height="150" rx="20" ry="20" fill="{colors["surface_fill"]}" />
  <image x="760" y="450" width="340" height="150" href="{quote_compare_uri}" preserveAspectRatio="xMidYMid slice" clip-path="url(#thumbA)" />
  <rect x="776" y="466" width="174" height="30" rx="15" ry="15" fill="{colors["chip_fill"]}" opacity="0.82" />
  <text x="794" y="487" font-family="{font_stack}" font-size="16" font-weight="700" fill="{colors["chip_text"]}">Quote-fit comparison</text>

  <rect x="1154" y="450" width="340" height="150" rx="20" ry="20" fill="{colors["surface_fill"]}" />
  <image x="1154" y="450" width="340" height="150" href="{smooth_surface_uri}" preserveAspectRatio="xMidYMid slice" clip-path="url(#thumbB)" />
  <rect x="1170" y="466" width="198" height="30" rx="15" ry="15" fill="{colors["chip_fill"]}" opacity="0.82" />
  <text x="1188" y="487" font-family="{font_stack}" font-size="16" font-weight="700" fill="{colors["chip_text"]}">Smoothed eSSVI surface</text>

  <rect x="760" y="628" width="734" height="178" rx="20" ry="20" fill="{colors["surface_fill"]}" />
  <image x="760" y="628" width="734" height="178" href="{poster_uri}" preserveAspectRatio="xMidYMid slice" clip-path="url(#thumbC)" />
  <rect x="776" y="644" width="248" height="30" rx="15" ry="15" fill="{colors["chip_fill"]}" opacity="0.82" />
  <text x="794" y="665" font-family="{font_stack}" font-size="16" font-weight="700" fill="{colors["chip_text"]}">Local-vol and PDE proof collage</text>
  <text x="760" y="850" font-family="{font_stack}" font-size="16" fill="{colors["muted"]}">Visual panel now inlines its supporting thumbnails so CI screenshots do not depend on nested browser fetches.</text>
</svg>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    return out_path


RENDERERS = {
    "surface_heatmap": lambda data, *, spec, out_path, dpi, theme: _surface_heatmap(
        next(iter(data.values())),
        spec=spec,
        out_path=out_path,
        dpi=dpi,
        theme=theme,
    ),
    "smile_slices": lambda data, *, spec, out_path, dpi, theme: _smile_slices(
        next(iter(data.values())),
        spec=spec,
        out_path=out_path,
        dpi=dpi,
        theme=theme,
    ),
    "quote_compare": lambda data, *, spec, out_path, dpi, theme: _quote_compare(
        next(iter(data.values())),
        spec=spec,
        out_path=out_path,
        dpi=dpi,
        theme=theme,
    ),
    "localvol_heatmap": lambda data, *, spec, out_path, dpi, theme: _localvol_heatmap(
        next(iter(data.values())),
        spec=spec,
        out_path=out_path,
        dpi=dpi,
        theme=theme,
    ),
    "diff_heatmap": lambda data, *, spec, out_path, dpi, theme: _diff_heatmap(
        next(iter(data.values())),
        spec=spec,
        out_path=out_path,
        dpi=dpi,
        theme=theme,
    ),
    "repricing_scatter": lambda data, *, spec, out_path, dpi, theme: _repricing_scatter(
        next(iter(data.values())),
        spec=spec,
        out_path=out_path,
        dpi=dpi,
        theme=theme,
    ),
    "convergence": lambda data, *, spec, out_path, dpi, theme: _convergence(
        next(iter(data.values())),
        spec=spec,
        out_path=out_path,
        dpi=dpi,
        theme=theme,
    ),
    "price_error_heatmap": lambda data, *, spec, out_path, dpi, theme: _price_error_heatmap(
        next(iter(data.values())),
        spec=spec,
        out_path=out_path,
        dpi=dpi,
        theme=theme,
    ),
    "poster_composite": _poster_composite,
    "story_triptych": _story_triptych,
    "smoothness_compare": lambda data, *, spec, out_path, dpi, theme: _smoothness_compare(
        next(iter(data.values())),
        spec=spec,
        out_path=out_path,
        dpi=dpi,
        theme=theme,
    ),
    "readme_proof_card": _readme_proof_card,
    "reviewer_proof_panel": _reviewer_proof_panel,
}


def render_plot_preset(
    manifest_or_path: BundleManifest | str | Path,
    *,
    preset: str,
    out_root: str | Path | None = None,
    dpi: int | None = None,
) -> list[Path]:
    manifest = _as_manifest(manifest_or_path)
    config = get_visual_build_config(manifest.profile)
    target_root = Path(out_root) if out_root is not None else config.readme_assets_dir
    target_dir = Path(target_root) / preset
    render_dpi = int(config.dpi if dpi is None else dpi)

    written: list[Path] = []
    for spec in build_plot_specs(preset=preset):
        datasets = {
            name: load_bundle_dataframe(manifest, name) for name in spec.datasets
        }
        renderer = RENDERERS[spec.renderer]
        variants = themed_asset_paths(target_dir / spec.filename)
        for theme in PUBLISHING_THEMES:
            path = renderer(
                datasets,
                spec=spec,
                out_path=variants.path_for(theme),
                dpi=render_dpi,
                theme=theme,
            )
            written.append(Path(path))
        written.append(copy_light_variant(variants))
    return written


def render_plot_presets(
    manifest_or_path: BundleManifest | str | Path,
    *,
    presets: list[str] | tuple[str, ...] | None = None,
    out_root: str | Path | None = None,
    dpi: int | None = None,
) -> list[Path]:
    manifest = _as_manifest(manifest_or_path)
    names = list(PRESET_SPECS) if presets is None else [str(name) for name in presets]
    written: list[Path] = []
    for name in names:
        written.extend(
            render_plot_preset(
                manifest,
                preset=name,
                out_root=out_root,
                dpi=dpi,
            )
        )
    return written
