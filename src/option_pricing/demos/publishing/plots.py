from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from option_pricing.viz.publishing import (
    PUBLISHING_THEMES,
    SVG_TEXT_FONT_STACK,
    SvgTextStyle,
    copy_light_variant,
    publishing_palette,
    publishing_style,
    render_svg_contained_raster,
    render_svg_text_block,
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
            filename="surface_repair_signature_composite.png",
            renderer="surface_repair_signature",
            datasets=(
                "surface/quote_surface_compare",
                "surface/svi_repaired_grid",
                "calibration/svi_fit_compare",
            ),
            title="Surface Repair Review Object",
            kwargs={},
        ),
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
            filename="essvi_handoff_signature_composite.png",
            renderer="essvi_handoff_signature",
            datasets=(
                "surface/essvi_smoothed_grid",
                "calibration/essvi_time_smoothness",
                "calibration/essvi_projection_summary",
            ),
            title="eSSVI Handoff Review Object",
            kwargs={},
        ),
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
            filename="homepage_essvi_surface_3d.png",
            renderer="surface_3d",
            datasets=("surface/essvi_smoothed_grid",),
            title="Smoothed eSSVI Surface",
            kwargs={
                "x_col": "moneyness",
                "value_col": "iv",
                "cmap": "viridis",
                "elev": 29.0,
                "azim": -58.0,
            },
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


def _choose_nearest_maturities(
    T_vals: np.ndarray,
    targets: tuple[float, ...],
) -> np.ndarray:
    Ts = np.asarray(sorted(set(np.asarray(T_vals, dtype=float))), dtype=float)
    picked: list[float] = []
    for target in targets:
        idx = int(np.argmin(np.abs(Ts - float(target))))
        candidate = float(Ts[idx])
        if not any(np.isclose(candidate, existing) for existing in picked):
            picked.append(candidate)
    return np.asarray(picked, dtype=float)


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


def _surface_3d(
    df: pd.DataFrame,
    *,
    spec: PlotSpec,
    out_path: Path,
    dpi: int,
    theme: str,
) -> Path:
    x_col = str(spec.kwargs.get("x_col", "moneyness"))
    value_col = str(spec.kwargs.get("value_col", "iv"))
    cmap = str(spec.kwargs.get("cmap", "viridis"))
    elev = float(spec.kwargs.get("elev", 28.0))
    azim = float(spec.kwargs.get("azim", -60.0))
    x_vals, T_vals, Z = _pivot_grid(df, x=x_col, y="T", value=value_col)
    if Z.size == 0:
        raise ValueError(f"{spec.filename}: empty grid")

    X, Y = np.meshgrid(x_vals, T_vals)
    z_min = float(np.nanmin(Z))
    z_max = float(np.nanmax(Z))
    z_span = max(z_max - z_min, 1e-6)
    palette = publishing_palette(theme)
    pane_fill = (
        (0.09, 0.14, 0.22, 0.82) if theme == "dark" else (0.97, 0.985, 1.0, 0.96)
    )

    with publishing_style(theme=theme) as plt:
        fig = plt.figure(figsize=(8.8, 6.4))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            X,
            Y,
            Z,
            cmap=cmap,
            linewidth=0.2,
            antialiased=True,
            shade=True,
            rcount=min(Z.shape[0], 64),
            ccount=min(Z.shape[1], 96),
        )
        ax.set_title(spec.title)
        ax.set_xlabel("Moneyness K/F" if x_col == "moneyness" else "Log-moneyness y")
        ax.set_ylabel("Maturity T")
        ax.set_zlabel(value_col.upper())
        ax.set_zlim(z_min - 0.02 * z_span, z_max + 0.04 * z_span)
        ax.set_box_aspect((1.35, 1.0, 0.58))
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(colors=palette["text"], pad=1, labelsize=8)

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor(pane_fill)
            axis.pane.set_edgecolor(palette["spine"])
            axis._axinfo["grid"]["color"] = palette["grid"]
            axis._axinfo["grid"]["linewidth"] = 0.8

        fig.subplots_adjust(left=0.0, right=0.98, bottom=0.04, top=0.94)
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


def _surface_repair_signature(
    datasets: dict[str, pd.DataFrame],
    *,
    spec: PlotSpec,
    out_path: Path,
    dpi: int,
    theme: str,
) -> Path:
    quotes = datasets["surface/quote_surface_compare"].copy()
    svi = datasets["surface/svi_repaired_grid"].copy()
    fit = datasets["calibration/svi_fit_compare"].copy()

    fit["T"] = pd.to_numeric(fit["T"], errors="coerce")
    fit["diag_ok_fx"] = fit["diag_ok_fx"].map(
        lambda value: (
            bool(value)
            if isinstance(value, (bool, np.bool_))
            else str(value).strip().lower() == "true"
        )
    )
    fit["min_g_fx"] = pd.to_numeric(fit["min_g_fx"], errors="coerce")
    fit["failure_reason_fx"] = fit["failure_reason_fx"].fillna("").astype(str)
    fit = fit.sort_values("T")

    quotes["T"] = pd.to_numeric(quotes["T"], errors="coerce")
    quotes["moneyness"] = pd.to_numeric(quotes["moneyness"], errors="coerce")
    quotes["y"] = pd.to_numeric(quotes["y"], errors="coerce")
    quotes["iv_obs"] = pd.to_numeric(quotes["iv_obs"], errors="coerce")
    quotes["iv_svi"] = pd.to_numeric(quotes["iv_svi"], errors="coerce")
    quotes["iv_resid_svi_bp"] = pd.to_numeric(
        quotes["iv_resid_svi_bp"],
        errors="coerce",
    )
    quotes = quotes.merge(fit[["T", "diag_ok_fx"]], on="T", how="left")
    quotes["diag_ok_fx"] = quotes["diag_ok_fx"].fillna(True).astype(bool)

    maturity_summary = (
        quotes.groupby("T", as_index=False)
        .agg(
            mae_bp=("iv_resid_svi_bp", lambda s: float(np.mean(np.abs(s)))),
            max_bp=("iv_resid_svi_bp", lambda s: float(np.max(np.abs(s)))),
        )
        .merge(
            fit[["T", "diag_ok_fx", "min_g_fx", "failure_reason_fx"]],
            on="T",
            how="left",
        )
        .sort_values("T")
    )
    maturity_summary["diag_ok_fx"] = (
        maturity_summary["diag_ok_fx"].fillna(True).astype(bool)
    )
    flagged_T = maturity_summary.loc[~maturity_summary["diag_ok_fx"], "T"].to_numpy(
        dtype=float
    )

    if flagged_T.size == 0:
        flagged_T = maturity_summary.nlargest(3, "mae_bp")["T"].to_numpy(dtype=float)

    x_vals, T_vals, Z = _pivot_grid(svi, x="moneyness", y="T", value="iv")
    if Z.size == 0:
        raise ValueError(f"{spec.filename}: empty repaired surface grid")

    X, Y = np.meshgrid(x_vals, T_vals)
    z_min = float(np.nanmin(Z))
    z_max = float(np.nanmax(Z))
    z_span = max(z_max - z_min, 1e-6)
    palette = publishing_palette(theme)

    if theme == "dark":
        colors = {
            "bar_pass": "#91E0D7",
            "bar_flag": "#F59E0B",
            "slice_colors": ("#F59E0B", "#F472B6", "#8CC9FF"),
            "pane_fill": (0.09, 0.14, 0.22, 0.82),
        }
    else:
        colors = {
            "bar_pass": "#0F766E",
            "bar_flag": "#C2410C",
            "slice_colors": ("#C2410C", "#8B3A86", "#0B5CAB"),
            "pane_fill": (0.97, 0.985, 1.0, 0.96),
        }

    with publishing_style(theme=theme) as plt:
        fig = plt.figure(figsize=(12.6, 7.2), constrained_layout=True)
        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=(1.55, 1.0),
            height_ratios=(0.9, 1.1),
        )
        ax_surface = fig.add_subplot(grid[:, 0], projection="3d")
        ax_residuals = fig.add_subplot(grid[0, 1])
        ax_slices = fig.add_subplot(grid[1, 1])

        ax_surface.plot_surface(
            X,
            Y,
            Z,
            cmap="viridis",
            linewidth=0.0,
            antialiased=True,
            shade=True,
            alpha=0.86,
            rcount=min(Z.shape[0], 64),
            ccount=min(Z.shape[1], 96),
        )
        ax_surface.set_xlabel("Moneyness K/F")
        ax_surface.set_ylabel("Maturity T")
        ax_surface.set_zlabel("IV")
        ax_surface.set_zlim(z_min - 0.02 * z_span, z_max + 0.04 * z_span)
        ax_surface.set_box_aspect((1.35, 1.0, 0.62))
        ax_surface.view_init(elev=26.0, azim=-57.0)
        ax_surface.tick_params(colors=palette["text"], pad=1, labelsize=8)
        ax_surface.text2D(
            0.02,
            0.97,
            "The surface stays clean here; stressed quote checks stay at right.",
            transform=ax_surface.transAxes,
            color=palette["muted_text"],
            fontsize=8,
            va="top",
        )

        for axis in (ax_surface.xaxis, ax_surface.yaxis, ax_surface.zaxis):
            axis.pane.set_facecolor(colors["pane_fill"])
            axis.pane.set_edgecolor(palette["spine"])
            axis._axinfo["grid"]["color"] = palette["grid"]
            axis._axinfo["grid"]["linewidth"] = 0.8

        maturity_positions = np.arange(len(maturity_summary), dtype=float)
        mae_values = maturity_summary["mae_bp"].to_numpy(dtype=float)
        diag_ok_values = maturity_summary["diag_ok_fx"].to_numpy(dtype=bool)
        bar_colors = [
            colors["bar_pass"] if passed else colors["bar_flag"]
            for passed in diag_ok_values
        ]
        ax_residuals.bar(
            maturity_positions,
            mae_values,
            color=bar_colors,
            width=0.72,
        )
        ax_residuals.set_title("Per-expiry residuals with static checks")
        ax_residuals.set_ylabel("Mean abs IV residual (bp)")
        ax_residuals.set_xlabel("Maturity T")
        ax_residuals.set_xticks(maturity_positions)
        ax_residuals.set_xticklabels(
            [f"{float(T):g}" for T in maturity_summary["T"].to_numpy(dtype=float)],
            rotation=45,
            ha="right",
        )
        ax_residuals.grid(axis="y", alpha=0.28)
        ax_residuals.text(
            0.02,
            0.97,
            f"{len(flagged_T)}/{len(maturity_summary)} expiries flagged by the static g-floor.",
            transform=ax_residuals.transAxes,
            color=palette["muted_text"],
            fontsize=8,
            va="top",
        )
        for idx, (mae_bp, diag_ok) in enumerate(
            zip(mae_values, diag_ok_values, strict=False)
        ):
            if not diag_ok:
                ax_residuals.text(
                    idx,
                    float(mae_bp) + 0.9,
                    "g<0",
                    color=colors["bar_flag"],
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        ax_slices.set_title("Flagged slices stay inspectable")
        for color, T in zip(colors["slice_colors"], flagged_T[:3], strict=False):
            maturity_slice = quotes.loc[np.isclose(quotes["T"], float(T))].sort_values(
                "moneyness"
            )
            ax_slices.plot(
                maturity_slice["moneyness"],
                maturity_slice["iv_svi"],
                color=color,
                linewidth=2.0,
                label=f"T={float(T):g} repaired",
            )
            ax_slices.scatter(
                maturity_slice["moneyness"],
                maturity_slice["iv_obs"],
                s=14,
                color=color,
                alpha=0.56,
            )
        ax_slices.set_xlabel("Moneyness K/F")
        ax_slices.set_ylabel("IV")
        ax_slices.set_xlim(
            float(np.nanmin(quotes["moneyness"])),
            float(np.nanmax(quotes["moneyness"])),
        )
        ax_slices.text(
            0.02,
            0.96,
            "Observed points remain attached to the stressed maturities.",
            transform=ax_slices.transAxes,
            color=palette["muted_text"],
            fontsize=8,
            va="top",
        )
        ax_slices.legend(loc="upper right", fontsize=7)
        fig.suptitle(
            spec.title,
            x=0.065,
            y=0.985,
            ha="left",
            fontsize=14,
            fontweight="bold",
        )
        return save_figure(fig, out_path, dpi=dpi)


def _essvi_handoff_signature(
    datasets: dict[str, pd.DataFrame],
    *,
    spec: PlotSpec,
    out_path: Path,
    dpi: int,
    theme: str,
) -> Path:
    smoothed = datasets["surface/essvi_smoothed_grid"].copy()
    smoothness = datasets["calibration/essvi_time_smoothness"].copy()
    projection = datasets["calibration/essvi_projection_summary"].copy()

    smoothed["T"] = pd.to_numeric(smoothed["T"], errors="coerce")
    smoothed["y"] = pd.to_numeric(smoothed["y"], errors="coerce")
    smoothed["moneyness"] = pd.to_numeric(smoothed["moneyness"], errors="coerce")
    smoothed["iv"] = pd.to_numeric(smoothed["iv"], errors="coerce")
    smoothed["w_T"] = pd.to_numeric(smoothed["w_T"], errors="coerce")

    smoothness["T_knot"] = pd.to_numeric(smoothness["T_knot"], errors="coerce")
    smoothness["max_abs_wT_jump_svi"] = pd.to_numeric(
        smoothness["max_abs_wT_jump_svi"],
        errors="coerce",
    )
    smoothness["max_abs_wT_jump_smoothed"] = pd.to_numeric(
        smoothness["max_abs_wT_jump_smoothed"],
        errors="coerce",
    )
    smoothness = smoothness.sort_values("T_knot")

    projection_row = projection.iloc[0]
    price_rmse = float(pd.to_numeric(projection_row["price_rmse"], errors="coerce"))
    max_abs_price_error = float(
        pd.to_numeric(projection_row["max_abs_price_error"], errors="coerce")
    )
    projection_invalid_count = int(
        pd.to_numeric(
            projection_row["projection_dupire_invalid_count"],
            errors="coerce",
        )
    )

    x_vals, T_vals, Z = _pivot_grid(smoothed, x="moneyness", y="T", value="iv")
    if Z.size == 0:
        raise ValueError(f"{spec.filename}: empty smoothed surface grid")

    X, Y = np.meshgrid(x_vals, T_vals)
    z_min = float(np.nanmin(Z))
    z_max = float(np.nanmax(Z))
    z_span = max(z_max - z_min, 1e-6)
    seam_floor = float(
        np.nanmin(
            smoothness["max_abs_wT_jump_smoothed"].to_numpy(dtype=float),
        )
    )
    wt_slices = _choose_nearest_maturities(
        smoothed["T"].to_numpy(dtype=float),
        targets=(0.2, 0.5, 1.0, 1.5),
    )
    palette = publishing_palette(theme)

    if theme == "dark":
        colors = {
            "seam_svi": "#7DB5FF",
            "seam_smoothed": "#F59E0B",
            "wt_slices": ("#F59E0B", "#F472B6", "#8CC9FF", "#91E0D7"),
            "pane_fill": (0.09, 0.14, 0.22, 0.82),
        }
    else:
        colors = {
            "seam_svi": "#1D4ED8",
            "seam_smoothed": "#C2410C",
            "wt_slices": ("#C2410C", "#9D2E8C", "#0B5CAB", "#0F766E"),
            "pane_fill": (0.97, 0.985, 1.0, 0.96),
        }

    with publishing_style(theme=theme) as plt:
        fig = plt.figure(figsize=(12.7, 7.3), constrained_layout=True)
        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=(1.58, 1.0),
            height_ratios=(0.88, 1.12),
        )
        ax_surface = fig.add_subplot(grid[:, 0], projection="3d")
        ax_seams = fig.add_subplot(grid[0, 1])
        ax_wt = fig.add_subplot(grid[1, 1])

        ax_surface.plot_surface(
            X,
            Y,
            Z,
            cmap="viridis",
            linewidth=0.0,
            antialiased=True,
            shade=True,
            alpha=0.9,
            rcount=min(Z.shape[0], 64),
            ccount=min(Z.shape[1], 96),
        )
        ax_surface.set_xlabel("Moneyness K/F")
        ax_surface.set_ylabel("Maturity T")
        ax_surface.set_zlabel("IV")
        ax_surface.set_zlim(z_min - 0.02 * z_span, z_max + 0.04 * z_span)
        ax_surface.set_box_aspect((1.35, 1.0, 0.62))
        ax_surface.view_init(elev=27.0, azim=-58.0)
        ax_surface.tick_params(colors=palette["text"], pad=1, labelsize=8)
        ax_surface.text2D(
            0.02,
            0.97,
            "eSSVI makes the Dupire handoff analytic in maturity.",
            transform=ax_surface.transAxes,
            color=palette["muted_text"],
            fontsize=8,
            va="top",
        )

        for axis in (ax_surface.xaxis, ax_surface.yaxis, ax_surface.zaxis):
            axis.pane.set_facecolor(colors["pane_fill"])
            axis.pane.set_edgecolor(palette["spine"])
            axis._axinfo["grid"]["color"] = palette["grid"]
            axis._axinfo["grid"]["linewidth"] = 0.8

        seam_x = smoothness["T_knot"].to_numpy(dtype=float)
        seam_svi = smoothness["max_abs_wT_jump_svi"].to_numpy(dtype=float)
        seam_smoothed = smoothness["max_abs_wT_jump_smoothed"].to_numpy(dtype=float)
        ax_seams.plot(
            seam_x,
            seam_svi,
            marker="o",
            linewidth=2.0,
            color=colors["seam_svi"],
            label="SVI repaired",
        )
        ax_seams.plot(
            seam_x,
            seam_smoothed,
            marker="o",
            linewidth=2.0,
            color=colors["seam_smoothed"],
            label="eSSVI smoothed",
        )
        ax_seams.set_yscale("log")
        ax_seams.set_ylim(seam_floor * 0.7, float(np.nanmax(seam_svi)) * 1.3)
        ax_seams.set_title("Seam jumps collapse after projection")
        ax_seams.set_xlabel("Knot maturity")
        ax_seams.set_ylabel("Max |jump in w_T|")
        ax_seams.grid(axis="y", alpha=0.28, which="both")
        ax_seams.legend(loc="upper right", fontsize=7)
        ax_seams.text(
            0.02,
            0.97,
            "Worst knot: 8.07e-02 -> 8.17e-05",
            transform=ax_seams.transAxes,
            color=palette["muted_text"],
            fontsize=8,
            va="top",
        )
        ax_seams.text(
            0.02,
            0.88,
            f"Dupire invalid count after projection: {projection_invalid_count}",
            transform=ax_seams.transAxes,
            color=palette["muted_text"],
            fontsize=8,
            va="top",
        )

        ax_wt.set_title("Analytic w_T becomes inspectable")
        for color, maturity in zip(colors["wt_slices"], wt_slices, strict=False):
            maturity_slice = smoothed.loc[
                np.isclose(smoothed["T"], float(maturity))
            ].sort_values("y")
            ax_wt.plot(
                maturity_slice["y"],
                maturity_slice["w_T"],
                color=color,
                linewidth=2.0,
                label=f"T≈{float(maturity):.2f}",
            )
        ax_wt.set_xlabel("Log-moneyness y")
        ax_wt.set_ylabel("w_T")
        ax_wt.grid(alpha=0.25)
        ax_wt.legend(loc="upper left", fontsize=7)
        ax_wt.text(
            0.02,
            0.05,
            f"price_rmse={price_rmse:.5f}; max_abs_price_error={max_abs_price_error:.5f}",
            transform=ax_wt.transAxes,
            color=palette["muted_text"],
            fontsize=8,
            va="bottom",
        )

        fig.suptitle(
            spec.title,
            x=0.065,
            y=0.985,
            ha="left",
            fontsize=14,
            fontweight="bold",
        )
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

    seam_match = re.search(
        r"\| Worst seam jump \| `T = 0\.15`: `([^`]+) -> ([^`]+)` \|",
        essvi_text,
        flags=re.MULTILINE,
    )
    if seam_match is None:
        seam_match = _require_match(
            r"\| `T = 0\.15` \| `([^`]+)` \| `([^`]+)` \|",
            essvi_text,
            label="published seam-jump pair",
        )

    projection_tradeoff_match = re.search(
        r"\| Projection tradeoff \| `price_rmse = ([^`]+)`; `max_abs_price_error = ([^`]+)` \|",
        essvi_text,
        flags=re.MULTILINE,
    )
    projection_invalid_count: str
    if projection_tradeoff_match is None:
        projection_summary_match = _require_match(
            r"\| Projection summary \| `price_rmse = ([^`]+)` \| `max_abs_price_error = ([^`]+)` \| `projection_dupire_invalid_count = ([^`]+)` \|",
            essvi_text,
            label="projection summary",
        )
        projection_invalid_count = projection_summary_match.group(3)
    else:
        projection_invalid_match = _require_match(
            r"\| Dupire readiness \| `projection_dupire_invalid_count = ([^`]+)` \|",
            essvi_text,
            label="Dupire readiness summary",
        )
        projection_invalid_count = projection_invalid_match.group(1)

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
        "projection_invalid_count": projection_invalid_count,
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


_README_PROOF_CARD_TILE_WIDTH = 552.0
_README_PROOF_CARD_TILE_HEIGHT = 232.0


@dataclass(frozen=True)
class _ReadmeProofCardTile:
    key: str
    title: str
    x: float
    y: float
    body: str
    headline: str | None = None
    footer: str | None = None
    headline_fill_key: str = "accent"
    width: float = _README_PROOF_CARD_TILE_WIDTH
    height: float = _README_PROOF_CARD_TILE_HEIGHT


def _build_readme_proof_card_tiles(
    metrics: dict[str, str],
) -> tuple[_ReadmeProofCardTile, ...]:
    return (
        _ReadmeProofCardTile(
            key="surface-repair",
            title="Surface repair",
            x=64,
            y=220,
            body=(
                "Quoted versus repaired surfaces stay visible.\n"
                "No-arbitrage checks and per-expiry SVI residuals stay reviewable."
            ),
            footer="Open: Surface workflow and decision guide",
        ),
        _ReadmeProofCardTile(
            key="smooth-dupire-handoff",
            title="Smooth Dupire handoff",
            x=664,
            y=220,
            headline=f'{metrics["seam_svi"]} -> {metrics["seam_smoothed"]}',
            body=(
                "Worst published seam jump after smoothing.\n"
                f'Dupire invalid-count check stays at {metrics["projection_invalid_count"]}.'
            ),
        ),
        _ReadmeProofCardTile(
            key="local-vol-and-pde-validation",
            title="Local-vol and PDE validation",
            x=64,
            y=484,
            headline=f'{metrics["repriced_options"]} repricings',
            body=(
                f'Mean abs price error {metrics["mean_abs_price_error"]}.\n'
                f'Max abs IV error {metrics["max_abs_iv_error"]} on the published sweep.'
            ),
        ),
        _ReadmeProofCardTile(
            key="benchmarks-and-delivery",
            title="Benchmarks and delivery",
            x=664,
            y=484,
            headline=f'{metrics["iv_speedup"]} IV slice speedup',
            body=(
                f'Published at {metrics["iv_strikes"]} strikes from committed benchmark artifacts.\n'
                "README, docs visuals, and proof pages are regenerated and checked in CI."
            ),
        ),
    )


def _render_readme_proof_card_tile(
    tile: _ReadmeProofCardTile,
    *,
    colors: dict[str, str],
    font_stack: str,
) -> tuple[str, str]:
    content_x = tile.x + 32.0
    content_width = tile.width - 64.0
    clip_x = tile.x + 24.0
    clip_y = tile.y + 24.0
    clip_width = tile.width - 48.0
    clip_height = tile.height - 48.0

    title_style = SvgTextStyle(
        font_family=font_stack,
        font_size=28,
        font_weight="700",
        fill=colors["text"],
        line_height=32,
    )
    headline_style = SvgTextStyle(
        font_family=font_stack,
        font_size=34,
        font_weight="700",
        fill=colors[tile.headline_fill_key],
        line_height=38,
    )
    body_style = SvgTextStyle(
        font_family=font_stack,
        font_size=18,
        font_weight="400",
        fill=colors["muted"],
        line_height=22,
    )
    footer_style = SvgTextStyle(
        font_family=font_stack,
        font_size=18,
        font_weight="700",
        fill=colors["accent"],
        line_height=22,
    )

    blocks = [
        render_svg_text_block(
            block_id=f"readme-card-{tile.key}-title",
            text=tile.title,
            x=content_x,
            y=tile.y + 48,
            max_width=content_width,
            max_height=32,
            style=title_style,
            overflow_label=f"README proof card block {tile.key}/title",
        )
    ]
    if tile.headline is not None:
        blocks.append(
            render_svg_text_block(
                block_id=f"readme-card-{tile.key}-headline",
                text=tile.headline,
                x=content_x,
                y=tile.y + 90,
                max_width=content_width,
                max_height=38,
                style=headline_style,
                overflow_label=f"README proof card block {tile.key}/headline",
            )
        )
        blocks.append(
            render_svg_text_block(
                block_id=f"readme-card-{tile.key}-body",
                text=tile.body,
                x=content_x,
                y=tile.y + 128,
                max_width=content_width,
                max_height=92,
                style=body_style,
                overflow_label=f"README proof card block {tile.key}/body",
            )
        )
    else:
        blocks.append(
            render_svg_text_block(
                block_id=f"readme-card-{tile.key}-body",
                text=tile.body,
                x=content_x,
                y=tile.y + 88,
                max_width=content_width,
                max_height=80,
                style=body_style,
                overflow_label=f"README proof card block {tile.key}/body",
            )
        )
    if tile.footer is not None:
        blocks.append(
            render_svg_text_block(
                block_id=f"readme-card-{tile.key}-footer",
                text=tile.footer,
                x=content_x,
                y=tile.y + 182,
                max_width=content_width,
                max_height=22,
                style=footer_style,
                overflow_label=f"README proof card block {tile.key}/footer",
            )
        )

    clip_path = (
        f'    <clipPath id="readmeCardClip-{tile.key}">\n'
        f'      <rect x="{clip_x:g}" y="{clip_y:g}" width="{clip_width:g}" '
        f'height="{clip_height:g}" rx="18" ry="18" />\n'
        "    </clipPath>"
    )
    card_svg = "\n".join(
        [
            (
                f'  <rect x="{tile.x:g}" y="{tile.y:g}" width="{tile.width:g}" '
                f'height="{tile.height:g}" rx="24" ry="24" fill="{colors["card_bg"]}" '
                f'stroke="{colors["card_stroke"]}" stroke-width="2" />'
            ),
            (
                f'  <g id="readme-card-{tile.key}" '
                f'clip-path="url(#readmeCardClip-{tile.key})">'
            ),
            *[f"    {block}" for block in blocks],
            "  </g>",
        ]
    )
    return clip_path, card_svg


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

    subtitle_svg = render_svg_text_block(
        block_id="readme-card-intro-subtitle",
        text=(
            "Readable at GitHub README width, generated from the same published "
            "proof pages and benchmark artifacts as the docs."
        ),
        x=64,
        y=184,
        max_width=1152,
        max_height=48,
        style=SvgTextStyle(
            font_family=font_stack,
            font_size=20,
            font_weight="400",
            fill=colors["muted"],
            line_height=24,
        ),
        overflow_label="README proof card block intro/subtitle",
    )
    clip_paths: list[str] = []
    card_blocks: list[str] = []
    for tile in _build_readme_proof_card_tiles(metrics):
        clip_path, card_svg = _render_readme_proof_card_tile(
            tile,
            colors=colors,
            font_stack=font_stack,
        )
        clip_paths.append(clip_path)
        card_blocks.append(card_svg)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="760" viewBox="0 0 1280 760" role="img" aria-labelledby="title desc">
  <title id="title">README proof card for the option pricing library</title>
  <desc id="desc">Four readable proof tiles for README width covering surface repair, smooth Dupire handoff, local-vol and PDE validation, and benchmark plus delivery evidence. Metrics are loaded from the published proof pages and committed performance artifacts.</desc>
  <defs>
{chr(10).join(clip_paths)}
  </defs>

  <rect width="1280" height="760" fill="{colors["page_bg"]}" />
  <rect x="24" y="24" width="1232" height="712" rx="30" ry="30" fill="{colors["panel_bg"]}" stroke="{colors["panel_stroke"]}" stroke-width="2" />

  <rect x="64" y="66" width="170" height="34" rx="17" ry="17" fill="{colors["pill_bg"]}" />
  <text x="86" y="89" font-family="{font_stack}" font-size="16" font-weight="700" letter-spacing="1" fill="{colors["pill_text"]}">PROOF AT A GLANCE</text>
  <text x="64" y="148" font-family="{font_stack}" font-size="42" font-weight="700" fill="{colors["text"]}">What is already proven in this repo</text>
  {subtitle_svg}
{chr(10).join(card_blocks)}
</svg>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    return out_path


@dataclass(frozen=True)
class _ReviewerProofThumbnail:
    key: str
    x: float
    y: float
    width: float
    height: float
    chip_text: str
    chip_width: float
    image_path: Path


def _render_reviewer_proof_thumbnail(
    thumbnail: _ReviewerProofThumbnail,
    *,
    colors: dict[str, str],
    font_stack: str,
) -> tuple[str, str]:
    raster_block = render_svg_contained_raster(
        block_id=f"reviewerProofThumb-{thumbnail.key}",
        image_path=thumbnail.image_path,
        slot_x=thumbnail.x,
        slot_y=thumbnail.y,
        slot_width=thumbnail.width,
        slot_height=thumbnail.height,
        frame_radius=20,
        frame_fill=colors["surface_fill"],
        source_label=str(thumbnail.image_path.relative_to(ROOT)).replace("\\", "/"),
    )
    svg = "\n".join(
        [
            raster_block.svg,
            (
                f'  <rect x="{thumbnail.x + 16:g}" y="{thumbnail.y + 16:g}" '
                f'width="{thumbnail.chip_width:g}" height="30" rx="15" ry="15" '
                f'fill="{colors["chip_fill"]}" opacity="0.82" />'
            ),
            (
                f'  <text x="{thumbnail.x + 34:g}" y="{thumbnail.y + 37:g}" '
                f'font-family="{font_stack}" font-size="16" font-weight="700" '
                f'fill="{colors["chip_text"]}">{thumbnail.chip_text}</text>'
            ),
        ]
    )
    return raster_block.clip_path, svg


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
    quote_compare_path = (
        ROOT
        / "docs"
        / "assets"
        / "generated"
        / "static"
        / f"quote_surface_compare.{image_suffix}.png"
    )
    smooth_surface_path = (
        ROOT
        / "docs"
        / "assets"
        / "generated"
        / "dupire"
        / f"essvi_smoothed_surface_heatmap.{image_suffix}.png"
    )
    poster_path = (
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

    thumbnails = (
        _ReviewerProofThumbnail(
            key="quote-compare",
            x=760,
            y=450,
            width=250,
            height=155,
            chip_text="Quote-fit comparison",
            chip_width=174,
            image_path=quote_compare_path,
        ),
        _ReviewerProofThumbnail(
            key="smoothed-surface",
            x=760,
            y=622,
            width=250,
            height=155,
            chip_text="Smoothed eSSVI surface",
            chip_width=198,
            image_path=smooth_surface_path,
        ),
        _ReviewerProofThumbnail(
            key="proof-collage",
            x=1030,
            y=450,
            width=464,
            height=327,
            chip_text="Local-vol and PDE proof collage",
            chip_width=248,
            image_path=poster_path,
        ),
    )
    thumbnail_clip_paths: list[str] = []
    thumbnail_blocks: list[str] = []
    for thumbnail in thumbnails:
        clip_path, thumbnail_svg = _render_reviewer_proof_thumbnail(
            thumbnail,
            colors=colors,
            font_stack=font_stack,
        )
        thumbnail_clip_paths.append(clip_path)
        thumbnail_blocks.append(thumbnail_svg)

    supporting_note_svg = render_svg_text_block(
        block_id="reviewer-proof-panel-supporting-note",
        text=(
            "Visual panel inlines tracked thumbnails so CI and docs screenshots do "
            "not depend on nested browser fetches."
        ),
        x=760,
        y=806,
        max_width=734,
        max_height=18,
        style=SvgTextStyle(
            font_family=font_stack,
            font_size=14,
            font_weight="400",
            fill=colors["muted"],
            line_height=18,
        ),
        overflow_label="Reviewer proof panel supporting note",
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900" role="img" aria-labelledby="title desc">
  <title id="title">Reviewer proof panel for the option pricing library</title>
  <desc id="desc">Workflow strip from static surface to eSSVI smoothing to local vol to PDE repricing, four published proof metrics, and three inlined thumbnails reused from tracked docs assets.</desc>
  <defs>
    <linearGradient id="headerBg" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="{colors["header_start"]}" />
      <stop offset="100%" stop-color="{colors["header_end"]}" />
    </linearGradient>
{chr(10).join(thumbnail_clip_paths)}
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

{chr(10).join(thumbnail_blocks)}
  {supporting_note_svg}
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
    "surface_3d": lambda data, *, spec, out_path, dpi, theme: _surface_3d(
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
    "surface_repair_signature": _surface_repair_signature,
    "essvi_handoff_signature": _essvi_handoff_signature,
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
