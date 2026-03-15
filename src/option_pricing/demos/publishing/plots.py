from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from option_pricing.viz.publishing import publishing_style, save_figure

from .bundle import load_bundle_dataframe
from .config import get_visual_build_config
from .types import BundleManifest, PlotSpec

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
    return np.unique(Ts[idx])


def _surface_heatmap(
    df: pd.DataFrame, *, spec: PlotSpec, out_path: Path, dpi: int
) -> Path:
    x_col = str(spec.kwargs.get("x_col", "y"))
    value_col = str(spec.kwargs.get("value_col", "iv"))
    cmap = str(spec.kwargs.get("cmap", "viridis"))
    banner = bool(spec.kwargs.get("banner", False))
    x_vals, T_vals, Z = _pivot_grid(df, x=x_col, y="T", value=value_col)
    if Z.size == 0:
        raise ValueError(f"{spec.filename}: empty grid")

    figsize = (14, 3.8) if banner else (8.4, 5.2)
    with publishing_style() as plt:
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
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=value_col)
        return save_figure(fig, out_path, dpi=dpi)


def _smile_slices(
    df: pd.DataFrame, *, spec: PlotSpec, out_path: Path, dpi: int
) -> Path:
    with publishing_style() as plt:
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
    df: pd.DataFrame, *, spec: PlotSpec, out_path: Path, dpi: int
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
    with publishing_style() as plt:
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
) -> Path:
    value_col = str(spec.kwargs.get("value_col", "sigma_loc"))
    cmap = str(spec.kwargs.get("cmap", "cividis"))
    clean = df.copy()
    clean.loc[clean["invalid"].astype(bool), value_col] = np.nan
    y_vals, T_vals, Z = _pivot_grid(clean, x="y", y="T", value=value_col)
    with publishing_style() as plt:
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
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=value_col)
        return save_figure(fig, out_path, dpi=dpi)


def _diff_heatmap(
    df: pd.DataFrame, *, spec: PlotSpec, out_path: Path, dpi: int
) -> Path:
    value_col = str(spec.kwargs.get("value_col", "diff_sigma_loc"))
    cmap = str(spec.kwargs.get("cmap", "coolwarm"))
    clean = df.copy()
    clean.loc[clean["invalid_union"].astype(bool), value_col] = np.nan
    K_vals, T_vals, Z = _pivot_grid(clean, x="K", y="T", value=value_col)
    vmax = float(np.nanmax(np.abs(Z))) if np.isfinite(Z).any() else 1.0
    with publishing_style() as plt:
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
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=value_col)
        return save_figure(fig, out_path, dpi=dpi)


def _repricing_scatter(
    df: pd.DataFrame, *, spec: PlotSpec, out_path: Path, dpi: int
) -> Path:
    x = df["target_price"].to_numpy(dtype=float)
    y = df["pde_price"].to_numpy(dtype=float)
    with publishing_style() as plt:
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        ax.scatter(x, y, s=12, alpha=0.85)
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        ax.plot([lo, hi], [lo, hi], color="#222222", linewidth=1.1)
        ax.set_title(spec.title)
        ax.set_xlabel("Target price")
        ax.set_ylabel("PDE price")
        ax.set_aspect("equal", adjustable="box")
        return save_figure(fig, out_path, dpi=dpi)


def _convergence(df: pd.DataFrame, *, spec: PlotSpec, out_path: Path, dpi: int) -> Path:
    data = df.sort_values("grid_points").copy()
    gp = data["grid_points"].to_numpy(dtype=float)
    pde = data["pde_price"].to_numpy(dtype=float)
    ref = float(pde[-1])
    err_self = np.abs(pde[:-1] - ref)
    gp_self = gp[:-1]
    with publishing_style() as plt:
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
) -> Path:
    K_vals, T_vals, Z = _pivot_grid(df, x="K", y="T", value="price_error")
    vmax = float(np.nanmax(np.abs(Z))) if np.isfinite(Z).any() else 1.0
    with publishing_style() as plt:
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
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="price_error")
        return save_figure(fig, out_path, dpi=dpi)


def _poster_composite(
    datasets: dict[str, pd.DataFrame],
    *,
    spec: PlotSpec,
    out_path: Path,
    dpi: int,
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

    with publishing_style() as plt:
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
        fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

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
        fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

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
        fig.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)

        x = repricing["target_price"].to_numpy(dtype=float)
        y = repricing["pde_price"].to_numpy(dtype=float)
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        axs[1, 1].scatter(x, y, s=10, alpha=0.85)
        axs[1, 1].plot([lo, hi], [lo, hi], color="#222222", linewidth=1.1)
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
) -> Path:
    svi = datasets["surface/svi_repaired_grid"]
    essvi = datasets["surface/essvi_smoothed_grid"]
    lv = datasets["localvol/gatheral_grid"]
    x0, T0, Z0 = _pivot_grid(svi, x="y", y="T", value="iv")
    x1, T1, Z1 = _pivot_grid(essvi, x="y", y="T", value="iv")
    x2, T2, Z2 = _pivot_grid(
        lv.loc[~lv["invalid"].astype(bool)], x="y", y="T", value="sigma_loc"
    )

    with publishing_style() as plt:
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
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return save_figure(fig, out_path, dpi=dpi)


def _smoothness_compare(
    df: pd.DataFrame, *, spec: PlotSpec, out_path: Path, dpi: int
) -> Path:
    with publishing_style() as plt:
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


RENDERERS = {
    "surface_heatmap": lambda data, *, spec, out_path, dpi: _surface_heatmap(
        next(iter(data.values())),
        spec=spec,
        out_path=out_path,
        dpi=dpi,
    ),
    "smile_slices": lambda data, *, spec, out_path, dpi: _smile_slices(
        next(iter(data.values())),
        spec=spec,
        out_path=out_path,
        dpi=dpi,
    ),
    "quote_compare": lambda data, *, spec, out_path, dpi: _quote_compare(
        next(iter(data.values())),
        spec=spec,
        out_path=out_path,
        dpi=dpi,
    ),
    "localvol_heatmap": lambda data, *, spec, out_path, dpi: _localvol_heatmap(
        next(iter(data.values())),
        spec=spec,
        out_path=out_path,
        dpi=dpi,
    ),
    "diff_heatmap": lambda data, *, spec, out_path, dpi: _diff_heatmap(
        next(iter(data.values())),
        spec=spec,
        out_path=out_path,
        dpi=dpi,
    ),
    "repricing_scatter": lambda data, *, spec, out_path, dpi: _repricing_scatter(
        next(iter(data.values())),
        spec=spec,
        out_path=out_path,
        dpi=dpi,
    ),
    "convergence": lambda data, *, spec, out_path, dpi: _convergence(
        next(iter(data.values())),
        spec=spec,
        out_path=out_path,
        dpi=dpi,
    ),
    "price_error_heatmap": lambda data, *, spec, out_path, dpi: _price_error_heatmap(
        next(iter(data.values())),
        spec=spec,
        out_path=out_path,
        dpi=dpi,
    ),
    "poster_composite": _poster_composite,
    "story_triptych": _story_triptych,
    "smoothness_compare": lambda data, *, spec, out_path, dpi: _smoothness_compare(
        next(iter(data.values())),
        spec=spec,
        out_path=out_path,
        dpi=dpi,
    ),
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
        out_path = target_dir / spec.filename
        path = renderer(datasets, spec=spec, out_path=out_path, dpi=render_dpi)
        written.append(Path(path))
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
