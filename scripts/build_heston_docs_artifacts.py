from __future__ import annotations

# ruff: noqa: E402
import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from option_pricing.diagnostics.heston import (
    HestonMCComparisonCase,
    HestonMCSweepConfig,
    build_market_like_heston_quote_set,
    plot_heston_mc_bias_vs_timestep,
    plot_heston_mc_runtime_vs_error,
    run_heston_calibration_fit_diagnostics,
    run_heston_mc_comparison_sweep,
    run_heston_vs_local_vol_comparison,
    summarize_bias_vs_timestep,
    summarize_runtime_vs_error,
)
from option_pricing.models.heston.calibration import calibrate_heston_multistart
from option_pricing.models.heston.calibration.heston_types import HestonQuoteSet
from option_pricing.models.heston.params import HestonParams
from option_pricing.numerics.quadrature import QuadratureConfig
from option_pricing.types import MarketData, OptionType
from option_pricing.viz.publishing import (
    PUBLISHING_THEMES,
    copy_light_variant,
    publishing_palette,
    publishing_style,
    save_figure,
    style_colorbar,
    themed_asset_paths,
)
from option_pricing.vol.ssvi import ESSVICalibrationConfig, ESSVIProjectionConfig

CAVEAT = (
    "Synthetic deterministic quote fixture; not real market data. "
    "Heston is compared against eSSVI/local vol on the same target, "
    "not presented as universally superior."
)
SHORT_CAVEAT = "Synthetic deterministic target; not market data."
MODEL_CHOICE_CAVEAT = (
    "Synthetic target; model-choice evidence, not universal superiority."
)

MODEL_COLORS = {
    "Heston": "#0f4c5c",
    "ESSVI local-vol proxy": "#d17a22",
    "Direct local-vol PDE": "#4d7c0f",
}
MODEL_COLORS_DARK = {
    "Heston": "#8cc9ff",
    "ESSVI local-vol proxy": "#f3b562",
    "Direct local-vol PDE": "#91e0d7",
}
EXPIRY_COLORS = ("#0f4c5c", "#3d7ea6", "#d17a22", "#7a9e7e", "#b85c38")
EXPIRY_COLORS_DARK = ("#8cc9ff", "#b9dbff", "#f3b562", "#91e0d7", "#f59e9e")
PARAMETER_COLORS = {
    "kappa": "#0f4c5c",
    "vbar": "#d17a22",
    "eta": "#7a9e7e",
    "rho": "#8e5572",
    "v": "#4d7c0f",
}
PARAMETER_COLORS_DARK = {
    "kappa": "#8cc9ff",
    "vbar": "#f3b562",
    "eta": "#91e0d7",
    "rho": "#f4a8c7",
    "v": "#b4e197",
}
PARAMETER_LABELS = {
    "kappa": "kappa",
    "vbar": "theta / vbar",
    "eta": "sigma / eta",
    "rho": "rho",
    "v": "v0 / v",
}
DATA_TABLE_SPECS = {
    "heston_comparison_fit_errors.csv": "fit_errors",
    "heston_comparison_error_summary.csv": "error_summary",
    "heston_comparison_heldout.csv": "held_out_comparison",
    "heston_comparison_direct_local_vol_pde.csv": "direct_local_vol_pde",
    "heston_comparison_direct_pde_matched_error_summary.csv": (
        "direct_local_vol_pde_matched_error_summary"
    ),
    "heston_comparison_tradeoff_summary.csv": "tradeoff_summary",
}
HESTON_ARTIFACT_FRESHNESS_MANIFEST = "heston_artifact_freshness.json"
HESTON_ARTIFACT_FRESHNESS_VERSION = 1
HESTON_ARTIFACT_REBUILD_COMMAND = (
    "python scripts/build_heston_docs_artifacts.py --profile release"
)
HESTON_ARTIFACT_SOURCE_INPUTS = (
    "scripts/build_heston_docs_artifacts.py",
    "src/option_pricing/diagnostics/heston/fixtures.py",
    "src/option_pricing/diagnostics/heston/calibration_fit.py",
    "src/option_pricing/diagnostics/heston/model_comparison.py",
    "src/option_pricing/diagnostics/heston/monte_carlo.py",
    "src/option_pricing/diagnostics/heston/plot.py",
    "src/option_pricing/models/heston/calibration/calibrate.py",
    "src/option_pricing/pricers/heston.py",
)


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    name: str
    expiries: tuple[float, ...]
    log_moneyness: tuple[float, ...]
    quad_cfg: QuadratureConfig
    calibration_max_seeds: int
    calibration_max_nfev: int
    objective_slice_grid_size: int
    essvi_max_nfev: int
    projection_cfg: ESSVIProjectionConfig
    direct_local_vol_pde_max_quotes: int
    direct_local_vol_pde_Nx: int
    direct_local_vol_pde_Nt: int
    mc_case_market: MarketData
    mc_case_params: HestonParams
    mc_case_tau: float
    mc_case_strike: float
    mc_n_steps_grid: tuple[int, ...]
    mc_n_paths: int
    mc_repeats: int
    mc_seed: int


@dataclass(frozen=True, slots=True)
class DiagnosticsBundle:
    profile: ProfileConfig
    seed: int
    quotes: HestonQuoteSet
    held_out_mask: np.ndarray
    held_out_indices: tuple[int, ...]
    held_out_labels: tuple[str, ...]
    calibration: Any
    comparison: Any
    mc_sweep: pd.DataFrame
    mc_bias_summary: pd.DataFrame
    mc_runtime_summary: pd.DataFrame
    mc_convergence_summary: pd.DataFrame
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ThemeSpec:
    name: str
    palette: dict[str, str]
    model_colors: dict[str, str]
    expiry_colors: tuple[str, ...]
    parameter_colors: dict[str, str]
    target_iv_color: str
    held_out_color: str
    annotation_face: str
    annotation_edge: str
    annotation_text: str
    cell_edge: str
    best_run_color: str


PROFILE_DEFAULTS: dict[str, ProfileConfig] = {
    "smoke": ProfileConfig(
        name="smoke",
        expiries=(0.5, 1.0, 2.0),
        log_moneyness=(-0.10, 0.0, 0.10),
        quad_cfg=QuadratureConfig(u_max=45.0, n_panels=5, nodes_per_panel=5),
        calibration_max_seeds=4,
        calibration_max_nfev=120,
        objective_slice_grid_size=2,
        essvi_max_nfev=300,
        projection_cfg=ESSVIProjectionConfig(
            validation_nt=9,
            validation_y_min=-0.40,
            validation_y_max=0.40,
            validation_ny=21,
            dupire_nt=7,
            dupire_y_min=-0.35,
            dupire_y_max=0.35,
            dupire_ny=17,
            strict_validation=False,
        ),
        direct_local_vol_pde_max_quotes=3,
        direct_local_vol_pde_Nx=31,
        direct_local_vol_pde_Nt=41,
        mc_case_market=MarketData(spot=100.0, rate=0.02, dividend_yield=0.0),
        mc_case_params=HestonParams(
            kappa=1.0,
            vbar=0.04,
            eta=1.25,
            rho=-0.90,
            v=0.08,
        ),
        mc_case_tau=2.0,
        mc_case_strike=100.0,
        mc_n_steps_grid=(4, 8, 16, 32),
        mc_n_paths=512,
        mc_repeats=2,
        mc_seed=30,
    ),
    "release": ProfileConfig(
        name="release",
        expiries=(0.25, 0.50, 1.00, 1.50, 2.00),
        log_moneyness=(-0.18, -0.12, -0.06, 0.0, 0.06, 0.12, 0.18),
        quad_cfg=QuadratureConfig(u_max=90.0, n_panels=12, nodes_per_panel=12),
        calibration_max_seeds=8,
        calibration_max_nfev=300,
        objective_slice_grid_size=3,
        essvi_max_nfev=1200,
        projection_cfg=ESSVIProjectionConfig(
            validation_nt=15,
            validation_y_min=-0.50,
            validation_y_max=0.50,
            validation_ny=31,
            dupire_nt=11,
            dupire_y_min=-0.45,
            dupire_y_max=0.45,
            dupire_ny=25,
            strict_validation=False,
        ),
        direct_local_vol_pde_max_quotes=9,
        direct_local_vol_pde_Nx=81,
        direct_local_vol_pde_Nt=121,
        mc_case_market=MarketData(spot=100.0, rate=0.02, dividend_yield=0.0),
        mc_case_params=HestonParams(
            kappa=1.0,
            vbar=0.04,
            eta=1.25,
            rho=-0.90,
            v=0.08,
        ),
        mc_case_tau=2.0,
        mc_case_strike=100.0,
        mc_n_steps_grid=(4, 8, 16, 32, 64),
        mc_n_paths=4096,
        mc_repeats=3,
        mc_seed=30,
    ),
}

CAPTIONS = {
    "heston_comparison_summary_card": (
        "Capstone 3 model-comparison summary. A deterministic synthetic quote "
        "target is fitted with Heston, checked for calibration stability, and "
        "compared against the existing eSSVI/local-vol stack. This is diagnostic "
        "evidence, not a claim that Heston is universally superior."
    ),
    "heston_model_comparison_smile_overlay": (
        "Heston and eSSVI/local-vol are compared on the same deterministic quote "
        "target. The plot highlights model-purpose tradeoffs: Heston provides "
        "stochastic-variance dynamics, while eSSVI/local vol can flexibly fit "
        "vanilla surface geometry."
    ),
    "heston_iv_residual_heatmap": (
        "Heston IV residuals by expiry and log-moneyness. Structured residuals "
        "are part of the diagnostic story: the model is interpretable, but not an "
        "unconstrained surface fitter."
    ),
    "heston_model_comparison_error_buckets": (
        "Error buckets summarize model fit by moneyness region. Heston and "
        "eSSVI full-set rows are retained; direct PDE rows are selected "
        "Dupire/PDE audit quotes, so proxy-vs-direct-PDE comparisons should "
        "use the matched-subset CSV."
    ),
    "heston_multistart_stability_panel": (
        "Heston calibration can be weakly identifiable. Multistart diagnostics "
        "make optimizer dependence and parameter stability visible before "
        "interpreting a fit."
    ),
    "heston_mc_vs_fourier_convergence": (
        "Heston Monte Carlo is checked against semi-analytic Fourier pricing. "
        "QE and Euler behavior are shown as numerical validation evidence, not as "
        "universal runtime claims."
    ),
    "heston_train_vs_heldout_comparison": (
        "Train/held-out comparison helps separate calibration fit from "
        "out-of-sample diagnostic behavior on the deterministic quote grid."
    ),
    "heston_workflow_architecture": (
        "Capstone 3 uses the Capstone 2 surface/local-vol/PDE stack as a "
        "comparison baseline for Heston stochastic-volatility calibration and "
        "validation."
    ),
}

DOCS_PAGES = {
    "heston_comparison_summary_card": [
        "README.template.md",
        "README.md",
        "docs/index.md",
        "docs/user_guides/heston_model_comparison.md",
    ],
    "heston_model_comparison_smile_overlay": [
        "docs/user_guides/heston_model_comparison.md",
    ],
    "heston_iv_residual_heatmap": [
        "docs/user_guides/heston_model_comparison.md",
        "docs/validation_matrix.md",
    ],
    "heston_model_comparison_error_buckets": [
        "docs/user_guides/heston_model_comparison.md",
        "docs/validation_matrix.md",
    ],
    "heston_multistart_stability_panel": [
        "docs/user_guides/heston_model_comparison.md",
        "docs/validation_matrix.md",
    ],
    "heston_mc_vs_fourier_convergence": [
        "docs/performance.md",
        "docs/user_guides/heston_model_comparison.md",
        "docs/validation_matrix.md",
    ],
    "heston_train_vs_heldout_comparison": [
        "docs/user_guides/heston_model_comparison.md",
    ],
    "heston_workflow_architecture": [
        "docs/user_guides/heston_model_comparison.md",
    ],
    "heston_comparison_fit_errors.csv": [
        "docs/user_guides/heston_model_comparison.md",
    ],
    "heston_comparison_error_summary.csv": [
        "docs/user_guides/heston_model_comparison.md",
        "docs/validation_matrix.md",
    ],
    "heston_comparison_heldout.csv": [
        "docs/user_guides/heston_model_comparison.md",
    ],
    "heston_comparison_direct_local_vol_pde.csv": [
        "docs/user_guides/heston_model_comparison.md",
    ],
    "heston_comparison_direct_pde_matched_error_summary.csv": [
        "docs/user_guides/heston_model_comparison.md",
        "docs/validation_matrix.md",
    ],
    "heston_comparison_tradeoff_summary.csv": [
        "docs/user_guides/heston_model_comparison.md",
    ],
    "heston_mc_convergence_summary.csv": [
        "docs/performance.md",
        "docs/validation_matrix.md",
    ],
    "heston_artifact_manifest.json": [
        "docs/user_guides/heston_model_comparison.md",
        "docs/validation_matrix.md",
    ],
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reproducible Heston docs comparison artifacts."
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_DEFAULTS),
        default="release",
        help="Artifact quality/runtime profile.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "assets" / "generated" / "heston",
        help="Root directory for generated Heston docs assets.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="svg,png",
        help="Comma-separated figure formats to write (for example svg,png).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the deterministic seed used for the MC validation case.",
    )
    parser.add_argument(
        "--skip-direct-pde",
        action="store_true",
        help="Skip the direct local-vol PDE subset repricing audit.",
    )
    return parser.parse_args(argv)


def _parse_formats(raw: str) -> tuple[str, ...]:
    formats: list[str] = []
    for item in raw.split(","):
        fmt = item.strip().lower()
        if not fmt:
            continue
        if fmt not in {"svg", "png"}:
            raise ValueError(f"Unsupported format: {fmt!r}")
        if fmt not in formats:
            formats.append(fmt)
    if not formats:
        raise ValueError("At least one figure format is required.")
    return tuple(formats)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfe",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#0f172a",
            "axes.titleweight": "semibold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "font.size": 10,
            "grid.color": "#e2e8f0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )


def _theme_spec(theme: str) -> ThemeSpec:
    normalized = str(theme).strip().lower()
    palette = publishing_palette(normalized)
    if normalized == "dark":
        return ThemeSpec(
            name=normalized,
            palette=palette,
            model_colors=MODEL_COLORS_DARK,
            expiry_colors=EXPIRY_COLORS_DARK,
            parameter_colors=PARAMETER_COLORS_DARK,
            target_iv_color=palette["reference"],
            held_out_color="#f87171",
            annotation_face="#162033",
            annotation_edge="#314056",
            annotation_text=palette["text"],
            cell_edge="#23344a",
            best_run_color="#ffd166",
        )
    return ThemeSpec(
        name=normalized,
        palette=palette,
        model_colors=MODEL_COLORS,
        expiry_colors=EXPIRY_COLORS,
        parameter_colors=PARAMETER_COLORS,
        target_iv_color="#111827",
        held_out_color="#c81d25",
        annotation_face="#ffffff",
        annotation_edge="#cbd5e1",
        annotation_text="#0f172a",
        cell_edge="#d7e0ea",
        best_run_color="#f59e0b",
    )


def _annotation_box(spec: ThemeSpec, *, alpha: float = 0.94) -> dict[str, Any]:
    return {
        "facecolor": spec.annotation_face,
        "edgecolor": spec.annotation_edge,
        "alpha": alpha,
        "boxstyle": "round,pad=0.35",
    }


def _axis_edges(values: np.ndarray) -> np.ndarray:
    coords = np.asarray(values, dtype=np.float64)
    if coords.ndim != 1 or coords.size == 0:
        raise ValueError("Heatmap coordinates must be a non-empty 1D array.")
    if coords.size == 1:
        return np.array([coords[0] - 0.5, coords[0] + 0.5], dtype=np.float64)
    deltas = np.diff(coords)
    left = coords[0] - deltas[0] / 2.0
    right = coords[-1] + deltas[-1] / 2.0
    mids = coords[:-1] + deltas / 2.0
    return np.concatenate(([left], mids, [right])).astype(np.float64)


def _artifact_key(filename: str) -> str:
    stem = Path(filename).stem
    if stem.endswith(".light"):
        return stem.removesuffix(".light")
    if stem.endswith(".dark"):
        return stem.removesuffix(".dark")
    return stem


def _artifact_theme_variant(filename: str) -> str | None:
    stem = Path(filename).stem
    if stem.endswith(".light"):
        return "light"
    if stem.endswith(".dark"):
        return "dark"
    return None


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _relative_to_dir(path: Path, directory: Path) -> str:
    return str(path.relative_to(directory)).replace("\\", "/")


def _normalize_source_newlines(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _file_sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in {".csv", ".json", ".py", ".svg", ".txt"}:
        data = _normalize_source_newlines(data)
    return hashlib.sha256(data).hexdigest()


def _heston_artifact_source_payload() -> dict[str, str]:
    return {
        relative_path: _file_sha256(ROOT / relative_path)
        for relative_path in HESTON_ARTIFACT_SOURCE_INPUTS
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            cwd=ROOT,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def _selected_expiries(expiries: pd.Series | np.ndarray, *, count: int) -> list[float]:
    unique = sorted({float(value) for value in np.asarray(expiries, dtype=np.float64)})
    if len(unique) <= count:
        return unique
    positions = np.unique(np.linspace(0, len(unique) - 1, count, dtype=int))
    return [unique[int(position)] for position in positions]


def _quote_label(quotes: HestonQuoteSet, index: int) -> str:
    if quotes.labels is not None:
        return str(quotes.labels[index])
    return f"quote_{index}"


def _build_deterministic_held_out_mask(
    quotes: HestonQuoteSet,
) -> tuple[np.ndarray, tuple[int, ...], tuple[str, ...]]:
    expiry_targets = _selected_expiries(
        quotes.expiry, count=min(3, len(np.unique(quotes.expiry)))
    )
    log_moneyness = np.asarray(quotes.log_moneyness, dtype=np.float64)
    target_log_m = (-0.10, 0.0, 0.10)

    selected: list[int] = []
    used: set[int] = set()
    for expiry, target in zip(expiry_targets, target_log_m, strict=False):
        expiry_rows = np.flatnonzero(np.isclose(quotes.expiry, float(expiry)))
        ordered = sorted(
            (int(row) for row in expiry_rows if int(row) not in used),
            key=lambda row: (abs(float(log_moneyness[row]) - float(target)), row),
        )
        if not ordered:
            continue
        chosen = ordered[0]
        selected.append(chosen)
        used.add(chosen)

    mask = np.zeros(quotes.n_quotes, dtype=np.bool_)
    if selected:
        mask[np.asarray(selected, dtype=np.int64)] = True
    labels = tuple(_quote_label(quotes, index) for index in selected)
    return mask, tuple(selected), labels


def _subset_quotes(quotes: HestonQuoteSet, keep_mask: np.ndarray) -> HestonQuoteSet:
    if keep_mask.shape != (quotes.n_quotes,):
        raise ValueError(f"keep_mask must have shape ({quotes.n_quotes},).")
    metadata = dict(quotes.metadata or {})
    metadata["subset_label"] = "heston_docs_artifacts_train_subset"
    labels = (
        None
        if quotes.labels is None
        else tuple(
            label
            for label, keep in zip(quotes.labels, keep_mask, strict=True)
            if bool(keep)
        )
    )
    return HestonQuoteSet(
        ctx=quotes.ctx,
        strike=np.asarray(quotes.strike[keep_mask], dtype=np.float64),
        expiry=np.asarray(quotes.expiry[keep_mask], dtype=np.float64),
        is_call=np.asarray(quotes.is_call[keep_mask], dtype=np.bool_),
        mid=np.asarray(quotes.mid[keep_mask], dtype=np.float64),
        bs_vega=(
            None
            if quotes.bs_vega is None
            else np.asarray(quotes.bs_vega[keep_mask], dtype=np.float64)
        ),
        sqrt_weights=(
            None
            if quotes.sqrt_weights is None
            else np.asarray(quotes.sqrt_weights[keep_mask], dtype=np.float64)
        ),
        bid=(
            None
            if quotes.bid is None
            else np.asarray(quotes.bid[keep_mask], dtype=np.float64)
        ),
        ask=(
            None
            if quotes.ask is None
            else np.asarray(quotes.ask[keep_mask], dtype=np.float64)
        ),
        iv_mid=(
            None
            if quotes.iv_mid is None
            else np.asarray(quotes.iv_mid[keep_mask], dtype=np.float64)
        ),
        labels=labels,
        metadata=metadata,
    )


def _merge_mc_summaries(
    bias_summary: pd.DataFrame,
    runtime_summary: pd.DataFrame,
) -> pd.DataFrame:
    merged = bias_summary.merge(
        runtime_summary,
        on=["scheme", "n_steps", "dt"],
        how="outer",
        suffixes=("", "_runtime"),
    )
    return merged.sort_values(["scheme", "n_steps"], kind="stable").reset_index(
        drop=True
    )


def _build_diagnostics(
    profile: ProfileConfig,
    *,
    seed_override: int | None,
    skip_direct_pde: bool,
) -> DiagnosticsBundle:
    quotes = build_market_like_heston_quote_set(
        expiries=np.asarray(profile.expiries, dtype=np.float64),
        log_moneyness=np.asarray(profile.log_moneyness, dtype=np.float64),
    )
    held_out_mask, held_out_indices, held_out_labels = (
        _build_deterministic_held_out_mask(quotes)
    )
    if not np.any(~held_out_mask):
        raise RuntimeError("Held-out policy left no training quotes for Heston fit.")

    training_quotes = _subset_quotes(quotes, ~held_out_mask)
    fit = calibrate_heston_multistart(
        quotes=training_quotes,
        quad_cfg=profile.quad_cfg,
        max_seeds=profile.calibration_max_seeds,
        max_nfev=profile.calibration_max_nfev,
    )

    calibration = run_heston_calibration_fit_diagnostics(
        quotes=quotes,
        fit=fit,
        held_out_mask=held_out_mask,
        quad_cfg=profile.quad_cfg,
        objective_slice_grid_size=profile.objective_slice_grid_size,
        fit_used_filtered_quotes=True,
    )
    comparison = run_heston_vs_local_vol_comparison(
        quotes=quotes,
        heston_fit=fit,
        held_out_mask=held_out_mask,
        heston_quad_cfg=profile.quad_cfg,
        essvi_cfg=ESSVICalibrationConfig(max_nfev=profile.essvi_max_nfev),
        essvi_projection_cfg=profile.projection_cfg,
        run_direct_local_vol_pde=not skip_direct_pde,
        local_vol_pde_max_quotes=profile.direct_local_vol_pde_max_quotes,
        local_vol_pde_Nx=profile.direct_local_vol_pde_Nx,
        local_vol_pde_Nt=profile.direct_local_vol_pde_Nt,
    )

    mc_seed = profile.mc_seed if seed_override is None else int(seed_override)
    mc_case = HestonMCComparisonCase(
        ctx=profile.mc_case_market.to_context(),
        params=profile.mc_case_params,
        kind=OptionType.CALL,
        strike=float(profile.mc_case_strike),
        tau=float(profile.mc_case_tau),
    )
    mc_sweep = run_heston_mc_comparison_sweep(
        mc_case,
        HestonMCSweepConfig(
            schemes=("euler_full_truncation", "quadratic_exponential"),
            n_steps_grid=profile.mc_n_steps_grid,
            n_paths=profile.mc_n_paths,
            seed=mc_seed,
            antithetic=True,
            repeats=profile.mc_repeats,
            use_control_variate=False,
        ),
        quad_cfg=profile.quad_cfg,
    )
    mc_bias_summary = summarize_bias_vs_timestep(mc_sweep)
    mc_runtime_summary = summarize_runtime_vs_error(mc_sweep)
    mc_convergence_summary = _merge_mc_summaries(mc_bias_summary, mc_runtime_summary)

    warnings: list[str] = []
    direct_table = comparison.tables["direct_local_vol_pde"]
    if direct_table.empty:
        warnings.append(
            "Direct local-vol PDE audit produced no rows; the CSV is empty but retained."
        )
    elif "pde_status" in direct_table.columns:
        failed = direct_table.loc[direct_table["pde_status"].astype(str) != "ok"]
        if not failed.empty:
            warnings.append(
                "Direct local-vol PDE audit has failed rows; see "
                "heston_comparison_direct_local_vol_pde.csv for details."
            )

    return DiagnosticsBundle(
        profile=profile,
        seed=mc_seed,
        quotes=quotes,
        held_out_mask=held_out_mask,
        held_out_indices=held_out_indices,
        held_out_labels=held_out_labels,
        calibration=calibration,
        comparison=comparison,
        mc_sweep=mc_sweep,
        mc_bias_summary=mc_bias_summary,
        mc_runtime_summary=mc_runtime_summary,
        mc_convergence_summary=mc_convergence_summary,
        warnings=tuple(warnings),
    )


def _save_themed_figure(
    *,
    stem: str,
    out_dir: Path,
    formats: tuple[str, ...],
    render,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        variants = themed_asset_paths(out_dir / f"{stem}.{fmt}")
        for theme in PUBLISHING_THEMES:
            with publishing_style(theme):
                fig = render(theme)
            dpi = 280 if fmt == "png" else 160
            save_figure(fig, variants.path_for(theme), dpi=dpi)
            paths.append(variants.path_for(theme))
        copy_light_variant(variants)
        paths.append(variants.base)
    return paths


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _bucket_label(bucket: str) -> str:
    return {
        "all": "All",
        "atm": "ATM",
        "downside_wing": "Down wing",
        "upside_wing": "Up wing",
    }.get(bucket, bucket.replace("_", " ").title())


def _fit_errors_for_model(bundle: DiagnosticsBundle, model_name: str) -> pd.DataFrame:
    table = bundle.comparison.tables["fit_errors"].copy()
    return table.loc[table["model"].astype(str) == model_name].copy()


def _plot_summary_smile_panel(
    ax: Axes, bundle: DiagnosticsBundle, spec: ThemeSpec
) -> None:
    table = _fit_errors_for_model(bundle, "Heston")
    market = table.drop_duplicates("quote_index").sort_values(
        ["expiry", "log_moneyness"], kind="stable"
    )
    expiry_values = _selected_expiries(
        market["expiry"], count=min(3, len(np.unique(market["expiry"])))
    )

    for index, expiry in enumerate(expiry_values):
        color = spec.expiry_colors[index % len(spec.expiry_colors)]
        market_slice = market.loc[np.isclose(market["expiry"], float(expiry))].copy()
        model_slice = table.loc[np.isclose(table["expiry"], float(expiry))].copy()
        if market_slice.empty or model_slice.empty:
            continue
        ax.scatter(
            market_slice["log_moneyness"],
            market_slice["market_iv"],
            color=color,
            s=24,
            label=f"T={float(expiry):g}",
            zorder=3,
        )
        ax.plot(
            model_slice["log_moneyness"],
            model_slice["model_iv"],
            color=color,
            linewidth=2.0,
        )

        held_out = market_slice.loc[market_slice["is_held_out"].astype(bool)]
        if not held_out.empty:
            ax.scatter(
                held_out["log_moneyness"],
                held_out["market_iv"],
                s=54,
                facecolors="none",
                edgecolors=spec.held_out_color,
                linewidths=1.4,
                zorder=4,
            )

    ax.set_title("Heston vs target smile", loc="left")
    ax.set_xlabel("Log-moneyness")
    ax.set_ylabel("Implied vol")
    ax.grid(True, alpha=0.45)
    ax.legend(title="Expiry", ncol=3, fontsize=8, title_fontsize=8, loc="upper right")
    ax.text(
        0.02,
        0.02,
        "Markers = target IV\nLines = Heston fit",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
        color=spec.annotation_text,
        bbox=_annotation_box(spec),
    )


def _plot_residual_heatmap(
    ax: Axes, bundle: DiagnosticsBundle, spec: ThemeSpec
) -> None:
    table = _fit_errors_for_model(bundle, "Heston")
    pivot = table.pivot_table(
        index="expiry",
        columns="log_moneyness",
        values="iv_residual_bps",
        aggfunc="mean",
    ).sort_index()
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    values = pivot.to_numpy(dtype=np.float64)
    x_values = np.asarray(pivot.columns, dtype=np.float64)
    y_values = np.asarray(pivot.index, dtype=np.float64)
    x_edges = _axis_edges(x_values)
    y_edges = _axis_edges(y_values)
    max_abs = float(np.nanmax(np.abs(values))) if values.size else 0.0
    finite_scale = max(max_abs, 1.0)
    image = ax.pcolormesh(
        x_edges,
        y_edges,
        values,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-finite_scale, vcenter=0.0, vmax=finite_scale),
        edgecolors=spec.cell_edge,
        linewidth=0.9,
        shading="flat",
    )
    ax.set_title("IV residual heatmap", loc="left")
    ax.set_xlabel("Log-moneyness")
    ax.set_ylabel("Expiry")
    ax.set_xticks(
        x_values,
        [f"{float(value):+.2f}" for value in x_values],
        rotation=25,
        ha="right",
    )
    ax.set_yticks(y_values, [f"{float(value):g}" for value in y_values])
    ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    ax.set_ylim(float(y_edges[0]), float(y_edges[-1]))
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="bps")
    style_colorbar(colorbar, theme=spec.name)

    if values.size <= 40:
        for row_index, expiry in enumerate(y_values):
            for col_index, log_m in enumerate(x_values):
                value = float(values[row_index, col_index])
                if not np.isfinite(value):
                    continue
                text_color = (
                    spec.palette["text"]
                    if abs(value) < 0.6 * finite_scale
                    else "#ffffff"
                )
                ax.text(
                    float(log_m),
                    float(expiry),
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )

    summary = bundle.comparison.tables["error_summary"]
    heston_all = summary.loc[
        (summary["model"].astype(str) == "Heston")
        & (summary["bucket"].astype(str) == "all")
    ]
    rmse = (
        float(heston_all["iv_rmse_bps"].iloc[0])
        if not heston_all.empty
        else float("nan")
    )
    ax.text(
        0.02,
        0.98,
        f"IV RMSE {rmse:.1f} bps\nMax |res| {max_abs:.1f} bps",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color=spec.annotation_text,
        bbox=_annotation_box(spec),
    )


def _plot_error_buckets(ax: Axes, bundle: DiagnosticsBundle, spec: ThemeSpec) -> None:
    table = bundle.comparison.tables["error_summary"].copy()
    metric = "iv_rmse_bps"
    bucket_order = ["all", "atm", "downside_wing", "upside_wing"]
    model_order = [
        model
        for model in ["Heston", "ESSVI local-vol proxy", "Direct local-vol PDE"]
        if model in set(table["model"].astype(str))
    ]
    x = np.arange(len(bucket_order), dtype=np.float64)
    width = 0.78 / max(len(model_order), 1)

    for index, model_name in enumerate(model_order):
        rows = table.loc[table["model"].astype(str) == model_name]
        values: list[float] = []
        counts: list[int | None] = []
        for bucket in bucket_order:
            match = rows.loc[rows["bucket"].astype(str) == bucket]
            values.append(
                float(match[metric].iloc[0]) if not match.empty else float("nan")
            )
            counts.append(int(match["n_quotes"].iloc[0]) if not match.empty else None)
        offset = (index - (len(model_order) - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=model_name,
            color=spec.model_colors[model_name],
        )
        for bar, value, n_quotes in zip(bars, values, counts, strict=True):
            if n_quotes is None or not np.isfinite(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                max(float(value), 0.0),
                f"n={n_quotes}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7,
                color=spec.palette["muted_text"],
            )

    ax.set_title("Model comparison buckets", loc="left")
    ax.set_xticks(
        x, [_bucket_label(bucket) for bucket in bucket_order], rotation=18, ha="right"
    )
    ax.set_ylabel("IV RMSE (bps)")
    ax.grid(True, axis="y", alpha=0.45)
    ax.margins(y=0.20)
    ax.legend(fontsize=8, loc="upper left")
    if "Direct local-vol PDE" in model_order:
        ax.text(
            0.98,
            0.98,
            "Direct PDE bars use selected audit quotes;\nuse matched CSV for proxy comparison",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=8,
            color=spec.annotation_text,
            bbox=_annotation_box(spec),
        )


def _plot_multistart_cost_panel(
    ax: Axes,
    bundle: DiagnosticsBundle,
    spec: ThemeSpec,
    *,
    title: str,
) -> None:
    table = bundle.calibration.tables["multistart_runs"].copy()
    ax.set_title(title, loc="left")
    ax.set_xlabel("Seed index")
    ax.set_ylabel("Cost")

    if table.empty:
        ax.text(
            0.5,
            0.5,
            "No multistart runs are available.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color=spec.palette["text"],
        )
        ax.grid(False)
        return

    seed_index = pd.to_numeric(table["seed_index"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    cost = pd.to_numeric(table["cost"], errors="coerce").to_numpy(dtype=np.float64)
    success = table["success"].astype(bool).to_numpy()
    best = table["best_run"].astype(bool).to_numpy()
    bar_colors = [
        (
            spec.best_run_color
            if is_best
            else spec.model_colors["Heston"] if is_success else spec.held_out_color
        )
        for is_success, is_best in zip(success, best, strict=True)
    ]
    ax.bar(
        seed_index,
        cost,
        color=bar_colors,
        edgecolor=spec.palette["reference"],
        linewidth=0.4,
    )
    finite_cost = cost[np.isfinite(cost)]
    if finite_cost.size and finite_cost.max() / max(finite_cost.min(), 1.0e-16) > 50.0:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.45)
    ax.text(
        0.02,
        0.98,
        "gold=best, blue=success, red=failed",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=spec.annotation_text,
        bbox=_annotation_box(spec),
    )


def _apply_footer(
    fig: Figure,
    text: str,
    spec: ThemeSpec,
    *,
    y: float = 0.022,
    bottom: float = 0.08,
    top: float = 0.94,
) -> None:
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, bottom, 1.0, top))
    fig.text(
        0.5,
        y,
        text,
        ha="center",
        va="bottom",
        fontsize=8,
        color=spec.palette["muted_text"],
    )


def _build_summary_card(bundle: DiagnosticsBundle, theme: str) -> Figure:
    spec = _theme_spec(theme)
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.5), constrained_layout=True)
    _plot_summary_smile_panel(axes[0, 0], bundle, spec)
    _plot_residual_heatmap(axes[0, 1], bundle, spec)
    _plot_multistart_cost_panel(
        axes[1, 0],
        bundle,
        spec,
        title="Calibration stability",
    )
    _plot_error_buckets(axes[1, 1], bundle, spec)
    fig.suptitle(
        "Capstone 3: Heston model-choice diagnostics",
        fontsize=15,
        fontweight="semibold",
        y=0.992,
    )
    _apply_footer(fig, MODEL_CHOICE_CAVEAT, spec, bottom=0.075, top=0.90)
    return fig


def _build_smile_overlay_figure(bundle: DiagnosticsBundle, theme: str) -> Figure:
    spec = _theme_spec(theme)
    fit_errors = bundle.comparison.tables["fit_errors"].copy()
    expiry_values = _selected_expiries(
        fit_errors["expiry"], count=min(5, len(np.unique(fit_errors["expiry"])))
    )
    n_panels = max(len(expiry_values) + 1, 1)
    n_cols = 3 if n_panels > 3 else n_panels
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 4.0 * n_rows),
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for axis, expiry in zip(axes_flat, expiry_values, strict=False):
        slice_table = fit_errors.loc[
            np.isclose(fit_errors["expiry"], float(expiry))
        ].copy()
        market = slice_table.drop_duplicates("quote_index").sort_values(
            "log_moneyness", kind="stable"
        )
        axis.scatter(
            market["log_moneyness"],
            market["market_iv"],
            color=spec.target_iv_color,
            s=24,
            label="target IV",
        )
        held_out = market.loc[market["is_held_out"].astype(bool)]
        if not held_out.empty:
            axis.scatter(
                held_out["log_moneyness"],
                held_out["market_iv"],
                facecolors="none",
                edgecolors=spec.held_out_color,
                s=54,
                linewidths=1.2,
                label="held-out target",
            )
        for model_name in ["Heston", "ESSVI local-vol proxy"]:
            model_slice = slice_table.loc[
                slice_table["model"].astype(str) == model_name
            ].sort_values("log_moneyness", kind="stable")
            axis.plot(
                model_slice["log_moneyness"],
                model_slice["model_iv"],
                linewidth=2.0,
                color=spec.model_colors[model_name],
                label=model_name,
            )

        direct_slice = slice_table.loc[
            slice_table["model"].astype(str) == "Direct local-vol PDE"
        ].sort_values("log_moneyness", kind="stable")
        if not direct_slice.empty:
            axis.scatter(
                direct_slice["log_moneyness"],
                direct_slice["model_iv"],
                color=spec.model_colors["Direct local-vol PDE"],
                marker="D",
                s=34,
                label="Direct local-vol PDE subset",
                zorder=4,
            )

        axis.set_title(f"Expiry {float(expiry):g}y", loc="left")
        axis.set_xlabel("Log-moneyness")
        axis.set_ylabel("Implied vol")
        axis.grid(True, alpha=0.45)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    deduped: dict[str, Any] = {}
    for handle, label in zip(handles, labels, strict=True):
        deduped.setdefault(label, handle)
    legend_axis = axes_flat[len(expiry_values)]
    legend_axis.axis("off")
    legend_axis.legend(
        deduped.values(),
        deduped.keys(),
        loc="center",
        ncol=2,
        fontsize=9,
    )
    legend_axis.text(
        0.5,
        0.16,
        "Direct local-vol PDE markers show the selected validation subset.",
        ha="center",
        va="center",
        fontsize=8,
        color=spec.palette["muted_text"],
        wrap=True,
    )
    for axis in axes_flat[len(expiry_values) + 1 :]:
        axis.axis("off")
    fig.suptitle(
        "Heston vs eSSVI/local-vol smile overlays", fontsize=15, fontweight="semibold"
    )
    _apply_footer(fig, SHORT_CAVEAT, spec, bottom=0.065)
    return fig


def _build_residual_heatmap_figure(bundle: DiagnosticsBundle, theme: str) -> Figure:
    spec = _theme_spec(theme)
    fig, ax = plt.subplots(figsize=(11.2, 7.0), constrained_layout=True)
    _plot_residual_heatmap(ax, bundle, spec)
    fig.suptitle(
        "Heston residual honesty by expiry and moneyness",
        fontsize=15,
        fontweight="semibold",
    )
    _apply_footer(fig, SHORT_CAVEAT, spec, bottom=0.07)
    return fig


def _build_error_buckets_figure(bundle: DiagnosticsBundle, theme: str) -> Figure:
    spec = _theme_spec(theme)
    fig, ax = plt.subplots(figsize=(10.0, 5.4), constrained_layout=True)
    _plot_error_buckets(ax, bundle, spec)
    fig.suptitle("Model-choice error buckets", fontsize=15, fontweight="semibold")
    _apply_footer(fig, MODEL_CHOICE_CAVEAT, spec, bottom=0.095)
    return fig


def _build_multistart_stability_figure(bundle: DiagnosticsBundle, theme: str) -> Figure:
    spec = _theme_spec(theme)
    table = bundle.calibration.tables["multistart_runs"].copy()
    fig = plt.figure(figsize=(13.2, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.05, 1.15))
    ax_left = fig.add_subplot(grid[0, 0])
    _plot_multistart_cost_panel(
        ax_left,
        bundle,
        spec,
        title="Objective by start",
    )
    right_grid = grid[0, 1].subgridspec(5, 1, hspace=0.08)
    parameter_names = ["kappa", "vbar", "eta", "rho", "v"]
    best_mask = (
        table["best_run"].astype(bool).to_numpy()
        if not table.empty
        else np.array([], dtype=bool)
    )
    seed_index = (
        table["seed_index"].to_numpy(dtype=np.float64)
        if not table.empty
        else np.array([], dtype=np.float64)
    )

    legend_handles = [
        Line2D(
            [],
            [],
            color=spec.palette["muted_text"],
            marker="x",
            linestyle="",
            label="seed",
        ),
        Line2D(
            [],
            [],
            color=spec.parameter_colors["kappa"],
            marker="o",
            linewidth=1.8,
            label="fitted",
        ),
        Line2D(
            [],
            [],
            color=spec.best_run_color,
            marker="o",
            linestyle="",
            label="best run",
        ),
    ]

    for row_index, name in enumerate(parameter_names):
        axis = fig.add_subplot(right_grid[row_index, 0])
        seed_values = (
            table[f"seed_{name}"].to_numpy(dtype=np.float64)
            if not table.empty
            else np.array([], dtype=np.float64)
        )
        fitted_values = (
            table[f"fitted_{name}"].to_numpy(dtype=np.float64)
            if not table.empty
            else np.array([], dtype=np.float64)
        )
        axis.plot(
            seed_index,
            fitted_values,
            color=spec.parameter_colors[name],
            marker="o",
            linewidth=1.8,
        )
        axis.scatter(
            seed_index,
            seed_values,
            color=spec.palette["muted_text"],
            marker="x",
            alpha=0.7,
        )
        if best_mask.any():
            axis.scatter(
                seed_index[best_mask],
                fitted_values[best_mask],
                s=66,
                facecolor=spec.best_run_color,
                edgecolor=spec.palette["reference"],
                zorder=4,
            )
        axis.set_ylabel(PARAMETER_LABELS[name], fontsize=9)
        axis.grid(True, alpha=0.45)
        if row_index < len(parameter_names) - 1:
            axis.set_xticklabels([])
        else:
            axis.set_xlabel("Seed index")
        if row_index == 0:
            axis.set_title("Fitted parameter spread by start", loc="left")
            axis.legend(handles=legend_handles, loc="upper right", ncol=3, fontsize=8)

    fig.suptitle("Heston multistart stability", fontsize=15, fontweight="semibold")
    _apply_footer(
        fig,
        "Synthetic target; calibration instability stays visible.",
        spec,
        bottom=0.08,
    )
    return fig


def _build_mc_convergence_figure(bundle: DiagnosticsBundle, theme: str) -> Figure:
    spec = _theme_spec(theme)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
    plot_heston_mc_bias_vs_timestep(
        bundle.mc_bias_summary,
        x="n_steps",
        ax=axes[0],
        title="Bias vs timestep",
    )
    plot_heston_mc_runtime_vs_error(
        bundle.mc_runtime_summary,
        ax=axes[1],
        title="Runtime vs error",
    )
    axes[0].set_ylabel("Bias (MC - Fourier)")
    axes[1].set_ylabel("Mean abs error")
    fig.suptitle(
        "Heston MC vs Fourier",
        fontsize=14,
        fontweight="semibold",
        y=0.992,
    )
    _apply_footer(
        fig,
        "Deterministic validation case; directional runtime signal only.",
        spec,
        bottom=0.11,
        top=0.88,
    )
    return fig


def _build_train_vs_heldout_figure(bundle: DiagnosticsBundle, theme: str) -> Figure:
    spec = _theme_spec(theme)
    table = bundle.comparison.tables["held_out_comparison"].copy()
    fig, ax = plt.subplots(figsize=(9.4, 5.1), constrained_layout=True)
    if table.empty:
        ax.text(
            0.5,
            0.5,
            "No held-out rows are available.",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.axis("off")
    else:
        sample_order = ["train", "held_out"]
        model_order = [str(value) for value in table["model"].drop_duplicates()]
        x = np.arange(len(model_order), dtype=np.float64)
        width = 0.75 / len(sample_order)
        for index, sample_name in enumerate(sample_order):
            values: list[float] = []
            for model_name in model_order:
                rows = table.loc[
                    (table["model"].astype(str) == model_name)
                    & (table["sample"].astype(str) == sample_name)
                ]
                values.append(
                    float(rows["iv_rmse_bps"].iloc[0])
                    if not rows.empty
                    else float("nan")
                )
            offset = (index - (len(sample_order) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                label=sample_name.title(),
                color=(
                    spec.model_colors["Heston"]
                    if sample_name == "train"
                    else spec.best_run_color
                ),
                alpha=(0.95 if sample_name == "train" else 0.72),
            )
        ax.set_xticks(x, model_order, rotation=12, ha="right")
        ax.set_ylabel("IV RMSE (bps)")
        ax.set_title("Train vs held-out comparison", loc="left")
        ax.grid(True, axis="y", alpha=0.45)
        ax.legend(loc="upper left")
    fig.suptitle("Held-out honesty check", fontsize=15, fontweight="semibold")
    _apply_footer(
        fig,
        "Held-out split: short wing, middle ATM, long wing.",
        spec,
        bottom=0.12,
    )
    return fig


def _write_architecture_svg(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1040\" height=\"650\" viewBox=\"0 0 1040 650\" role=\"img\" aria-labelledby=\"title desc\">
  <title id=\"title\">Heston workflow architecture</title>
  <desc id=\"desc\">Synthetic quote target flows into Heston Fourier calibration, QE Monte Carlo validation, and the eSSVI/local-vol/PDE stack before a final model-choice comparison layer.</desc>
  <rect x=\"0\" y=\"0\" width=\"1040\" height=\"650\" rx=\"24\" fill=\"#ffffff\"/>
  <rect x=\"80\" y=\"38\" width=\"880\" height=\"92\" rx=\"18\" fill=\"#eef6f8\" stroke=\"#0f4c5c\" stroke-width=\"2\"/>
  <text x=\"520\" y=\"78\" text-anchor=\"middle\" font-size=\"26\" font-family=\"DejaVu Sans, Arial, sans-serif\" fill=\"#0f172a\" font-weight=\"600\">Synthetic quote target</text>
  <text x=\"520\" y=\"106\" text-anchor=\"middle\" font-size=\"16\" font-family=\"DejaVu Sans, Arial, sans-serif\" fill=\"#334155\">Deterministic market-like fixture for model-choice diagnostics</text>
  <line x1=\"520\" y1=\"130\" x2=\"520\" y2=\"175\" stroke=\"#64748b\" stroke-width=\"3\"/>
  <polygon points=\"520,188 512,172 528,172\" fill=\"#64748b\"/>
  <rect x=\"60\" y=\"210\" width=\"270\" height=\"165\" rx=\"18\" fill=\"#eef6f8\" stroke=\"#0f4c5c\" stroke-width=\"2\"/>
  <text x=\"195\" y=\"248\" text-anchor=\"middle\" font-size=\"22\" font-family=\"DejaVu Sans, Arial, sans-serif\" fill=\"#0f172a\" font-weight=\"600\">Heston Fourier pricing</text>
  <text x=\"195\" y=\"282\" text-anchor=\"middle\" font-size=\"16\" font-family=\"DejaVu Sans, Arial, sans-serif\" fill=\"#334155\">Multistart calibration</text>
  <text x=\"195\" y=\"306\" text-anchor=\"middle\" font-size=\"16\" font-family=\"DejaVu Sans, Arial, sans-serif\" fill=\"#334155\">Residual heatmaps</text>
  <text x=\"195\" y=\"330\" text-anchor=\"middle\" font-size=\"16\" font-family=\"DejaVu Sans, Arial, sans-serif\" fill=\"#334155\">Objective and seed stability</text>
  <rect x=\"385\" y=\"210\" width=\"270\" height=\"165\" rx=\"18\" fill=\"#fdf4ea\" stroke=\"#d17a22\" stroke-width=\"2\"/>
  <text x=\"520\" y=\"248\" text-anchor=\"middle\" font-size=\"22\" font-family=\"DejaVu Sans, Arial, sans-serif\" fill=\"#0f172a\" font-weight=\"600\">QE Monte Carlo</text>
  <text x=\"520\" y=\"282\" text-anchor=\"middle\" font-size=\"16\" font-family=\"DejaVu Sans, Arial, sans-serif\" fill=\"#334155\">MC vs Fourier validation</text>
  <text x=\"520\" y=\"306\" text-anchor=\"middle\" font-size=\"16\" font-family=\"DejaVu Sans, Arial, sans-serif\" fill=\"#334155\">Bias, CI, and runtime/error tradeoff</text>
  <text x=\"520\" y=\"330\" text-anchor=\"middle\" font-size=\"16\" font-family=\"DejaVu Sans, Arial, sans-serif\" fill=\"#334155\">Directional evidence, not universal speed claims</text>
  <rect x=\"710\" y=\"210\" width=\"270\" height=\"165\" rx=\"18\" fill=\"#f3f8eb\" stroke=\"#4d7c0f\" stroke-width=\"2\"/>
  <text x=\"845\" y=\"248\" text-anchor=\"middle\" font-size=\"22\" font-family=\"DejaVu Sans, Arial, sans-serif\" fill=\"#0f172a\" font-weight=\"600\">eSSVI / local-vol stack</text>
  <text x=\"845\" y=\"282\" text-anchor=\"middle\" font-size=\"16\" font-family=\"DejaVu Sans, Arial, sans-serif\" fill=\"#334155\">Flexible vanilla surface fit</text>
  <text x=\"845\" y=\"306\" text-anchor=\"middle\" font-size=\"16\" font-family=\"DejaVu Sans, Arial, sans-serif\" fill=\"#334155\">Direct Dupire/PDE repricing audit</text>
  <text x=\"845\" y=\"330\" text-anchor=\"middle\" font-size=\"16\" font-family=\"DejaVu Sans, Arial, sans-serif\" fill=\"#334155\">Capstone 2 baseline, reused not rebuilt</text>
  <line x1=\"195\" y1=\"375\" x2=\"195\" y2=\"455\" stroke=\"#64748b\" stroke-width=\"3\"/>
  <line x1=\"520\" y1=\"375\" x2=\"520\" y2=\"455\" stroke=\"#64748b\" stroke-width=\"3\"/>
  <line x1=\"845\" y1=\"375\" x2=\"845\" y2=\"455\" stroke=\"#64748b\" stroke-width=\"3\"/>
  <polygon points=\"195,468 187,452 203,452\" fill=\"#64748b\"/>
  <polygon points=\"520,468 512,452 528,452\" fill=\"#64748b\"/>
  <polygon points=\"845,468 837,452 853,452\" fill=\"#64748b\"/>
  <rect x=\"160\" y=\"480\" width=\"720\" height=\"112\" rx=\"22\" fill=\"#f8fafc\" stroke=\"#1e293b\" stroke-width=\"2.5\"/>
  <text x=\"520\" y=\"520\" text-anchor=\"middle\" font-size=\"28\" font-family=\"DejaVu Sans, Arial, sans-serif\" fill=\"#0f172a\" font-weight=\"700\">Model-choice comparison</text>
  <text x=\"520\" y=\"552\" text-anchor=\"middle\" font-size=\"18\" font-family=\"DejaVu Sans, Arial, sans-serif\" fill=\"#334155\">fit quality | stability | dynamics | failure modes</text>
  <text x=\"520\" y=\"625\" text-anchor=\"middle\" font-size=\"13\" font-family=\"DejaVu Sans, Arial, sans-serif\" fill=\"#475569\">{SHORT_CAVEAT} Capstone 3 compares Heston against the same eSSVI/local-vol baseline rather than declaring a universal winner.</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")
    return path


def _write_themed_architecture_svgs(out_dir: Path) -> list[Path]:
    variants = themed_asset_paths(out_dir / "heston_workflow_architecture.svg")
    paths: list[Path] = []
    for theme in PUBLISHING_THEMES:
        spec = _theme_spec(theme)
        if theme == "dark":
            colors = {
                "page": "#08101f",
                "surface": "#0f172a",
                "soft": "#162033",
                "stroke": "#314056",
                "text": spec.palette["text"],
                "muted": spec.palette["muted_text"],
                "accent": "#8cc9ff",
                "warm": "#f3b562",
                "green": "#91e0d7",
            }
        else:
            colors = {
                "page": "#ffffff",
                "surface": "#f8fafc",
                "soft": "#eef6f8",
                "stroke": "#d7e0ea",
                "text": spec.palette["text"],
                "muted": spec.palette["muted_text"],
                "accent": "#0f4c5c",
                "warm": "#d17a22",
                "green": "#4d7c0f",
            }
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="650" viewBox="0 0 1040 650" role="img" aria-labelledby="title desc">
  <title id="title">Heston workflow architecture</title>
  <desc id="desc">Synthetic quote target flows into Heston Fourier calibration, QE Monte Carlo validation, and the eSSVI/local-vol/PDE stack before a final model-choice comparison layer.</desc>
  <rect x="0" y="0" width="1040" height="650" rx="24" fill="{colors['page']}"/>
  <rect x="80" y="38" width="880" height="92" rx="18" fill="{colors['soft']}" stroke="{colors['accent']}" stroke-width="2"/>
  <text x="520" y="78" text-anchor="middle" font-size="26" font-family="DejaVu Sans, Arial, sans-serif" fill="{colors['text']}" font-weight="600">Synthetic quote target</text>
  <text x="520" y="106" text-anchor="middle" font-size="16" font-family="DejaVu Sans, Arial, sans-serif" fill="{colors['muted']}">Deterministic market-like fixture for model-choice diagnostics</text>
  <line x1="520" y1="130" x2="520" y2="175" stroke="{colors['muted']}" stroke-width="3"/>
  <polygon points="520,188 512,172 528,172" fill="{colors['muted']}"/>
  <rect x="60" y="210" width="270" height="165" rx="18" fill="{colors['soft']}" stroke="{colors['accent']}" stroke-width="2"/>
  <text x="195" y="248" text-anchor="middle" font-size="22" font-family="DejaVu Sans, Arial, sans-serif" fill="{colors['text']}" font-weight="600">Heston Fourier pricing</text>
  <text x="195" y="282" text-anchor="middle" font-size="16" font-family="DejaVu Sans, Arial, sans-serif" fill="{colors['muted']}">Multistart calibration</text>
  <text x="195" y="306" text-anchor="middle" font-size="16" font-family="DejaVu Sans, Arial, sans-serif" fill="{colors['muted']}">Residual heatmaps</text>
  <text x="195" y="330" text-anchor="middle" font-size="16" font-family="DejaVu Sans, Arial, sans-serif" fill="{colors['muted']}">Objective and seed stability</text>
  <rect x="385" y="210" width="270" height="165" rx="18" fill="{colors['surface']}" stroke="{colors['warm']}" stroke-width="2"/>
  <text x="520" y="248" text-anchor="middle" font-size="22" font-family="DejaVu Sans, Arial, sans-serif" fill="{colors['text']}" font-weight="600">QE Monte Carlo</text>
  <text x="520" y="282" text-anchor="middle" font-size="16" font-family="DejaVu Sans, Arial, sans-serif" fill="{colors['muted']}">MC vs Fourier validation</text>
  <text x="520" y="306" text-anchor="middle" font-size="16" font-family="DejaVu Sans, Arial, sans-serif" fill="{colors['muted']}">Bias, CI, and runtime/error tradeoff</text>
  <text x="520" y="330" text-anchor="middle" font-size="16" font-family="DejaVu Sans, Arial, sans-serif" fill="{colors['muted']}">Directional evidence, not universal speed claims</text>
  <rect x="710" y="210" width="270" height="165" rx="18" fill="{colors['surface']}" stroke="{colors['green']}" stroke-width="2"/>
  <text x="845" y="248" text-anchor="middle" font-size="22" font-family="DejaVu Sans, Arial, sans-serif" fill="{colors['text']}" font-weight="600">eSSVI / local-vol stack</text>
  <text x="845" y="282" text-anchor="middle" font-size="16" font-family="DejaVu Sans, Arial, sans-serif" fill="{colors['muted']}">Flexible vanilla surface fit</text>
  <text x="845" y="306" text-anchor="middle" font-size="16" font-family="DejaVu Sans, Arial, sans-serif" fill="{colors['muted']}">Direct Dupire/PDE repricing audit</text>
  <text x="845" y="330" text-anchor="middle" font-size="16" font-family="DejaVu Sans, Arial, sans-serif" fill="{colors['muted']}">Capstone 2 baseline, reused not rebuilt</text>
  <line x1="195" y1="375" x2="195" y2="455" stroke="{colors['muted']}" stroke-width="3"/>
  <line x1="520" y1="375" x2="520" y2="455" stroke="{colors['muted']}" stroke-width="3"/>
  <line x1="845" y1="375" x2="845" y2="455" stroke="{colors['muted']}" stroke-width="3"/>
  <polygon points="195,468 187,452 203,452" fill="{colors['muted']}"/>
  <polygon points="520,468 512,452 528,452" fill="{colors['muted']}"/>
  <polygon points="845,468 837,452 853,452" fill="{colors['muted']}"/>
  <rect x="160" y="480" width="720" height="112" rx="22" fill="{colors['surface']}" stroke="{colors['stroke']}" stroke-width="2.5"/>
  <text x="520" y="520" text-anchor="middle" font-size="28" font-family="DejaVu Sans, Arial, sans-serif" fill="{colors['text']}" font-weight="700">Model-choice comparison</text>
  <text x="520" y="552" text-anchor="middle" font-size="18" font-family="DejaVu Sans, Arial, sans-serif" fill="{colors['muted']}">fit quality | stability | dynamics | failure modes</text>
  <text x="520" y="625" text-anchor="middle" font-size="13" font-family="DejaVu Sans, Arial, sans-serif" fill="{colors['muted']}">{SHORT_CAVEAT} Capstone 3 compares Heston against the same eSSVI/local-vol baseline rather than declaring a universal winner.</text>
</svg>
"""
        path = variants.path_for(theme)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg, encoding="utf-8")
        paths.append(path)
    copy_light_variant(variants)
    paths.append(variants.base)
    return paths


def _write_data_tables(bundle: DiagnosticsBundle, data_dir: Path) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        filename: _write_frame(
            bundle.comparison.tables[table_name], data_dir / filename
        )
        for filename, table_name in DATA_TABLE_SPECS.items()
    }
    outputs["heston_mc_convergence_summary.csv"] = _write_frame(
        bundle.mc_convergence_summary,
        data_dir / "heston_mc_convergence_summary.csv",
    )
    return outputs


def _artifact_entry(
    *,
    filename: str,
    artifact_type: str,
    source: str,
    bundle: DiagnosticsBundle,
    repo_commit: str | None,
    generated_at: str,
) -> dict[str, Any]:
    artifact_key = _artifact_key(filename)
    return {
        "filename": filename,
        "artifact_type": artifact_type,
        "source": source,
        "theme_variant": _artifact_theme_variant(filename),
        "fixture_label": str(bundle.quotes.metadata.get("fixture_label", "")),
        "fixture_is_synthetic": bool(
            bundle.quotes.metadata.get("data_source") == "synthetic_not_market_data"
        ),
        "generated_from_heston": bool(
            bundle.quotes.metadata.get("generated_from_heston", False)
        ),
        "seed": int(bundle.seed),
        "profile": bundle.profile.name,
        "repo_commit": repo_commit,
        "generated_at": generated_at,
        "caption": CAPTIONS.get(artifact_key, CAPTIONS.get(filename, "")),
        "caveats": [CAVEAT],
        "docs_pages": DOCS_PAGES.get(filename, DOCS_PAGES.get(artifact_key, [])),
    }


def _write_manifest(
    *,
    bundle: DiagnosticsBundle,
    out_dir: Path,
    written_figure_paths: dict[str, list[Path]],
    data_paths: dict[str, Path],
    architecture_paths: list[Path],
) -> Path:
    generated_at = datetime.now(tz=UTC).isoformat()
    repo_commit = _git_commit()
    manifest_path = out_dir / "data" / "heston_artifact_manifest.json"
    artifacts: list[dict[str, Any]] = []
    for stem, paths in written_figure_paths.items():
        for path in paths:
            artifacts.append(
                _artifact_entry(
                    filename=path.name,
                    artifact_type="figure",
                    source=stem,
                    bundle=bundle,
                    repo_commit=repo_commit,
                    generated_at=generated_at,
                )
            )
    for name, path in data_paths.items():
        artifacts.append(
            _artifact_entry(
                filename=path.name,
                artifact_type="data",
                source=name,
                bundle=bundle,
                repo_commit=repo_commit,
                generated_at=generated_at,
            )
        )
    for architecture_path in architecture_paths:
        artifacts.append(
            _artifact_entry(
                filename=architecture_path.name,
                artifact_type="diagram",
                source="heston_workflow_architecture",
                bundle=bundle,
                repo_commit=repo_commit,
                generated_at=generated_at,
            )
        )
    artifacts.append(
        _artifact_entry(
            filename=manifest_path.name,
            artifact_type="data",
            source="heston_artifact_manifest.json",
            bundle=bundle,
            repo_commit=repo_commit,
            generated_at=generated_at,
        )
    )
    artifacts.append(
        _artifact_entry(
            filename=HESTON_ARTIFACT_FRESHNESS_MANIFEST,
            artifact_type="data",
            source=HESTON_ARTIFACT_FRESHNESS_MANIFEST,
            bundle=bundle,
            repo_commit=repo_commit,
            generated_at=generated_at,
        )
    )
    payload = {
        "version": 1,
        "profile": bundle.profile.name,
        "seed": int(bundle.seed),
        "generated_at": generated_at,
        "repo_commit": repo_commit,
        "fixture": {
            "label": str(bundle.quotes.metadata.get("fixture_label", "")),
            "construction": str(bundle.quotes.metadata.get("construction", "")),
            "data_source": str(bundle.quotes.metadata.get("data_source", "")),
            "generated_from_heston": bool(
                bundle.quotes.metadata.get("generated_from_heston", False)
            ),
            "quote_count": int(bundle.quotes.n_quotes),
            "expiries": [float(value) for value in np.unique(bundle.quotes.expiry)],
            "log_moneyness": [
                float(value) for value in np.unique(bundle.quotes.log_moneyness)
            ],
        },
        "held_out_policy": {
            "description": (
                "Deterministic three-point split: one downside wing at short expiry, "
                "one ATM point at middle expiry, and one upside wing at long expiry."
            ),
            "indices": [int(index) for index in bundle.held_out_indices],
            "labels": list(bundle.held_out_labels),
        },
        "caveat": CAVEAT,
        "warnings": list(bundle.warnings),
        "artifacts": artifacts,
    }
    return _write_json(payload, manifest_path)


def _required_paths(
    out_dir: Path,
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    figure_stems = [
        "heston_comparison_summary_card",
        "heston_model_comparison_smile_overlay",
        "heston_iv_residual_heatmap",
        "heston_model_comparison_error_buckets",
        "heston_multistart_stability_panel",
        "heston_mc_vs_fourier_convergence",
        "heston_train_vs_heldout_comparison",
    ]
    required: list[Path] = []
    for stem in figure_stems:
        for fmt in formats:
            variants = themed_asset_paths(out_dir / f"{stem}.{fmt}")
            required.extend([variants.base, variants.light, variants.dark])
    architecture_variants = themed_asset_paths(
        out_dir / "heston_workflow_architecture.svg"
    )
    required.extend(
        [
            architecture_variants.base,
            architecture_variants.light,
            architecture_variants.dark,
            out_dir / "data" / "heston_comparison_fit_errors.csv",
            out_dir / "data" / "heston_comparison_error_summary.csv",
            out_dir / "data" / "heston_comparison_heldout.csv",
            out_dir / "data" / "heston_comparison_direct_local_vol_pde.csv",
            out_dir / "data" / "heston_comparison_direct_pde_matched_error_summary.csv",
            out_dir / "data" / "heston_comparison_tradeoff_summary.csv",
            out_dir / "data" / "heston_mc_convergence_summary.csv",
            out_dir / "data" / "heston_artifact_manifest.json",
            out_dir / "data" / HESTON_ARTIFACT_FRESHNESS_MANIFEST,
        ]
    )
    return required


def _freshness_tracked_paths(
    out_dir: Path,
    *,
    formats: tuple[str, ...],
) -> list[Path]:
    return [
        path
        for path in _required_paths(out_dir, formats=formats)
        if path.name != HESTON_ARTIFACT_FRESHNESS_MANIFEST
    ]


def _write_heston_artifact_freshness_manifest(
    out_dir: Path,
    *,
    formats: tuple[str, ...],
    profile_name: str,
) -> Path:
    manifest_path = out_dir / "data" / HESTON_ARTIFACT_FRESHNESS_MANIFEST
    payload = {
        "version": HESTON_ARTIFACT_FRESHNESS_VERSION,
        "profile": profile_name,
        "formats": list(formats),
        "rebuild_command": HESTON_ARTIFACT_REBUILD_COMMAND,
        "source_inputs": _heston_artifact_source_payload(),
        "generated_files": [
            {
                "path": _relative_to_dir(path, out_dir),
                "sha256": _file_sha256(path),
            }
            for path in sorted(
                _freshness_tracked_paths(out_dir, formats=formats),
                key=lambda candidate: _relative_to_dir(candidate, out_dir),
            )
        ],
    }
    return _write_json(payload, manifest_path)


def _generate_artifacts(
    bundle: DiagnosticsBundle,
    *,
    out_dir: Path,
    formats: tuple[str, ...],
) -> Path:
    data_dir = out_dir / "data"
    data_paths = _write_data_tables(bundle, data_dir)
    written_figures = {
        "heston_comparison_summary_card": _save_themed_figure(
            stem="heston_comparison_summary_card",
            out_dir=out_dir,
            formats=formats,
            render=lambda theme: _build_summary_card(bundle, theme),
        ),
        "heston_model_comparison_smile_overlay": _save_themed_figure(
            stem="heston_model_comparison_smile_overlay",
            out_dir=out_dir,
            formats=formats,
            render=lambda theme: _build_smile_overlay_figure(bundle, theme),
        ),
        "heston_iv_residual_heatmap": _save_themed_figure(
            stem="heston_iv_residual_heatmap",
            out_dir=out_dir,
            formats=formats,
            render=lambda theme: _build_residual_heatmap_figure(bundle, theme),
        ),
        "heston_model_comparison_error_buckets": _save_themed_figure(
            stem="heston_model_comparison_error_buckets",
            out_dir=out_dir,
            formats=formats,
            render=lambda theme: _build_error_buckets_figure(bundle, theme),
        ),
        "heston_multistart_stability_panel": _save_themed_figure(
            stem="heston_multistart_stability_panel",
            out_dir=out_dir,
            formats=formats,
            render=lambda theme: _build_multistart_stability_figure(bundle, theme),
        ),
        "heston_mc_vs_fourier_convergence": _save_themed_figure(
            stem="heston_mc_vs_fourier_convergence",
            out_dir=out_dir,
            formats=formats,
            render=lambda theme: _build_mc_convergence_figure(bundle, theme),
        ),
        "heston_train_vs_heldout_comparison": _save_themed_figure(
            stem="heston_train_vs_heldout_comparison",
            out_dir=out_dir,
            formats=formats,
            render=lambda theme: _build_train_vs_heldout_figure(bundle, theme),
        ),
    }
    architecture_paths = _write_themed_architecture_svgs(out_dir)
    manifest_path = _write_manifest(
        bundle=bundle,
        out_dir=out_dir,
        written_figure_paths=written_figures,
        data_paths=data_paths,
        architecture_paths=architecture_paths,
    )
    _write_heston_artifact_freshness_manifest(
        out_dir,
        formats=formats,
        profile_name=bundle.profile.name,
    )

    missing = [
        path for path in _required_paths(out_dir, formats=formats) if not path.exists()
    ]
    if missing:
        missing_rel = ", ".join(_relative(path) for path in missing)
        raise RuntimeError(
            f"Required Heston docs artifacts were not generated: {missing_rel}"
        )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    _configure_matplotlib()
    args = _parse_args(argv)
    profile = PROFILE_DEFAULTS[str(args.profile)]
    formats = _parse_formats(str(args.formats))
    out_dir = args.output_dir.resolve()
    bundle = _build_diagnostics(
        profile,
        seed_override=args.seed,
        skip_direct_pde=bool(args.skip_direct_pde),
    )
    manifest_path = _generate_artifacts(bundle, out_dir=out_dir, formats=formats)
    print(_relative(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
