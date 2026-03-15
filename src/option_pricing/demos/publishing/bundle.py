from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import SupportsFloat, cast

import numpy as np
import pandas as pd

from option_pricing.diagnostics.vol_surface import localvol as vs_localvol
from option_pricing.vol.local_vol_types import LVInvalidReason

from ..integration import run_surface_to_localvol_pde_integration
from .config import BuildProfileName, VisualBuildConfig, get_visual_build_config
from .types import MANIFEST_VERSION, BundleManifest, DatasetManifest


@dataclass(frozen=True, slots=True)
class VisualBundleResult:
    config: VisualBuildConfig
    manifest: BundleManifest


def _bundle_root(
    *,
    cfg: VisualBuildConfig,
    seed: int,
    bundle_dir: str | Path | None,
) -> Path:
    if bundle_dir is not None:
        return Path(bundle_dir).resolve()
    return (cfg.bundle_root / f"profile_{cfg.profile}_seed_{int(seed)}").resolve()


def _write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _reasons_to_str(mask: int) -> str:
    if mask == 0:
        return ""
    parts: list[str] = []
    for reason in LVInvalidReason:
        value = int(reason)
        if value != 0 and (int(mask) & value):
            if reason.name is not None:
                parts.append(reason.name)
    return "|".join(parts)


def _surface_grid(
    *,
    T_grid: np.ndarray,
    y_grid: np.ndarray,
    forward,
    iv_fn,
    w_fn=None,
    derivs_fn=None,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for T in np.asarray(T_grid, dtype=np.float64):
        F = float(forward(float(T)))
        K = F * np.exp(y_grid)
        iv = np.asarray(iv_fn(K, y_grid, float(T)), dtype=np.float64).reshape(-1)
        if w_fn is not None:
            w = np.asarray(w_fn(K, y_grid, float(T)), dtype=np.float64).reshape(-1)
        else:
            w = (iv * iv) * float(T)

        derivs: tuple[np.ndarray, ...] | None = None
        if derivs_fn is not None:
            derivs = tuple(
                np.asarray(arr, dtype=np.float64).reshape(-1)
                for arr in derivs_fn(K, y_grid, float(T))
            )

        for idx, y in enumerate(y_grid):
            row: dict[str, float] = {
                "T": float(T),
                "F": float(F),
                "y": float(y),
                "moneyness": float(np.exp(y)),
                "K": float(K[idx]),
                "iv": float(iv[idx]),
                "w": float(w[idx]),
            }
            if derivs is not None:
                row["w_y"] = float(derivs[1][idx])
                row["w_yy"] = float(derivs[2][idx])
                row["w_T"] = float(derivs[3][idx])
            rows.append(row)
    return pd.DataFrame(rows)


def _build_quote_surface_compare(artifacts) -> pd.DataFrame:
    surface = artifacts.surface_demo.surfaces["svi_repaired"]
    compare = pd.DataFrame(artifacts.tables["essvi_quote_compare"]).copy()
    compare["moneyness"] = compare["K"].astype(float) / compare["F"].astype(float)

    iv_svi = np.full((len(compare),), np.nan, dtype=np.float64)
    for tau_key, idx in compare.groupby("T").groups.items():
        ii = np.asarray(list(idx), dtype=int)
        strikes = compare.loc[ii, "K"].to_numpy(dtype=np.float64)
        tau = float(cast(SupportsFloat, tau_key))
        iv_svi[ii] = np.asarray(
            surface.iv(strikes, tau),
            dtype=np.float64,
        ).reshape(-1)

    compare["iv_svi"] = iv_svi
    compare["iv_essvi_nodal"] = compare["iv_nodal"].astype(float)
    compare["iv_essvi_smoothed"] = compare["iv_smoothed"].astype(float)
    compare["price_essvi_nodal"] = compare["price_nodal"].astype(float)
    compare["price_essvi_smoothed"] = compare["price_smoothed"].astype(float)
    compare["iv_resid_svi_bp"] = 1e4 * (compare["iv_svi"] - compare["iv_obs"])
    compare["iv_resid_essvi_nodal_bp"] = 1e4 * (
        compare["iv_essvi_nodal"] - compare["iv_obs"]
    )
    compare["iv_resid_essvi_smoothed_bp"] = 1e4 * (
        compare["iv_essvi_smoothed"] - compare["iv_obs"]
    )
    compare["price_resid_essvi_nodal"] = (
        compare["price_essvi_nodal"] - compare["price_mkt"]
    )
    compare["price_resid_essvi_smoothed"] = (
        compare["price_essvi_smoothed"] - compare["price_mkt"]
    )

    keep = [
        "T",
        "F",
        "y",
        "moneyness",
        "K",
        "iv_obs",
        "iv_true",
        "iv_svi",
        "iv_essvi_nodal",
        "iv_essvi_smoothed",
        "iv_resid_svi_bp",
        "iv_resid_essvi_nodal_bp",
        "iv_resid_essvi_smoothed_bp",
        "price_mkt",
        "price_essvi_nodal",
        "price_essvi_smoothed",
        "price_resid_essvi_nodal",
        "price_resid_essvi_smoothed",
        "iv_noise_bp",
        "kind",
        "is_call",
    ]
    return compare[keep].sort_values(["T", "K"]).reset_index(drop=True)


def _long_localvol_grid(report) -> pd.DataFrame:
    rows: list[dict[str, float | bool | str]] = []
    Ts = np.asarray(report.expiries, dtype=np.float64)
    y_grid = np.asarray(report.y, dtype=np.float64)
    K_grid = np.asarray(report.K, dtype=np.float64)
    sigma = np.asarray(report.sigma, dtype=np.float64)
    local_var = np.asarray(report.local_var, dtype=np.float64)
    denom = np.asarray(report.denom, dtype=np.float64)
    invalid = np.asarray(report.invalid, dtype=bool)
    reason = np.asarray(report.reason, dtype=np.uint32)

    for i, T in enumerate(Ts):
        for j, y in enumerate(y_grid):
            K = float(K_grid[i, j]) if K_grid.ndim == 2 else float(K_grid[j])
            F = K / float(np.exp(y))
            rows.append(
                {
                    "T": float(T),
                    "F": float(F),
                    "y": float(y),
                    "K": K,
                    "moneyness": float(np.exp(y)),
                    "sigma_loc": float(sigma[i, j]),
                    "local_var": float(local_var[i, j]),
                    "denom": float(denom[i, j]),
                    "invalid": bool(invalid[i, j]),
                    "reasons": _reasons_to_str(int(reason[i, j])),
                }
            )
    return pd.DataFrame(rows)


def _long_localvol_compare(compare) -> pd.DataFrame:
    rows: list[dict[str, float | bool | str]] = []
    Ts = np.asarray(compare.expiries, dtype=np.float64)
    Ks = np.asarray(compare.strikes, dtype=np.float64)
    Fs = np.asarray(compare.forwards, dtype=np.float64)
    y = np.asarray(compare.y, dtype=np.float64)
    g_sigma = np.asarray(compare.g_sigma, dtype=np.float64)
    g_lv = np.asarray(compare.g_local_var, dtype=np.float64)
    g_denom = np.asarray(compare.g_denom, dtype=np.float64)
    g_invalid = np.asarray(compare.g_invalid, dtype=bool)
    g_reason = np.asarray(compare.g_reason, dtype=np.uint32)
    d_sigma = np.asarray(compare.dupire.sigma, dtype=np.float64)
    d_lv = np.asarray(compare.dupire.local_var, dtype=np.float64)
    d_denom = np.asarray(compare.dupire.denom, dtype=np.float64)
    d_invalid = np.asarray(compare.dupire.invalid, dtype=bool)
    d_reason = np.asarray(compare.dupire.reason, dtype=np.uint32)
    diff_sigma = np.asarray(compare.diff_sigma, dtype=np.float64)
    diff_lv = np.asarray(compare.diff_local_var, dtype=np.float64)
    invalid_union = np.asarray(compare.invalid_union, dtype=bool)

    for i, T in enumerate(Ts):
        F = float(Fs[i]) if i < len(Fs) else float("nan")
        for j, K in enumerate(Ks):
            rows.append(
                {
                    "T": float(T),
                    "F": F,
                    "y": float(y[i, j]),
                    "K": float(K),
                    "sigma_loc_gatheral": float(g_sigma[i, j]),
                    "sigma_loc_dupire": float(d_sigma[i, j]),
                    "local_var_gatheral": float(g_lv[i, j]),
                    "local_var_dupire": float(d_lv[i, j]),
                    "denom_gatheral": float(g_denom[i, j]),
                    "denom_dupire": float(d_denom[i, j]),
                    "diff_sigma_loc": float(diff_sigma[i, j]),
                    "diff_local_var": float(diff_lv[i, j]),
                    "invalid_gatheral": bool(g_invalid[i, j]),
                    "invalid_dupire": bool(d_invalid[i, j]),
                    "invalid_union": bool(invalid_union[i, j]),
                    "reasons_gatheral": _reasons_to_str(int(g_reason[i, j])),
                    "reasons_dupire": _reasons_to_str(int(d_reason[i, j])),
                }
            )
    return pd.DataFrame(rows)


def _build_gatheral_report(artifacts, cfg: VisualBuildConfig):
    expiries = np.linspace(
        float(np.min(artifacts.scenario.expiries)),
        float(np.max(artifacts.scenario.expiries)),
        cfg.localvol_nt,
        dtype=np.float64,
    )
    y_grid = np.linspace(
        cfg.localvol_y_min,
        cfg.localvol_y_max,
        cfg.localvol_ny,
        dtype=np.float64,
    )
    return vs_localvol.localvol_grid_diagnostics(
        artifacts.localvol,
        expiries=expiries.tolist(),
        y_grid=y_grid,
        eps_w=1e-12,
        eps_denom=1e-12,
        top_n=12,
    )


def _build_gatheral_vs_dupire_report(artifacts, cfg: VisualBuildConfig):
    strikes = np.asarray(artifacts.scenario.cfg["SHARED_STRIKES"], dtype=np.float64)
    expiries = np.linspace(
        float(np.min(artifacts.scenario.expiries)),
        float(np.max(artifacts.scenario.expiries)),
        cfg.localvol_nt,
        dtype=np.float64,
    )
    strike_grid = np.linspace(
        float(np.min(strikes)),
        float(np.max(strikes)),
        cfg.localvol_ny,
        dtype=np.float64,
    )
    return vs_localvol.localvol_compare_gatheral_vs_dupire(
        artifacts.localvol,
        expiries=expiries.tolist(),
        strikes=strike_grid,
        market=artifacts.scenario.market,
        price_convention="discounted",
        strike_coordinate="logK",
        trim_t=1,
        trim_k=1,
        top_n=12,
    )


def _build_manifest(
    bundle_root: Path, datasets: list[DatasetManifest], *, cfg, seed: int
):
    manifest = BundleManifest(
        version=MANIFEST_VERSION,
        profile=cfg.profile,
        workflow_profile=cfg.workflow_profile,
        seed=int(seed),
        bundle_root=bundle_root,
        datasets=tuple(datasets),
    )
    manifest.write()
    return manifest


def load_bundle_manifest(bundle_dir: str | Path) -> BundleManifest:
    return BundleManifest.load(Path(bundle_dir) / "meta" / "manifest.json")


def load_bundle_dataframe(
    manifest: BundleManifest,
    dataset_name: str,
) -> pd.DataFrame:
    dataset = manifest.resolve_dataset(dataset_name)
    return pd.read_csv(manifest.bundle_root / dataset.relative_path)


def build_visual_bundle(
    *,
    profile: BuildProfileName = "ci",
    seed: int = 7,
    bundle_dir: str | Path | None = None,
) -> VisualBundleResult:
    cfg = get_visual_build_config(profile)
    out_root = _bundle_root(cfg=cfg, seed=seed, bundle_dir=bundle_dir)

    artifacts = run_surface_to_localvol_pde_integration(
        profile=cfg.workflow_profile,
        seed=seed,
        overrides=cfg.overrides,
    )

    y_grid = np.linspace(
        cfg.surface_y_min,
        cfg.surface_y_max,
        cfg.surface_ny,
        dtype=np.float64,
    )
    T_grid = np.linspace(
        float(np.min(artifacts.scenario.expiries)),
        float(np.max(artifacts.scenario.expiries)),
        cfg.surface_nt,
        dtype=np.float64,
    )

    svi_grid = _surface_grid(
        T_grid=T_grid,
        y_grid=y_grid,
        forward=artifacts.scenario.forward,
        iv_fn=lambda K, _y, T: artifacts.surface_demo.surfaces["svi_repaired"].iv(K, T),
    )
    essvi_nodal_grid = _surface_grid(
        T_grid=T_grid,
        y_grid=y_grid,
        forward=artifacts.scenario.forward,
        iv_fn=lambda _K, y, T: artifacts.essvi_bridge.nodal_surface.iv(y, T),
        w_fn=lambda _K, y, T: artifacts.essvi_bridge.nodal_surface.w(y, T),
    )
    essvi_smoothed_grid = _surface_grid(
        T_grid=T_grid,
        y_grid=y_grid,
        forward=artifacts.scenario.forward,
        iv_fn=lambda _K, y, T: artifacts.essvi_bridge.smoothed_surface.iv(y, T),
        w_fn=lambda _K, y, T: artifacts.essvi_bridge.smoothed_surface.w(y, T),
        derivs_fn=lambda _K, y, T: artifacts.essvi_bridge.smoothed_surface.w_and_derivs(
            y,
            T,
        ),
    )

    quote_compare = _build_quote_surface_compare(artifacts)

    gatheral_report = _build_gatheral_report(artifacts, cfg)
    gatheral_grid = _long_localvol_grid(gatheral_report)
    gatheral_vs_dupire = _long_localvol_compare(
        _build_gatheral_vs_dupire_report(artifacts, cfg)
    )

    repricing_grid = artifacts.tables["repricing_grid"].copy()
    repricing_grid["moneyness"] = repricing_grid["K"].astype(float) / repricing_grid[
        "F"
    ].astype(float)
    repricing_grid["price_error"] = repricing_grid["pde_price"].astype(
        float
    ) - repricing_grid["target_price"].astype(float)
    repricing_grid["iv_error_bp"] = 1e4 * (
        repricing_grid["pde_iv"].astype(float)
        - repricing_grid["target_iv"].astype(float)
    )
    repricing_grid = repricing_grid.rename(
        columns={
            "abs_price_error": "price_abs_error",
            "abs_iv_error_bp": "iv_abs_error_bp",
        }
    )

    repricing_summary = artifacts.tables["repricing_summary"].copy()
    convergence_grid = artifacts.tables["convergence_grid"].copy()

    outputs: list[tuple[pd.DataFrame, DatasetManifest]] = [
        (
            svi_grid,
            DatasetManifest(
                name="surface/svi_repaired_grid",
                relative_path="surface/svi_repaired_grid.csv",
                story="static",
                rectangular_grid=True,
                axis_columns={"i": "y", "j": "T", "x": "y", "y": "T", "z": "iv"},
                default_scalar_fields=("iv", "w"),
                source_object_type=type(
                    artifacts.surface_demo.surfaces["svi_repaired"]
                ).__name__,
                aliases=("svi_repaired_grid",),
                description="Uniform (T, y) sample of the repaired SVI surface.",
            ),
        ),
        (
            essvi_nodal_grid,
            DatasetManifest(
                name="surface/essvi_nodal_grid",
                relative_path="surface/essvi_nodal_grid.csv",
                story="dupire",
                rectangular_grid=True,
                axis_columns={"i": "y", "j": "T", "x": "y", "y": "T", "z": "iv"},
                default_scalar_fields=("iv", "w"),
                source_object_type=type(artifacts.essvi_bridge.nodal_surface).__name__,
                aliases=("essvi_nodal_grid",),
                description="Uniform (T, y) sample of the exact nodal eSSVI surface.",
            ),
        ),
        (
            essvi_smoothed_grid,
            DatasetManifest(
                name="surface/essvi_smoothed_grid",
                relative_path="surface/essvi_smoothed_grid.csv",
                story="dupire",
                rectangular_grid=True,
                axis_columns={"i": "y", "j": "T", "x": "y", "y": "T", "z": "iv"},
                default_scalar_fields=("iv", "w", "w_T", "w_y", "w_yy"),
                source_object_type=type(
                    artifacts.essvi_bridge.smoothed_surface
                ).__name__,
                aliases=("essvi_smoothed_grid",),
                description="Uniform (T, y) sample of the smoothed eSSVI surface.",
            ),
        ),
        (
            quote_compare,
            DatasetManifest(
                name="surface/quote_surface_compare",
                relative_path="surface/quote_surface_compare.csv",
                story="static",
                rectangular_grid=False,
                axis_columns={"x": "y", "y": "T"},
                default_scalar_fields=(
                    "iv_obs",
                    "iv_svi",
                    "iv_essvi_nodal",
                    "iv_essvi_smoothed",
                ),
                source_object_type="DataFrame",
                aliases=("quote_surface_compare",),
                description="Observed quotes compared against SVI and eSSVI surfaces.",
            ),
        ),
        (
            artifacts.tables["svi_fit_compare"].copy(),
            DatasetManifest(
                name="calibration/svi_fit_compare",
                relative_path="calibration/svi_fit_compare.csv",
                story="static",
                rectangular_grid=False,
                axis_columns={"x": "T"},
                default_scalar_fields=("rmse_w_nr", "rmse_w_fx"),
                source_object_type="DataFrame",
                aliases=("svi_fit_compare",),
                description="Per-expiry SVI fit diagnostics before and after repair.",
            ),
        ),
        (
            artifacts.tables["essvi_nodes"].copy(),
            DatasetManifest(
                name="calibration/essvi_nodes",
                relative_path="calibration/essvi_nodes.csv",
                story="dupire",
                rectangular_grid=False,
                axis_columns={"x": "T"},
                default_scalar_fields=("theta", "psi", "rho", "eta"),
                source_object_type="ESSVINodeSet",
                aliases=("essvi_nodes",),
                description="Calibrated eSSVI nodes.",
            ),
        ),
        (
            artifacts.tables["essvi_projection_summary"].copy(),
            DatasetManifest(
                name="calibration/essvi_projection_summary",
                relative_path="calibration/essvi_projection_summary.csv",
                story="dupire",
                rectangular_grid=False,
                axis_columns={},
                default_scalar_fields=(
                    "price_rmse",
                    "max_abs_price_error",
                    "projection_dupire_invalid_count",
                ),
                source_object_type="ESSVIProjectionResult",
                aliases=("essvi_projection_summary",),
                description="Summary of the nodal-to-smoothed eSSVI projection.",
            ),
        ),
        (
            artifacts.tables["essvi_time_smoothness_compare"].copy(),
            DatasetManifest(
                name="calibration/essvi_time_smoothness",
                relative_path="calibration/essvi_time_smoothness.csv",
                story="dupire",
                rectangular_grid=False,
                axis_columns={"x": "T_knot"},
                default_scalar_fields=(
                    "max_abs_wT_jump_svi",
                    "max_abs_wT_jump_smoothed",
                ),
                source_object_type="DataFrame",
                aliases=("essvi_time_smoothness",),
                description="Seam comparison between repaired SVI and smoothed eSSVI.",
            ),
        ),
        (
            gatheral_grid,
            DatasetManifest(
                name="localvol/gatheral_grid",
                relative_path="localvol/gatheral_grid.csv",
                story="dupire",
                rectangular_grid=True,
                axis_columns={
                    "i": "y",
                    "j": "T",
                    "x": "y",
                    "y": "T",
                    "z": "sigma_loc",
                },
                default_scalar_fields=("sigma_loc", "local_var", "denom"),
                source_object_type=type(gatheral_report).__name__,
                aliases=("gatheral_grid",),
                description="Uniform local-vol diagnostics grid from Gatheral formula.",
            ),
        ),
        (
            gatheral_vs_dupire,
            DatasetManifest(
                name="localvol/gatheral_vs_dupire_grid",
                relative_path="localvol/gatheral_vs_dupire_grid.csv",
                story="numerics",
                rectangular_grid=True,
                axis_columns={
                    "i": "K",
                    "j": "T",
                    "x": "K",
                    "y": "T",
                    "z": "diff_sigma_loc",
                },
                default_scalar_fields=(
                    "sigma_loc_gatheral",
                    "sigma_loc_dupire",
                    "diff_sigma_loc",
                ),
                source_object_type="LocalVolCompareReport",
                aliases=("gatheral_vs_dupire_grid",),
                description="Strike-space local-vol compare grid for Gatheral vs Dupire.",
            ),
        ),
        (
            repricing_grid,
            DatasetManifest(
                name="repricing/pde_roundtrip_grid",
                relative_path="repricing/pde_roundtrip_grid.csv",
                story="numerics",
                rectangular_grid=True,
                axis_columns={
                    "i": "K",
                    "j": "T",
                    "x": "K",
                    "y": "T",
                    "z": "price_error",
                },
                default_scalar_fields=(
                    "target_price",
                    "pde_price",
                    "price_error",
                    "iv_error_bp",
                ),
                source_object_type="LocalVolRepricingResult",
                aliases=("pde_roundtrip_grid",),
                description="PDE repricing grid against the originating smoothed eSSVI surface.",
            ),
        ),
        (
            repricing_summary,
            DatasetManifest(
                name="repricing/pde_roundtrip_summary",
                relative_path="repricing/pde_roundtrip_summary.csv",
                story="numerics",
                rectangular_grid=False,
                axis_columns={},
                default_scalar_fields=tuple(repricing_summary.columns[:4]),
                source_object_type="DataFrame",
                aliases=("pde_roundtrip_summary",),
                description="Aggregate PDE repricing summary metrics.",
            ),
        ),
        (
            convergence_grid,
            DatasetManifest(
                name="repricing/convergence_grid",
                relative_path="repricing/convergence_grid.csv",
                story="numerics",
                rectangular_grid=False,
                axis_columns={"x": "grid_points"},
                default_scalar_fields=("pde_price",),
                source_object_type="LocalVolConvergenceSweepResult",
                aliases=("convergence_grid",),
                description="Single-option local-vol PDE convergence sweep.",
            ),
        ),
    ]

    dataset_entries: list[DatasetManifest] = []
    for df, dataset in outputs:
        _write_df(df, out_root / dataset.relative_path)
        dataset_entries.append(dataset)

    manifest = _build_manifest(out_root, dataset_entries, cfg=cfg, seed=seed)
    return VisualBundleResult(config=cfg, manifest=manifest)
