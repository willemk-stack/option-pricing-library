from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from option_pricing.diagnostics.vol_surface import localvol as vs_localvol
from option_pricing.pricers.black_scholes import bs_price_from_ctx
from option_pricing.types import OptionType
from option_pricing.vol import (
    ESSVICalibrationConfig,
    ESSVINodalSurface,
    ESSVIProjectionConfig,
    LocalVolSurface,
    calibrate_essvi,
    project_essvi_nodes,
)

from .scenario import (
    SharedDemoScenario,
    build_shared_demo_scenario,
    resolve_run_flag,
)
from .surface_flagship import SurfaceDemoArtifacts, build_surface_demo_artifacts


@dataclass(frozen=True, slots=True)
class ESSVIBridgeArtifacts:
    profile: str
    cfg: dict[str, Any]
    scenario: SharedDemoScenario
    fit: Any
    nodal_surface: ESSVINodalSurface
    projection: Any
    smoothed_surface: Any
    localvol: LocalVolSurface
    tables: dict[str, pd.DataFrame]
    reports: dict[str, Any]
    meta: dict[str, Any]


def _coerce_float(value: object, *, name: str) -> float:
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"{name} must be numeric, got {type(value).__name__}")


def _finite_diff_wT(surface, *, y: float, T: float, dt: float) -> float:
    w1 = float(np.asarray(surface.w(np.asarray([y], dtype=np.float64), T + dt))[0])
    w0 = float(np.asarray(surface.w(np.asarray([y], dtype=np.float64), T))[0])
    return (w1 - w0) / dt


def _quote_comparison_table(
    *,
    price_quotes_df: pd.DataFrame,
    nodal_surface: ESSVINodalSurface,
    smoothed_surface,
    market,
) -> pd.DataFrame:
    compare = price_quotes_df.copy()
    T = compare["T"].to_numpy(dtype=np.float64)
    K = compare["K"].to_numpy(dtype=np.float64)
    iv_obs = compare["iv_obs"].to_numpy(dtype=np.float64)
    is_call = compare["is_call"].to_numpy(dtype=bool)
    ctx = market.to_context()

    iv_nodal = np.full((len(compare),), np.nan, dtype=np.float64)
    iv_smoothed = np.full((len(compare),), np.nan, dtype=np.float64)
    for tau, idx in compare.groupby("T").groups.items():
        ii = np.asarray(list(idx), dtype=int)
        y_slice = compare.loc[ii, "y"].to_numpy(dtype=np.float64)
        tau_value = _coerce_float(tau, name="expiry")
        iv_nodal[ii] = nodal_surface.iv(y_slice, tau_value)
        iv_smoothed[ii] = smoothed_surface.iv(y_slice, tau_value)

    compare["iv_nodal"] = iv_nodal
    compare["iv_smoothed"] = iv_smoothed
    compare["abs_iv_error_bp_nodal"] = 1e4 * np.abs(compare["iv_nodal"] - iv_obs)
    compare["abs_iv_error_bp_smoothed"] = 1e4 * np.abs(compare["iv_smoothed"] - iv_obs)

    nodal_prices = []
    smoothed_prices = []
    for strike, tau, nodal_iv, smoothed_iv, call_flag in zip(
        K,
        T,
        compare["iv_nodal"].to_numpy(dtype=np.float64),
        compare["iv_smoothed"].to_numpy(dtype=np.float64),
        is_call,
        strict=True,
    ):
        kind = OptionType.CALL if call_flag else OptionType.PUT
        nodal_prices.append(
            bs_price_from_ctx(
                kind=kind,
                strike=float(strike),
                sigma=float(nodal_iv),
                tau=float(tau),
                ctx=ctx,
            )
        )
        smoothed_prices.append(
            bs_price_from_ctx(
                kind=kind,
                strike=float(strike),
                sigma=float(smoothed_iv),
                tau=float(tau),
                ctx=ctx,
            )
        )

    compare["price_nodal"] = np.asarray(nodal_prices, dtype=np.float64)
    compare["price_smoothed"] = np.asarray(smoothed_prices, dtype=np.float64)
    compare["abs_price_error_nodal"] = np.abs(
        compare["price_nodal"] - compare["price_mkt"]
    )
    compare["abs_price_error_smoothed"] = np.abs(
        compare["price_smoothed"] - compare["price_mkt"]
    )
    return compare


def _quote_summary(compare: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "mean_abs_iv_error_bp_nodal": float(
                    compare["abs_iv_error_bp_nodal"].mean()
                ),
                "max_abs_iv_error_bp_nodal": float(
                    compare["abs_iv_error_bp_nodal"].max()
                ),
                "mean_abs_iv_error_bp_smoothed": float(
                    compare["abs_iv_error_bp_smoothed"].mean()
                ),
                "max_abs_iv_error_bp_smoothed": float(
                    compare["abs_iv_error_bp_smoothed"].max()
                ),
                "mean_abs_price_error_nodal": float(
                    compare["abs_price_error_nodal"].mean()
                ),
                "max_abs_price_error_nodal": float(
                    compare["abs_price_error_nodal"].max()
                ),
                "mean_abs_price_error_smoothed": float(
                    compare["abs_price_error_smoothed"].mean()
                ),
                "max_abs_price_error_smoothed": float(
                    compare["abs_price_error_smoothed"].max()
                ),
            }
        ]
    )


def _projection_summary_table(fit, projection) -> pd.DataFrame:
    validation = getattr(projection.diag, "validation", None)
    validation_ok = bool(getattr(validation, "ok", False)) if validation else False
    return pd.DataFrame(
        [
            {
                "fit_success": bool(fit.diag.success),
                "price_rmse": float(fit.diag.price_rmse),
                "max_abs_price_error": float(fit.diag.max_abs_price_error),
                "node_validation_ok": bool(fit.diag.node_validation.ok),
                "projection_success": bool(projection.success),
                "projection_validation_ok": validation_ok,
                "projection_dupire_invalid_count": int(
                    projection.diag.dupire_invalid_count
                ),
                "projection_total_points": int(projection.diag.dupire_total_points),
                "projection_message": str(projection.diag.message),
            }
        ]
    )


def _time_smoothness_compare(
    *,
    surface_artifacts: SurfaceDemoArtifacts | None,
    smoothed_surface,
    expiries: np.ndarray,
) -> pd.DataFrame:
    if surface_artifacts is None:
        return pd.DataFrame()

    svi_surface = surface_artifacts.surfaces["svi_repaired"]
    dt = 1e-4
    y_points = (-0.30, -0.15, 0.15, 0.30)
    rows: list[dict[str, float]] = []
    for T_knot in expiries[1:-1]:
        svi_diffs = []
        smoothed_diffs = []
        for y in y_points:
            svi_left = _finite_diff_wT(
                svi_surface,
                y=float(y),
                T=float(T_knot) - 2 * dt,
                dt=dt,
            )
            svi_right = _finite_diff_wT(
                svi_surface,
                y=float(y),
                T=float(T_knot) + dt,
                dt=dt,
            )
            sm_left = _finite_diff_wT(
                smoothed_surface,
                y=float(y),
                T=float(T_knot) - 2 * dt,
                dt=dt,
            )
            sm_right = _finite_diff_wT(
                smoothed_surface,
                y=float(y),
                T=float(T_knot) + dt,
                dt=dt,
            )
            svi_diffs.append(abs(svi_right - svi_left))
            smoothed_diffs.append(abs(sm_right - sm_left))
        rows.append(
            {
                "T_knot": float(T_knot),
                "max_abs_wT_jump_svi": float(max(svi_diffs)),
                "mean_abs_wT_jump_svi": float(np.mean(svi_diffs)),
                "max_abs_wT_jump_smoothed": float(max(smoothed_diffs)),
                "mean_abs_wT_jump_smoothed": float(np.mean(smoothed_diffs)),
            }
        )
    return pd.DataFrame(rows)


def build_essvi_bridge_artifacts(
    *,
    profile: str = "quick",
    seed: int = 7,
    overrides: dict[str, Any] | None = None,
    scenario: SharedDemoScenario | None = None,
    surface_artifacts: SurfaceDemoArtifacts | None = None,
) -> ESSVIBridgeArtifacts:
    shared = (
        build_shared_demo_scenario(profile=profile, seed=seed, overrides=overrides)
        if scenario is None
        else scenario
    )
    surface_demo = (
        build_surface_demo_artifacts(
            profile=shared.profile,
            seed=shared.seed,
            overrides=overrides,
            scenario=shared,
        )
        if surface_artifacts is None
        else surface_artifacts
    )

    calib_cfg = ESSVICalibrationConfig(**shared.cfg.get("ESSVI_CALIBRATION_CFG", {}))
    fit = calibrate_essvi(
        y=shared.price_quotes_df["y"].to_numpy(dtype=np.float64),
        T=shared.price_quotes_df["T"].to_numpy(dtype=np.float64),
        price_mkt=shared.price_quotes_df["price_mkt"].to_numpy(dtype=np.float64),
        market=shared.market,
        is_call=shared.price_quotes_df["is_call"].to_numpy(dtype=bool),
        cfg=calib_cfg,
    )

    nodal_surface = ESSVINodalSurface(fit.nodes)
    projection = project_essvi_nodes(
        fit.nodes,
        cfg=ESSVIProjectionConfig(**shared.cfg.get("ESSVI_PROJECTION_CFG", {})),
    )
    if projection.surface is None:
        raise ValueError(projection.diag.message)

    smoothed_surface = projection.surface
    localvol = LocalVolSurface.from_implied(
        smoothed_surface,
        forward=shared.forward,
        discount=shared.discount,
    )

    compare = _quote_comparison_table(
        price_quotes_df=shared.price_quotes_df,
        nodal_surface=nodal_surface,
        smoothed_surface=smoothed_surface,
        market=shared.market,
    )
    handoff_report = vs_localvol.localvol_grid_diagnostics(
        localvol,
        expiries=shared.cfg["LV_DIAG_EXPIRIES"],
        y_grid=np.linspace(-0.20, 0.20, 21, dtype=np.float64),
        eps_w=1e-12,
        eps_denom=1e-12,
        top_n=8,
    )

    node_table = pd.DataFrame(
        {
            "T": fit.nodes.expiries,
            "theta": fit.nodes.theta,
            "psi": fit.nodes.psi,
            "rho": fit.nodes.rho,
            "eta": fit.nodes.eta,
            "g_plus": fit.nodes.g_plus,
            "g_minus": fit.nodes.g_minus,
        }
    )

    time_smoothness = (
        _time_smoothness_compare(
            surface_artifacts=surface_demo,
            smoothed_surface=smoothed_surface,
            expiries=shared.expiries,
        )
        if resolve_run_flag(
            name="RUN_ESSVI_TIME_SMOOTHNESS_COMPARE",
            default=True,
            overrides=overrides,
        )
        else pd.DataFrame()
    )

    tables: dict[str, pd.DataFrame] = {
        "quotes_df": shared.quotes_df,
        "quote_summary": shared.quote_summary,
        "essvi_nodes": node_table,
        "essvi_projection_summary": _projection_summary_table(fit, projection),
        "essvi_quote_compare": compare,
        "essvi_quote_summary": _quote_summary(compare),
        "essvi_time_smoothness_compare": time_smoothness,
        "essvi_localvol_handoff_summary": pd.DataFrame(
            [vs_localvol.localvol_summary(handoff_report)]
        ),
        "essvi_localvol_handoff_worst": handoff_report.worst_points,
    }

    reports: dict[str, Any] = {
        "fit": fit,
        "projection": projection,
        "surface_artifacts": surface_demo,
        "localvol_handoff": handoff_report,
    }
    meta = {
        "profile": shared.profile,
        "projection_success": bool(projection.success),
        "dupire_ready_surface": type(smoothed_surface).__name__,
    }

    return ESSVIBridgeArtifacts(
        profile=shared.profile,
        cfg=shared.cfg,
        scenario=shared,
        fit=fit,
        nodal_surface=nodal_surface,
        projection=projection,
        smoothed_surface=smoothed_surface,
        localvol=localvol,
        tables=tables,
        reports=reports,
        meta=meta,
    )


__all__ = ["ESSVIBridgeArtifacts", "build_essvi_bridge_artifacts"]
