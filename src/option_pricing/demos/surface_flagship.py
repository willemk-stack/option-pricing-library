from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from option_pricing.diagnostics.vol_surface import report as vs_report
from option_pricing.diagnostics.vol_surface import svi as vs_svi
from option_pricing.diagnostics.vol_surface.recipes import (
    build_svi_surface_with_fallback,
    default_svi_repair_candidates,
    gj_example51_check_table,
    gj_example51_comparison_table,
    run_explicit_svi_repair_demo,
)
from option_pricing.vol import VolSurface, check_surface_noarb
from option_pricing.vol.svi import run_gj_example51_repair_sanity_check

from .scenario import (
    SharedDemoScenario,
    build_shared_demo_scenario,
    resolve_run_flag,
)


@dataclass(frozen=True, slots=True)
class SurfaceDemoArtifacts:
    profile: str
    cfg: dict[str, Any]
    scenario: SharedDemoScenario
    surfaces: dict[str, VolSurface]
    tables: dict[str, pd.DataFrame]
    reports: dict[str, Any]
    meta: dict[str, Any]


def _noarb_summary_rows(
    *,
    truth_report,
    grid_report,
    svi_no_repair_report,
    svi_repaired_report,
) -> pd.DataFrame:
    summary_rows = []
    for name, rep in [
        ("truth_grid (latent benchmark)", truth_report),
        ("grid (observed quotes)", grid_report),
        ("svi_no_repair", svi_no_repair_report),
        ("svi_repaired", svi_repaired_report),
    ]:
        cal = getattr(rep, "calendar_total_variance", None)
        summary_rows.append(
            {
                "surface": name,
                "ok": bool(getattr(rep, "ok", True)),
                "message": str(getattr(rep, "message", "")),
                "n_smiles": len(getattr(rep, "smile_monotonicity", [])),
                "calendar_performed": (
                    bool(getattr(cal, "performed", False)) if cal is not None else False
                ),
                "calendar_ok": (
                    bool(getattr(cal, "ok", True)) if cal is not None else np.nan
                ),
                "calendar_max_violation": (
                    float(getattr(cal, "max_violation", 0.0))
                    if (cal is not None and getattr(cal, "performed", False))
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(summary_rows)


def _repair_failure_summary(attempts: pd.DataFrame) -> pd.DataFrame:
    if attempts.empty:
        return pd.DataFrame()
    failures = attempts.loc[~attempts["ok"].astype(bool), "error"].astype(str)
    if failures.empty:
        return pd.DataFrame()
    error_type = failures.str.split(":", n=1, expand=True)[0].str.strip()
    return error_type.value_counts().rename_axis("error_type").reset_index(name="count")


def _pick_focus_expiry(svi_fit: pd.DataFrame, expiries: np.ndarray) -> float:
    if "rmse_w" in svi_fit.columns and svi_fit["rmse_w"].notna().any():
        return float(svi_fit.sort_values("rmse_w", ascending=False).iloc[0]["T"])
    return float(expiries[len(expiries) // 2])


def _pick_interpolation_probe(expiries: np.ndarray) -> float:
    mid = len(expiries) // 2
    if mid == 0:
        return float(expiries[0])
    left = float(expiries[mid - 1])
    right = float(expiries[mid])
    return 0.5 * (left + right)


def _interpolation_choice_table(
    *,
    surface: VolSurface,
    T_probe: float,
    forward,
    y_grid: np.ndarray,
) -> pd.DataFrame:
    noarb = surface.slice(float(T_probe), method="no_arb")
    linear = surface.slice(float(T_probe), method="linear_w")
    y = np.asarray(y_grid, dtype=np.float64)
    F = float(forward(float(T_probe)))
    K = F * np.exp(y)

    return pd.DataFrame(
        {
            "T_probe": float(T_probe),
            "y": y,
            "K": K,
            "iv_no_arb": np.asarray(noarb.iv_at(y), dtype=np.float64),
            "iv_linear_w": np.asarray(linear.iv_at(y), dtype=np.float64),
            "abs_iv_gap_bp": 1e4
            * np.abs(
                np.asarray(noarb.iv_at(y), dtype=np.float64)
                - np.asarray(linear.iv_at(y), dtype=np.float64)
            ),
        }
    )


def build_surface_demo_artifacts(
    *,
    profile: str = "quick",
    seed: int = 7,
    overrides: dict[str, Any] | None = None,
    scenario: SharedDemoScenario | None = None,
) -> SurfaceDemoArtifacts:
    shared = (
        build_shared_demo_scenario(profile=profile, seed=seed, overrides=overrides)
        if scenario is None
        else scenario
    )
    cfg = shared.cfg

    surface_grid = VolSurface.from_grid(shared.rows_obs, forward=shared.forward)
    surface_svi_norepair = VolSurface.from_svi(
        shared.rows_obs,
        forward=shared.forward,
        calibrate_kwargs=cfg["SVI_CALIB_NO_REPAIR"],
    )

    candidates = cfg.get("SVI_CALIB_REPAIR_CANDIDATES")
    if candidates is None:
        candidates = default_svi_repair_candidates(robust_data_only=True)

    surface_svi_repaired, repair_mode_used, repair_attempts = (
        build_svi_surface_with_fallback(
            shared.rows_obs,
            forward=shared.forward,
            candidates=candidates,
            fallback_surface=surface_svi_norepair,
        )
    )

    noarb_grid = check_surface_noarb(surface_grid, df=shared.discount)
    noarb_svi_norepair = check_surface_noarb(surface_svi_norepair, df=shared.discount)
    noarb_svi_repaired = check_surface_noarb(surface_svi_repaired, df=shared.discount)

    noarb_summary = _noarb_summary_rows(
        truth_report=shared.noarb_true,
        grid_report=noarb_grid,
        svi_no_repair_report=noarb_svi_norepair,
        svi_repaired_report=noarb_svi_repaired,
    )

    quotes_for_diag = shared.quotes_df.rename(columns={"iv_obs": "iv"})[
        ["T", "K", "iv"]
    ]
    diag_grid = None
    diag_svi_repaired = None
    if shared.profile == "full":
        diag_grid = vs_report.run_surface_diagnostics(
            surface_grid,
            forward=shared.forward,
            df=shared.discount,
            quotes_df=quotes_for_diag,
            include_svi=True,
            include_domain=True,
            include_calendar_arrays=False,
        )
        diag_svi_repaired = vs_report.run_surface_diagnostics(
            surface_svi_repaired,
            forward=shared.forward,
            df=shared.discount,
            quotes_df=quotes_for_diag,
            include_svi=True,
            include_domain=True,
            include_calendar_arrays=False,
        )

    svi_fit_norepair = vs_svi.svi_fit_table(surface_svi_norepair)
    svi_fit_repaired = vs_svi.svi_fit_table(surface_svi_repaired)
    compare_cols = [
        c
        for c in [
            "T",
            "diag_ok",
            "rmse_w",
            "mae_w",
            "min_g",
            "lee_ok",
            "butterfly_ok",
            "failure_reason",
        ]
        if c in svi_fit_norepair.columns
    ]
    svi_fit_compare = (
        svi_fit_norepair[compare_cols]
        .merge(
            svi_fit_repaired[compare_cols],
            on="T",
            how="outer",
            suffixes=("_nr", "_fx"),
        )
        .sort_values("T")
        .reset_index(drop=True)
    )

    focus_T = _pick_focus_expiry(svi_fit_norepair, shared.expiries)
    repair_demo = None
    if resolve_run_flag(
        name="RUN_EXPLICIT_SVI_REPAIR_DEMO",
        default=True,
        overrides=overrides,
    ):
        repair_demo = run_explicit_svi_repair_demo(
            focus_T=focus_T,
            repaired_surface=surface_svi_repaired,
            w_floor=1e-12,
            g_floor=0.0,
            plot_grid_size=801,
            y_obj_size=101,
            y_penalty_size=301,
            repair_scan_size=101,
            repair_bisect_steps=50,
        )

    interpolation_probe_T = _pick_interpolation_probe(shared.expiries)
    interpolation_choice = _interpolation_choice_table(
        surface=surface_svi_repaired,
        T_probe=interpolation_probe_T,
        forward=shared.forward,
        y_grid=np.linspace(-0.25, 0.25, 9, dtype=np.float64),
    )

    gj51 = None
    if resolve_run_flag(
        name="RUN_GJ_PAPER_SANITY_CHECK",
        default=True,
        overrides=overrides,
    ):
        gj51 = run_gj_example51_repair_sanity_check(
            T=1.0,
            y_domain_hint=(-1.5, 1.5),
            w_floor=1e-12,
            y_obj=np.linspace(-1.5, 1.5, 301, dtype=np.float64),
            y_penalty=np.linspace(-1.5, 1.5, 1201, dtype=np.float64),
            y_plot=np.linspace(-1.5, 1.5, 801, dtype=np.float64),
            lambda_price=1.0,
            lambda_g=1e7,
            g_floor=0.0,
            g_scale=0.02,
            lambda_wfloor=0.0,
            max_nfev=2000,
        )

    tables: dict[str, pd.DataFrame] = {
        "quotes_df": shared.quotes_df,
        "quote_summary": shared.quote_summary,
        "surface_noarb_summary": noarb_summary,
        "svi_repair_attempts": repair_attempts,
        "svi_repair_failure_summary": _repair_failure_summary(repair_attempts),
        "svi_fit_norepair": svi_fit_norepair,
        "svi_fit_repaired": svi_fit_repaired,
        "svi_fit_compare": svi_fit_compare,
        "interpolation_choice": interpolation_choice,
    }
    if repair_demo is not None:
        tables["explicit_repair_metadata"] = repair_demo.metadata
        tables["explicit_repair_attempts"] = repair_demo.repair_attempts
        tables["explicit_repair_summary"] = repair_demo.summary
    if gj51 is not None:
        tables["gj51_comparison"] = gj_example51_comparison_table(gj51)
        tables["gj51_checks"] = gj_example51_check_table(gj51)

    if diag_grid is not None:
        tables["diag_noarb_grid"] = diag_grid.tables.get("noarb_smiles", pd.DataFrame())
    if diag_svi_repaired is not None:
        tables["diag_noarb_svi_repaired"] = diag_svi_repaired.tables.get(
            "noarb_smiles",
            pd.DataFrame(),
        )
        if "svi_fit" in diag_svi_repaired.tables:
            tables["diag_svi_fit"] = diag_svi_repaired.tables["svi_fit"]

    reports: dict[str, Any] = {
        "noarb_true": shared.noarb_true,
        "noarb_grid": noarb_grid,
        "noarb_svi_norepair": noarb_svi_norepair,
        "noarb_svi_repaired": noarb_svi_repaired,
        "diag_grid": diag_grid,
        "diag_svi_repaired": diag_svi_repaired,
        "explicit_repair_demo": repair_demo,
        "gj51": gj51,
    }

    surfaces = {
        "grid": surface_grid,
        "svi_norepair": surface_svi_norepair,
        "svi_repaired": surface_svi_repaired,
        "true_grid": shared.surface_true,
    }
    meta = {
        "profile": shared.profile,
        "focus_T": float(focus_T),
        "interpolation_probe_T": float(interpolation_probe_T),
        "svi_repair_mode_used": str(repair_mode_used),
        "svi_repair_fallback_used": str(repair_mode_used) == "FALLBACK",
    }

    return SurfaceDemoArtifacts(
        profile=shared.profile,
        cfg=cfg,
        scenario=shared,
        surfaces=surfaces,
        tables=tables,
        reports=reports,
        meta=meta,
    )


__all__ = ["SurfaceDemoArtifacts", "build_surface_demo_artifacts"]
