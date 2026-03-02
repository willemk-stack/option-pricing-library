"""Capstone 2 demo recipe: surface -> SVI -> local vol -> PDE diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from option_pricing import MarketData, OptionType, PricingInputs
from option_pricing.data_generators.recipes import (
    generate_synthetic_surface_latent_noarb,
)
from option_pricing.diagnostics.pde_vs_digital import run_demo_baseline_sweep
from option_pricing.diagnostics.vol_surface import localvol as vs_localvol
from option_pricing.diagnostics.vol_surface import report as vs_report
from option_pricing.diagnostics.vol_surface import svi as vs_svi
from option_pricing.diagnostics.vol_surface.pde_repricing import (
    LocalVolConvergenceSweepResult,
    LocalVolRepricingResult,
    localvol_pde_repricing_grid,
    localvol_pde_single_option_convergence_sweep,
)
from option_pricing.diagnostics.vol_surface.recipes import (
    build_svi_surface_with_fallback,
    default_svi_repair_candidates,
    gj_example51_check_table,
    gj_example51_comparison_table,
    run_explicit_svi_repair_demo,
)
from option_pricing.types import DigitalSpec
from option_pricing.vol import LocalVolSurface, VolSurface, check_surface_noarb
from option_pricing.vol.svi.diagnostics import run_gj_example51_repair_sanity_check

from ._capstone2_defaults import get_capstone2_defaults


@dataclass(frozen=True)
class Capstone2Artifacts:
    """Bundle of data + diagnostics for the Capstone 2 demo."""

    profile: str
    cfg: dict[str, Any]
    flags: dict[str, bool]
    synthetic: dict[str, Any]
    surfaces: dict[str, VolSurface]
    localvol: LocalVolSurface
    tables: dict[str, pd.DataFrame]
    reports: dict[str, Any]
    repricing: LocalVolRepricingResult | None
    convergence: LocalVolConvergenceSweepResult | None
    digital_baseline: Any | None
    gj51: Any | None
    meta: dict[str, Any]


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **val}
        else:
            out[key] = val
    return out


def _apply_overrides(
    cfg: dict[str, Any], overrides: dict[str, Any] | None
) -> dict[str, Any]:
    if not overrides:
        return dict(cfg)

    out = dict(cfg)
    for key, val in overrides.items():
        if key == "RUN_*":
            continue
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _merge_dict(out[key], val)
        else:
            out[key] = val
    return out


def run_capstone2(
    profile: str = "quick",
    seed: int = 7,
    overrides: dict | None = None,
) -> Capstone2Artifacts:
    """Run the Capstone 2 demo recipe and return notebook-friendly artifacts."""
    profile = str(profile).lower().strip()
    if profile not in {"quick", "full"}:
        raise ValueError("profile must be 'quick' or 'full'")

    cfg = get_capstone2_defaults(seed)
    cfg = _apply_overrides(cfg, overrides)

    run_flags = {
        "RUN_DUPIRE_VS_GATHERAL_COMPARE": profile == "full",
        "RUN_DIGITAL_PDE_BASELINE": profile == "full",
        "RUN_LOCALVOL_REPRICING": True,
        "RUN_LOCALVOL_CONVERGENCE_SWEEP": profile == "full",
        "RUN_GJ_EXAMPLE51": profile == "full",
        "RUN_EXPLICIT_SVI_REPAIR_DEMO": profile == "full",
    }

    if overrides and isinstance(overrides.get("RUN_*"), dict):
        for k, v in overrides["RUN_*"].items():
            if k in run_flags:
                run_flags[k] = bool(v)

    syn = generate_synthetic_surface_latent_noarb(
        enforce=bool(cfg.get("ENFORCE_ARB_FREE_LATENT_TRUTH", True)),
        max_rounds=int(cfg.get("SYNTH_MAX_ROUNDS", 8)),
        **cfg["SYNTH_CFG"],
    )

    synthetic = syn.synthetic
    surface_true_grid = syn.surface_true
    noarb_true_grid = syn.noarb_true
    rows_true = syn.rows_true
    synth_cfg_used = syn.cfg_used
    synth_tuning_log = syn.tuning_log

    quotes_df = (
        pd.DataFrame(
            {
                "T": synthetic.T,
                "x": synthetic.x,
                "K": synthetic.K,
                "F": synthetic.F,
                "iv_obs": synthetic.iv_obs,
                "iv_true": synthetic.iv_true,
            }
        )
        .sort_values(["T", "K"])
        .reset_index(drop=True)
    )

    quotes_df["y"] = np.log(quotes_df["K"] / quotes_df["F"])
    quotes_df["w_obs"] = quotes_df["T"] * quotes_df["iv_obs"] ** 2
    quotes_df["w_true"] = quotes_df["T"] * quotes_df["iv_true"] ** 2
    quotes_df["iv_noise_bp"] = 1e4 * (quotes_df["iv_obs"] - quotes_df["iv_true"])

    rows_obs = [(float(t), float(k), float(iv)) for t, k, iv in synthetic.rows_obs]
    forward = synthetic.forward
    df_curve = synthetic.df

    surface_grid = VolSurface.from_grid(rows_obs, forward=forward)

    surface_svi_norepair = VolSurface.from_svi(
        rows_obs,
        forward=forward,
        calibrate_kwargs=cfg["SVI_CALIB_NO_REPAIR"],
    )

    candidates = cfg.get("SVI_CALIB_REPAIR_CANDIDATES")
    if candidates is None:
        candidates = default_svi_repair_candidates(robust_data_only=True)

    surface_svi_repaired, svi_repair_mode_used, svi_repair_attempts = (
        build_svi_surface_with_fallback(
            rows_obs,
            forward=forward,
            candidates=candidates,
            fallback_surface=surface_svi_norepair,
        )
    )

    svi_repair_fallback_used = str(svi_repair_mode_used) == "FALLBACK"
    svi_repair_failure_summary = pd.DataFrame()
    if not svi_repair_attempts.empty:
        fail = svi_repair_attempts.loc[
            ~svi_repair_attempts["ok"].astype(bool), "error"
        ].astype(str)
        if not fail.empty:
            err_type = fail.str.split(":", n=1, expand=True)[0].str.strip()
            svi_repair_failure_summary = (
                err_type.value_counts()
                .rename_axis("error_type")
                .reset_index(name="count")
            )

    noarb_true = noarb_true_grid
    noarb_grid = check_surface_noarb(surface_grid, df=df_curve)
    noarb_svi_nr = check_surface_noarb(surface_svi_norepair, df=df_curve)
    noarb_svi_fx = check_surface_noarb(surface_svi_repaired, df=df_curve)

    summary_rows = []
    for name, rep in [
        ("truth_grid (latent benchmark)", noarb_true),
        ("grid (observed quotes)", noarb_grid),
        ("svi_no_repair", noarb_svi_nr),
        ("svi_repaired", noarb_svi_fx),
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

    noarb_summary = pd.DataFrame(summary_rows)

    quote_summary = (
        quotes_df.groupby("T")
        .agg(
            n_quotes=("K", "size"),
            iv_obs_min=("iv_obs", "min"),
            iv_obs_max=("iv_obs", "max"),
            mean_abs_noise_bp=("iv_noise_bp", lambda s: float(np.mean(np.abs(s)))),
        )
        .reset_index()
    )

    quotes_for_diag = quotes_df.rename(columns={"iv_obs": "iv"})[["T", "K", "iv"]]

    diag_grid = None
    diag_svi_nr = None
    diag_svi_fx = None
    if profile == "full":
        diag_grid = vs_report.run_surface_diagnostics(
            surface_grid,
            forward=forward,
            df=df_curve,
            quotes_df=quotes_for_diag,
            include_svi=True,
            include_domain=True,
            include_calendar_arrays=False,
        )
        diag_svi_nr = vs_report.run_surface_diagnostics(
            surface_svi_norepair,
            forward=forward,
            df=df_curve,
            quotes_df=quotes_for_diag,
            include_svi=True,
            include_domain=True,
            include_calendar_arrays=False,
        )
        diag_svi_fx = vs_report.run_surface_diagnostics(
            surface_svi_repaired,
            forward=forward,
            df=df_curve,
            quotes_df=quotes_for_diag,
            include_svi=True,
            include_domain=True,
            include_calendar_arrays=False,
        )

    svi_fit_nr = vs_svi.svi_fit_table(surface_svi_norepair)
    svi_fit_fx = vs_svi.svi_fit_table(surface_svi_repaired)

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
        if c in svi_fit_nr.columns
    ]
    svi_fit_compare = (
        svi_fit_nr[compare_cols]
        .merge(
            svi_fit_fx[compare_cols],
            on="T",
            how="outer",
            suffixes=("_nr", "_fx"),
        )
        .sort_values("T")
    )

    if "rmse_w" in svi_fit_nr.columns and svi_fit_nr["rmse_w"].notna().any():
        focus_T = float(svi_fit_nr.sort_values("rmse_w", ascending=False).iloc[0]["T"])
    else:
        focus_T = float(
            np.asarray(surface_svi_repaired.expiries, dtype=float)[
                len(surface_svi_repaired.expiries) // 2
            ]
        )

    explicit_repair_demo = None
    if run_flags["RUN_EXPLICIT_SVI_REPAIR_DEMO"]:
        explicit_repair_demo = run_explicit_svi_repair_demo(
            focus_T=float(focus_T),
            repaired_surface=surface_svi_repaired,
            w_floor=1e-12,
            g_floor=0.0,
            plot_grid_size=801,
            y_obj_size=101,
            y_penalty_size=301,
            repair_scan_size=101,
            repair_bisect_steps=50,
        )

    localvol = LocalVolSurface.from_implied(
        surface_svi_repaired,
        forward=forward,
        discount=df_curve,
    )

    lv_rep = vs_localvol.localvol_grid_diagnostics(
        localvol,
        expiries=cfg["LV_DIAG_EXPIRIES"],
        y_grid=cfg["LV_DIAG_YGRID"],
        eps_w=1e-12,
        eps_denom=1e-12,
        top_n=15,
    )

    localvol_summary = pd.DataFrame([vs_localvol.localvol_summary(lv_rep)])

    reason_counts_df = pd.DataFrame(
        [{"reason": k, "count": v} for k, v in lv_rep.reason_counts.items()]
    )
    if not reason_counts_df.empty:
        reason_counts_df = reason_counts_df.sort_values("count", ascending=False)

    lv_cmp = None
    if run_flags["RUN_DUPIRE_VS_GATHERAL_COMPARE"]:
        mkt = MarketData(
            spot=cfg["SYNTH_CFG"]["spot"],
            rate=cfg["SYNTH_CFG"]["r"],
            dividend_yield=cfg["SYNTH_CFG"]["q"],
        )
        lv_cmp = vs_localvol.localvol_compare_gatheral_vs_dupire(
            localvol,
            expiries=cfg["SHARED_EXPIRIES"],
            strikes=cfg["SHARED_STRIKES"],
            market=mkt,
            price_convention="discounted",
            strike_coordinate="logK",
            trim_t=1,
            trim_k=1,
            top_n=12,
        )

    digital_baseline = None
    if run_flags["RUN_DIGITAL_PDE_BASELINE"]:
        sigma_bs = 0.22
        p_dig = PricingInputs(
            spec=DigitalSpec(
                kind=OptionType.CALL,
                strike=100.0,
                expiry=1.0,
                payout=1.0,
            ),
            market=MarketData(spot=100.0, rate=0.02, dividend_yield=0.00),
            sigma=sigma_bs,
        )

        digital_baseline = run_demo_baseline_sweep(
            p_dig,
            methods=("cn", "rannacher"),
            advections=("central", "upwind"),
            n_sigmas=(4.0, 5.0),
            Nx_list=(101, 151, 201),
            Nt_list=(101, 201, 401),
            coord=cfg["LV_PDE_SOLVER_CFG"].get("coord", None),
            spacing="uniform",
            ic_remedy="cell_avg",
            tol_abs=5e-4,
            tol_rel=5e-3,
            budget_ms=250.0,
        )

    repricing = None
    if run_flags["RUN_LOCALVOL_REPRICING"]:
        Nx = 201 if profile == "full" else 151
        Nt = 401 if profile == "full" else 301
        stride = 2 if profile == "full" else 3
        strikes = cfg["SHARED_STRIKES"][2:-2:stride]

        repricing = localvol_pde_repricing_grid(
            lv=localvol,
            market=MarketData(
                spot=cfg["SYNTH_CFG"]["spot"],
                rate=cfg["SYNTH_CFG"]["r"],
                dividend_yield=cfg["SYNTH_CFG"]["q"],
            ),
            strikes=strikes,
            expiries=cfg["SHARED_EXPIRIES"],
            Nx=Nx,
            Nt=Nt,
            solver_cfg=cfg["LV_PDE_SOLVER_CFG"],
            kind=OptionType.CALL,
            target="black76_from_implied",
            compute_implied_vol=True,
        )

    convergence = None
    if run_flags["RUN_LOCALVOL_CONVERGENCE_SWEEP"]:
        T0 = 1.0
        K0 = float(forward(T0))
        mkt = MarketData(
            spot=cfg["SYNTH_CFG"]["spot"],
            rate=cfg["SYNTH_CFG"]["r"],
            dividend_yield=cfg["SYNTH_CFG"]["q"],
        )
        convergence = localvol_pde_single_option_convergence_sweep(
            lv=localvol,
            market=mkt,
            strike=K0,
            expiry=T0,
            grids=cfg["LV_SWEEP_GRIDS"],
            solver_cfg=cfg["LV_PDE_SOLVER_CFG"],
            kind=OptionType.CALL,
        )

    gj51 = None
    if run_flags["RUN_GJ_EXAMPLE51"]:
        gj51 = run_gj_example51_repair_sanity_check(
            T=1.0,
            y_domain_hint=(-1.5, 1.5),
            w_floor=1e-12,
            y_obj=np.linspace(-1.5, 1.5, 301),
            y_penalty=np.linspace(-1.5, 1.5, 1201),
            y_plot=np.linspace(-1.5, 1.5, 801),
            lambda_price=1.0,
            lambda_g=1e7,
            g_floor=0.0,
            g_scale=0.02,
            lambda_wfloor=0.0,
            max_nfev=2000,
        )

    tables: dict[str, pd.DataFrame] = {
        "synth_tuning_log": pd.DataFrame(synth_tuning_log),
        "latent_truth_noarb": pd.DataFrame(
            [
                {
                    "latent_truth_ok": bool(noarb_true_grid.ok),
                    "message": str(noarb_true_grid.message),
                    "calendar_ok": (
                        bool(noarb_true_grid.calendar_total_variance.ok)
                        if noarb_true_grid.calendar_total_variance.performed
                        else np.nan
                    ),
                    "calendar_max_violation": (
                        float(noarb_true_grid.calendar_total_variance.max_violation)
                        if noarb_true_grid.calendar_total_variance.performed
                        else np.nan
                    ),
                }
            ]
        ),
        "synth_cfg_used": pd.DataFrame([synth_cfg_used]),
        "quotes_df": quotes_df,
        "quote_summary": quote_summary,
        "svi_repair_attempts": svi_repair_attempts,
        "svi_repair_failure_summary": svi_repair_failure_summary,
        "surface_noarb_summary": noarb_summary,
        "svi_fit_compare": svi_fit_compare,
        "svi_fit_norepair": svi_fit_nr,
        "localvol_summary": localvol_summary,
        "localvol_reason_counts": reason_counts_df,
        "localvol_worst_points": lv_rep.worst_points,
    }

    if explicit_repair_demo is not None:
        tables["explicit_repair_metadata"] = explicit_repair_demo.metadata
        tables["explicit_repair_attempts"] = explicit_repair_demo.repair_attempts
        tables["explicit_repair_summary"] = explicit_repair_demo.summary

    if diag_grid is not None:
        tables["diag_noarb_grid"] = diag_grid.tables.get("noarb_smiles", pd.DataFrame())
    if diag_svi_fx is not None:
        tables["diag_noarb_svi_fx"] = diag_svi_fx.tables.get(
            "noarb_smiles", pd.DataFrame()
        )
        if "svi_fit" in diag_svi_fx.tables:
            tables["diag_svi_fit"] = diag_svi_fx.tables["svi_fit"]

    if repricing is not None:
        tables["repricing_grid"] = repricing.grid
        tables["repricing_summary"] = repricing.summary

    if convergence is not None:
        tables["convergence_grid"] = convergence.grid

    if digital_baseline is not None:
        tables["digital_ok"] = digital_baseline.ok
        tables["digital_errors"] = digital_baseline.errors
        tables["digital_grouped"] = digital_baseline.grouped
        tables["digital_frontier"] = digital_baseline.frontier

    if lv_cmp is not None:
        tables["lv_compare_summary"] = pd.DataFrame([lv_cmp.summary])
        tables["lv_compare_worst_diffs"] = lv_cmp.worst_diffs
        tables["lv_compare_gatheral_reasons"] = pd.DataFrame(
            [lv_cmp.gatheral_reason_counts]
        )
        tables["lv_compare_dupire_reasons"] = pd.DataFrame(
            [lv_cmp.dupire_reason_counts]
        )

    if gj51 is not None:
        tables["gj51_comparison"] = gj_example51_comparison_table(gj51)
        tables["gj51_checks"] = gj_example51_check_table(gj51)

    reports: dict[str, Any] = {
        "noarb_true": noarb_true,
        "noarb_grid": noarb_grid,
        "noarb_svi_nr": noarb_svi_nr,
        "noarb_svi_fx": noarb_svi_fx,
        "lv_rep": lv_rep,
        "diag_grid": diag_grid,
        "diag_svi_nr": diag_svi_nr,
        "diag_svi_fx": diag_svi_fx,
        "explicit_repair_demo": explicit_repair_demo,
        "lv_compare": lv_cmp,
    }

    surfaces = {
        "grid": surface_grid,
        "svi_norepair": surface_svi_norepair,
        "svi_repaired": surface_svi_repaired,
        "true_grid": surface_true_grid,
    }

    synthetic_bundle = {
        "quotes_df": quotes_df,
        "rows_obs": rows_obs,
        "rows_true": rows_true,
        "surface_true_grid": surface_true_grid,
        "noarb_true_grid": noarb_true_grid,
        "forward": forward,
        "df_curve": df_curve,
        "synth_cfg_used": synth_cfg_used,
        "synth_tuning_log": synth_tuning_log,
    }

    meta = {
        "svi_repair_mode_used": svi_repair_mode_used,
        "svi_repair_fallback_used": bool(svi_repair_fallback_used),
        "focus_T": float(focus_T),
        "profile": profile,
    }

    return Capstone2Artifacts(
        profile=profile,
        cfg=cfg,
        flags=run_flags,
        synthetic=synthetic_bundle,
        surfaces=surfaces,
        localvol=localvol,
        tables=tables,
        reports=reports,
        repricing=repricing,
        convergence=convergence,
        digital_baseline=digital_baseline,
        gj51=gj51,
        meta=meta,
    )
