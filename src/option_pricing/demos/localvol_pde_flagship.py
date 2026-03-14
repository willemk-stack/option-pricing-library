from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from option_pricing.diagnostics.pde_vs_digital import run_demo_baseline_sweep
from option_pricing.diagnostics.vol_surface import localvol as vs_localvol
from option_pricing.diagnostics.vol_surface.pde_repricing import (
    LocalVolConvergenceSweepResult,
    LocalVolRepricingResult,
    localvol_pde_repricing_grid,
    localvol_pde_single_option_convergence_sweep,
)
from option_pricing.numerics.pde.domain import Coord
from option_pricing.types import DigitalSpec, OptionType, PricingInputs

from .essvi_bridge import ESSVIBridgeArtifacts, build_essvi_bridge_artifacts
from .scenario import (
    SharedDemoScenario,
    build_shared_demo_scenario,
    resolve_run_flag,
)


@dataclass(frozen=True, slots=True)
class LocalVolPDEDemoArtifacts:
    profile: str
    cfg: dict[str, Any]
    scenario: SharedDemoScenario
    bridge: ESSVIBridgeArtifacts
    localvol: Any
    repricing: LocalVolRepricingResult | None
    convergence: LocalVolConvergenceSweepResult | None
    digital_baseline: Any | None
    tables: dict[str, pd.DataFrame]
    reports: dict[str, Any]
    meta: dict[str, Any]


def _solver_coord(solver_cfg: dict[str, Any]) -> Coord | str:
    coord = solver_cfg.get("coord", Coord.LOG_S)
    if isinstance(coord, Coord | str):
        return coord
    raise TypeError(f"coord must be a Coord or str, got {type(coord).__name__}")


def _reason_counts_table(reason_counts: dict[str, int]) -> pd.DataFrame:
    df = pd.DataFrame([{"reason": k, "count": v} for k, v in reason_counts.items()])
    if df.empty:
        return df
    return df.sort_values("count", ascending=False).reset_index(drop=True)


def _repricing_axes(cfg: dict[str, Any], profile: str) -> tuple[np.ndarray, np.ndarray]:
    strikes = np.asarray(cfg["SHARED_STRIKES"], dtype=np.float64)
    expiries = np.asarray(cfg["SHARED_EXPIRIES"], dtype=np.float64)
    if profile == "full":
        return strikes[2:-2:2], expiries
    return strikes[4:-4:4], expiries[1::2]


def _repricing_grid_shape(profile: str) -> tuple[int, int]:
    return (151, 301) if profile == "full" else (101, 201)


def _convergence_grids(cfg: dict[str, Any], profile: str) -> list[tuple[int, int]]:
    grids = list(cfg["LV_SWEEP_GRIDS"])
    return grids if profile == "full" else grids[:3]


def _run_pde_anchor(*, profile: str, solver_cfg: dict[str, Any], market):
    coord = _solver_coord(solver_cfg)
    p_dig = PricingInputs(
        spec=DigitalSpec(
            kind=OptionType.CALL,
            strike=100.0,
            expiry=1.0,
            payout=1.0,
        ),
        market=market,
        sigma=0.22,
    )
    if profile == "full":
        return run_demo_baseline_sweep(
            p_dig,
            methods=("cn", "rannacher"),
            advections=("central", "upwind"),
            n_sigmas=(4.0, 5.0),
            Nx_list=(101, 151),
            Nt_list=(201, 401),
            coord=coord,
            spacing="uniform",
            ic_remedy="cell_avg",
            tol_abs=5e-4,
            tol_rel=5e-3,
            budget_ms=250.0,
        )
    return run_demo_baseline_sweep(
        p_dig,
        methods=("rannacher",),
        advections=("central",),
        n_sigmas=(5.0,),
        Nx_list=(101,),
        Nt_list=(201,),
        coord=coord,
        spacing="uniform",
        ic_remedy="cell_avg",
        tol_abs=5e-4,
        tol_rel=5e-3,
        budget_ms=250.0,
    )


def build_localvol_pde_demo_artifacts(
    *,
    profile: str = "quick",
    seed: int = 7,
    overrides: dict[str, Any] | None = None,
    scenario: SharedDemoScenario | None = None,
    bridge_artifacts: ESSVIBridgeArtifacts | None = None,
) -> LocalVolPDEDemoArtifacts:
    shared = (
        build_shared_demo_scenario(profile=profile, seed=seed, overrides=overrides)
        if scenario is None
        else scenario
    )
    bridge = (
        build_essvi_bridge_artifacts(
            profile=shared.profile,
            seed=shared.seed,
            overrides=overrides,
            scenario=shared,
        )
        if bridge_artifacts is None
        else bridge_artifacts
    )

    localvol = bridge.localvol
    lv_rep = vs_localvol.localvol_grid_diagnostics(
        localvol,
        expiries=shared.cfg["LV_DIAG_EXPIRIES"],
        y_grid=shared.cfg["LV_DIAG_YGRID"],
        eps_w=1e-12,
        eps_denom=1e-12,
        top_n=15,
    )

    lv_compare = None
    if resolve_run_flag(
        name="RUN_DUPIRE_VS_GATHERAL_COMPARE",
        default=shared.profile == "full",
        overrides=overrides,
    ):
        repricing_strikes, repricing_expiries = _repricing_axes(
            shared.cfg, shared.profile
        )
        lv_compare = vs_localvol.localvol_compare_gatheral_vs_dupire(
            localvol,
            expiries=repricing_expiries.tolist(),
            strikes=repricing_strikes,
            market=shared.market,
            price_convention="discounted",
            strike_coordinate="logK",
            trim_t=1,
            trim_k=1,
            top_n=12,
        )

    solver_cfg = shared.cfg["LV_PDE_SOLVER_CFG"]
    digital_baseline = None
    if resolve_run_flag(
        name="RUN_DIGITAL_PDE_BASELINE",
        default=True,
        overrides=overrides,
    ):
        digital_baseline = _run_pde_anchor(
            profile=shared.profile,
            solver_cfg=solver_cfg,
            market=shared.market,
        )

    repricing_strikes, repricing_expiries = _repricing_axes(shared.cfg, shared.profile)
    repricing = None
    if resolve_run_flag(
        name="RUN_LOCALVOL_REPRICING",
        default=True,
        overrides=overrides,
    ):
        Nx, Nt = _repricing_grid_shape(shared.profile)
        repricing = localvol_pde_repricing_grid(
            lv=localvol,
            market=shared.market,
            strikes=repricing_strikes,
            expiries=repricing_expiries,
            Nx=Nx,
            Nt=Nt,
            solver_cfg=solver_cfg,
            kind=OptionType.CALL,
            target="black76_from_implied",
            compute_implied_vol=True,
        )

    T0 = 1.0
    K0 = float(shared.forward(T0))
    convergence = None
    if resolve_run_flag(
        name="RUN_LOCALVOL_CONVERGENCE_SWEEP",
        default=True,
        overrides=overrides,
    ):
        convergence = localvol_pde_single_option_convergence_sweep(
            lv=localvol,
            market=shared.market,
            strike=K0,
            expiry=T0,
            grids=_convergence_grids(shared.cfg, shared.profile),
            solver_cfg=solver_cfg,
            kind=OptionType.CALL,
        )

    tables: dict[str, pd.DataFrame] = {
        "quote_summary": shared.quote_summary,
        "localvol_summary": pd.DataFrame([vs_localvol.localvol_summary(lv_rep)]),
        "localvol_reason_counts": _reason_counts_table(lv_rep.reason_counts),
        "localvol_worst_points": lv_rep.worst_points,
    }
    if digital_baseline is not None:
        tables["pde_anchor_ok"] = digital_baseline.ok
        tables["pde_anchor_errors"] = digital_baseline.errors
        tables["pde_anchor_grouped"] = digital_baseline.grouped
        tables["pde_anchor_frontier"] = digital_baseline.frontier
    if repricing is not None:
        tables["repricing_grid"] = repricing.grid
        tables["repricing_summary"] = repricing.summary
    if convergence is not None:
        tables["convergence_grid"] = convergence.grid

    if lv_compare is not None:
        tables["lv_compare_summary"] = pd.DataFrame([lv_compare.summary])
        tables["lv_compare_worst_diffs"] = lv_compare.worst_diffs
        tables["lv_compare_gatheral_reasons"] = pd.DataFrame(
            [lv_compare.gatheral_reason_counts]
        )
        tables["lv_compare_dupire_reasons"] = pd.DataFrame(
            [lv_compare.dupire_reason_counts]
        )

    reports: dict[str, Any] = {
        "lv_rep": lv_rep,
        "lv_compare": lv_compare,
        "digital_baseline": digital_baseline,
    }
    meta = {
        "profile": shared.profile,
        "input_surface_type": type(bridge.smoothed_surface).__name__,
        "repricing_target_surface_type": type(localvol.implied).__name__,
        "repricing_strikes": repricing_strikes.tolist(),
        "repricing_expiries": repricing_expiries.tolist(),
    }

    return LocalVolPDEDemoArtifacts(
        profile=shared.profile,
        cfg=shared.cfg,
        scenario=shared,
        bridge=bridge,
        localvol=localvol,
        repricing=repricing,
        convergence=convergence,
        digital_baseline=digital_baseline,
        tables=tables,
        reports=reports,
        meta=meta,
    )


__all__ = ["LocalVolPDEDemoArtifacts", "build_localvol_pde_demo_artifacts"]
