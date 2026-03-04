"""Notebook-friendly orchestration helpers for digital PDE diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from option_pricing.numerics.grids import SpacingPolicy
from option_pricing.numerics.pde.domain import Coord
from option_pricing.pricers.pde.domain import BSDomainConfig, BSDomainPolicy
from option_pricing.types import DigitalSpec, PricingInputs

from .._pde_compare.tables import (
    add_decision_columns,
    order_columns,
    pareto_frontier,
    to_frame,
)
from . import compute as pde_dig


@dataclass(frozen=True, slots=True)
class DigitalPdeBaselineResult:
    """Result bundle for the demo-7 digital PDE baseline sweep."""

    frame: pd.DataFrame
    ok: pd.DataFrame
    errors: pd.DataFrame
    grouped: pd.DataFrame
    frontier: pd.DataFrame


def run_demo_baseline_sweep(
    p: PricingInputs[DigitalSpec],
    *,
    methods: tuple[str, ...] = ("cn", "rannacher"),
    advections: tuple[str, ...] = ("central", "upwind"),
    n_sigmas: tuple[float, ...] = (4.0, 5.0),
    Nx_list: tuple[int, ...] = (101, 151, 201),
    Nt_list: tuple[int, ...] = (101, 201, 401),
    coord: Coord | str = Coord.LOG_S,
    spacing: str = "uniform",
    ic_remedy: str = "cell_avg",
    tol_abs: float = 5e-4,
    tol_rel: float = 5e-3,
    budget_ms: float = 250.0,
) -> DigitalPdeBaselineResult:
    """Run the compact baseline sweep showcased in demo 7.

    This owns the nested-method/grid/domain loops so notebooks can focus on the
    resulting tables and plots.
    """

    digital_records: list[dict[str, object]] = []

    for method in methods:
        for advection in advections:
            for n_sigma in n_sigmas:
                domain_cfg = BSDomainConfig(
                    policy=BSDomainPolicy.LOG_NSIGMA,
                    n_sigma=float(n_sigma),
                    center="strike",
                    spacing=SpacingPolicy.UNIFORM,
                )
                recs = pde_dig.sweep_nx_nt(
                    p,
                    domain_cfg=domain_cfg,
                    Nx_list=Nx_list,
                    Nt_list=Nt_list,
                    coord=coord,
                    method=method,
                    advection=advection,
                    spacing=spacing,
                    ic_remedy=ic_remedy,
                )
                for rec in recs:
                    rec["domain_n_sigma"] = float(n_sigma)
                digital_records.extend(recs)

    frame = to_frame(digital_records)
    if "status" in frame.columns:
        ok = frame[frame["status"].fillna("ok") == "ok"].copy()
        errors = frame[frame["status"].fillna("ok") != "ok"].copy()
    else:
        ok = frame.copy()
        errors = frame.iloc[0:0].copy()

    ok = order_columns(
        add_decision_columns(
            ok,
            tol_abs=tol_abs,
            tol_rel=tol_rel,
            budget_ms=budget_ms,
        )
    )

    grouped = (
        ok.groupby(["method", "advection", "Nx", "Nt"], as_index=False)
        .agg(
            n=("abs_err", "size"),
            abs_err_mean=("abs_err", "mean"),
            abs_err_max=("abs_err", "max"),
            runtime_ms_mean=("runtime_ms", "mean"),
            feasible_rate=("feasible_and_within_budget", "mean"),
        )
        .sort_values(["abs_err_mean", "runtime_ms_mean"])
        .reset_index(drop=True)
    )
    frontier = pareto_frontier(ok).reset_index(drop=True)

    return DigitalPdeBaselineResult(
        frame=frame,
        ok=ok,
        errors=errors,
        grouped=grouped,
        frontier=frontier,
    )


__all__ = ["DigitalPdeBaselineResult", "run_demo_baseline_sweep"]
