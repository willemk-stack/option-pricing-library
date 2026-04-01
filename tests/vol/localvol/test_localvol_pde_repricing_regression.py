from __future__ import annotations

import pytest

from option_pricing.diagnostics.vol_surface.pde_repricing import (
    localvol_pde_repricing_grid,
    localvol_pde_single_option_convergence_sweep,
)
from option_pricing.types import OptionType


def test_localvol_pde_reprices_smoothed_surface_on_small_vanilla_grid(
    quick_demo08_case,
) -> None:
    """Regression guard for demo08 vanilla local-vol repricing.

    This exercises the non-constant implied-surface -> local-vol -> PDE repricing
    path on a small grid representative of the demo notebook. The tolerances are
    intentionally loose enough for a stable demo, but tight enough to catch the
    current blown-out repricing output.
    """

    scenario, bridge, strikes, expiries = quick_demo08_case

    result = localvol_pde_repricing_grid(
        lv=bridge.localvol,
        market=scenario.market,
        strikes=strikes,
        expiries=expiries,
        Nx=101,
        Nt=201,
        solver_cfg=scenario.cfg["LV_PDE_SOLVER_CFG"],
        kind=OptionType.CALL,
        target="black76_from_implied",
        compute_implied_vol=True,
    )

    grid = result.grid
    summary = result.summary.iloc[0]

    assert not grid.empty
    assert len(grid) == len(strikes) * len(expiries)
    assert grid["iv_ok"].all()

    # Demo-level repricing should stay reasonably close to the source surface.
    assert float(summary["mean_abs_price_error"]) <= 0.10
    assert float(summary["max_abs_price_error"]) <= 0.35
    assert float(summary["mean_abs_iv_error_bp"]) <= 100.0
    assert float(summary["max_abs_iv_error_bp"]) <= 250.0


@pytest.mark.slow
def test_localvol_pde_worst_node_gap_closes_under_grid_refinement(
    quick_demo08_case,
) -> None:
    """Diagnostic guard for the demo08 worst repricing node.

    After the local-vol time-convention fix, the worst coarse-grid node should
    remain numerically stable under refinement and the fine-grid PDE price
    should sit close to the implied-surface target price.
    """

    scenario, bridge, strikes, expiries = quick_demo08_case
    coarse_grid = (101, 201)
    reference_grid = (401, 801)

    repricing = localvol_pde_repricing_grid(
        lv=bridge.localvol,
        market=scenario.market,
        strikes=strikes,
        expiries=expiries,
        Nx=coarse_grid[0],
        Nt=coarse_grid[1],
        solver_cfg=scenario.cfg["LV_PDE_SOLVER_CFG"],
        kind=OptionType.CALL,
        target="black76_from_implied",
        compute_implied_vol=True,
    )
    worst_row = repricing.grid.sort_values("abs_price_error", ascending=False).iloc[0]

    sweep = localvol_pde_single_option_convergence_sweep(
        lv=bridge.localvol,
        market=scenario.market,
        strike=float(worst_row["K"]),
        expiry=float(worst_row["T"]),
        grids=[coarse_grid],
        solver_cfg=scenario.cfg["LV_PDE_SOLVER_CFG"],
        kind=OptionType.CALL,
        reference_grid=reference_grid,
    )

    coarse_row = sweep.grid.iloc[0]
    fine_grid_target_gap = abs(
        float(sweep.meta["reference_price"]) - float(sweep.meta["implied_target_price"])
    )

    assert int(coarse_row["Nx"]) == coarse_grid[0]
    assert int(coarse_row["Nt"]) == coarse_grid[1]
    assert float(coarse_row["abs_error"]) <= 0.01
    assert fine_grid_target_gap <= 0.05
    assert fine_grid_target_gap <= 10.0 * float(coarse_row["abs_error"])
