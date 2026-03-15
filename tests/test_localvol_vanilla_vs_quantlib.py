from __future__ import annotations

import math

import numpy as np
import pytest

from option_pricing.numerics.pde.domain import Coord
from option_pricing.pricers.pde_pricer import local_vol_price_pde_european
from option_pricing.types import OptionSpec, OptionType, PricingInputs

pytestmark = pytest.mark.slow


def _our_localvol_vanilla_price_from_smoothed_essvi(
    workflow,
    *,
    K: float,
    T: float,
    Nx: int,
    Nt: int,
) -> float:
    p = PricingInputs(
        spec=OptionSpec(kind=OptionType.CALL, strike=float(K), expiry=float(T)),
        market=workflow.shared.market,
        sigma=0.25,
        t=0.0,
    )

    return float(
        local_vol_price_pde_european(
            p,
            lv=workflow.localvol,
            coord=Coord(workflow.solver_cfg["coord"]),
            domain_cfg=workflow.solver_cfg["domain_cfg"],
            Nx=int(Nx),
            Nt=int(Nt),
            method=str(workflow.solver_cfg["method"]),
            advection=workflow.solver_cfg["advection"],
        )
    )


def _mc_vanilla_call_price_from_smoothed_essvi_localvol(
    workflow,
    *,
    K: float,
    T: float,
    n_paths: int = 120_000,
    n_steps: int = 180,
    seed: int = 12345,
) -> tuple[float, float]:
    """Monte Carlo vanilla call under the workflow's local-vol surface.

    This uses the same local-vol surface that feeds the PDE pricer, which makes
    it a cleaner numerical oracle than a sampled external BlackVarianceSurface.
    """
    market = workflow.shared.market
    rng = np.random.default_rng(seed)
    dt = float(T) / float(n_steps)
    sqrt_dt = math.sqrt(dt)
    half_paths = int(n_paths) // 2
    z = rng.standard_normal((int(n_steps), half_paths))
    z = np.concatenate([z, -z], axis=1)

    S = np.full(z.shape[1], float(market.spot), dtype=float)
    drift = float(market.rate - market.dividend_yield)

    for step in range(int(n_steps)):
        calendar_time = min(max(step * dt, 1.0e-8), float(T))
        sigma = np.asarray(workflow.localvol.local_vol(S, calendar_time), dtype=float)
        S *= np.exp((drift - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * z[step])

    payoff = np.maximum(S - float(K), 0.0)
    discount = math.exp(-float(market.rate) * float(T))
    price = discount * float(np.mean(payoff))
    se = discount * float(np.std(payoff, ddof=1) / math.sqrt(payoff.size))
    return price, se


def test_localvol_vanilla_matches_localvol_mc_on_smoothed_essvi_surface_nodes(
    quick_smoothed_localvol_workflow,
) -> None:
    """Cross-check the smoothed-eSSVI local-vol PDE against Monte Carlo.

    The earlier QuantLib-via-sampled-BlackVarianceSurface cross-check turned out to be
    sensitive to the strike sampling grid used to build the external surface. For the
    smoothed eSSVI workflow we therefore validate the PDE directly against Monte Carlo
    under the *same* local-vol surface.

      synthetic quotes -> eSSVI calibration -> smooth projection -> LocalVolSurface
      -> vanilla PDE repricing

    This keeps the regression focused on PDE numerics rather than on an external local-vol
    reconstruction stack built from a sampled volatility surface.
    """
    workflow = quick_smoothed_localvol_workflow

    assert type(workflow.smoothed_surface).__name__ == "ESSVISmoothedSurface"
    assert workflow.localvol.implied is workflow.smoothed_surface

    # First make sure the PDE itself is reasonably grid-stable at a representative node.
    T_anchor = 1.0
    K_anchor = float(workflow.shared.forward(T_anchor))

    ours_coarse = _our_localvol_vanilla_price_from_smoothed_essvi(
        workflow,
        K=K_anchor,
        T=T_anchor,
        Nx=101,
        Nt=201,
    )
    ours_fine = _our_localvol_vanilla_price_from_smoothed_essvi(
        workflow,
        K=K_anchor,
        T=T_anchor,
        Nx=151,
        Nt=301,
    )

    ours_step = abs(ours_fine - ours_coarse)
    assert ours_step < 6.0e-2, (
        f"ours not stable enough: coarse={ours_coarse:.8f}, fine={ours_fine:.8f}, "
        f"step={ours_step:.3g}"
    )

    # Use the nodes where the old QuantLib cross-check showed the largest disagreement.
    # If the PDE still matches Monte Carlo on these contracts, the remaining mismatch is
    # not a vanilla-PDE discretization bug.
    strikes_all = np.asarray(workflow.shared.cfg["SHARED_STRIKES"], dtype=float)
    expiries_all = np.asarray(workflow.shared.expiries, dtype=float)
    nodes = [
        (float(strikes_all[12]), float(expiries_all[8])),
        (float(strikes_all[16]), float(expiries_all[8])),
    ]

    rows: list[dict[str, float]] = []
    for K, T in nodes:
        ours = _our_localvol_vanilla_price_from_smoothed_essvi(
            workflow,
            K=K,
            T=T,
            Nx=151,
            Nt=301,
        )
        mc_price, mc_se = _mc_vanilla_call_price_from_smoothed_essvi_localvol(
            workflow,
            K=K,
            T=T,
        )
        rows.append(
            {
                "T": float(T),
                "K": float(K),
                "ours": float(ours),
                "mc": float(mc_price),
                "mc_se": float(mc_se),
                "abs_diff": float(abs(ours - mc_price)),
            }
        )

    diffs = np.asarray([row["abs_diff"] for row in rows], dtype=float)
    mean_diff = float(np.mean(diffs))
    max_diff = float(np.max(diffs))
    max_mc_se = float(max(row["mc_se"] for row in rows))

    # Basic sanity: all prices should be within vanilla no-arbitrage bounds.
    market = workflow.shared.market
    for row in rows:
        K = float(row["K"])
        T = float(row["T"])
        df = math.exp(-float(market.rate) * T)
        lower = max(
            0.0,
            float(market.spot) * math.exp(-float(market.dividend_yield) * T) - K * df,
        )
        upper = float(market.spot) * math.exp(-float(market.dividend_yield) * T)
        assert lower - 1e-8 <= row["ours"] <= upper + 1e-8
        assert lower - 1e-8 <= row["mc"] <= upper + 1e-8

    tol = max(1.2e-1, 3.0 * max_mc_se)
    assert max_diff < tol, (
        f"max abs price diff too large: max_diff={max_diff:.4f}, tol={tol:.4f}, "
        f"mean_diff={mean_diff:.4f}, rows={rows}"
    )
    assert mean_diff < 0.75 * tol, (
        f"mean abs price diff too large: mean_diff={mean_diff:.4f}, "
        f"tol={0.75 * tol:.4f}, rows={rows}"
    )
