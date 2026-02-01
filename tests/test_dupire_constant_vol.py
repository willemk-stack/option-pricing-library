from __future__ import annotations

import numpy as np
import pytest

from option_pricing.pricers.black_scholes import black76_call_prices_vec_from_ctx
from option_pricing.types import MarketData
from option_pricing.vol.dupire import local_vol_from_call_grid


@pytest.mark.parametrize("strike_coordinate", ["logK", "K"])
@pytest.mark.parametrize("price_convention", ["discounted", "forward"])
def test_dupire_recovers_constant_vol(
    strike_coordinate: str, price_convention: str
) -> None:
    """In a constant-vol Black-Scholes world, Dupire local vol should be ~ constant."""

    S0 = 100.0
    r = 0.02
    q = 0.01
    sigma0 = 0.20

    market = MarketData(spot=S0, rate=r, dividend_yield=q)
    ctx = market.to_context()

    # Dupire implementation requires >=3 maturities and >=3 strikes.
    taus = np.asarray([0.25, 0.50, 1.00, 2.00, 3.00], dtype=float)
    # Use a denser, more uniform maturity grid to improve the T-derivative.
    taus = np.linspace(0.25, 3.00, 23, dtype=float)  # step ≈ 0.125

    # Use more strikes (smaller dK) for a much tighter curvature estimate.
    # Keep a moderate range to avoid very small curvature in the far wings.
    strikes = np.linspace(0.85 * S0, 1.15 * S0, 121, dtype=float)  # step ≈ 0.25

    # Build call-price grid from Black-76 (discounted PV), then convert if needed.
    call_pv = np.vstack(
        [
            black76_call_prices_vec_from_ctx(
                ctx=ctx,
                strikes=strikes,
                sigma=sigma0,
                tau=float(tau),
            )
            for tau in taus
        ]
    ).astype(float, copy=False)

    if price_convention == "discounted":
        call_grid = call_pv
    elif price_convention == "forward":
        df = np.asarray([ctx.df(float(tau)) for tau in taus], dtype=float)[:, None]
        call_grid = call_pv / df
    else:  # pragma: no cover
        raise ValueError("Unexpected price_convention in test")

    res = local_vol_from_call_grid(
        call=call_grid,
        strikes=strikes,
        taus=taus,
        market=market,
        price_convention=price_convention,  # type: ignore[arg-type]
        strike_coordinate=strike_coordinate,  # type: ignore[arg-type]
        # Trimming is part of the intended guardrails.
        trim_t=1,
        trim_k=1,
    )

    # In a clean BS grid, we expect only the trimmed boundary cells to be invalid.
    nT, nK = call_grid.shape
    expected_invalid = 2 * nK + 2 * (nT - 2)
    assert int(res.invalid_count) == expected_invalid

    # Compare only on valid (untrimmed) points.
    mask = ~np.asarray(res.invalid, dtype=bool)
    assert mask.any()
    sigma_est = np.asarray(res.sigma, dtype=float)[mask]

    # Dupire uses finite differences; require a tight-but-not-fragile tolerance.
    assert float(np.nanmean(sigma_est)) == pytest.approx(sigma0, rel=1e-2, abs=5e-4)
    assert float(np.nanmax(np.abs(sigma_est - sigma0))) <= 3e-3
