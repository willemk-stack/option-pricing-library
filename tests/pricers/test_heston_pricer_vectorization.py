from __future__ import annotations

import numpy as np
import pytest

from option_pricing.models.heston import HestonParams
from option_pricing.pricers.heston import (
    heston_price_call_from_ctx,
    heston_price_from_ctx,
    heston_price_put_from_ctx,
)
from option_pricing.types import MarketData, OptionType


def _ctx():
    return MarketData(spot=100.0, rate=0.02, dividend_yield=0.0).to_context()


def _params() -> HestonParams:
    return HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05)


def test_heston_call_slice_matches_scalar_loop() -> None:
    ctx = _ctx()
    params = _params()
    tau = 1.0
    strikes = np.linspace(80.0, 120.0, 7)

    slice_prices = heston_price_call_from_ctx(
        strike=strikes,
        ctx=ctx,
        tau=tau,
        params=params,
    )
    scalar_loop_prices = np.asarray(
        [
            heston_price_call_from_ctx(
                strike=float(strike),
                ctx=ctx,
                tau=tau,
                params=params,
            )
            for strike in strikes
        ],
        dtype=float,
    )

    assert isinstance(slice_prices, np.ndarray)
    assert slice_prices.shape == strikes.shape
    assert np.allclose(slice_prices, scalar_loop_prices, atol=1e-10, rtol=0.0)


def test_heston_put_slice_matches_scalar_loop() -> None:
    ctx = _ctx()
    params = _params()
    tau = 1.0
    strikes = np.linspace(80.0, 120.0, 7)

    slice_prices = heston_price_put_from_ctx(
        strike=strikes,
        tau=tau,
        ctx=ctx,
        params=params,
    )
    scalar_loop_prices = np.asarray(
        [
            heston_price_put_from_ctx(
                strike=float(strike),
                tau=tau,
                ctx=ctx,
                params=params,
            )
            for strike in strikes
        ],
        dtype=float,
    )

    assert isinstance(slice_prices, np.ndarray)
    assert slice_prices.shape == strikes.shape
    assert np.allclose(slice_prices, scalar_loop_prices, atol=1e-10, rtol=0.0)


def test_heston_pricer_preserves_scalar_return_type() -> None:
    ctx = _ctx()
    params = _params()

    price = heston_price_from_ctx(
        kind=OptionType.CALL,
        strike=100.0,
        tau=1.0,
        ctx=ctx,
        params=params,
    )

    assert isinstance(price, float)


def test_heston_pricer_rejects_nonpositive_strikes_in_array() -> None:
    ctx = _ctx()
    params = _params()

    with pytest.raises(ValueError, match="strike\\(s\\) must be positive"):
        heston_price_from_ctx(
            kind=OptionType.PUT,
            strike=np.array([80.0, 0.0, 100.0], dtype=np.float64),
            tau=1.0,
            ctx=ctx,
            params=params,
        )
