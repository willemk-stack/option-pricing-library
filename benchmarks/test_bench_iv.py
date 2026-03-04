from __future__ import annotations

import numpy as np
import pytest

from option_pricing.models.black_scholes.bs import black76_call_price_vec
from option_pricing.pricers.black_scholes import bs_price_from_ctx
from option_pricing.types import MarketData, OptionSpec, OptionType
from option_pricing.vol.implied_vol_scalar import implied_vol_bs_result
from option_pricing.vol.implied_vol_slice import implied_vol_black76_slice


@pytest.mark.parametrize("nK", [21, 61, 201, 801])
def test_bench_iv_slice_scaling(benchmark, nK: int) -> None:
    forward = 100.0
    df = 0.99
    tau = 0.5
    sigma = 0.2

    strikes = np.linspace(60.0, 140.0, nK, dtype=float)
    prices = black76_call_price_vec(
        forward=forward, strikes=strikes, sigma=sigma, tau=tau, df=df
    )
    benchmark(
        implied_vol_black76_slice,
        forward=forward,
        strikes=strikes,
        tau=tau,
        df=df,
        prices=prices,
        is_call=True,
        initial_sigma=sigma,
        return_result=False,
    )


@pytest.mark.parametrize(
    "strike,tau",
    [(100.0, 1.0), (100.0, 0.02), (140.0, 1.0)],
)
def test_bench_iv_scalar_scenarios(benchmark, strike: float, tau: float) -> None:
    market = MarketData(spot=100.0, rate=0.01, dividend_yield=0.0)
    ctx = market.to_context()
    sigma = 0.2

    price = bs_price_from_ctx(
        kind=OptionType.CALL, strike=strike, sigma=sigma, tau=tau, ctx=ctx
    )
    spec = OptionSpec(kind=OptionType.CALL, strike=strike, expiry=tau)

    def _run() -> object:
        res = implied_vol_bs_result(price, spec, market)
        _ = res.root_result
        return res

    benchmark(_run)
