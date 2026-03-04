from __future__ import annotations

import math

from option_pricing import (
    ExerciseStyle,
    FlatCarryForwardCurve,
    FlatDiscountCurve,
    MarketData,
    OptionSpec,
    OptionType,
    PricingContext,
    PricingInputs,
    VanillaOption,
    bs_price,
    bs_price_from_ctx,
    bs_price_instrument,
)


def test_smoke_convenience_api() -> None:
    market = MarketData(spot=100.0, rate=0.03, dividend_yield=0.01)
    spec = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)
    p = PricingInputs(spec=spec, market=market, sigma=0.2, t=0.0)

    price = float(bs_price(p))
    assert math.isfinite(price)
    assert price > 0.0


def test_smoke_instrument_api() -> None:
    inst = VanillaOption(
        expiry=1.0,
        strike=100.0,
        kind=OptionType.CALL,
        exercise=ExerciseStyle.EUROPEAN,
    )
    market = MarketData(spot=100.0, rate=0.03, dividend_yield=0.01)

    price = float(bs_price_instrument(inst, market=market, sigma=0.2))
    assert math.isfinite(price)
    assert price > 0.0


def test_smoke_curves_first_api() -> None:
    spot = 100.0
    r = 0.03
    q = 0.01
    sigma = 0.2
    tau = 1.0
    strike = 100.0

    discount = FlatDiscountCurve(r)
    forward = FlatCarryForwardCurve(spot=spot, r=r, q=q)
    ctx = PricingContext(spot=spot, discount=discount, forward=forward)

    price = float(
        bs_price_from_ctx(
            kind=OptionType.CALL, strike=strike, sigma=sigma, tau=tau, ctx=ctx
        )
    )
    assert math.isfinite(price)
    assert price > 0.0
