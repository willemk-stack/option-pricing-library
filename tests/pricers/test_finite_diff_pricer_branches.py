import math

import pytest

from option_pricing.instruments.vanilla import VanillaOption
from option_pricing.pricers.finite_diff import (
    finite_diff_greeks,
    finite_diff_greeks_instrument,
)
from option_pricing.types import MarketData, OptionSpec, OptionType, PricingInputs


def _pricing_inputs(*, t: float) -> PricingInputs:
    spec = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.0)
    return PricingInputs(spec=spec, market=market, sigma=0.2, t=t)


def test_finite_diff_greeks_forward_and_central_theta():
    greeks_forward = finite_diff_greeks(_pricing_inputs(t=0.0))
    assert all(math.isfinite(v) for v in greeks_forward.values())

    greeks_central = finite_diff_greeks(_pricing_inputs(t=0.5))
    assert all(math.isfinite(v) for v in greeks_central.values())


def test_finite_diff_greeks_validation_errors():
    bad_market = MarketData(spot=0.0, rate=0.02, dividend_yield=0.0)
    spec = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)
    market_ok = MarketData(spot=100.0, rate=0.02, dividend_yield=0.0)
    with pytest.raises(ValueError):
        finite_diff_greeks(PricingInputs(spec=spec, market=bad_market, sigma=0.2))

    bad_strike = OptionSpec(kind=OptionType.CALL, strike=0.0, expiry=1.0)
    with pytest.raises(ValueError):
        finite_diff_greeks(PricingInputs(spec=bad_strike, market=market_ok, sigma=0.2))

    with pytest.raises(ValueError):
        finite_diff_greeks(PricingInputs(spec=spec, market=market_ok, sigma=0.0))

    with pytest.raises(ValueError):
        finite_diff_greeks(_pricing_inputs(t=1.0))


def test_finite_diff_greeks_instrument_smoke():
    inst = VanillaOption(expiry=0.75, strike=100.0, kind=OptionType.CALL)
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.0)
    greeks = finite_diff_greeks_instrument(inst, market=market, sigma=0.2)
    assert all(math.isfinite(v) for v in greeks.values())

    with pytest.raises(ValueError):
        finite_diff_greeks_instrument(inst, market=market, sigma=0.0)

    with pytest.raises(ValueError):
        finite_diff_greeks_instrument(
            inst,
            market=MarketData(spot=0.0, rate=0.02, dividend_yield=0.0),
            sigma=0.2,
        )

    with pytest.raises(ValueError):
        finite_diff_greeks_instrument(
            VanillaOption(expiry=0.0, strike=100.0, kind=OptionType.CALL),
            market=market,
            sigma=0.2,
        )
