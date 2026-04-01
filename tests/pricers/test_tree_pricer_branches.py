import math

import pytest

from option_pricing.instruments.base import ExerciseStyle
from option_pricing.instruments.digital import DigitalOption
from option_pricing.market.curves import PricingContext
from option_pricing.pricers.tree import (
    binom_price_from_ctx,
    binom_price_instrument_from_ctx,
)
from option_pricing.types import MarketData, OptionType


def _ctx() -> PricingContext:
    return MarketData(spot=100.0, rate=0.05, dividend_yield=0.0).to_context()


def test_tree_invalid_params_raise():
    ctx = _ctx()

    with pytest.raises(ValueError):
        binom_price_from_ctx(
            kind=OptionType.CALL,
            strike=100.0,
            sigma=0.2,
            tau=1.0,
            ctx=ctx,
            n_steps=0,
        )

    with pytest.raises(ValueError):
        binom_price_from_ctx(
            kind=OptionType.CALL,
            strike=100.0,
            sigma=0.0,
            tau=1.0,
            ctx=ctx,
            n_steps=50,
        )

    with pytest.raises(ValueError):
        binom_price_from_ctx(
            kind=OptionType.CALL,
            strike=100.0,
            sigma=0.2,
            tau=0.0,
            ctx=ctx,
            n_steps=50,
        )

    with pytest.raises(ValueError):
        binom_price_from_ctx(
            kind="bad",
            strike=100.0,
            sigma=0.2,
            tau=1.0,
            ctx=ctx,
            n_steps=50,
        )

    with pytest.raises(ValueError):
        binom_price_from_ctx(
            kind=OptionType.CALL,
            strike=100.0,
            sigma=0.2,
            tau=1.0,
            ctx=ctx,
            n_steps=50,
            american=True,
            method="closed_form",
        )


def test_tree_euro_vs_american_put_call():
    ctx = _ctx()
    sigma = 0.2
    tau = 1.0
    strike = 100.0
    n_steps = 50

    euro_call = binom_price_from_ctx(
        kind=OptionType.CALL,
        strike=strike,
        sigma=sigma,
        tau=tau,
        ctx=ctx,
        n_steps=n_steps,
        american=False,
        method="tree",
    )
    am_call = binom_price_from_ctx(
        kind=OptionType.CALL,
        strike=strike,
        sigma=sigma,
        tau=tau,
        ctx=ctx,
        n_steps=n_steps,
        american=True,
        method="tree",
    )
    assert am_call == pytest.approx(euro_call, rel=2e-2)

    euro_put = binom_price_from_ctx(
        kind=OptionType.PUT,
        strike=strike,
        sigma=sigma,
        tau=tau,
        ctx=ctx,
        n_steps=n_steps,
        american=False,
        method="tree",
    )
    am_put = binom_price_from_ctx(
        kind=OptionType.PUT,
        strike=strike,
        sigma=sigma,
        tau=tau,
        ctx=ctx,
        n_steps=n_steps,
        american=True,
        method="tree",
    )
    assert am_put >= euro_put - 1e-12


def test_tree_closed_form_paths():
    ctx = MarketData(spot=100.0, rate=0.03, dividend_yield=0.01).to_context()
    sigma = 0.25
    tau = 0.75
    strike = 95.0
    n_steps = 60

    for kind in (OptionType.CALL, OptionType.PUT):
        price = binom_price_from_ctx(
            kind=kind,
            strike=strike,
            sigma=sigma,
            tau=tau,
            ctx=ctx,
            n_steps=n_steps,
            american=False,
            method="closed_form",
        )
        assert math.isfinite(price)


def test_tree_digital_instrument_smoke():
    ctx = _ctx()
    inst = DigitalOption(
        expiry=1.0,
        strike=100.0,
        payout=1.0,
        kind=OptionType.CALL,
        exercise=ExerciseStyle.EUROPEAN,
    )
    price = binom_price_instrument_from_ctx(
        ctx=ctx,
        inst=inst,
        sigma=0.2,
        n_steps=25,
        method="tree",
    )
    assert math.isfinite(float(price))
    assert price >= 0.0
