import math

import pytest

from option_pricing.market.curves import (
    FlatCarryForwardCurve,
    FlatDiscountCurve,
    PricingContext,
    avg_carry_from_forward,
    avg_rate_from_df,
)


class _ConstForward:
    def __init__(self, value: float) -> None:
        self._value = float(value)

    def fwd(self, _tau: float) -> float:
        return float(self._value)


class _ConstDiscount:
    def __init__(self, value: float) -> None:
        self._value = float(value)

    def df(self, _tau: float) -> float:
        return float(self._value)


def test_flat_curves_validation_and_call():
    curve = FlatDiscountCurve(0.03)
    assert curve(1.0) == pytest.approx(curve.df(1.0))
    with pytest.raises(ValueError):
        curve.df(-0.1)

    fwd = FlatCarryForwardCurve(spot=100.0, r=0.05, q=0.02)
    assert fwd(0.5) == pytest.approx(fwd.fwd(0.5))
    with pytest.raises(ValueError):
        fwd.fwd(-0.5)
    with pytest.raises(ValueError):
        FlatCarryForwardCurve(spot=0.0, r=0.01).fwd(0.5)
    with pytest.raises(ValueError):
        fwd.forward(T=0.0, t=0.1)


def test_pricing_context_avg_rates_and_df_q():
    discount = FlatDiscountCurve(0.03)
    forward = FlatCarryForwardCurve(spot=100.0, r=0.03, q=0.01)
    ctx = PricingContext(spot=100.0, discount=discount, forward=forward)

    tau = 1.2
    df = ctx.df(tau)
    fwd = ctx.fwd(tau)
    assert math.isfinite(df)
    assert math.isfinite(fwd)
    assert math.isfinite(ctx.prepaid_forward(tau))

    assert ctx.r_avg(0.0) == 0.0
    assert ctx.b_avg(0.0) == 0.0
    assert ctx.df_q(0.0) == 1.0

    r_avg = ctx.r_avg(tau)
    b_avg = ctx.b_avg(tau)
    q_avg = ctx.q_avg(tau)

    assert r_avg == pytest.approx(0.03, rel=1e-3)
    assert b_avg == pytest.approx(0.02, rel=1e-3)
    assert q_avg == pytest.approx(0.01, rel=1e-3)
    assert ctx.df_q(tau) == pytest.approx(math.exp(-0.01 * tau), rel=1e-6)


def test_pricing_context_invalid_inputs_raise():
    discount = _ConstDiscount(1.0)
    forward = _ConstForward(100.0)
    ctx = PricingContext(spot=100.0, discount=discount, forward=forward)

    with pytest.raises(ValueError):
        ctx.r_avg(-0.1)
    with pytest.raises(ValueError):
        ctx.b_avg(-0.1)
    with pytest.raises(ValueError):
        ctx.df_q(-0.1)

    bad_df_ctx = PricingContext(
        spot=100.0, discount=_ConstDiscount(-1.0), forward=forward
    )
    with pytest.raises(ValueError):
        bad_df_ctx.r_avg(1.0)

    bad_spot_ctx = PricingContext(spot=-1.0, discount=discount, forward=forward)
    with pytest.raises(ValueError):
        bad_spot_ctx.b_avg(1.0)

    bad_fwd_ctx = PricingContext(
        spot=100.0, discount=discount, forward=_ConstForward(-1.0)
    )
    with pytest.raises(ValueError):
        bad_fwd_ctx.b_avg(1.0)


def test_avg_rate_and_carry_validation():
    with pytest.raises(ValueError):
        avg_rate_from_df(df=0.99, tau=0.0)
    with pytest.raises(ValueError):
        avg_rate_from_df(df=0.0, tau=1.0)

    with pytest.raises(ValueError):
        avg_carry_from_forward(spot=100.0, forward=101.0, tau=0.0)
    with pytest.raises(ValueError):
        avg_carry_from_forward(spot=0.0, forward=101.0, tau=1.0)
    with pytest.raises(ValueError):
        avg_carry_from_forward(spot=100.0, forward=0.0, tau=1.0)

    assert math.isfinite(avg_rate_from_df(df=math.exp(-0.05), tau=1.0))
    assert math.isfinite(avg_carry_from_forward(spot=100.0, forward=105.0, tau=1.0))
