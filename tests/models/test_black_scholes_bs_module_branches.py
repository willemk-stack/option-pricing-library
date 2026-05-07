import math

import numpy as np
import pytest

from option_pricing.instruments.digital import DigitalOption
from option_pricing.market.curves import (
    FlatCarryForwardCurve,
    FlatDiscountCurve,
    PricingContext,
)
from option_pricing.models.black_scholes.bs import (
    black76_call_price_vec,
    black76_call_price_vega_vec,
    black76_put_price_vec,
    call_greeks,
    d1_d2_black_76,
    d1_d2_from_spot,
    digital_call_price,
    digital_put_price,
    discount_factor,
    put_greeks,
)
from option_pricing.types import OptionType


def test_discount_factor_and_d1_d2_validation():
    with pytest.raises(ValueError):
        discount_factor(0.01, -0.5)

    with pytest.raises(ValueError):
        d1_d2_from_spot(spot=0.0, strike=100.0, r=0.0, q=0.0, sigma=0.2, tau=1.0)
    with pytest.raises(ValueError):
        d1_d2_from_spot(spot=100.0, strike=0.0, r=0.0, q=0.0, sigma=0.2, tau=1.0)

    with pytest.raises(ValueError):
        d1_d2_black_76(forward=0.0, strike=100.0, tau=1.0, sigma=0.2)
    with pytest.raises(ValueError):
        d1_d2_black_76(forward=100.0, strike=0.0, tau=1.0, sigma=0.2)
    with pytest.raises(ValueError):
        d1_d2_black_76(forward=100.0, strike=100.0, tau=0.0, sigma=0.2)
    with pytest.raises(ValueError):
        d1_d2_black_76(forward=100.0, strike=100.0, tau=1.0, sigma=0.0)


def test_greeks_validation_branches():
    with pytest.raises(ValueError):
        call_greeks(spot=100.0, strike=100.0, r=0.01, q=0.0, sigma=0.0, tau=1.0)
    with pytest.raises(ValueError):
        call_greeks(spot=100.0, strike=100.0, r=0.01, q=0.0, sigma=0.2, tau=0.0)

    with pytest.raises(ValueError):
        put_greeks(spot=100.0, strike=100.0, r=0.01, q=0.0, sigma=0.0, tau=1.0)
    with pytest.raises(ValueError):
        put_greeks(spot=100.0, strike=100.0, r=0.01, q=0.0, sigma=0.2, tau=0.0)


def test_vectorized_black76_prices_and_vega_mask():
    strikes = np.array([90.0, 100.0, 110.0])
    sigma = np.array([0.2, 0.0, 0.3])
    tau = 0.5
    df = 0.98

    call = black76_call_price_vec(
        forward=100.0, strikes=strikes, sigma=sigma, tau=tau, df=df
    )
    put = black76_put_price_vec(
        forward=100.0, strikes=strikes, sigma=sigma, tau=tau, df=df
    )

    assert call.shape == strikes.shape
    assert put.shape == strikes.shape
    assert np.all(np.isfinite(call))
    assert np.all(np.isfinite(put))

    intrinsic = df * np.maximum(100.0 - strikes, 0.0)
    assert call[1] == pytest.approx(intrinsic[1])

    price, vega = black76_call_price_vega_vec(
        forward=100.0, strikes=strikes, sigma=sigma, tau=tau, df=df
    )
    assert price.shape == strikes.shape
    assert vega.shape == strikes.shape
    assert vega[1] == pytest.approx(0.0)


def test_digital_prices_scalar_and_array_sigma():
    discount = FlatDiscountCurve(0.01)
    forward = FlatCarryForwardCurve(spot=100.0, r=0.01, q=0.0)
    ctx = PricingContext(spot=100.0, discount=discount, forward=forward)

    call = DigitalOption(expiry=1.0, strike=100.0, payout=1.0, kind=OptionType.CALL)
    put = DigitalOption(expiry=1.0, strike=100.0, payout=1.0, kind=OptionType.PUT)

    price_call = digital_call_price(call, ctx, sigma=0.2)
    assert math.isfinite(float(price_call))

    price_put = digital_put_price(put, ctx, sigma=np.array([0.2, 0.3]))
    price_put_arr = np.asarray(price_put, dtype=float)
    assert price_put_arr.shape == (2,)
    assert np.all(np.isfinite(price_put_arr))
