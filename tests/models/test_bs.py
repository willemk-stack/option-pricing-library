import math

import numpy as np

from option_pricing.pricers.black_scholes import bs_price_call, bs_price_put
from option_pricing.types import OptionType


def test_put_call_parity_with_dividends(make_inputs):
    """C - P = S*exp(-q*tau) - K*exp(-r*tau) for European options."""
    p_call = make_inputs(
        S=100.0,
        K=105.0,
        r=0.03,
        q=0.02,
        sigma=0.25,
        T=1.2,
        t=0.0,
        kind=OptionType.CALL,
    )
    p_put = make_inputs(
        S=p_call.S,
        K=p_call.K,
        r=p_call.market.rate,
        q=p_call.market.dividend_yield,
        sigma=p_call.sigma,
        T=p_call.T,
        t=p_call.t,
        kind=OptionType.PUT,
    )

    tau = p_call.tau
    df_r = math.exp(-p_call.market.rate * tau)
    df_q = math.exp(-p_call.market.dividend_yield * tau)

    C = float(bs_price_call(p_call))
    P = float(bs_price_put(p_put))

    assert abs((C - P) - (p_call.S * df_q - p_call.K * df_r)) < 1e-8


def test_call_bounds(make_inputs):
    """max(S*e^{-q tau}-K*e^{-r tau},0) <= C <= S*e^{-q tau}."""
    p = make_inputs(
        S=120.0, K=100.0, r=0.04, q=0.01, sigma=0.3, T=0.75, kind=OptionType.CALL
    )
    tau = p.tau
    df_r = math.exp(-p.market.rate * tau)
    df_q = math.exp(-p.market.dividend_yield * tau)

    C = float(bs_price_call(p))
    lower = max(p.S * df_q - p.K * df_r, 0.0)
    upper = p.S * df_q

    assert lower - 1e-12 <= C <= upper + 1e-12


def test_put_bounds(make_inputs):
    """max(K*e^{-r tau}-S*e^{-q tau},0) <= P <= K*e^{-r tau}."""
    p = make_inputs(
        S=80.0, K=100.0, r=0.02, q=0.01, sigma=0.35, T=1.4, kind=OptionType.PUT
    )
    tau = p.tau
    df_r = math.exp(-p.market.rate * tau)
    df_q = math.exp(-p.market.dividend_yield * tau)

    P = float(bs_price_put(p))
    lower = max(p.K * df_r - p.S * df_q, 0.0)
    upper = p.K * df_r

    assert lower - 1e-12 <= P <= upper + 1e-12


def test_call_monotone_decreasing_in_strike(make_inputs):
    """For fixed (S,tau,r,q,sigma), call price should be non-increasing in strike."""
    S = 100.0
    r = 0.05
    q = 0.01
    sigma = 0.2
    T = 1.0

    strikes = np.array([60, 80, 100, 120, 140], dtype=float)
    prices = np.array(
        [
            bs_price_call(
                make_inputs(
                    S=S, K=float(K), r=r, q=q, sigma=sigma, T=T, kind=OptionType.CALL
                )
            )
            for K in strikes
        ],
        dtype=float,
    )

    assert np.all(np.diff(prices) <= 1e-10)


def test_put_monotone_increasing_in_strike(make_inputs):
    """For fixed (S,tau,r,q,sigma), put price should be non-decreasing in strike."""
    S = 100.0
    r = 0.05
    q = 0.01
    sigma = 0.2
    T = 1.0

    strikes = np.array([60, 80, 100, 120, 140], dtype=float)
    prices = np.array(
        [
            bs_price_put(
                make_inputs(
                    S=S, K=float(K), r=r, q=q, sigma=sigma, T=T, kind=OptionType.PUT
                )
            )
            for K in strikes
        ],
        dtype=float,
    )

    assert np.all(np.diff(prices) >= -1e-10)
