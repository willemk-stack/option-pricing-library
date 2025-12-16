import math

import numpy as np

from option_pricing.models.bs import bs_call, bs_put


def test_put_call_parity_no_dividends():
    """C - P = S - K*exp(-rT) for European options with q=0."""
    t = 0.0
    S = 100.0
    K = 105.0
    r = 0.03
    sigma = 0.25
    T = 1.2

    df = math.exp(-r * (T - t))
    C = float(bs_call(t=t, x=S, K=K, r=r, sigma=sigma, T=T))
    P = float(bs_put(t=t, x=S, K=K, r=r, sigma=sigma, T=T))

    assert abs((C - P) - (S - K * df)) < 1e-8


def test_call_bounds():
    """max(S-K*df,0) <= C <= S."""
    t = 0.0
    S = 120.0
    K = 100.0
    r = 0.04
    sigma = 0.3
    T = 0.75

    df = math.exp(-r * (T - t))
    C = float(bs_call(t=t, x=S, K=K, r=r, sigma=sigma, T=T))

    lower = max(S - K * df, 0.0)
    upper = S

    assert lower - 1e-12 <= C <= upper + 1e-12


def test_put_bounds():
    """max(K*df-S,0) <= P <= K*df."""
    t = 0.0
    S = 80.0
    K = 100.0
    r = 0.02
    sigma = 0.35
    T = 1.4

    df = math.exp(-r * (T - t))
    P = float(bs_put(t=t, x=S, K=K, r=r, sigma=sigma, T=T))

    lower = max(K * df - S, 0.0)
    upper = K * df

    assert lower - 1e-12 <= P <= upper + 1e-12


def test_call_monotone_decreasing_in_strike():
    """For fixed (S,T,r,sigma), call price should be non-increasing in strike."""
    t = 0.0
    S = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0

    strikes = np.array([60, 80, 100, 120, 140], dtype=float)
    prices = np.array(
        [bs_call(t=t, x=S, K=float(K), r=r, sigma=sigma, T=T) for K in strikes],
        dtype=float,
    )

    assert np.all(np.diff(prices) <= 1e-10)


def test_put_monotone_increasing_in_strike():
    """For fixed (S,T,r,sigma), put price should be non-decreasing in strike."""
    t = 0.0
    S = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0

    strikes = np.array([60, 80, 100, 120, 140], dtype=float)
    prices = np.array(
        [bs_put(t=t, x=S, K=float(K), r=r, sigma=sigma, T=T) for K in strikes],
        dtype=float,
    )

    assert np.all(np.diff(prices) >= -1e-10)
