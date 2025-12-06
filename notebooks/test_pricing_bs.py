# tests/test_pricing_bs.py

import math
import numpy as np
import pytest

from option_pricing.pricing_bs import bs_call, bs_put


@pytest.mark.parametrize(
    "t, x, K, r, sigma, T",
    [
        (0.0, 100.0, 100.0, 0.01, 0.2, 1.0),
        (0.0, 120.0, 100.0, 0.03, 0.25, 2.0),
        (0.5, 100.0, 90.0, 0.02, 0.3, 1.5),
    ],
)
def test_put_call_parity(t, x, K, r, sigma, T):
    """
    Check put–call parity:
        C(t, x) - P(t, x) = x - K * exp(-r * (T - t))
    for several parameter choices.
    """
    call_price = bs_call(t, x, K, r, sigma, T)
    put_price = bs_put(t, x, K, r, sigma, T)

    lhs = call_price - put_price
    rhs = x - K * math.exp(-r * (T - t))

    # Analytic formula, so we can use a tight tolerance
    assert lhs == pytest.approx(rhs, rel=1e-10, abs=1e-10)


def test_call_price_decreases_with_strike():
    """
    For fixed (t, x, r, sigma, T), the European call price should be
    decreasing in K (non-increasing mathematically, but strictly
    decreasing in practice for a reasonable range of strikes).
    """
    t = 0.0
    x = 100.0
    r = 0.01
    sigma = 0.2
    T = 1.0

    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

    prices = np.array(
        [bs_call(t, x, K, r, sigma, T) for K in strikes]
    )

    # Check that prices are non-increasing as K increases
    diffs = np.diff(prices)  # prices[i+1] - prices[i]
    assert np.all(diffs <= 1e-10)  # allow tiny numerical noise

    # And at least one strictly negative difference (so not all equal)
    assert np.any(diffs < -1e-5)