"""Heuristic seed generation for implied volatility root finding."""

from __future__ import annotations

from math import exp, log, pi, sqrt

from ..market.curves import PricingContext
from ..types import MarketData, OptionSpec, OptionType
from .implied_vol_common import _to_ctx


def _iv_seed_from_time_value(
    mkt_price: float,
    spec: OptionSpec,
    market: MarketData | PricingContext,
    tau: float,
    *,
    sigma_lo: float,
    sigma_hi: float,
) -> float:
    """Heuristic initial guess for implied volatility from time value.

    Produces a robust starting volatility for numerical inversion by combining:

    - an ATM time-value approximation, effective for small log-moneyness, and
    - a moneyness-based (MK-style) scaling for far-from-ATM options,

    then clamping the result to the solver domain ``[sigma_lo, sigma_hi]``.

    Parameters
    ----------
    mkt_price : float
        Observed market option price (discounted).
    spec : OptionSpec
        Option specification (kind, strike, expiry).
    market : MarketData
        Market observables (spot, rate, dividend yield).
    tau : float
        Time to expiry.
    sigma_lo : float
        Lower bound of the volatility search interval.
    sigma_hi : float
        Upper bound of the volatility search interval.

    Returns
    -------
    float
        Initial volatility guess within ``[sigma_lo, sigma_hi]``.

    Notes
    -----
    The function works with an *undiscounted* price ``u = mkt_price / df`` and forward
    ``F = S*exp((r-q)*tau)``. It computes an undiscounted intrinsic value and uses the
    remaining amount as time value. If the option is essentially pure intrinsic, it
    returns ``sigma_lo`` to avoid Newton steps in a near-zero-vega region.
    """
    ctx = _to_ctx(market)

    # market quantities
    df = ctx.df(tau)
    F = ctx.fwd(tau)

    # undiscounted option price
    u = mkt_price / df

    K = spec.strike
    k = log(F / K)

    # undiscounted intrinsic and time value
    if spec.kind == OptionType.CALL:
        intr = max(F - K, 0.0)
    else:  # PUT
        intr = max(K - F, 0.0)

    tv = max(u - intr, 0.0)

    # If essentially pure intrinsic, implied vol ~ 0 and vega is tiny.
    # Starting Newton in this region is numerically awkward, so return a floor.
    if tv <= 1e-16 * max(1.0, F):
        return float(sigma_lo)

    # ATM time-value seed (works best near k ~ 0)
    sigma_atm = sqrt(2.0 * pi / tau) * (tv / F)

    # MK-style moneyness seed (keeps you sane away from ATM)
    sigma_mk = sqrt(2.0 * abs(k) / tau)

    # Smooth blend: near ATM -> mostly sigma_atm; far -> mostly sigma_mk
    w = exp(-abs(k) / 0.10)  # 0.10 is a reasonable "near-ATM" log-moneyness scale
    sigma0 = w * sigma_atm + (1.0 - w) * sigma_mk

    # Clamp into solver domain
    return float(min(sigma_hi, max(sigma_lo, sigma0)))
