"""Common helper functions for implied volatility computations."""

from __future__ import annotations

from math import exp

from ..exceptions import InvalidOptionPriceError
from ..market.curves import PricingContext
from ..types import MarketData, OptionSpec, OptionType


def _to_ctx(market: MarketData | PricingContext) -> PricingContext:
    """Normalize flat MarketData or an already-curves-first PricingContext to PricingContext."""
    if isinstance(market, PricingContext):
        return market
    return market.to_context()


def _df(r: float, tau: float) -> float:
    """Discount factor for a cash payoff at expiry.

    Parameters
    ----------
    r : float
        Continuously-compounded risk-free rate (annualized).
    tau : float
        Time to expiry (in years or consistent time units).

    Returns
    -------
    float
        Discount factor ``exp(-r * tau)``.
    """
    return exp(-r * tau)


def _prepaid_forward(spot: float, q: float, tau: float) -> float:
    """Prepaid forward price under continuous dividend yield.

    The prepaid forward is the value today of receiving the underlying at expiry,
    assuming a continuous dividend yield ``q``:

    ``Fp = S * exp(-q * tau)``.

    Parameters
    ----------
    spot : float
        Spot price of the underlying, :math:`S`.
    q : float
        Continuously-compounded dividend yield (annualized), :math:`q`.
    tau : float
        Time to expiry.

    Returns
    -------
    float
        Prepaid forward ``S * exp(-q * tau)``.
    """
    return spot * exp(-q * tau)


def _bounds(
    spec: OptionSpec,
    market: MarketData | PricingContext,
    t: float,
) -> tuple[float, float, float]:
    """Compute tight no-arbitrage bounds for a European option price.

    Bounds are computed under continuous compounding with dividend yield. Let:

    - ``tau = spec.expiry - t``
    - ``df = exp(-r*tau)``
    - ``Fp = S*exp(-q*tau)`` (prepaid forward)
    - ``K_df = K*df``

    Then the tight bounds are:

    - Call: ``max(Fp - K_df, 0) <= C <= Fp``
    - Put : ``max(K_df - Fp, 0) <= P <= K_df``

    Parameters
    ----------
    spec : OptionSpec
        Option specification (kind, strike, expiry), where ``expiry`` is the
        absolute expiry in the same units as ``t``.
    market : MarketData
        Market observables (spot, rate, dividend yield).
    t : float
        Valuation time in the same units as ``spec.expiry``.

    Returns
    -------
    lb : float
        Lower bound for the option price.
    ub : float
        Upper bound for the option price.
    tau : float
        Time to expiry ``spec.expiry - t``.

    Raises
    ------
    ValueError
        If ``spec.expiry - t <= 0``.
    ValueError
        If ``spec.kind`` is not a supported :class:`OptionType`.
    """
    tau = spec.expiry - t
    if tau <= 0.0:
        raise ValueError("Need expiry > t")

    ctx = _to_ctx(market)
    df = ctx.df(tau)
    fp = ctx.prepaid_forward(tau)
    K_df = spec.strike * df

    if spec.kind == OptionType.CALL:
        lb = max(fp - K_df, 0.0)
        ub = fp
    elif spec.kind == OptionType.PUT:
        lb = max(K_df - fp, 0.0)
        ub = K_df
    else:
        raise ValueError(f"Unknown option kind: {spec.kind!r}")

    return lb, ub, tau


def _validate_bounds(
    price: float,
    spec: OptionSpec,
    market: MarketData | PricingContext,
    t: float,
    *,
    eps: float = 1e-12,
) -> tuple[float, float, float]:
    """Validate that a market option price satisfies no-arbitrage bounds.

    Parameters
    ----------
    price : float
        Market option price to validate.
    spec : OptionSpec
        Option specification (kind, strike, expiry), where ``expiry`` is the
        absolute expiry in the same units as ``t``.
    market : MarketData
        Market observables (spot, rate, dividend yield).
    t : float
        Valuation time in the same units as ``spec.expiry``.
    eps : float, default 1e-12
        Numerical tolerance. Prices within ``eps`` of the bounds are accepted.

    Returns
    -------
    lb : float
        Lower bound for the option price.
    ub : float
        Upper bound for the option price.
    tau : float
        Time to expiry ``spec.expiry - t``.

    Raises
    ------
    InvalidOptionPriceError
        If ``price`` lies outside ``[lb - eps, ub + eps]``.
    ValueError
        If expiry is not strictly greater than ``t`` (propagated from :func:`_bounds`).

    Notes
    -----
    This check helps avoid root-finder failures due to an infeasible implied
    volatility problem.
    """
    lb, ub, tau = _bounds(spec, market, t)
    ctx = _to_ctx(market)
    if price < lb - eps or price > ub + eps:
        raise InvalidOptionPriceError(
            f"Option price out of bounds: price={price:.12g}, bounds=[{lb:.12g}, {ub:.12g}], "
            f"spot={ctx.spot:.12g}, K={spec.strike:.12g}, df={ctx.df(tau):.12g}, F={ctx.fwd(tau):.12g}, "
            f"tau={tau:.12g}"
        )
    return lb, ub, tau
