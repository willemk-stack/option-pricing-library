from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import exp, log, pi, sqrt

from option_pricing import MarketData, OptionSpec, OptionType, PricingInputs, bs_greeks
from option_pricing.numerics.root_finding import RootResult


class InvalidOptionPriceError(ValueError):
    """Raised when an input option price violates no-arbitrage bounds.

    This error is raised by :func:`_validate_bounds` (and therefore by
    :func:`implied_vol_bs_result` / :func:`implied_vol_bs`) when the provided market
    option price is inconsistent with tight no-arbitrage bounds for European options
    under continuous rates and dividend yield.

    Notes
    -----
    The bounds used are:

    - Call: ``max(Fp - K*df, 0) <= C <= Fp``
    - Put : ``max(K*df - Fp, 0) <= P <= K*df``

    where ``df = exp(-r*tau)`` and ``Fp = S*exp(-q*tau)`` is the prepaid forward.
    """


@dataclass(frozen=True, slots=True)
class ImpliedVolResult:
    """Result container for Black-Scholes implied volatility inversion.

    Parameters
    ----------
    vol : float
        Implied volatility corresponding to the root found by the solver.
    root_result : RootResult
        Diagnostics returned by the chosen root-finding method (iterations, status,
        final residual, etc.).
    mkt_price : float
        The input market option price used for inversion.
    bounds : tuple[float, float]
        No-arbitrage bounds ``(lb, ub)`` for the option price.
    tau : float
        Time to expiry used in inversion, ``tau = spec.expiry - t``.

    Notes
    -----
    This is returned by :func:`implied_vol_bs_result`.
    """

    vol: float
    root_result: RootResult
    mkt_price: float
    bounds: tuple[float, float]  # (lb, ub)
    tau: float


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
    market: MarketData,
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
        Option specification (kind, strike, expiry).
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

    df = _df(market.rate, tau)
    fp = _prepaid_forward(market.spot, market.dividend_yield, tau)
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
    market: MarketData,
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
        Option specification (kind, strike, expiry).
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
    if price < lb - eps or price > ub + eps:
        raise InvalidOptionPriceError(
            f"Option price out of bounds: price={price:.12g}, bounds=[{lb:.12g}, {ub:.12g}], "
            f"S={market.spot:.12g}, K={spec.strike:.12g}, r={market.rate:.12g}, q={market.dividend_yield:.12g}, "
            f"tau={tau:.12g}"
        )
    return lb, ub, tau


def _iv_seed_from_time_value(
    mkt_price: float,
    spec: OptionSpec,
    market: MarketData,
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
    # discount factors
    df = _df(market.rate, tau)

    # prepaid forward and forward
    fp = _prepaid_forward(market.spot, market.dividend_yield, tau)  # S*e^{-q tau}
    F = fp / df  # = S*e^{(r-q)tau}

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
    w = exp(-abs(k) / 0.10)  # 0.10 is a reasonable “near-ATM” log-moneyness scale
    sigma0 = w * sigma_atm + (1.0 - w) * sigma_mk

    # Clamp into solver domain
    return float(min(sigma_hi, max(sigma_lo, sigma0)))


def implied_vol_bs_result(
    mkt_price: float,
    spec: OptionSpec,
    market: MarketData,
    root_method: Callable[..., RootResult],
    *,
    t: float = 0.0,
    sigma0: float | None = None,
    sigma_lo: float = 1e-8,
    sigma_hi: float = 5.0,
    tol_f: float = 1e-10,
    tol_x: float = 1e-10,
    max_iter: int | None = None,
) -> ImpliedVolResult:
    """Compute Black-Scholes implied volatility and return diagnostics.

    This function inverts the Black-Scholes price for volatility by solving:

    ``BS_price(sigma) - mkt_price = 0``

    using a user-supplied root-finding routine.

    Parameters
    ----------
    mkt_price : float
        Observed market option price.
    spec : OptionSpec
        Option specification (kind, strike, expiry).
    market : MarketData
        Market observables (spot, rate, dividend yield).
    root_method : Callable[..., RootResult]
        Root-finder with signature compatible with::

            root_method(Fn, x_lo, x_hi, *, x0=..., dFn=..., tol_f=..., tol_x=..., max_iter=...)

        The solver is expected to return a :class:`~option_pricing.numerics.root_finding.RootResult`
        containing at least ``root``.
    t : float, default 0.0
        Valuation time in the same units as ``spec.expiry``.
    sigma0 : float or None, default None
        Initial volatility guess. If ``None``, a heuristic seed is computed via
        :func:`_iv_seed_from_time_value`.
    sigma_lo : float, default 1e-8
        Lower bound of the volatility search interval.
    sigma_hi : float, default 5.0
        Upper bound of the volatility search interval.
    tol_f : float, default 1e-10
        Function tolerance passed to the root solver.
    tol_x : float, default 1e-10
        Parameter tolerance passed to the root solver.
    max_iter : int or None, default None
        Maximum iterations passed to the root solver (if provided).

    Returns
    -------
    ImpliedVolResult
        Container with the implied volatility and root-finder diagnostics.

    Raises
    ------
    InvalidOptionPriceError
        If the input price violates no-arbitrage bounds (see :func:`_validate_bounds`).
    ValueError
        If expiry is not strictly greater than ``t``.
    Exception
        Any exception raised by the provided `root_method` is propagated.

    Notes
    -----
    The objective uses :func:`option_pricing.bs_greeks` for both price and vega.
    Vega is provided as the derivative ``dFn`` to accelerate Newton/secan hybrids.
    """
    lb, ub, tau = _validate_bounds(mkt_price, spec, market, t)

    if sigma0 is None:
        sigma0 = _iv_seed_from_time_value(
            mkt_price=mkt_price,
            spec=spec,
            market=market,
            tau=tau,
            sigma_lo=sigma_lo,
            sigma_hi=sigma_hi,
        )

    p0 = PricingInputs(spec=spec, market=market, sigma=float(sigma0), t=t)

    def Fn(sigma: float) -> float:
        px = replace(p0, sigma=float(sigma))
        return float(bs_greeks(px)["price"]) - mkt_price

    def dFn(sigma: float) -> float:
        px = replace(p0, sigma=float(sigma))
        return float(bs_greeks(px)["vega"])

    kwargs: dict[str, object] = dict(x0=p0.sigma, dFn=dFn, tol_f=tol_f, tol_x=tol_x)
    if max_iter is not None:
        kwargs["max_iter"] = int(max_iter)

    rr = root_method(Fn, float(sigma_lo), float(sigma_hi), **kwargs)

    return ImpliedVolResult(
        vol=float(rr.root),
        root_result=rr,
        mkt_price=float(mkt_price),
        bounds=(float(lb), float(ub)),
        tau=float(tau),
    )


def implied_vol_bs(
    mkt_price: float,
    spec: OptionSpec,
    market: MarketData,
    root_method: Callable[..., RootResult],
    *,
    t: float = 0.0,
    sigma0: float | None = None,
    sigma_lo: float = 1e-8,
    sigma_hi: float = 5.0,
    tol_f: float = 1e-10,
    tol_x: float = 1e-10,
    max_iter: int | None = None,
) -> float:
    """Compute Black-Scholes implied volatility.

    Convenience wrapper around :func:`implied_vol_bs_result` that returns only the
    implied volatility.

    Parameters
    ----------
    mkt_price : float
        Observed market option price.
    spec : OptionSpec
        Option specification (kind, strike, expiry).
    market : MarketData
        Market observables (spot, rate, dividend yield).
    root_method : Callable[..., RootResult]
        Root-finder compatible with :func:`implied_vol_bs_result`.
    t : float, default 0.0
        Valuation time in the same units as ``spec.expiry``.
    sigma0 : float or None, default None
        Initial volatility guess. If ``None``, a heuristic seed is computed.
    sigma_lo : float, default 1e-8
        Lower bound of the volatility search interval.
    sigma_hi : float, default 5.0
        Upper bound of the volatility search interval.
    tol_f : float, default 1e-10
        Function tolerance passed to the root solver.
    tol_x : float, default 1e-10
        Parameter tolerance passed to the root solver.
    max_iter : int or None, default None
        Maximum iterations passed to the root solver (if provided).

    Returns
    -------
    float
        Implied volatility.

    Raises
    ------
    InvalidOptionPriceError
        If the input price violates no-arbitrage bounds.
    ValueError
        If expiry is not strictly greater than ``t``.
    """
    return implied_vol_bs_result(
        mkt_price=mkt_price,
        spec=spec,
        market=market,
        root_method=root_method,
        t=t,
        sigma0=sigma0,
        sigma_lo=sigma_lo,
        sigma_hi=sigma_hi,
        tol_f=tol_f,
        tol_x=tol_x,
        max_iter=max_iter,
    ).vol
