from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, pi, sqrt

import numpy as np

from ..config import ImpliedVolConfig, SeedStrategy
from ..exceptions import InvalidOptionPriceError
from ..market.curves import PricingContext
from ..models.black_scholes.bs import black76_call_price_vega_vec
from ..numerics.root_finding import RootResult, get_root_method
from ..pricers.black_scholes import bs_greeks_from_ctx
from ..types import MarketData, OptionSpec, OptionType
from ..typing import ArrayLike, FloatArray


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
    ctx = _to_ctx(market)
    if price < lb - eps or price > ub + eps:
        raise InvalidOptionPriceError(
            f"Option price out of bounds: price={price:.12g}, bounds=[{lb:.12g}, {ub:.12g}], "
            f"spot={ctx.spot:.12g}, K={spec.strike:.12g}, df={ctx.df(tau):.12g}, F={ctx.fwd(tau):.12g}, "
            f"tau={tau:.12g}"
        )
    return lb, ub, tau


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
    w = exp(-abs(k) / 0.10)  # 0.10 is a reasonable “near-ATM” log-moneyness scale
    sigma0 = w * sigma_atm + (1.0 - w) * sigma_mk

    # Clamp into solver domain
    return float(min(sigma_hi, max(sigma_lo, sigma0)))


def implied_vol_bs_result(
    mkt_price: float,
    spec: OptionSpec,
    market: MarketData | PricingContext,
    *,
    cfg: ImpliedVolConfig | None = None,
    t: float = 0.0,
    sigma0: float | None = None,
) -> ImpliedVolResult:
    """Compute Black-Scholes implied volatility and return diagnostics.

    This function inverts the Black-Scholes price for volatility by solving:

    ``BS_price(sigma) - mkt_price = 0``

    using a root-finding method selected by ``cfg.root_method``.

    Parameters
    ----------
    mkt_price : float
        Observed market option price.
    spec : OptionSpec
        Option specification (kind, strike, expiry).
    market : MarketData
        Market observables (spot, rate, dividend yield).
    cfg : ImpliedVolConfig or None, optional
        Configuration for implied-vol inversion. If ``None``, ``ImpliedVolConfig()``
        is used. Relevant fields:

        - ``root_method`` : RootMethod
            Internal solver selector (dispatched via :func:`get_root_method`).
        - ``sigma_lo``, ``sigma_hi`` : float
            Volatility search interval.
        - ``bounds_eps`` : float
            Tolerance for no-arbitrage bound checks.
        - ``seed_strategy`` : SeedStrategy
            How to select the initial guess when ``sigma0`` is not provided.
        - ``numerics`` : NumericsConfig
            Tolerances and iteration limits (``abs_tol``, ``rel_tol``, ``max_iter``),
            plus safety clamps (``min_vega``).
    t : float, default 0.0
        Valuation time in the same units as ``spec.expiry``.
    sigma0 : float or None, default None
        Initial volatility guess. If ``None``, the seed is chosen according to
        ``cfg.seed_strategy``:

        - ``HEURISTIC``: compute a robust seed from time value.
        - ``USE_GUESS`` or ``LAST_SOLUTION``: requires `sigma0` to be provided.

    Returns
    -------
    ImpliedVolResult
        Container with implied volatility and root-finder diagnostics.

    Raises
    ------
    InvalidOptionPriceError
        If the input price violates no-arbitrage bounds (see :func:`_validate_bounds`).
    ValueError
        If expiry is not strictly greater than ``t``.
    ValueError
        If ``sigma0`` is required by the configured seed strategy but not provided.
    Exception
        Any exception raised by the root solver is propagated.

    Notes
    -----
    This function is intentionally "config-driven": bounds, tolerances, and the
    solver method are taken from `cfg`.

    One-off overrides should be done by creating a modified config, e.g.::

        from dataclasses import replace
        cfg2 = replace(cfg, sigma_hi=10.0)
        iv  = implied_vol_bs_result(price, spec, market, cfg=cfg2)

    The objective uses :func:`option_pricing.bs_greeks` for both price and vega.
    Vega is clamped below by ``cfg.numerics.min_vega`` for robustness.
    """
    cfg = ImpliedVolConfig() if cfg is None else cfg
    num = cfg.numerics

    sigma_lo = float(cfg.sigma_lo)
    sigma_hi = float(cfg.sigma_hi)

    # Validate bounds with configured epsilon
    lb, ub, tau = _validate_bounds(
        mkt_price, spec, market, t, eps=float(cfg.bounds_eps)
    )

    # Resolve internal root solver from enum
    solver = get_root_method(cfg.root_method)

    # Seed logic
    if sigma0 is None:
        if cfg.seed_strategy in (SeedStrategy.USE_GUESS, SeedStrategy.LAST_SOLUTION):
            raise ValueError(
                "sigma0 must be provided when seed_strategy is USE_GUESS or LAST_SOLUTION"
            )
        sigma0 = _iv_seed_from_time_value(
            mkt_price=mkt_price,
            spec=spec,
            market=market,
            tau=tau,
            sigma_lo=sigma_lo,
            sigma_hi=sigma_hi,
        )
    else:
        sigma0 = float(sigma0)

    # Clamp seed into solver domain
    sigma0 = float(min(sigma_hi, max(sigma_lo, sigma0)))

    ctx = _to_ctx(market)

    def Fn(sigma: float) -> float:
        g = bs_greeks_from_ctx(
            kind=spec.kind,
            strike=spec.strike,
            sigma=float(sigma),
            tau=float(tau),
            ctx=ctx,
        )
        return float(g["price"]) - float(mkt_price)

    def dFn(sigma: float) -> float:
        g = bs_greeks_from_ctx(
            kind=spec.kind,
            strike=spec.strike,
            sigma=float(sigma),
            tau=float(tau),
            ctx=ctx,
        )
        vega = float(g["vega"])
        return max(float(num.min_vega), vega)

    rr = solver(
        Fn,
        sigma_lo,
        sigma_hi,
        x0=sigma0,
        dFn=dFn,
        tol_f=float(num.abs_tol),
        tol_x=float(num.rel_tol),
        max_iter=int(num.max_iter),
    )

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
    market: MarketData | PricingContext,
    *,
    cfg: ImpliedVolConfig | None = None,
    t: float = 0.0,
    sigma0: float | None = None,
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
    cfg : ImpliedVolConfig or None, optional
        Configuration for implied-vol inversion. If ``None``, ``ImpliedVolConfig()``
        is used.
    t : float, default 0.0
        Valuation time in the same units as ``spec.expiry``.
    sigma0 : float or None, default None
        Initial volatility guess. If ``None``, the seed is chosen according to
        ``cfg.seed_strategy``.

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
    ValueError
        If ``sigma0`` is required by the configured seed strategy but not provided.
    """
    return implied_vol_bs_result(
        mkt_price=mkt_price,
        spec=spec,
        market=market,
        cfg=cfg,
        t=t,
        sigma0=sigma0,
    ).vol


@dataclass(frozen=True, slots=True)
class ImpliedVolSliceResult:
    vol: ArrayLike
    converged: ArrayLike
    iterations: ArrayLike
    f_at_root: ArrayLike
    bracket_lo: ArrayLike
    bracket_hi: ArrayLike
    status: ArrayLike  # 0=OK, 1=CLIPPED_LOW, 2=CLIPPED_HIGH, 3=INVALID_PRICE, 4=NO_CONVERGENCE


def implied_vol_black76_slice(
    forward: float,
    strikes: ArrayLike,
    tau: float,
    df: float,
    prices: ArrayLike,
    is_call: bool | ArrayLike,
    initial_sigma: float | ArrayLike = 0.2,
    sigma_lo: float = 1e-6,
    sigma_hi: float = 5.0,
    max_iter: int = 100,
    tol: float = 1e-10,
    return_result: bool = False,
) -> FloatArray | tuple[FloatArray, ImpliedVolSliceResult]:
    F = float(forward)
    if F <= 0.0:
        raise ValueError("forward must be positive")
    tau = float(tau)
    if tau < 0.0:
        raise ValueError("tau must be >= 0")
    df = float(df)
    if df <= 0.0:
        raise ValueError("df must be positive")

    K = np.asarray(strikes, dtype=float)
    P = np.asarray(prices, dtype=float)
    if K.shape != P.shape:
        raise ValueError("strikes and prices must have the same shape")
    if np.any(K <= 0.0):
        raise ValueError("strikes must be positive")

    # call/put flags (broadcastable)
    if is_call is None:
        is_call_arr = np.ones_like(K, dtype=bool)
    else:
        is_call_arr = np.asarray(is_call, dtype=bool)
        if is_call_arr.shape != K.shape:
            is_call_arr = np.broadcast_to(is_call_arr, K.shape).astype(bool)

    # Convert puts -> call targets via parity
    call_target = np.where(is_call_arr, P, P + df * (F - K))

    intrinsic = df * np.maximum(F - K, 0.0)
    upper = df * F

    status = np.zeros_like(K, dtype=np.int8)
    vol = np.full_like(K, np.nan, dtype=float)
    converged = np.zeros_like(K, dtype=bool)
    iterations = np.zeros_like(K, dtype=np.int32)
    f_at_root = np.full_like(K, np.nan, dtype=float)

    invalid = (
        ~np.isfinite(call_target)
        | (call_target < intrinsic - 1e-12)
        | (call_target > upper + 1e-12)
    )
    status[invalid] = 3  # INVALID_PRICE

    # tau==0: only intrinsic is admissible
    if tau == 0.0:
        ok0 = ~invalid & (np.abs(call_target - intrinsic) <= tol)
        vol[ok0] = sigma_lo
        converged[ok0] = True
        f_at_root[ok0] = call_target[ok0] - intrinsic[ok0]
        status[ok0] = 0
        if return_result:
            res = ImpliedVolSliceResult(
                vol=vol,
                converged=converged,
                iterations=iterations,
                f_at_root=f_at_root,
                bracket_lo=np.full_like(K, sigma_lo),
                bracket_hi=np.full_like(K, sigma_hi),
                status=status,
            )
            return vol, res
        return vol

    # Per-strike brackets
    a = np.full_like(K, sigma_lo, dtype=float)
    b = np.full_like(K, sigma_hi, dtype=float)

    # f(a), f(b)
    pa, _ = black76_call_price_vega_vec(forward=F, strikes=K, sigma=a, tau=tau, df=df)
    pb, _ = black76_call_price_vega_vec(forward=F, strikes=K, sigma=b, tau=tau, df=df)
    fa = pa - call_target
    fb = pb - call_target

    ok = ~invalid
    low_clip = ok & (fa > tol)  # root < sigma_lo
    high_clip = ok & (fb < -tol)  # root > sigma_hi

    if np.any(low_clip):
        vol[low_clip] = sigma_lo
        converged[low_clip] = True
        status[low_clip] = 1
        f_at_root[low_clip] = fa[low_clip]
        iterations[low_clip] = 0

    if np.any(high_clip):
        vol[high_clip] = sigma_hi
        converged[high_clip] = True
        status[high_clip] = 2
        f_at_root[high_clip] = fb[high_clip]
        iterations[high_clip] = 0

    active = ok & ~low_clip & ~high_clip & (fa <= tol) & (fb >= -tol)

    # Initial guess
    x0 = np.asarray(initial_sigma, dtype=float)
    if x0.shape != K.shape:
        x0 = np.broadcast_to(x0, K.shape).astype(float)
    x = np.clip(x0, a, b)

    tol_x = 1e-12
    vega_min = 1e-14

    for it in range(1, max_iter + 1):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break

        # One kernel call per iteration for all active points
        price, vega = black76_call_price_vega_vec(
            forward=F, strikes=K[idx], sigma=x[idx], tau=tau, df=df
        )
        fx = price - call_target[idx]

        # 1) residual convergence (at current x)
        done = np.abs(fx) <= tol
        if np.any(done):
            j = idx[done]
            vol[j] = x[j]
            converged[j] = True
            status[j] = 0
            f_at_root[j] = fx[done]
            iterations[j] = it - 1
            active[j] = False

        idx = idx[~done]
        if idx.size == 0:
            continue

        fx = fx[~done]
        vega = vega[~done]

        # 2) bracket update using sign change between a and x
        fa_idx = fa[idx]
        left = (fa_idx * fx) < 0.0  # root in [a, x] => b := x
        if np.any(left):
            jl = idx[left]
            b[jl] = x[jl]
            fb[jl] = fx[left]
        if np.any(~left):
            jr = idx[~left]
            a[jr] = x[jr]
            fa[jr] = fx[~left]

        # 3) bracket-width convergence
        width_done = ((b[idx] - a[idx]) * 0.5) <= tol_x
        if np.any(width_done):
            j = idx[width_done]
            xm = 0.5 * (a[j] + b[j])
            pm, _ = black76_call_price_vega_vec(
                forward=F, strikes=K[j], sigma=xm, tau=tau, df=df
            )
            fm = pm - call_target[j]
            vol[j] = xm
            converged[j] = True
            status[j] = 0
            f_at_root[j] = fm
            iterations[j] = it
            active[j] = False

        idx = idx[~width_done]
        if idx.size == 0:
            continue

        fx = fx[~width_done]
        vega = vega[~width_done]

        # 4) compute next sigma (Newton if safe & inside bracket, else bisection)
        newton_ok = np.isfinite(vega) & (vega > vega_min) & np.isfinite(fx)
        x_curr = x[idx]
        cand = x_curr - fx / vega
        mid = 0.5 * (a[idx] + b[idx])

        inside = (cand > a[idx]) & (cand < b[idx])
        x_new = np.where(newton_ok & inside, cand, mid)

        # 5) step convergence (sigma stops moving)
        step_small = np.abs(x_new - x_curr) <= tol_x * np.maximum(1.0, np.abs(x_new))
        if np.any(step_small):
            j = idx[step_small]
            vol[j] = x_new[step_small]
            converged[j] = True
            status[j] = 0
            pn, _ = black76_call_price_vega_vec(
                forward=F, strikes=K[j], sigma=vol[j], tau=tau, df=df
            )
            f_at_root[j] = pn - call_target[j]
            iterations[j] = it
            active[j] = False

        # update x only for still-active points (aligned indexing)
        idx = idx[~step_small]
        if idx.size:
            x[idx] = np.clip(x_new[~step_small], a[idx], b[idx])

    # anything still active: no convergence
    still = active
    if np.any(still):
        status[still] = 4
        vol[still] = 0.5 * (a[still] + b[still])
        pr, _ = black76_call_price_vega_vec(
            forward=F, strikes=K[still], sigma=vol[still], tau=tau, df=df
        )
        f_at_root[still] = pr - call_target[still]
        iterations[still] = max_iter

    if return_result:
        res = ImpliedVolSliceResult(
            vol=vol,
            converged=converged,
            iterations=iterations,
            f_at_root=f_at_root,
            bracket_lo=a,
            bracket_hi=b,
            status=status,
        )
        return vol, res

    return vol
