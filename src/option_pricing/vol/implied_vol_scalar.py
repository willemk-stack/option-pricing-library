"""Scalar implied volatility computation for individual options."""

from __future__ import annotations

from ..config import ImpliedVolConfig, SeedStrategy
from ..market.curves import PricingContext
from ..numerics.root_finding import get_root_method
from ..pricers.black_scholes import bs_greeks_from_ctx
from ..types import MarketData, OptionSpec
from .implied_vol_common import _to_ctx, _validate_bounds
from .implied_vol_seed import _iv_seed_from_time_value
from .implied_vol_types import ImpliedVolResult


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
