from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import exp

from option_pricing import MarketData, OptionSpec, OptionType, PricingInputs, bs_greeks
from option_pricing.numerics.root_finding import RootResult


class InvalidOptionPriceError(ValueError):
    """Raised when an input option price violates no-arbitrage bounds."""


@dataclass(frozen=True, slots=True)
class ImpliedVolResult:
    vol: float
    root_result: RootResult
    mkt_price: float
    bounds: tuple[float, float]  # (lb, ub)
    tau: float


def _df(r: float, tau: float) -> float:
    """Discount factor for a cash payoff at expiry."""
    return exp(-r * tau)


def _prepaid_forward(spot: float, q: float, tau: float) -> float:
    """
    Prepaid forward under continuous dividend yield q:
        F_prepaid = S * exp(-q * tau)
    """
    return spot * exp(-q * tau)


def _bounds(
    spec: OptionSpec,
    market: MarketData,
    t: float,
) -> tuple[float, float, float]:
    """
    Tight no-arbitrage bounds for European options under continuous dividend yield.

    Let tau = expiry - t, df = exp(-r*tau), Fp = S*exp(-q*tau) (prepaid forward).

      Call: max(Fp - K*df, 0) <= C <= Fp
      Put : max(K*df - Fp, 0) <= P <= K*df

    Returns (lb, ub, tau).
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
    lb, ub, tau = _bounds(spec, market, t)
    if price < lb - eps or price > ub + eps:
        raise InvalidOptionPriceError(
            f"Option price out of bounds: price={price:.12g}, bounds=[{lb:.12g}, {ub:.12g}], "
            f"S={market.spot:.12g}, K={spec.strike:.12g}, r={market.rate:.12g}, q={market.dividend_yield:.12g}, "
            f"tau={tau:.12g}"
        )
    return lb, ub, tau


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
    """
    Black–Scholes implied vol inversion that returns diagnostics.

    Contract (Option A):
      - root_method returns RootResult
      - implied_vol_bs_result returns ImpliedVolResult(root_result=RootResult)
    """
    lb, ub, tau = _validate_bounds(mkt_price, spec, market, t)

    p0 = PricingInputs(spec=spec, market=market, sigma=float(sigma0 or 0.2), t=t)

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
    """Float-only convenience wrapper around implied_vol_bs_result()."""
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
