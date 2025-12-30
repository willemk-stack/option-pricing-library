from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..models import bs as bs_model
from ..types import MarketData, OptionType, PricingInputs


# -------------------------
# BSM wrappers (scalar)
# -------------------------
def bs_price_call(p: PricingInputs) -> float:
    return bs_model.call_price(
        spot=p.S,
        strike=p.K,
        r=p.r,
        q=p.q,
        sigma=p.sigma,
        tau=p.tau,
    )


def bs_price_put(p: PricingInputs) -> float:
    return bs_model.put_price(
        spot=p.S,
        strike=p.K,
        r=p.r,
        q=p.q,
        sigma=p.sigma,
        tau=p.tau,
    )


def bs_price(p: PricingInputs) -> float:
    if p.spec.kind == OptionType.CALL:
        return bs_price_call(p)
    if p.spec.kind == OptionType.PUT:
        return bs_price_put(p)
    raise ValueError(f"Unsupported option kind: {p.spec.kind}")


def bs_call_greeks(p: PricingInputs) -> dict[str, float]:
    return bs_model.call_greeks(
        spot=p.S,
        strike=p.K,
        r=p.r,
        q=p.q,
        sigma=p.sigma,
        tau=p.tau,
    )


def bs_put_greeks(p: PricingInputs) -> dict[str, float]:
    return bs_model.put_greeks(
        spot=p.S,
        strike=p.K,
        r=p.r,
        q=p.q,
        sigma=p.sigma,
        tau=p.tau,
    )


def bs_greeks(p: PricingInputs) -> dict[str, float]:
    if p.spec.kind == OptionType.CALL:
        return bs_call_greeks(p)
    if p.spec.kind == OptionType.PUT:
        return bs_put_greeks(p)
    raise ValueError(f"Unsupported option kind: {p.spec.kind}")


# -------------------------
# Black-76 wrappers (vectorized; used by surface checks)
# -------------------------
def black76_call_prices_vec_from_curves(
    *,
    strikes: np.ndarray,
    sigma: float | np.ndarray,
    tau: float,
    forward_fn: Callable[[float], float],
    df_fn: Callable[[float], float] | None = None,
) -> np.ndarray:
    """
    Price discounted Black-76 calls for one maturity tau on a vector of strikes,
    using curve-like callables forward_fn(tau) and df_fn(tau).

    If df_fn is None, df is taken as 1.0 (useful for some sanity checks).
    """
    tau = float(tau)
    F = float(forward_fn(tau))
    df = float(df_fn(tau)) if df_fn is not None else 1.0

    return bs_model.black76_call_price_vec(
        forward=F,
        strikes=np.asarray(strikes, dtype=np.float64),
        sigma=sigma,
        tau=tau,
        df=df,
    )


def black76_put_prices_vec_from_curves(
    *,
    strikes: np.ndarray,
    sigma: float | np.ndarray,
    tau: float,
    forward_fn: Callable[[float], float],
    df_fn: Callable[[float], float] | None = None,
) -> np.ndarray:
    """Vectorized discounted Black-76 puts using forward/df callables."""
    tau = float(tau)
    F = float(forward_fn(tau))
    df = float(df_fn(tau)) if df_fn is not None else 1.0

    return bs_model.black76_put_price_vec(
        forward=F,
        strikes=np.asarray(strikes, dtype=np.float64),
        sigma=sigma,
        tau=tau,
        df=df,
    )


def black76_call_prices_vec_from_market(
    *,
    market: MarketData,
    strikes: np.ndarray,
    sigma: float | np.ndarray,
    tau: float,
) -> np.ndarray:
    """
    Vectorized discounted Black-76 calls for one maturity tau using MarketData methods:
      F = market.forward(T=tau)
      df = market.df(T=tau)
    """
    tau = float(tau)
    F = float(market.forward(T=tau))
    df = float(market.df(T=tau))

    return bs_model.black76_call_price_vec(
        forward=F,
        strikes=np.asarray(strikes, dtype=np.float64),
        sigma=sigma,
        tau=tau,
        df=df,
    )


def black76_put_prices_vec_from_market(
    *,
    market: MarketData,
    strikes: np.ndarray,
    sigma: float | np.ndarray,
    tau: float,
) -> np.ndarray:
    """Vectorized discounted Black-76 puts for one maturity tau using MarketData."""
    tau = float(tau)
    F = float(market.forward(T=tau))
    df = float(market.df(T=tau))

    return bs_model.black76_put_price_vec(
        forward=F,
        strikes=np.asarray(strikes, dtype=np.float64),
        sigma=sigma,
        tau=tau,
        df=df,
    )
