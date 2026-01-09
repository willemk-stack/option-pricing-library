from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..instruments.base import ExerciseStyle
from ..instruments.vanilla import VanillaOption
from ..market.curves import PricingContext
from ..models import bs as bs_model
from ..types import MarketData, OptionType, PricingInputs

# -------------------------
# Curves-first BSM/Black-76 pricing
# -------------------------


def bs_price_call_from_ctx(
    *, strike: float, sigma: float, tau: float, ctx: PricingContext
) -> float:
    """European call price using (forward, discount) from a :class:`PricingContext`."""
    tau = float(tau)
    df = ctx.df(tau)
    F = ctx.fwd(tau)
    return bs_model.black76_call_price(
        forward=F, strike=float(strike), sigma=float(sigma), tau=tau, df=df
    )


def bs_price_put_from_ctx(
    *, strike: float, sigma: float, tau: float, ctx: PricingContext
) -> float:
    """European put price using (forward, discount) from a :class:`PricingContext`."""
    tau = float(tau)
    df = ctx.df(tau)
    F = ctx.fwd(tau)
    return bs_model.black76_put_price(
        forward=F, strike=float(strike), sigma=float(sigma), tau=tau, df=df
    )


def bs_price_from_ctx(
    *, kind: OptionType, strike: float, sigma: float, tau: float, ctx: PricingContext
) -> float:
    if kind == OptionType.CALL:
        return bs_price_call_from_ctx(strike=strike, sigma=sigma, tau=tau, ctx=ctx)
    if kind == OptionType.PUT:
        return bs_price_put_from_ctx(strike=strike, sigma=sigma, tau=tau, ctx=ctx)
    raise ValueError(f"Unsupported option kind: {kind}")


def bs_call_greeks_from_ctx(
    *, strike: float, sigma: float, tau: float, ctx: PricingContext
) -> dict[str, float]:
    tau = float(tau)
    df = ctx.df(tau)
    F = ctx.fwd(tau)
    return bs_model.black76_call_greeks_from_curves(
        spot=ctx.spot,
        forward=F,
        strike=float(strike),
        sigma=float(sigma),
        tau=tau,
        df=df,
    )


def bs_put_greeks_from_ctx(
    *, strike: float, sigma: float, tau: float, ctx: PricingContext
) -> dict[str, float]:
    tau = float(tau)
    df = ctx.df(tau)
    F = ctx.fwd(tau)
    return bs_model.black76_put_greeks_from_curves(
        spot=ctx.spot,
        forward=F,
        strike=float(strike),
        sigma=float(sigma),
        tau=tau,
        df=df,
    )


def bs_greeks_from_ctx(
    *, kind: OptionType, strike: float, sigma: float, tau: float, ctx: PricingContext
) -> dict[str, float]:
    if kind == OptionType.CALL:
        return bs_call_greeks_from_ctx(strike=strike, sigma=sigma, tau=tau, ctx=ctx)
    if kind == OptionType.PUT:
        return bs_put_greeks_from_ctx(strike=strike, sigma=sigma, tau=tau, ctx=ctx)
    raise ValueError(f"Unsupported option kind: {kind}")


def _to_ctx(market: MarketData | PricingContext) -> PricingContext:
    """Normalize flat MarketData or an already-curves-first PricingContext."""
    if isinstance(market, PricingContext):
        return market
    return market.to_context()


def bs_price_instrument_from_ctx(
    *, inst: VanillaOption, sigma: float, ctx: PricingContext
) -> float:
    """Black-76 price for a vanilla instrument using curves-first inputs.

    Notes
    -----
    - ``inst.expiry`` is interpreted as time-to-expiry (tau).
    - Only European exercise is supported by this closed-form pricer.
    """
    if inst.exercise != ExerciseStyle.EUROPEAN:
        raise ValueError("Black-76 closed-form supports European exercise only")
    return bs_price_from_ctx(
        kind=inst.kind,
        strike=inst.strike,
        sigma=float(sigma),
        tau=float(inst.expiry),
        ctx=ctx,
    )


def bs_greeks_instrument_from_ctx(
    *, inst: VanillaOption, sigma: float, ctx: PricingContext
) -> dict[str, float]:
    """Black-76 Greeks for a vanilla instrument using curves-first inputs."""
    if inst.exercise != ExerciseStyle.EUROPEAN:
        raise ValueError("Black-76 greeks support European exercise only")
    return bs_greeks_from_ctx(
        kind=inst.kind,
        strike=inst.strike,
        sigma=float(sigma),
        tau=float(inst.expiry),
        ctx=ctx,
    )


def bs_price_instrument(
    inst: VanillaOption, *, market: MarketData | PricingContext, sigma: float
) -> float:
    """Convenience wrapper accepting flat :class:`~option_pricing.types.MarketData`."""
    return bs_price_instrument_from_ctx(inst=inst, sigma=sigma, ctx=_to_ctx(market))


def bs_greeks_instrument(
    inst: VanillaOption, *, market: MarketData | PricingContext, sigma: float
) -> dict[str, float]:
    """Convenience wrapper accepting flat :class:`~option_pricing.types.MarketData`."""
    return bs_greeks_instrument_from_ctx(inst=inst, sigma=sigma, ctx=_to_ctx(market))


# -------------------------
# Backwards-compatible wrappers (PricingInputs)
# -------------------------


def bs_price_call(p: PricingInputs) -> float:
    return bs_price_call_from_ctx(strike=p.K, sigma=p.sigma, tau=p.tau, ctx=p.ctx)


def bs_price_put(p: PricingInputs) -> float:
    return bs_price_put_from_ctx(strike=p.K, sigma=p.sigma, tau=p.tau, ctx=p.ctx)


def bs_price(p: PricingInputs) -> float:
    return bs_price_from_ctx(
        kind=p.spec.kind, strike=p.K, sigma=p.sigma, tau=p.tau, ctx=p.ctx
    )


def bs_call_greeks(p: PricingInputs) -> dict[str, float]:
    return bs_call_greeks_from_ctx(strike=p.K, sigma=p.sigma, tau=p.tau, ctx=p.ctx)


def bs_put_greeks(p: PricingInputs) -> dict[str, float]:
    return bs_put_greeks_from_ctx(strike=p.K, sigma=p.sigma, tau=p.tau, ctx=p.ctx)


def bs_greeks(p: PricingInputs) -> dict[str, float]:
    return bs_greeks_from_ctx(
        kind=p.spec.kind, strike=p.K, sigma=p.sigma, tau=p.tau, ctx=p.ctx
    )


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
    """Price discounted Black-76 calls for one maturity tau on a vector of strikes,
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


def black76_call_prices_vec_from_ctx(
    *,
    ctx: PricingContext,
    strikes: np.ndarray,
    sigma: float | np.ndarray,
    tau: float,
) -> np.ndarray:
    """Vectorized discounted Black-76 calls for one maturity tau using PricingContext."""
    return black76_call_prices_vec_from_curves(
        strikes=strikes,
        sigma=sigma,
        tau=tau,
        forward_fn=ctx.fwd,
        df_fn=ctx.df,
    )


def black76_put_prices_vec_from_ctx(
    *,
    ctx: PricingContext,
    strikes: np.ndarray,
    sigma: float | np.ndarray,
    tau: float,
) -> np.ndarray:
    """Vectorized discounted Black-76 puts for one maturity tau using PricingContext."""
    return black76_put_prices_vec_from_curves(
        strikes=strikes,
        sigma=sigma,
        tau=tau,
        forward_fn=ctx.fwd,
        df_fn=ctx.df,
    )


def black76_call_prices_vec_from_market(
    *,
    market: MarketData,
    strikes: np.ndarray,
    sigma: float | np.ndarray,
    tau: float,
) -> np.ndarray:
    """Vectorized discounted Black-76 calls for one maturity tau using MarketData."""
    return black76_call_prices_vec_from_ctx(
        ctx=market.to_context(),
        strikes=strikes,
        sigma=sigma,
        tau=tau,
    )


def black76_put_prices_vec_from_market(
    *,
    market: MarketData,
    strikes: np.ndarray,
    sigma: float | np.ndarray,
    tau: float,
) -> np.ndarray:
    """Vectorized discounted Black-76 puts for one maturity tau using MarketData."""
    return black76_put_prices_vec_from_ctx(
        ctx=market.to_context(),
        strikes=strikes,
        sigma=sigma,
        tau=tau,
    )
