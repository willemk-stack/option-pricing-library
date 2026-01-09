from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from ..instruments.base import TerminalInstrument
from ..pricers.black_scholes import (  # or bs_price_put, etc.
    bs_price_call,
    bs_price_instrument,
)
from ..types import MarketData, PricingInputs


def finite_diff_greeks(
    p: PricingInputs,
    *,
    price_fn: Callable[[PricingInputs], float] = bs_price_call,
    h_x: float | None = None,
    h_sigma: float | None = None,
    h_t: float | None = None,
) -> dict[str, float]:
    """
    Finite-difference Greeks for a pricer that takes PricingInputs -> price.

    Returns dict with price, delta, gamma, vega, theta,
    where theta ≈ ∂V/∂t (calendar time, holding expiry fixed).
    """
    # --- basic validation (use your own validation if you already have it)
    if p.S <= 0.0:
        raise ValueError("spot must be positive")
    if p.K <= 0.0:
        raise ValueError("strike must be positive")
    if p.sigma <= 0.0:
        raise ValueError("sigma must be positive")

    tau = p.T - p.t
    if tau <= 0.0:
        raise ValueError("Need expiry > t")

    # --- step sizes
    h_x = h_x or (0.01 * p.S)  # 1% of spot
    h_sigma = h_sigma or (0.01 * p.sigma)  # 1% of vol
    h_t = h_t or (1.0 / 365.0)  # 1 day in years

    h_x = min(h_x, 0.5 * p.S)  # keep S-h_x positive
    h_sigma = min(h_sigma, 0.5 * p.sigma)  # keep sigma-h_sigma positive
    h_t = min(h_t, 0.5 * tau)

    def with_spot(pi: PricingInputs, spot: float) -> PricingInputs:
        return replace(pi, market=replace(pi.market, spot=spot))

    # --- base price
    V = price_fn(p)

    # --- delta, gamma (bump spot)
    V_up_x = price_fn(with_spot(p, p.S + h_x))
    V_down_x = price_fn(with_spot(p, p.S - h_x))

    delta = (V_up_x - V_down_x) / (2.0 * h_x)
    gamma = (V_up_x - 2.0 * V + V_down_x) / (h_x**2)

    # --- vega (bump sigma)
    V_up_sigma = price_fn(replace(p, sigma=p.sigma + h_sigma))
    V_down_sigma = price_fn(replace(p, sigma=p.sigma - h_sigma))
    vega = (V_up_sigma - V_down_sigma) / (2.0 * h_sigma)

    # --- theta (bump calendar time t, holding expiry fixed)
    theta: float
    t0 = p.t
    T = p.T

    if (t0 - h_t) >= 0.0 and (t0 + h_t) < T:
        V_up_t = price_fn(replace(p, t=t0 + h_t))
        V_down_t = price_fn(replace(p, t=t0 - h_t))
        theta = (V_up_t - V_down_t) / (2.0 * h_t)
    elif (t0 + h_t) < T:
        V_up_t = price_fn(replace(p, t=t0 + h_t))
        theta = (V_up_t - V) / h_t
    elif (t0 - h_t) >= 0.0:
        V_down_t = price_fn(replace(p, t=t0 - h_t))
        theta = (V - V_down_t) / h_t
    else:
        raise ValueError("Cannot compute theta: time steps violate 0 <= t < expiry.")

    return {"price": V, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def finite_diff_greeks_instrument(
    inst: TerminalInstrument,
    *,
    market: MarketData,
    sigma: float,
    price_fn: Callable[..., float] = bs_price_instrument,
    h_x: float | None = None,
    h_sigma: float | None = None,
    h_tau: float | None = None,
) -> dict[str, float]:
    """Finite-difference Greeks for *tau-based* instrument pricers.

    This is a companion to :func:`finite_diff_greeks` for the instrument-based API.

    Differences vs :func:`finite_diff_greeks`
    ---------------------------------------
    - Here we only have **time to expiry** (``tau = inst.expiry``), so the time
      sensitivity returned is ``theta_tau = ∂V/∂tau``.
    - The instrument must be compatible with :func:`dataclasses.replace` for the
      expiry bump (e.g. :class:`~option_pricing.instruments.vanilla.VanillaOption`).
    """
    if market.spot <= 0.0:
        raise ValueError("spot must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    tau = float(inst.expiry)
    if tau <= 0.0:
        raise ValueError("Need expiry > 0")

    h_x = h_x or (0.01 * market.spot)
    h_sigma = h_sigma or (0.01 * sigma)
    h_tau = h_tau or (1.0 / 365.0)

    h_x = min(h_x, 0.5 * market.spot)
    h_sigma = min(h_sigma, 0.5 * sigma)
    h_tau = min(h_tau, 0.5 * tau)

    def price(_inst: TerminalInstrument, _market: MarketData, _sigma: float) -> float:
        return float(price_fn(_inst, market=_market, sigma=_sigma))

    V = price(inst, market, sigma)

    # --- delta, gamma (bump spot)
    V_up_x = price(inst, replace(market, spot=market.spot + h_x), sigma)
    V_down_x = price(inst, replace(market, spot=market.spot - h_x), sigma)
    delta = (V_up_x - V_down_x) / (2.0 * h_x)
    gamma = (V_up_x - 2.0 * V + V_down_x) / (h_x**2)

    # --- vega (bump sigma)
    V_up_sigma = price(inst, market, sigma + h_sigma)
    V_down_sigma = price(inst, market, sigma - h_sigma)
    vega = (V_up_sigma - V_down_sigma) / (2.0 * h_sigma)

    # --- theta_tau (bump tau)
    inst_up = replace(inst, expiry=tau + h_tau)
    inst_down = replace(inst, expiry=tau - h_tau)
    V_up_tau = price(inst_up, market, sigma)
    V_down_tau = price(inst_down, market, sigma)
    theta_tau = (V_up_tau - V_down_tau) / (2.0 * h_tau)

    return {
        "price": V,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta_tau": theta_tau,
    }
