from __future__ import annotations

import math

from scipy.stats import norm


def _validate_scalar_inputs(
    *, spot: float, strike: float, sigma: float, tau: float
) -> None:
    if spot <= 0.0:
        raise ValueError("spot must be positive")
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if tau <= 0.0:
        raise ValueError("tau must be positive")


def discount_factor(rate: float, tau: float) -> float:
    return math.exp(-rate * tau)


def forward(spot: float, r: float, q: float, tau: float) -> float:
    # F = S * e^{(r-q) tau}
    return spot * math.exp((r - q) * tau)


def d1_d2_from_spot(
    *, spot: float, strike: float, r: float, q: float, sigma: float, tau: float
) -> tuple[float, float]:
    _validate_scalar_inputs(spot=spot, strike=strike, sigma=sigma, tau=tau)
    vol_sqrt_t = sigma * math.sqrt(tau)
    num = math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * tau
    d1 = num / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return float(d1), float(d2)


def call_price(
    *, spot: float, strike: float, r: float, q: float, sigma: float, tau: float
) -> float:
    """
    Black–Scholes European call with continuous dividend yield q.
    """
    d1, d2 = d1_d2_from_spot(spot=spot, strike=strike, r=r, q=q, sigma=sigma, tau=tau)
    df_r = discount_factor(r, tau)
    df_q = discount_factor(q, tau)
    return spot * df_q * norm.cdf(d1) - strike * df_r * norm.cdf(d2)


def put_price(
    *, spot: float, strike: float, r: float, q: float, sigma: float, tau: float
) -> float:
    """
    Black–Scholes European put with continuous dividend yield q.
    """
    d1, d2 = d1_d2_from_spot(spot=spot, strike=strike, r=r, q=q, sigma=sigma, tau=tau)
    df_r = discount_factor(r, tau)
    df_q = discount_factor(q, tau)
    return strike * df_r * norm.cdf(-d2) - spot * df_q * norm.cdf(-d1)


def call_greeks(
    *, spot: float, strike: float, r: float, q: float, sigma: float, tau: float
) -> dict[str, float]:
    """
    Analytic Greeks for BS European call (with dividend yield q).

    theta is ∂Price/∂t (calendar time, holding expiry fixed), per year.
    """
    d1, d2 = d1_d2_from_spot(spot=spot, strike=strike, r=r, q=q, sigma=sigma, tau=tau)
    sqrt_tau = math.sqrt(tau)
    df_r = discount_factor(r, tau)
    df_q = discount_factor(q, tau)

    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    phi_d1 = norm.pdf(d1)

    price = spot * df_q * Nd1 - strike * df_r * Nd2
    delta = df_q * Nd1
    gamma = df_q * phi_d1 / (spot * sigma * sqrt_tau)
    vega = spot * df_q * phi_d1 * sqrt_tau
    theta = (
        -(spot * df_q * phi_d1 * sigma) / (2.0 * sqrt_tau)
        - r * strike * df_r * Nd2
        + q * spot * df_q * Nd1
    )

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
    }


def put_greeks(
    *, spot: float, strike: float, r: float, q: float, sigma: float, tau: float
) -> dict[str, float]:
    """
    Analytic Greeks for BS European put (with dividend yield q).

    theta is ∂Price/∂t (calendar time, holding expiry fixed), per year.
    """
    d1, d2 = d1_d2_from_spot(spot=spot, strike=strike, r=r, q=q, sigma=sigma, tau=tau)
    sqrt_tau = math.sqrt(tau)
    df_r = discount_factor(r, tau)
    df_q = discount_factor(q, tau)

    Nmd1 = norm.cdf(-d1)
    Nmd2 = norm.cdf(-d2)
    phi_d1 = norm.pdf(d1)

    price = strike * df_r * Nmd2 - spot * df_q * Nmd1
    delta = df_q * (norm.cdf(d1) - 1.0)
    gamma = df_q * phi_d1 / (spot * sigma * sqrt_tau)
    vega = spot * df_q * phi_d1 * sqrt_tau
    theta = (
        -(spot * df_q * phi_d1 * sigma) / (2.0 * sqrt_tau)
        + r * strike * df_r * Nmd2
        - q * spot * df_q * Nmd1
    )

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
    }
