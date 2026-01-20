from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from ..instruments.digital import DigitalOption
from ..types import PricingContext
from ..typing import ArrayLike, FloatDType


# -------------------------
# Pure helpers (math-only)
# -------------------------
def discount_factor(rate: float, tau: float) -> float:
    """Continuous-compounding discount factor: exp(-rate * tau)."""
    tau = float(tau)
    if tau < 0.0:
        raise ValueError("tau must be >= 0")
    return math.exp(-float(rate) * tau)


def forward_from_spot(spot: float, r: float, q: float, tau: float) -> float:
    """Cost-of-carry forward: F = S * exp((r-q) * tau)."""
    spot = float(spot)
    tau = float(tau)
    if spot <= 0.0:
        raise ValueError("spot must be positive")
    if tau < 0.0:
        raise ValueError("tau must be >= 0")
    return spot * math.exp((float(r) - float(q)) * tau)


def _validate_pos_spot_strike(spot: float, strike: float) -> None:
    if spot <= 0.0:
        raise ValueError("spot must be positive")
    if strike <= 0.0:
        raise ValueError("strike must be positive")


# -------------------------
# Black–Scholes–Merton (spot, r, q)
# -------------------------
def d1_d2_from_spot(
    *, spot: float, strike: float, r: float, q: float, sigma: float, tau: float
) -> tuple[float, float]:
    _validate_pos_spot_strike(float(spot), float(strike))
    sigma = float(sigma)
    tau = float(tau)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if tau <= 0.0:
        raise ValueError("tau must be positive")

    vol_sqrt_t = sigma * math.sqrt(tau)
    num = (
        math.log(float(spot) / float(strike))
        + (float(r) - float(q) + 0.5 * sigma * sigma) * tau
    )
    d1 = num / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return float(d1), float(d2)


def call_price(
    *, spot: float, strike: float, r: float, q: float, sigma: float, tau: float
) -> float:
    """
    Black–Scholes–Merton European call with continuous dividend yield q.
    """
    _validate_pos_spot_strike(float(spot), float(strike))
    sigma = float(sigma)
    tau = float(tau)

    # allow tau==0 / sigma==0 as intrinsic
    if tau == 0.0 or sigma == 0.0:
        return max(float(spot) - float(strike), 0.0)

    d1, d2 = d1_d2_from_spot(spot=spot, strike=strike, r=r, q=q, sigma=sigma, tau=tau)
    df_r = discount_factor(r, tau)
    df_q = discount_factor(q, tau)
    return float(spot) * df_q * float(norm.cdf(d1)) - float(strike) * df_r * float(
        norm.cdf(d2)
    )


def put_price(
    *, spot: float, strike: float, r: float, q: float, sigma: float, tau: float
) -> float:
    """
    Black–Scholes–Merton European put with continuous dividend yield q.
    """
    _validate_pos_spot_strike(float(spot), float(strike))
    sigma = float(sigma)
    tau = float(tau)

    if tau == 0.0 or sigma == 0.0:
        return max(float(strike) - float(spot), 0.0)

    d1, d2 = d1_d2_from_spot(spot=spot, strike=strike, r=r, q=q, sigma=sigma, tau=tau)
    df_r = discount_factor(r, tau)
    df_q = discount_factor(q, tau)
    return float(strike) * df_r * float(norm.cdf(-d2)) - float(spot) * df_q * float(
        norm.cdf(-d1)
    )


def call_greeks(
    *, spot: float, strike: float, r: float, q: float, sigma: float, tau: float
) -> dict[str, float]:
    """
    Analytic Greeks for BSM European call (continuous dividend yield q).
    Theta is ∂Price/∂t (calendar time, holding expiry fixed), per year.
    """
    _validate_pos_spot_strike(float(spot), float(strike))
    sigma = float(sigma)
    tau = float(tau)
    if sigma <= 0.0 or tau <= 0.0:
        raise ValueError("sigma and tau must be positive for analytic greeks")

    d1, d2 = d1_d2_from_spot(spot=spot, strike=strike, r=r, q=q, sigma=sigma, tau=tau)
    sqrt_tau = math.sqrt(tau)
    df_r = discount_factor(r, tau)
    df_q = discount_factor(q, tau)

    Nd1 = float(norm.cdf(d1))
    Nd2 = float(norm.cdf(d2))
    phi_d1 = float(norm.pdf(d1))

    price = float(spot) * df_q * Nd1 - float(strike) * df_r * Nd2
    delta = df_q * Nd1
    gamma = df_q * phi_d1 / (float(spot) * sigma * sqrt_tau)
    vega = float(spot) * df_q * phi_d1 * sqrt_tau
    theta = (
        -(float(spot) * df_q * phi_d1 * sigma) / (2.0 * sqrt_tau)
        - float(r) * float(strike) * df_r * Nd2
        + float(q) * float(spot) * df_q * Nd1
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
    Analytic Greeks for BSM European put (continuous dividend yield q).
    Theta is ∂Price/∂t (calendar time, holding expiry fixed), per year.
    """
    _validate_pos_spot_strike(float(spot), float(strike))
    sigma = float(sigma)
    tau = float(tau)
    if sigma <= 0.0 or tau <= 0.0:
        raise ValueError("sigma and tau must be positive for analytic greeks")

    d1, d2 = d1_d2_from_spot(spot=spot, strike=strike, r=r, q=q, sigma=sigma, tau=tau)
    sqrt_tau = math.sqrt(tau)
    df_r = discount_factor(r, tau)
    df_q = discount_factor(q, tau)

    Nmd1 = float(norm.cdf(-d1))
    Nmd2 = float(norm.cdf(-d2))
    phi_d1 = float(norm.pdf(d1))

    price = float(strike) * df_r * Nmd2 - float(spot) * df_q * Nmd1
    delta = df_q * (float(norm.cdf(d1)) - 1.0)
    gamma = df_q * phi_d1 / (float(spot) * sigma * sqrt_tau)
    vega = float(spot) * df_q * phi_d1 * sqrt_tau
    theta = (
        -(float(spot) * df_q * phi_d1 * sigma) / (2.0 * sqrt_tau)
        + float(r) * float(strike) * df_r * Nmd2
        - float(q) * float(spot) * df_q * Nmd1
    )

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
    }


# -------------------------


# -------------------------
# Black-76 (forward, df) — scalar helpers
# -------------------------
def _implied_rq_from_forward_df(
    *, spot: float, forward: float, df: float, tau: float
) -> tuple[float, float]:
    """Infer flat r and q consistent with (spot, forward, df) over tau."""
    tau = float(tau)
    if tau <= 0.0:
        return 0.0, 0.0
    df = float(df)
    if df <= 0.0:
        raise ValueError("df must be > 0")
    spot = float(spot)
    forward = float(forward)
    if spot <= 0.0 or forward <= 0.0:
        raise ValueError("spot and forward must be > 0")
    r = -math.log(df) / tau
    b = math.log(forward / spot) / tau  # carry = r - q
    q = r - b
    return float(r), float(q)


def black76_call_price(
    *, forward: float, strike: float, sigma: float, tau: float, df: float = 1.0
) -> float:
    """Discounted Black-76 European call price."""
    forward = float(forward)
    strike = float(strike)
    sigma = float(sigma)
    tau = float(tau)
    df = float(df)
    if forward <= 0.0 or strike <= 0.0:
        raise ValueError("forward and strike must be > 0")
    if df <= 0.0:
        raise ValueError("df must be > 0")
    if tau <= 0.0:
        return df * max(forward - strike, 0.0)
    if sigma <= 0.0:
        return df * max(forward - strike, 0.0)
    vol_sqrt_t = sigma * math.sqrt(tau)
    d1 = (math.log(forward / strike) + 0.5 * sigma * sigma * tau) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return df * (forward * float(norm.cdf(d1)) - strike * float(norm.cdf(d2)))


def black76_put_price(
    *, forward: float, strike: float, sigma: float, tau: float, df: float = 1.0
) -> float:
    """Discounted Black-76 European put price (via parity)."""
    call = black76_call_price(
        forward=forward, strike=strike, sigma=sigma, tau=tau, df=df
    )
    return call - float(df) * (float(forward) - float(strike))


def black76_call_greeks_from_curves(
    *,
    spot: float,
    forward: float,
    strike: float,
    sigma: float,
    tau: float,
    df: float = 1.0,
) -> dict[str, float]:
    """BSM call Greeks implied by (spot, forward, df)."""
    r, q = _implied_rq_from_forward_df(
        spot=float(spot), forward=float(forward), df=float(df), tau=float(tau)
    )
    return call_greeks(
        spot=float(spot),
        strike=float(strike),
        r=r,
        q=q,
        sigma=float(sigma),
        tau=float(tau),
    )


def black76_put_greeks_from_curves(
    *,
    spot: float,
    forward: float,
    strike: float,
    sigma: float,
    tau: float,
    df: float = 1.0,
) -> dict[str, float]:
    """BSM put Greeks implied by (spot, forward, df)."""
    r, q = _implied_rq_from_forward_df(
        spot=float(spot), forward=float(forward), df=float(df), tau=float(tau)
    )
    return put_greeks(
        spot=float(spot),
        strike=float(strike),
        r=r,
        q=q,
        sigma=float(sigma),
        tau=float(tau),
    )


# Black-76 (forward, df) — vectorized (used for checks/surfaces)
# -------------------------
def black76_call_price_vec(
    *,
    forward: float,
    strikes: np.ndarray,
    sigma: float | np.ndarray,
    tau: float,
    df: float = 1.0,
) -> np.ndarray:
    """
    Vectorized discounted Black-76 call: df * (F N(d1) - K N(d2)).
    Accepts sigma scalar or array broadcastable to strikes.
    """
    F = float(forward)
    if F <= 0.0:
        raise ValueError("forward must be positive")

    K = np.asarray(strikes, dtype=FloatDType)
    if np.any(K <= 0.0):
        raise ValueError("strikes must be positive")

    tau = float(tau)
    if tau < 0.0:
        raise ValueError("tau must be >= 0")

    df = float(df)
    if df <= 0.0:
        raise ValueError("df must be positive")

    sig = np.asarray(sigma, dtype=FloatDType)
    if np.any(sig < 0.0):
        raise ValueError("sigma must be non-negative")

    # tau == 0 => intrinsic (discounted)
    intrinsic = df * np.maximum(F - K, 0.0)
    if tau == 0.0:
        return intrinsic

    sqrt_tau = np.sqrt(tau)
    sig_sqrt_tau = sig * sqrt_tau

    # safe mask for near-zero vols
    mask = sig_sqrt_tau > 1e-16
    if not np.any(mask):
        return intrinsic

    # Compute with errstate to avoid warnings when sigma==0 in other entries
    with np.errstate(divide="ignore", invalid="ignore"):
        logFK = np.log(F / K)
        d1 = (logFK + 0.5 * (sig**2) * tau) / sig_sqrt_tau
        d2 = d1 - sig_sqrt_tau
        C = df * (F * norm.cdf(d1) - K * norm.cdf(d2))

    return np.where(mask, C, intrinsic)


def black76_put_price_vec(
    *,
    forward: float,
    strikes: np.ndarray,
    sigma: float | np.ndarray,
    tau: float,
    df: float = 1.0,
) -> np.ndarray:
    """Vectorized discounted Black-76 put via parity: P = C - df*(F-K)."""
    K = np.asarray(strikes, dtype=FloatDType)
    C = black76_call_price_vec(forward=forward, strikes=K, sigma=sigma, tau=tau, df=df)
    return C - df * (float(forward) - K)


###
# Digital option
###


def d1_d2_black_76(
    *, forward: float, strike: float, tau: float, sigma: ArrayLike
) -> tuple[ArrayLike, ArrayLike]:
    """
    Calculates d1 and d2 for Black-76 using Python 3.12 type hinting.
    """
    # Validation logic
    if forward <= 0 or strike <= 0:
        raise ValueError("Forward and strike prices must be positive")

    # Using numpy to allow the function to handle scalar floats or arrays
    # if the user wants to pass a volatility surface/vector
    s = np.asanyarray(sigma)

    if np.any(s <= 0) or tau <= 0:
        raise ValueError("Sigma and tau must be positive")

    vol_sqrt_tau = s * math.sqrt(tau)

    # Black-76 numerator: ln(F/K) + (0.5 * sigma^2 * tau)
    num = np.log(forward / strike) + (0.5 * s**2 * tau)

    d1 = num / vol_sqrt_tau
    d2 = d1 - vol_sqrt_tau

    # Return as original type (float if scalar, ndarray if vector)
    return (d1.item() if d1.size == 1 else d1, d2.item() if d2.size == 1 else d2)


def digital_call_price(
    instr: DigitalOption,
    market: PricingContext,
    sigma: float,
) -> ArrayLike:
    Q = instr.payout
    T = instr.expiry
    strike = instr.strike
    fwd = market.fwd
    df = market.df
    _, d2 = d1_d2_black_76(forward=fwd, strike=strike, tau=T, sigma=sigma)

    price = Q * df * norm.cdf(d2)
    return price


def digital_put_price(
    instr: DigitalOption,
    market: PricingContext,
    sigma: float,
) -> ArrayLike:
    Q = instr.payout
    T = instr.expiry
    strike = instr.strike
    fwd = market.fwd
    df = market.df
    _, d2 = d1_d2_black_76(forward=fwd, strike=strike, tau=T, sigma=sigma)

    price = Q * df * norm.cdf(-d2)
    return price
