from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..market.curves import PricingContext
from ..types import MarketData

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LocalVolResult:
    sigma: FloatArray  # (nT, nK)
    local_var: FloatArray  # (nT, nK)
    invalid: NDArray[np.bool_]  # (nT, nK)


def _to_ctx(market: MarketData | PricingContext) -> PricingContext:
    return market if isinstance(market, PricingContext) else market.to_context()


def _fd_weights_3pt(
    x0: float, x1: float, x2: float, x_eval: float, deriv: int
) -> tuple[float, float, float]:
    """
    Quadratic Lagrange 3-point finite-difference weights for derivative at x_eval
    using points x0, x1, x2.
    deriv: 1 or 2
    """
    if deriv == 1:
        w0 = (2.0 * x_eval - x1 - x2) / ((x0 - x1) * (x0 - x2))
        w1 = (2.0 * x_eval - x0 - x2) / ((x1 - x0) * (x1 - x2))
        w2 = (2.0 * x_eval - x0 - x1) / ((x2 - x0) * (x2 - x1))
        return w0, w1, w2
    if deriv == 2:
        w0 = 2.0 / ((x0 - x1) * (x0 - x2))
        w1 = 2.0 / ((x1 - x0) * (x1 - x2))
        w2 = 2.0 / ((x2 - x0) * (x2 - x1))
        return w0, w1, w2
    raise ValueError("deriv must be 1 or 2")


def _weights_first_derivative_nonuniform(
    h_m: FloatArray, h_p: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Interior 3-point weights for first derivative on non-uniform grid."""
    beta_m = -h_p / (h_m * (h_m + h_p))
    beta = (h_p - h_m) / (h_m * h_p)
    beta_p = h_m / (h_p * (h_m + h_p))
    return beta_m, beta, beta_p


def _weights_second_derivative_nonuniform(
    h_m: FloatArray, h_p: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Interior 3-point weights for second derivative on non-uniform grid."""
    alpha_m = 2.0 / (h_m * (h_m + h_p))
    alpha = -2.0 / (h_m * h_p)
    alpha_p = 2.0 / (h_p * (h_m + h_p))
    return alpha_m, alpha, alpha_p


def _fdm_comp(
    prices: FloatArray,  # (nT, nK)
    taus: FloatArray,  # (nT,)
    strikes: FloatArray,  # (nK,)
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Returns:
      C_T  : (nT, nK)  dC/dtau
      C_K  : (nT, nK)  dC/dK
      C_KK : (nT, nK)  d²C/dK²
    """
    prices = np.asarray(prices, dtype=np.float64)
    T = np.asarray(taus, dtype=np.float64)
    K = np.asarray(strikes, dtype=np.float64)

    if prices.ndim != 2:
        raise ValueError(f"prices must be 2D (nT,nK), got shape {prices.shape}")
    nT, nK = prices.shape
    if T.shape != (nT,) or K.shape != (nK,):
        raise ValueError(
            f"Shape mismatch: prices {prices.shape}, taus {T.shape}, strikes {K.shape}"
        )
    if nT < 3 or nK < 3:
        raise ValueError(
            "Need at least 3 maturities and 3 strikes for 3-point stencils."
        )
    if not (np.all(np.diff(T) > 0) and np.all(np.diff(K) > 0)):
        raise ValueError("taus and strikes must be strictly increasing.")
    if np.any(K <= 0):
        raise ValueError("strikes must be > 0.")

    C_T = np.empty_like(prices)
    C_K = np.empty_like(prices)
    C_KK = np.empty_like(prices)

    # ---- C_T along tau (axis 0) ----
    h_m_T = T[1:-1] - T[:-2]
    h_p_T = T[2:] - T[1:-1]
    beta_m, beta, beta_p = _weights_first_derivative_nonuniform(h_m_T, h_p_T)

    beta_m = beta_m[:, None]
    beta = beta[:, None]
    beta_p = beta_p[:, None]
    C_T[1:-1, :] = (
        beta_m * prices[:-2, :] + beta * prices[1:-1, :] + beta_p * prices[2:, :]
    )

    w0, w1, w2 = _fd_weights_3pt(T[0], T[1], T[2], T[0], deriv=1)
    C_T[0, :] = w0 * prices[0, :] + w1 * prices[1, :] + w2 * prices[2, :]

    w0, w1, w2 = _fd_weights_3pt(T[-3], T[-2], T[-1], T[-1], deriv=1)
    C_T[-1, :] = w0 * prices[-3, :] + w1 * prices[-2, :] + w2 * prices[-1, :]

    # ---- along K (axis 1) ----
    h_m_K = K[1:-1] - K[:-2]
    h_p_K = K[2:] - K[1:-1]

    gamma_m, gamma, gamma_p = _weights_first_derivative_nonuniform(h_m_K, h_p_K)
    C_K[:, 1:-1] = (
        gamma_m * prices[:, :-2] + gamma * prices[:, 1:-1] + gamma_p * prices[:, 2:]
    )

    w0, w1, w2 = _fd_weights_3pt(K[0], K[1], K[2], K[0], deriv=1)
    C_K[:, 0] = w0 * prices[:, 0] + w1 * prices[:, 1] + w2 * prices[:, 2]

    w0, w1, w2 = _fd_weights_3pt(K[-3], K[-2], K[-1], K[-1], deriv=1)
    C_K[:, -1] = w0 * prices[:, -3] + w1 * prices[:, -2] + w2 * prices[:, -1]

    alpha_m, alpha, alpha_p = _weights_second_derivative_nonuniform(h_m_K, h_p_K)
    C_KK[:, 1:-1] = (
        alpha_m * prices[:, :-2] + alpha * prices[:, 1:-1] + alpha_p * prices[:, 2:]
    )

    w0, w1, w2 = _fd_weights_3pt(K[0], K[1], K[2], K[0], deriv=2)
    C_KK[:, 0] = w0 * prices[:, 0] + w1 * prices[:, 1] + w2 * prices[:, 2]

    w0, w1, w2 = _fd_weights_3pt(K[-3], K[-2], K[-1], K[-1], deriv=2)
    C_KK[:, -1] = w0 * prices[:, -3] + w1 * prices[:, -2] + w2 * prices[:, -1]

    return C_T, C_K, C_KK


def _fdm_comp_logk(
    prices: FloatArray,  # (nT, nK)
    taus: FloatArray,  # (nT,)
    strikes: FloatArray,  # (nK,)
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Returns:
      C_T  : (nT, nK)  dC/dtau
      C_x  : (nT, nK)  dC/dx   where x = log(K)
      C_xx : (nT, nK)  d²C/dx²
    """
    prices = np.asarray(prices, dtype=np.float64)
    T = np.asarray(taus, dtype=np.float64)
    K = np.asarray(strikes, dtype=np.float64)

    if np.any(K <= 0):
        raise ValueError("strikes must be > 0 to use logK coordinate.")
    x = np.log(K)

    if prices.ndim != 2:
        raise ValueError(f"prices must be 2D (nT,nK), got shape {prices.shape}")
    nT, nK = prices.shape
    if T.shape != (nT,) or K.shape != (nK,):
        raise ValueError(
            f"Shape mismatch: prices {prices.shape}, taus {T.shape}, strikes {K.shape}"
        )
    if nT < 3 or nK < 3:
        raise ValueError(
            "Need at least 3 maturities and 3 strikes for 3-point stencils."
        )
    if not (np.all(np.diff(T) > 0) and np.all(np.diff(K) > 0)):
        raise ValueError("taus and strikes must be strictly increasing.")

    # ---- time derivative same as in _fdm_comp ----
    C_T = np.empty_like(prices)
    h_m_T = T[1:-1] - T[:-2]
    h_p_T = T[2:] - T[1:-1]
    beta_m, beta, beta_p = _weights_first_derivative_nonuniform(h_m_T, h_p_T)
    beta_m = beta_m[:, None]
    beta = beta[:, None]
    beta_p = beta_p[:, None]
    C_T[1:-1, :] = (
        beta_m * prices[:-2, :] + beta * prices[1:-1, :] + beta_p * prices[2:, :]
    )

    w0, w1, w2 = _fd_weights_3pt(T[0], T[1], T[2], T[0], deriv=1)
    C_T[0, :] = w0 * prices[0, :] + w1 * prices[1, :] + w2 * prices[2, :]

    w0, w1, w2 = _fd_weights_3pt(T[-3], T[-2], T[-1], T[-1], deriv=1)
    C_T[-1, :] = w0 * prices[-3, :] + w1 * prices[-2, :] + w2 * prices[-1, :]

    # ---- strike derivatives along x=logK ----
    C_x = np.empty_like(prices)
    C_xx = np.empty_like(prices)

    h_m_x = x[1:-1] - x[:-2]
    h_p_x = x[2:] - x[1:-1]

    gamma_m, gamma, gamma_p = _weights_first_derivative_nonuniform(h_m_x, h_p_x)
    C_x[:, 1:-1] = (
        gamma_m * prices[:, :-2] + gamma * prices[:, 1:-1] + gamma_p * prices[:, 2:]
    )

    w0, w1, w2 = _fd_weights_3pt(x[0], x[1], x[2], x[0], deriv=1)
    C_x[:, 0] = w0 * prices[:, 0] + w1 * prices[:, 1] + w2 * prices[:, 2]

    w0, w1, w2 = _fd_weights_3pt(x[-3], x[-2], x[-1], x[-1], deriv=1)
    C_x[:, -1] = w0 * prices[:, -3] + w1 * prices[:, -2] + w2 * prices[:, -1]

    alpha_m, alpha, alpha_p = _weights_second_derivative_nonuniform(h_m_x, h_p_x)
    C_xx[:, 1:-1] = (
        alpha_m * prices[:, :-2] + alpha * prices[:, 1:-1] + alpha_p * prices[:, 2:]
    )

    w0, w1, w2 = _fd_weights_3pt(x[0], x[1], x[2], x[0], deriv=2)
    C_xx[:, 0] = w0 * prices[:, 0] + w1 * prices[:, 1] + w2 * prices[:, 2]

    w0, w1, w2 = _fd_weights_3pt(x[-3], x[-2], x[-1], x[-1], deriv=2)
    C_xx[:, -1] = w0 * prices[:, -3] + w1 * prices[:, -2] + w2 * prices[:, -1]

    return C_T, C_x, C_xx


def _local_vol_from_call_grid_ctx(
    call: FloatArray,  # (nT, nK)
    strikes: FloatArray,  # (nK,)
    taus: FloatArray,  # (nT,)
    *,
    ctx: PricingContext,
    price_convention: Literal["discounted", "forward"] = "discounted",
    strike_coordinate: Literal["K", "logK"] = "logK",
    # guardrails:
    trim_t: int = 1,
    trim_k: int = 1,
    eps_rel: float = 1e-12,
    eps_gamma_rel: float = 1e-12,
) -> LocalVolResult:
    """
    Full-numerator Dupire local vol on a (tau, K) grid.

    Coordinate options
    ------------------
    strike_coordinate="K":
        Uses (C_K, C_KK) and denom = K^2 * C_KK.

    strike_coordinate="logK":
        Uses x=log(K), computes (C_x, C_xx) and denom = (C_xx - C_x)
        since K*C_K = C_x and K^2*C_KK = C_xx - C_x.

    Price conventions
    -----------------
    discounted:
        C = PV call. Uses: sigma^2 = 2*(C_tau + b*K*C_K + q*C) / (K^2*C_KK)
        or in logK:        sigma^2 = 2*(C_tau + b*C_x + q*C) / (C_xx - C_x)

    forward:
        c = C/df (undiscounted). Uses: sigma^2 = 2*(c_tau + b*(K*c_K - c)) / (K^2*c_KK)
        or in logK:              sigma^2 = 2*(c_tau + b*(c_x - c)) / (c_xx - c_x)
    """
    C_in = np.asarray(call, dtype=np.float64)
    T = np.asarray(taus, dtype=np.float64)
    K = np.asarray(strikes, dtype=np.float64)

    if C_in.ndim != 2:
        raise ValueError(f"call must be 2D (nT,nK), got {C_in.shape}")
    if T.shape != (C_in.shape[0],) or K.shape != (C_in.shape[1],):
        raise ValueError(
            f"Shape mismatch: call {C_in.shape}, taus {T.shape}, strikes {K.shape}"
        )

    # per-maturity carry b(tau)=r-q (avg approximation)
    b = np.array([ctx.b_avg(float(tau)) for tau in T], dtype=np.float64)[
        :, None
    ]  # (nT,1)

    # Choose derivative coordinate
    if strike_coordinate == "K":
        C_T, C_K, C_KK = _fdm_comp(prices=C_in, taus=T, strikes=K)
        K_row = K[None, :]
        denom = (K_row * K_row) * C_KK

        if price_convention == "discounted":
            q = np.array([ctx.q_avg(float(tau)) for tau in T], dtype=np.float64)[
                :, None
            ]
            num = C_T + b * K_row * C_K + q * C_in
        elif price_convention == "forward":
            num = C_T + b * (K_row * C_K - C_in)
        else:
            raise ValueError("price_convention must be 'discounted' or 'forward'.")

        curvature = C_KK  # for scaling / diagnostics

    elif strike_coordinate == "logK":
        C_T, C_x, C_xx = _fdm_comp_logk(prices=C_in, taus=T, strikes=K)
        denom = C_xx - C_x

        if price_convention == "discounted":
            q = np.array([ctx.q_avg(float(tau)) for tau in T], dtype=np.float64)[
                :, None
            ]
            num = C_T + b * C_x + q * C_in
        elif price_convention == "forward":
            num = C_T + b * (C_x - C_in)
        else:
            raise ValueError("price_convention must be 'discounted' or 'forward'.")

        curvature = denom  # denom is the key convexity object in logK mode
    else:
        raise ValueError("strike_coordinate must be 'K' or 'logK'.")

    with np.errstate(divide="ignore", invalid="ignore"):
        local_var = 2.0 * num / denom

    # robust relative scales
    denom_scale = np.nanmax(np.abs(denom))
    if not np.isfinite(denom_scale) or denom_scale == 0.0:
        denom_scale = 1.0

    curv_scale = np.nanmax(np.abs(curvature))
    if not np.isfinite(curv_scale) or curv_scale == 0.0:
        curv_scale = 1.0

    invalid = (
        (~np.isfinite(local_var))
        | (local_var < 0.0)
        | (~np.isfinite(denom))
        | (np.abs(denom) <= eps_rel * denom_scale)
        # convexity/positivity condition:
        | (denom <= 0.0)
        | (np.abs(curvature) <= eps_gamma_rel * curv_scale)
    )

    # trim unstable boundaries (highly recommended)
    if trim_t > 0:
        invalid[:trim_t, :] = True
        invalid[-trim_t:, :] = True
    if trim_k > 0:
        invalid[:, :trim_k] = True
        invalid[:, -trim_k:] = True

    local_var = np.where(invalid, np.nan, local_var)
    with np.errstate(invalid="ignore"):
        sigma = np.sqrt(local_var)

    return LocalVolResult(sigma=sigma, local_var=local_var, invalid=invalid)


def local_vol_from_call_grid(
    call: FloatArray,
    strikes: FloatArray,
    taus: FloatArray,
    *,
    market: MarketData | PricingContext,
    price_convention: Literal["discounted", "forward"] = "discounted",
    strike_coordinate: Literal["K", "logK"] = "logK",
    trim_t: int = 1,
    trim_k: int = 1,
    eps_rel: float = 1e-12,
    eps_gamma_rel: float = 1e-12,
) -> LocalVolResult:
    """
    Public wrapper: accepts MarketData or PricingContext, normalizes to ctx, then runs core.
    """
    ctx = _to_ctx(market)
    return _local_vol_from_call_grid_ctx(
        call=call,
        strikes=strikes,
        taus=taus,
        ctx=ctx,
        price_convention=price_convention,
        strike_coordinate=strike_coordinate,
        trim_t=trim_t,
        trim_k=trim_k,
        eps_rel=eps_rel,
        eps_gamma_rel=eps_gamma_rel,
    )
