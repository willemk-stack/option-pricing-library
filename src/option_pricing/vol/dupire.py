from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..market.curves import PricingContext
from ..numerics.fd.diff import diff1_nonuniform, diff2_nonuniform
from ..types import MarketData

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LocalVolResult:
    sigma: FloatArray  # (nT, nK)
    local_var: FloatArray  # (nT, nK)
    invalid: NDArray[np.bool_]  # (nT, nK)
    invalid_count: NDArray


def _to_ctx(market: MarketData | PricingContext) -> PricingContext:
    return market if isinstance(market, PricingContext) else market.to_context()


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

    # ---- C_T along tau (axis 0) ----
    C_T = diff1_nonuniform(y=prices, x=taus, axis=0)

    # ---- along K (axis 1) ----
    C_K = diff1_nonuniform(y=prices, x=strikes, axis=1)
    C_KK = diff2_nonuniform(y=prices, x=strikes, axis=1)

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
    # ---- C_T along tau (axis 0) ----
    C_T = diff1_nonuniform(y=prices, x=taus, axis=0)

    # ---- along K (axis 1) ----
    C_x = diff1_nonuniform(y=prices, x=x, axis=1)
    C_xx = diff2_nonuniform(y=prices, x=x, axis=1)

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

    invalid_count = np.sum(invalid)

    return LocalVolResult(
        sigma=sigma, local_var=local_var, invalid=invalid, invalid_count=invalid_count
    )


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
