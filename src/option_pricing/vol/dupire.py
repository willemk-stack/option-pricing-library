from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ..market.curves import PricingContext
from ..numerics.fd.diff import diff1_nonuniform, diff2_nonuniform
from ..types import MarketData

FloatArray = NDArray[np.floating[Any]]


@dataclass(frozen=True, slots=True)
class LocalVolResult:
    sigma: FloatArray  # (nT, nK)
    local_var: FloatArray  # (nT, nK)
    invalid: NDArray[np.bool_]  # (nT, nK)
    invalid_count: int


class LVInvalidReason(IntFlag):
    """Reason codes for invalid local-vol values (bitmask)."""

    NONFINITE_INPUT = 1 << 0
    W_TOO_SMALL = 1 << 1

    DENOM_NONFINITE = 1 << 2
    DENOM_TOO_SMALL = 1 << 3
    DENOM_NONPOSITIVE = 1 << 4

    CURVATURE_TOO_SMALL = 1 << 5

    LOCALVAR_NONFINITE = 1 << 6
    LOCALVAR_NEGATIVE = 1 << 7

    TRIM_T = 1 << 8
    TRIM_K = 1 << 9


@dataclass(frozen=True, slots=True)
class GatheralLVReport:
    """Diagnostics for Gatheral local variance computed from total variance."""

    y: FloatArray
    w: FloatArray
    w_y: FloatArray
    w_yy: FloatArray
    w_T: FloatArray

    denom: FloatArray
    local_var: FloatArray
    sigma: FloatArray

    invalid: NDArray[np.bool_]
    reason: NDArray[np.uint32]
    invalid_count: int


@dataclass(frozen=True, slots=True)
class DupireLVReport:
    """Diagnostics for Dupire local variance computed from a (tau, strike) call grid."""

    sigma: FloatArray
    local_var: FloatArray
    num: FloatArray
    denom: FloatArray
    curvature: FloatArray

    invalid: NDArray[np.bool_]
    reason: NDArray[np.uint32]
    invalid_count: int

    trim_t: int
    trim_k: int
    price_convention: str
    strike_coordinate: str


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


def _local_vol_from_call_grid_ctx_diagnostics(
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
) -> DupireLVReport:
    """Same computation as :func:`local_vol_from_call_grid`, but returns diagnostics.

    This is intended for notebooks / debugging: you get numerator, denominator,
    curvature proxy and reason-coded invalid masks.
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

    b = np.array([ctx.b_avg(float(tau)) for tau in T], dtype=np.float64)[:, None]

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

        curvature = C_KK

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

        curvature = denom
    else:
        raise ValueError("strike_coordinate must be 'K' or 'logK'.")

    with np.errstate(divide="ignore", invalid="ignore"):
        local_var = 2.0 * num / denom

    denom_scale = np.nanmax(np.abs(denom))
    if not np.isfinite(denom_scale) or denom_scale == 0.0:
        denom_scale = 1.0

    curv_scale = np.nanmax(np.abs(curvature))
    if not np.isfinite(curv_scale) or curv_scale == 0.0:
        curv_scale = 1.0

    reason = np.zeros_like(local_var, dtype=np.uint32)

    input_nonfinite = ~np.isfinite(C_in)
    if np.any(input_nonfinite):
        reason |= np.where(
            input_nonfinite, np.uint32(LVInvalidReason.NONFINITE_INPUT), 0
        )

    denom_nonfinite = ~np.isfinite(denom)
    reason |= np.where(denom_nonfinite, np.uint32(LVInvalidReason.DENOM_NONFINITE), 0)

    denom_small = np.abs(denom) <= eps_rel * denom_scale
    reason |= np.where(denom_small, np.uint32(LVInvalidReason.DENOM_TOO_SMALL), 0)

    denom_nonpos = denom <= 0.0
    reason |= np.where(denom_nonpos, np.uint32(LVInvalidReason.DENOM_NONPOSITIVE), 0)

    curv_small = np.abs(curvature) <= eps_gamma_rel * curv_scale
    reason |= np.where(curv_small, np.uint32(LVInvalidReason.CURVATURE_TOO_SMALL), 0)

    lv_nonfinite = ~np.isfinite(local_var)
    reason |= np.where(lv_nonfinite, np.uint32(LVInvalidReason.LOCALVAR_NONFINITE), 0)

    lv_neg = local_var < 0.0
    reason |= np.where(lv_neg, np.uint32(LVInvalidReason.LOCALVAR_NEGATIVE), 0)

    invalid = reason != 0

    # trim unstable boundaries (highly recommended)
    if trim_t > 0:
        invalid[:trim_t, :] = True
        invalid[-trim_t:, :] = True
        reason[:trim_t, :] |= np.uint32(LVInvalidReason.TRIM_T)
        reason[-trim_t:, :] |= np.uint32(LVInvalidReason.TRIM_T)

    if trim_k > 0:
        invalid[:, :trim_k] = True
        invalid[:, -trim_k:] = True
        reason[:, :trim_k] |= np.uint32(LVInvalidReason.TRIM_K)
        reason[:, -trim_k:] |= np.uint32(LVInvalidReason.TRIM_K)

    local_var = np.where(invalid, np.nan, local_var)
    with np.errstate(invalid="ignore"):
        sigma = np.sqrt(local_var)

    invalid_count = int(np.sum(invalid))

    return DupireLVReport(
        sigma=sigma,
        local_var=local_var,
        num=np.asarray(num, dtype=np.float64),
        denom=np.asarray(denom, dtype=np.float64),
        curvature=np.asarray(curvature, dtype=np.float64),
        invalid=np.asarray(invalid, dtype=bool),
        reason=np.asarray(reason, dtype=np.uint32),
        invalid_count=invalid_count,
        trim_t=int(trim_t),
        trim_k=int(trim_k),
        price_convention=str(price_convention),
        strike_coordinate=str(strike_coordinate),
    )


def local_vol_from_call_grid_diagnostics(
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
) -> DupireLVReport:
    """Public diagnostics wrapper for Dupire local vol from a call grid."""
    ctx = _to_ctx(market)
    return _local_vol_from_call_grid_ctx_diagnostics(
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


def gatheral_local_var_diagnostics(
    *,
    y: FloatArray,
    w: FloatArray,
    w_y: FloatArray,
    w_yy: FloatArray,
    w_T: FloatArray,
    eps_w: float = 1e-12,
    eps_denom: float = 1e-12,
) -> GatheralLVReport:
    """Compute local variance using Gatheral's total-variance formula, with diagnostics.

    Returns a report including the denominator, invalid mask and reason-coded flags.
    """
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    w_y = np.asarray(w_y, dtype=np.float64)
    w_yy = np.asarray(w_yy, dtype=np.float64)
    w_T = np.asarray(w_T, dtype=np.float64)

    reason = np.zeros_like(w, dtype=np.uint32)

    nonfinite = (
        (~np.isfinite(y))
        | (~np.isfinite(w))
        | (~np.isfinite(w_y))
        | (~np.isfinite(w_yy))
        | (~np.isfinite(w_T))
    )
    reason |= np.where(nonfinite, np.uint32(LVInvalidReason.NONFINITE_INPUT), 0)

    w_small = w <= eps_w
    reason |= np.where(w_small, np.uint32(LVInvalidReason.W_TOO_SMALL), 0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inv_w = 1.0 / w
        y_over_w = y * inv_w
        wy2 = w_y * w_y
        term = -0.25 - inv_w + (y * y) * (inv_w * inv_w)
        denom = 1.0 - y_over_w * w_y + 0.25 * term * wy2 + 0.5 * w_yy
        local_var = w_T / denom

    denom_nonfinite = ~np.isfinite(denom)
    reason |= np.where(denom_nonfinite, np.uint32(LVInvalidReason.DENOM_NONFINITE), 0)

    denom_small = np.abs(denom) <= eps_denom
    reason |= np.where(denom_small, np.uint32(LVInvalidReason.DENOM_TOO_SMALL), 0)

    denom_nonpos = denom <= 0.0
    reason |= np.where(denom_nonpos, np.uint32(LVInvalidReason.DENOM_NONPOSITIVE), 0)

    lv_nonfinite = ~np.isfinite(local_var)
    reason |= np.where(lv_nonfinite, np.uint32(LVInvalidReason.LOCALVAR_NONFINITE), 0)

    lv_neg = local_var < 0.0
    reason |= np.where(lv_neg, np.uint32(LVInvalidReason.LOCALVAR_NEGATIVE), 0)

    invalid = reason != 0

    local_var = np.where(invalid, np.nan, local_var)
    with np.errstate(invalid="ignore"):
        sigma = np.sqrt(local_var)

    invalid_count = int(np.sum(invalid))

    return GatheralLVReport(
        y=y,
        w=w,
        w_y=w_y,
        w_yy=w_yy,
        w_T=w_T,
        denom=np.asarray(denom, dtype=np.float64),
        local_var=np.asarray(local_var, dtype=np.float64),
        sigma=np.asarray(sigma, dtype=np.float64),
        invalid=np.asarray(invalid, dtype=bool),
        reason=np.asarray(reason, dtype=np.uint32),
        invalid_count=invalid_count,
    )


def _gatheral_local_var_from_w(
    *,
    y: FloatArray,
    w: FloatArray,
    w_y: FloatArray,
    w_yy: FloatArray,
    w_T: FloatArray,
    eps_w: float = 1e-12,
    eps_denom: float = 1e-12,
) -> tuple[FloatArray, np.ndarray]:
    """Compute local variance using the (Gatheral) formula in total variance.

    Uses y = ln(K/F(T)) and w(y,T) = T * sigma_imp(y,T)^2.

    Returns (local_var, invalid_mask).

    Notes
    -----
    This is a *pointwise* formula. In practice, stability depends heavily on the
    smoothness/consistency of the implied surface (especially in T).
    """

    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    w_y = np.asarray(w_y, dtype=np.float64)
    w_yy = np.asarray(w_yy, dtype=np.float64)
    w_T = np.asarray(w_T, dtype=np.float64)

    invalid = (
        (~np.isfinite(y))
        | (~np.isfinite(w))
        | (~np.isfinite(w_y))
        | (~np.isfinite(w_yy))
        | (~np.isfinite(w_T))
        | (w <= eps_w)
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inv_w = 1.0 / w
        y_over_w = y * inv_w
        wy2 = w_y * w_y
        term = -0.25 - inv_w + (y * y) * (inv_w * inv_w)
        denom = 1.0 - y_over_w * w_y + 0.25 * term * wy2 + 0.5 * w_yy
        local_var = w_T / denom

    invalid = (
        invalid
        | (~np.isfinite(denom))
        | (np.abs(denom) <= eps_denom)
        | (denom <= 0.0)
        | (~np.isfinite(local_var))
        | (local_var < 0.0)
    )

    local_var = np.where(invalid, np.nan, local_var)
    return np.asarray(local_var, dtype=np.float64), np.asarray(invalid, dtype=bool)
