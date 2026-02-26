"""Dupire local volatility computation from call option grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ..market.curves import PricingContext
from ..types import MarketData
from .local_vol_fd import _fdm_comp, _fdm_comp_logk
from .local_vol_types import DupireLVReport, LocalVolResult, LVInvalidReason

FloatArray = NDArray[np.floating[Any]]


@dataclass(frozen=True, slots=True)
class DupireTerms:
    dC_dT: FloatArray
    dC_dK: FloatArray
    d2C_dK2: FloatArray
    numerator: FloatArray
    denominator: FloatArray
    curvature: FloatArray
    local_var_raw: FloatArray


def _compute_dupire_terms(
    *,
    call: FloatArray,
    strikes: FloatArray,
    taus: FloatArray,
    ctx: PricingContext,
    price_convention: Literal["discounted", "forward"],
    strike_coordinate: Literal["K", "logK"],
) -> DupireTerms:
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
        dC_dK = C_K
        d2C_dK2 = C_KK

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
        dC_dK = C_x
        d2C_dK2 = C_xx
    else:
        raise ValueError("strike_coordinate must be 'K' or 'logK'.")

    with np.errstate(divide="ignore", invalid="ignore"):
        local_var_raw = 2.0 * num / denom

    return DupireTerms(
        dC_dT=np.asarray(C_T, dtype=np.float64),
        dC_dK=np.asarray(dC_dK, dtype=np.float64),
        d2C_dK2=np.asarray(d2C_dK2, dtype=np.float64),
        numerator=np.asarray(num, dtype=np.float64),
        denominator=np.asarray(denom, dtype=np.float64),
        curvature=np.asarray(curvature, dtype=np.float64),
        local_var_raw=np.asarray(local_var_raw, dtype=np.float64),
    )


def _dupire_invalid_mask(
    terms: DupireTerms,
    *,
    trim_t: int,
    trim_k: int,
    eps_rel: float,
    eps_gamma_rel: float,
) -> np.ndarray:
    denom = terms.denominator
    curvature = terms.curvature
    local_var = terms.local_var_raw

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
        | (denom <= 0.0)
        | (np.abs(curvature) <= eps_gamma_rel * curv_scale)
    )

    if trim_t > 0:
        invalid[:trim_t, :] = True
        invalid[-trim_t:, :] = True
    if trim_k > 0:
        invalid[:, :trim_k] = True
        invalid[:, -trim_k:] = True

    return np.asarray(invalid, dtype=bool)


def _dupire_reason_codes(
    *,
    terms: DupireTerms,
    call: FloatArray,
    strikes: FloatArray,
    taus: FloatArray,
    trim_t: int,
    trim_k: int,
    eps_rel: float,
    eps_gamma_rel: float,
) -> np.ndarray:
    reason = np.zeros_like(terms.local_var_raw, dtype=np.uint32)

    inputs = (
        ~np.isfinite(call)
        | ~np.isfinite(strikes[None, :])
        | ~np.isfinite(taus[:, None])
    )
    reason |= np.where(inputs, np.uint32(LVInvalidReason.NONFINITE_INPUT), 0)

    denom = terms.denominator
    curvature = terms.curvature
    local_var = terms.local_var_raw

    denom_nonfinite = ~np.isfinite(denom)
    reason |= np.where(denom_nonfinite, np.uint32(LVInvalidReason.DENOM_NONFINITE), 0)

    denom_scale = np.nanmax(np.abs(denom))
    if not np.isfinite(denom_scale) or denom_scale == 0.0:
        denom_scale = 1.0
    denom_small = np.abs(denom) <= eps_rel * denom_scale
    reason |= np.where(denom_small, np.uint32(LVInvalidReason.DENOM_TOO_SMALL), 0)

    denom_nonpos = denom <= 0.0
    reason |= np.where(denom_nonpos, np.uint32(LVInvalidReason.DENOM_NONPOSITIVE), 0)

    curv_scale = np.nanmax(np.abs(curvature))
    if not np.isfinite(curv_scale) or curv_scale == 0.0:
        curv_scale = 1.0
    curv_small = np.abs(curvature) <= eps_gamma_rel * curv_scale
    reason |= np.where(curv_small, np.uint32(LVInvalidReason.CURVATURE_TOO_SMALL), 0)

    lv_nonfinite = ~np.isfinite(local_var)
    reason |= np.where(lv_nonfinite, np.uint32(LVInvalidReason.LOCALVAR_NONFINITE), 0)

    lv_neg = local_var < 0.0
    reason |= np.where(lv_neg, np.uint32(LVInvalidReason.LOCALVAR_NEGATIVE), 0)

    if trim_t > 0:
        reason[:trim_t, :] |= np.uint32(LVInvalidReason.TRIM_T)
        reason[-trim_t:, :] |= np.uint32(LVInvalidReason.TRIM_T)
    if trim_k > 0:
        reason[:, :trim_k] |= np.uint32(LVInvalidReason.TRIM_K)
        reason[:, -trim_k:] |= np.uint32(LVInvalidReason.TRIM_K)

    return reason


def _to_ctx(market: MarketData | PricingContext) -> PricingContext:
    """Normalize flat MarketData or an already-curves-first PricingContext to PricingContext."""
    return market if isinstance(market, PricingContext) else market.to_context()


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
    terms = _compute_dupire_terms(
        call=call,
        strikes=strikes,
        taus=taus,
        ctx=ctx,
        price_convention=price_convention,
        strike_coordinate=strike_coordinate,
    )

    invalid = _dupire_invalid_mask(
        terms,
        trim_t=trim_t,
        trim_k=trim_k,
        eps_rel=eps_rel,
        eps_gamma_rel=eps_gamma_rel,
    )

    local_var = np.where(invalid, np.nan, terms.local_var_raw)
    with np.errstate(invalid="ignore"):
        sigma = np.sqrt(local_var)

    invalid_count = int(np.sum(invalid))

    return LocalVolResult(
        sigma=np.asarray(sigma, dtype=np.float64),
        local_var=np.asarray(local_var, dtype=np.float64),
        invalid=np.asarray(invalid, dtype=bool),
        invalid_count=invalid_count,
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
    """Same computation as :func:`local_vol_from_call_grid`, but returns diagnostics."""
    terms = _compute_dupire_terms(
        call=call,
        strikes=strikes,
        taus=taus,
        ctx=ctx,
        price_convention=price_convention,
        strike_coordinate=strike_coordinate,
    )

    reason = _dupire_reason_codes(
        terms=terms,
        call=np.asarray(call, dtype=np.float64),
        strikes=np.asarray(strikes, dtype=np.float64),
        taus=np.asarray(taus, dtype=np.float64),
        trim_t=trim_t,
        trim_k=trim_k,
        eps_rel=eps_rel,
        eps_gamma_rel=eps_gamma_rel,
    )
    invalid = reason != 0

    local_var = np.where(invalid, np.nan, terms.local_var_raw)
    with np.errstate(invalid="ignore"):
        sigma = np.sqrt(local_var)

    invalid_count = int(np.sum(invalid))

    return DupireLVReport(
        sigma=np.asarray(sigma, dtype=np.float64),
        local_var=np.asarray(local_var, dtype=np.float64),
        num=terms.numerator,
        denom=terms.denominator,
        curvature=terms.curvature,
        invalid=np.asarray(invalid, dtype=bool),
        reason=np.asarray(reason, dtype=np.uint32),
        invalid_count=invalid_count,
        trim_t=trim_t,
        trim_k=trim_k,
        price_convention=price_convention,
        strike_coordinate=strike_coordinate,
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
