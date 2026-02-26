"""Gatheral local variance computation from total variance surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .local_vol_types import GatheralLVReport, LVInvalidReason

FloatArray = NDArray[np.floating[Any]]


@dataclass(frozen=True, slots=True)
class GatheralTerms:
    y: FloatArray
    w: FloatArray
    w_y: FloatArray
    w_yy: FloatArray
    w_T: FloatArray
    denom: FloatArray
    local_var_raw: FloatArray


def _compute_gatheral_terms(
    *,
    y: FloatArray,
    w: FloatArray,
    w_y: FloatArray,
    w_yy: FloatArray,
    w_T: FloatArray,
) -> GatheralTerms:
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    w_y = np.asarray(w_y, dtype=np.float64)
    w_yy = np.asarray(w_yy, dtype=np.float64)
    w_T = np.asarray(w_T, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inv_w = 1.0 / w
        y_over_w = y * inv_w
        wy2 = w_y * w_y
        term = -0.25 - inv_w + (y * y) * (inv_w * inv_w)
        denom = 1.0 - y_over_w * w_y + 0.25 * term * wy2 + 0.5 * w_yy
        local_var_raw = w_T / denom

    return GatheralTerms(
        y=y,
        w=w,
        w_y=w_y,
        w_yy=w_yy,
        w_T=w_T,
        denom=np.asarray(denom, dtype=np.float64),
        local_var_raw=np.asarray(local_var_raw, dtype=np.float64),
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

    Parameters
    ----------
    y : FloatArray
        Log-moneyness y = ln(K/F(T))
    w : FloatArray
        Total variance w(y, T) = T * sigma_impl(y, T)^2, same shape as y
    w_y : FloatArray
        Derivative dw/dy
    w_yy : FloatArray
        Second derivative d²w/dy²
    w_T : FloatArray
        Time derivative dw/dT
    eps_w : float
        Minimum total variance threshold
    eps_denom : float
        Minimum denominator threshold

    Returns
    -------
    GatheralLVReport
        Diagnostics including the local variance, denominator, and reason-coded invalid mask
    """
    terms = _compute_gatheral_terms(y=y, w=w, w_y=w_y, w_yy=w_yy, w_T=w_T)

    reason = np.zeros_like(terms.w, dtype=np.uint32)

    nonfinite = (
        (~np.isfinite(terms.y))
        | (~np.isfinite(terms.w))
        | (~np.isfinite(terms.w_y))
        | (~np.isfinite(terms.w_yy))
        | (~np.isfinite(terms.w_T))
    )
    reason |= np.where(nonfinite, np.uint32(LVInvalidReason.NONFINITE_INPUT), 0)

    w_small = terms.w <= eps_w
    reason |= np.where(w_small, np.uint32(LVInvalidReason.W_TOO_SMALL), 0)

    denom_nonfinite = ~np.isfinite(terms.denom)
    reason |= np.where(denom_nonfinite, np.uint32(LVInvalidReason.DENOM_NONFINITE), 0)

    denom_small = np.abs(terms.denom) <= eps_denom
    reason |= np.where(denom_small, np.uint32(LVInvalidReason.DENOM_TOO_SMALL), 0)

    denom_nonpos = terms.denom <= 0.0
    reason |= np.where(denom_nonpos, np.uint32(LVInvalidReason.DENOM_NONPOSITIVE), 0)

    lv_nonfinite = ~np.isfinite(terms.local_var_raw)
    reason |= np.where(lv_nonfinite, np.uint32(LVInvalidReason.LOCALVAR_NONFINITE), 0)

    lv_neg = terms.local_var_raw < 0.0
    reason |= np.where(lv_neg, np.uint32(LVInvalidReason.LOCALVAR_NEGATIVE), 0)

    invalid = reason != 0

    local_var = np.where(invalid, np.nan, terms.local_var_raw)
    with np.errstate(invalid="ignore"):
        sigma = np.sqrt(local_var)

    invalid_count = int(np.sum(invalid))

    return GatheralLVReport(
        y=terms.y,
        w=terms.w,
        w_y=terms.w_y,
        w_yy=terms.w_yy,
        w_T=terms.w_T,
        denom=np.asarray(terms.denom, dtype=np.float64),
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

    terms = _compute_gatheral_terms(y=y, w=w, w_y=w_y, w_yy=w_yy, w_T=w_T)

    invalid = (
        (~np.isfinite(terms.y))
        | (~np.isfinite(terms.w))
        | (~np.isfinite(terms.w_y))
        | (~np.isfinite(terms.w_yy))
        | (~np.isfinite(terms.w_T))
        | (terms.w <= eps_w)
        | (~np.isfinite(terms.denom))
        | (np.abs(terms.denom) <= eps_denom)
        | (terms.denom <= 0.0)
        | (~np.isfinite(terms.local_var_raw))
        | (terms.local_var_raw < 0.0)
    )

    local_var = np.where(invalid, np.nan, terms.local_var_raw)
    return np.asarray(local_var, dtype=np.float64), np.asarray(invalid, dtype=bool)
