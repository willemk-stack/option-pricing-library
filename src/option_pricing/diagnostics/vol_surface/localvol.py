from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from option_pricing.vol.local_vol_dupire import local_vol_from_call_grid_diagnostics
from option_pricing.vol.local_vol_surface import LocalVolSurface
from option_pricing.vol.local_vol_types import (
    GatheralLVReport,
    LVInvalidReason,
)

from .contracts import Black76Module
from .models import LocalVolCompareReport, LocalVolGridReport
from .pricing import _default_black76, call_prices_from_surface_on_strikes


def _reasons_to_str(mask: int) -> str:
    if mask == 0:
        return ""
    parts: list[str] = []
    for r in LVInvalidReason:
        if int(r) != 0 and (mask & int(r)):
            if r.name is not None:
                parts.append(r.name)
    return "|".join(parts)


def _reason_counts(reason: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    r = np.asarray(reason, dtype=np.uint32)
    for flag in LVInvalidReason:
        if int(flag) == 0:
            continue
        c = int(np.sum((r & np.uint32(int(flag))) != 0))
        if c:
            name = flag.name
            if name is not None:
                out[name] = c
    return out


def localvol_grid_diagnostics(
    localvol: LocalVolSurface,
    *,
    expiries: Sequence[float],
    y_grid: np.ndarray,
    eps_w: float = 1e-12,
    eps_denom: float = 1e-12,
    top_n: int = 10,
) -> LocalVolGridReport:
    """Sample Gatheral local vol on a grid and return a diagnostics report."""

    Ts = np.asarray(list(expiries), dtype=float)
    y = np.asarray(y_grid, dtype=float)
    if Ts.ndim != 1 or Ts.size == 0:
        raise ValueError("expiries must be a non-empty 1D sequence")
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y_grid must be a non-empty 1D array")

    nT = int(Ts.size)
    ny = int(y.size)
    K_grid = np.empty((nT, ny), dtype=float)
    lv = np.empty((nT, ny), dtype=float)
    sig = np.empty((nT, ny), dtype=float)
    denom = np.empty((nT, ny), dtype=float)
    invalid = np.empty((nT, ny), dtype=bool)
    reason = np.empty((nT, ny), dtype=np.uint32)

    for i, T in enumerate(Ts):
        T_ = float(T)
        F = float(localvol.forward(T_))
        K = F * np.exp(y)
        K_grid[i, :] = K

        rep: GatheralLVReport = localvol.local_var_diagnostics(
            K, T_, eps_w=eps_w, eps_denom=eps_denom
        )
        lv[i, :] = np.asarray(rep.local_var, dtype=float)
        sig[i, :] = np.asarray(rep.sigma, dtype=float)
        denom[i, :] = np.asarray(rep.denom, dtype=float)
        invalid[i, :] = np.asarray(rep.invalid, dtype=bool)
        reason[i, :] = np.asarray(rep.reason, dtype=np.uint32)

    invalid_count = int(np.sum(invalid))
    invalid_frac = float(invalid_count) / float(nT * ny)

    reason_counts = _reason_counts(reason)

    denom_abs = np.abs(denom)
    finite = np.isfinite(denom_abs)
    flat_idx = np.argsort(np.where(finite, denom_abs, np.inf).ravel())
    rows: list[dict[str, float | str | int]] = []
    taken = 0
    for k in flat_idx:
        if taken >= int(top_n):
            break
        ii = int(k // ny)
        jj = int(k % ny)
        if not finite[ii, jj]:
            continue
        rows.append(
            {
                "rank": taken + 1,
                "T": float(Ts[ii]),
                "y": float(y[jj]),
                "K": float(K_grid[ii, jj]),
                "denom": float(denom[ii, jj]),
                "local_var": float(lv[ii, jj]) if np.isfinite(lv[ii, jj]) else np.nan,
                "sigma": float(sig[ii, jj]) if np.isfinite(sig[ii, jj]) else np.nan,
                "invalid": bool(invalid[ii, jj]),
                "reasons": _reasons_to_str(int(reason[ii, jj])),
            }
        )
        taken += 1

    worst_df = pd.DataFrame(rows)

    return LocalVolGridReport(
        expiries=Ts,
        y=y,
        K=K_grid,
        local_var=lv,
        sigma=sig,
        denom=denom,
        invalid=invalid,
        reason=reason,
        invalid_count=invalid_count,
        invalid_frac=invalid_frac,
        reason_counts=reason_counts,
        worst_points=worst_df,
    )


def localvol_summary(rep: LocalVolGridReport) -> dict[str, float | int]:
    """Notebook-friendly summary stats for a LocalVolGridReport."""
    sig = np.asarray(rep.sigma, dtype=float)
    mask = np.asarray(rep.invalid, dtype=bool) | (~np.isfinite(sig))
    safe = np.where(mask, np.nan, sig)

    return {
        "invalid_count": int(rep.invalid_count),
        "invalid_frac": float(rep.invalid_frac),
        "sigma_min": (
            float(np.nanmin(safe)) if np.isfinite(np.nanmin(safe)) else float("nan")
        ),
        "sigma_median": (
            float(np.nanmedian(safe))
            if np.isfinite(np.nanmedian(safe))
            else float("nan")
        ),
        "sigma_max": (
            float(np.nanmax(safe)) if np.isfinite(np.nanmax(safe)) else float("nan")
        ),
        "denom_abs_min": (
            float(np.nanmin(np.abs(rep.denom)))
            if np.isfinite(np.nanmin(np.abs(rep.denom)))
            else float("nan")
        ),
    }


def localvol_compare_gatheral_vs_dupire(
    localvol: LocalVolSurface,
    *,
    expiries: Sequence[float],
    strikes: np.ndarray,
    market: Any,
    eps_w: float = 1e-12,
    eps_denom: float = 1e-12,
    price_convention: str = "discounted",
    strike_coordinate: str = "logK",
    trim_t: int = 1,
    trim_k: int = 1,
    eps_rel: float = 1e-12,
    eps_gamma_rel: float = 1e-12,
    top_n: int = 10,
    bs_model: Black76Module | None = None,
) -> LocalVolCompareReport:
    """Compare Gatheral local vol against Dupire local vol on a shared grid."""

    Ts = np.asarray(list(expiries), dtype=float)
    K = np.asarray(strikes, dtype=float)
    if Ts.ndim != 1 or Ts.size == 0:
        raise ValueError("expiries must be a non-empty 1D sequence")
    if K.ndim != 1 or K.size == 0:
        raise ValueError("strikes must be a non-empty 1D array")
    if Ts.size < 3 or K.size < 3:
        raise ValueError(
            "Need at least 3 maturities and 3 strikes to compare Dupire vs Gatheral."
        )
    if not np.all(np.diff(Ts) > 0):
        raise ValueError("expiries must be strictly increasing")
    if not np.all(np.diff(K) > 0):
        raise ValueError("strikes must be strictly increasing")
    if np.any(K <= 0):
        raise ValueError("strikes must be > 0")

    if bs_model is None:
        bs_model = _default_black76()

    nT = int(Ts.size)
    nK = int(K.size)

    y = np.empty((nT, nK), dtype=float)
    g_sigma = np.empty((nT, nK), dtype=float)
    g_lv = np.empty((nT, nK), dtype=float)
    g_denom = np.empty((nT, nK), dtype=float)
    g_invalid = np.empty((nT, nK), dtype=bool)
    g_reason = np.empty((nT, nK), dtype=np.uint32)
    forwards = np.empty((nT,), dtype=float)

    for i, T in enumerate(Ts):
        rep: GatheralLVReport = localvol.local_var_diagnostics(
            K, float(T), eps_w=eps_w, eps_denom=eps_denom
        )
        y[i, :] = np.asarray(rep.y, dtype=float)
        g_sigma[i, :] = np.asarray(rep.sigma, dtype=float)
        g_lv[i, :] = np.asarray(rep.local_var, dtype=float)
        g_denom[i, :] = np.asarray(rep.denom, dtype=float)
        g_invalid[i, :] = np.asarray(rep.invalid, dtype=bool)
        g_reason[i, :] = np.asarray(rep.reason, dtype=np.uint32)
        forwards[i] = float(localvol.forward(float(T)))

    _K, _Ts, calls, _iv, _forwards2 = call_prices_from_surface_on_strikes(
        localvol.implied,
        expiries=Ts.tolist(),
        strikes=K,
        forward=localvol.forward,
        df=localvol.discount,
        bs_model=bs_model,
    )
    forwards = np.asarray(_forwards2, dtype=float)

    dup = local_vol_from_call_grid_diagnostics(
        calls,
        strikes=K,
        taus=Ts,
        market=market,
        price_convention=cast(Any, price_convention),
        strike_coordinate=cast(Any, strike_coordinate),
        trim_t=int(trim_t),
        trim_k=int(trim_k),
        eps_rel=float(eps_rel),
        eps_gamma_rel=float(eps_gamma_rel),
    )

    diff_sigma = np.asarray(dup.sigma, dtype=float) - np.asarray(g_sigma, dtype=float)
    diff_lv = np.asarray(dup.local_var, dtype=float) - np.asarray(g_lv, dtype=float)
    invalid_union = (
        np.asarray(g_invalid, dtype=bool)
        | np.asarray(dup.invalid, dtype=bool)
        | (~np.isfinite(diff_sigma))
    )

    diff_sigma = np.where(invalid_union, np.nan, diff_sigma)
    diff_lv = np.where(invalid_union, np.nan, diff_lv)

    valid = ~invalid_union
    n_total = int(nT * nK)
    n_valid = int(np.sum(valid))

    abs_diff = np.abs(diff_sigma)
    rmse = (
        float(np.sqrt(np.nanmean(diff_sigma * diff_sigma))) if n_valid else float("nan")
    )
    mae = float(np.nanmean(abs_diff)) if n_valid else float("nan")
    max_abs = float(np.nanmax(abs_diff)) if n_valid else float("nan")

    summary: dict[str, float | int] = {
        "n_total": n_total,
        "n_compared": n_valid,
        "compared_frac": float(n_valid) / float(n_total) if n_total else 0.0,
        "gatheral_invalid_frac": (
            float(np.sum(g_invalid)) / float(n_total) if n_total else 0.0
        ),
        "dupire_invalid_frac": (
            float(np.sum(dup.invalid)) / float(n_total) if n_total else 0.0
        ),
        "union_invalid_frac": (
            float(np.sum(invalid_union)) / float(n_total) if n_total else 0.0
        ),
        "diff_sigma_rmse": rmse,
        "diff_sigma_mae": mae,
        "diff_sigma_max_abs": max_abs,
    }

    rows: list[dict[str, Any]] = []
    if n_valid:
        flat = np.argsort(np.where(valid, abs_diff, -np.inf).ravel())[::-1]
        taken = 0
        for idx in flat:
            if taken >= int(top_n):
                break
            ii = int(idx // nK)
            jj = int(idx % nK)
            if not valid[ii, jj]:
                continue
            rows.append(
                {
                    "rank": taken + 1,
                    "T": float(Ts[ii]),
                    "K": float(K[jj]),
                    "y": float(y[ii, jj]),
                    "sigma_gatheral": float(g_sigma[ii, jj]),
                    "sigma_dupire": float(dup.sigma[ii, jj]),
                    "diff_sigma": float(diff_sigma[ii, jj]),
                    "denom_gatheral": float(g_denom[ii, jj]),
                    "denom_dupire": float(dup.denom[ii, jj]),
                }
            )
            taken += 1

    worst_df = pd.DataFrame(rows)

    return LocalVolCompareReport(
        expiries=Ts,
        strikes=K,
        forwards=forwards,
        y=y,
        g_sigma=g_sigma,
        g_local_var=g_lv,
        g_denom=g_denom,
        g_invalid=g_invalid,
        g_reason=g_reason,
        dupire=dup,
        diff_sigma=diff_sigma,
        diff_local_var=diff_lv,
        invalid_union=invalid_union,
        summary=summary,
        worst_diffs=worst_df,
        gatheral_reason_counts=_reason_counts(g_reason),
        dupire_reason_counts=_reason_counts(dup.reason),
    )
