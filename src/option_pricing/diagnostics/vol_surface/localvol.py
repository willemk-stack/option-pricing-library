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


def fixed_strikes_from_forward_y(
    y_grid: np.ndarray,
    *,
    forward: Any,
    reference_expiry: float | None = None,
    reference_forward: float | None = None,
) -> np.ndarray:
    """Build a fixed strike grid from a forward-log-moneyness grid.

    ``local_vol_from_call_grid_diagnostics`` differentiates call prices across
    maturities at fixed strikes. eSSVI surfaces, however, are parameterized by
    ``y = log(K / F(T))``. This helper chooses one reference forward and returns
    ``K = F_ref * exp(y_grid)``; each maturity then has its own effective
    forward log-moneyness ``log(K / F(T))`` when the surface is evaluated.
    """

    y = np.asarray(y_grid, dtype=float)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y_grid must be a non-empty 1D array")
    if not np.all(np.isfinite(y)):
        raise ValueError("y_grid must contain finite values")
    if not np.all(np.diff(y) > 0):
        raise ValueError("y_grid must be strictly increasing")

    if reference_forward is None:
        if reference_expiry is None:
            raise ValueError(
                "Pass either reference_forward or reference_expiry to set the "
                "fixed strike-grid anchor."
            )
        reference_forward = float(forward(float(reference_expiry)))

    f_ref = float(reference_forward)
    if not np.isfinite(f_ref) or f_ref <= 0.0:
        raise ValueError("reference_forward must be positive and finite")

    strikes = np.asarray(f_ref * np.exp(y), dtype=float)
    if not np.all(np.diff(strikes) > 0):  # pragma: no cover - guarded by y monotone
        raise ValueError("constructed strikes must be strictly increasing")
    return strikes


def _boundary_mask(
    shape: tuple[int, int],
    *,
    trim_t: int,
    trim_k: int,
) -> np.ndarray:
    nT, nK = shape
    boundary = np.zeros(shape, dtype=bool)
    if trim_t > 0:
        trim_t_eff = min(int(trim_t), nT)
        boundary[:trim_t_eff, :] = True
        boundary[nT - trim_t_eff :, :] = True
    if trim_k > 0:
        trim_k_eff = min(int(trim_k), nK)
        boundary[:, :trim_k_eff] = True
        boundary[:, nK - trim_k_eff :] = True
    return boundary


def _invalid_points_table(
    *,
    invalid: np.ndarray,
    reason: np.ndarray,
    expiries: np.ndarray,
    strikes: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    local_var: np.ndarray,
    denom: np.ndarray,
    trim_t: int,
    trim_k: int,
    numerator: np.ndarray | None = None,
    curvature: np.ndarray | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    invalid_mask = np.asarray(invalid, dtype=bool)
    if invalid_mask.ndim != 2:
        raise ValueError("invalid must be a 2D mask")

    Ts = np.asarray(expiries, dtype=float)
    K = np.asarray(strikes, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    reason_arr = np.asarray(reason, dtype=np.uint32)
    boundary = _boundary_mask(
        invalid_mask.shape,
        trim_t=int(trim_t),
        trim_k=int(trim_k),
    )

    rows: list[dict[str, Any]] = []
    nT, nK = invalid_mask.shape
    for i in range(nT):
        for j in range(nK):
            if not invalid_mask[i, j]:
                continue
            K_ij = float(K[i, j]) if K.ndim == 2 else float(K[j])
            row: dict[str, Any] = {
                "T": float(Ts[i]),
                "K": K_ij,
                "y": float(y_arr[i, j]),
                "i_t": int(i),
                "i_k": int(j),
                "is_boundary": bool(boundary[i, j]),
                "is_interior": bool(not boundary[i, j]),
                "sigma": float(sigma[i, j]) if np.isfinite(sigma[i, j]) else np.nan,
                "local_var": (
                    float(local_var[i, j]) if np.isfinite(local_var[i, j]) else np.nan
                ),
                "denom": float(denom[i, j]) if np.isfinite(denom[i, j]) else np.nan,
                "reason_code": int(reason_arr[i, j]),
                "reasons": _reasons_to_str(int(reason_arr[i, j])),
            }
            if numerator is not None:
                row["numerator"] = (
                    float(numerator[i, j]) if np.isfinite(numerator[i, j]) else np.nan
                )
            if curvature is not None:
                row["curvature"] = (
                    float(curvature[i, j]) if np.isfinite(curvature[i, j]) else np.nan
                )
            rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values(
        ["is_interior", "T", "K"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    if top_n is not None:
        return table.head(int(top_n)).copy()
    return table


def _boundary_summary(
    *,
    invalid: np.ndarray,
    trim_t: int,
    trim_k: int,
) -> dict[str, float | int | bool]:
    invalid_mask = np.asarray(invalid, dtype=bool)
    boundary = _boundary_mask(
        invalid_mask.shape,
        trim_t=int(trim_t),
        trim_k=int(trim_k),
    )
    total = int(invalid_mask.size)
    invalid_count = int(np.sum(invalid_mask))
    boundary_invalid = int(np.sum(invalid_mask & boundary))
    interior_invalid = int(np.sum(invalid_mask & ~boundary))
    boundary_cells = int(np.sum(boundary))
    interior_cells = int(total - boundary_cells)
    return {
        "n_total": total,
        "invalid_count": invalid_count,
        "boundary_cell_count": boundary_cells,
        "interior_cell_count": interior_cells,
        "boundary_invalid_count": boundary_invalid,
        "interior_invalid_count": interior_invalid,
        "boundary_invalid_frac": (
            float(boundary_invalid) / float(boundary_cells) if boundary_cells else 0.0
        ),
        "interior_invalid_frac": (
            float(interior_invalid) / float(interior_cells) if interior_cells else 0.0
        ),
        "invalids_are_boundary_only": bool(invalid_count > 0 and interior_invalid == 0),
    }


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
    invalid_points = _invalid_points_table(
        invalid=invalid,
        reason=reason,
        expiries=Ts,
        strikes=K_grid,
        y=np.broadcast_to(y[None, :], (nT, ny)),
        sigma=sig,
        local_var=lv,
        denom=denom,
        trim_t=0,
        trim_k=0,
        top_n=None,
    )

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
        invalid_points=invalid_points,
        coordinate_conventions={
            "analytic_gatheral": "y = log(K / F(T)); K grid is rebuilt as F(T) * exp(y) for each maturity.",
            "surface_derivatives": "LocalVolSurface.local_var_diagnostics uses implied.w_and_derivs(y, T) when available.",
        },
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

    summary: dict[str, float | int | bool] = {
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
    boundary_summary = _boundary_summary(
        invalid=dup.invalid,
        trim_t=int(trim_t),
        trim_k=int(trim_k),
    )
    summary.update(boundary_summary)

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
    invalid_points = _invalid_points_table(
        invalid=dup.invalid,
        reason=dup.reason,
        expiries=Ts,
        strikes=K,
        y=y,
        sigma=dup.sigma,
        local_var=dup.local_var,
        denom=dup.denom,
        numerator=dup.num,
        curvature=dup.curvature,
        trim_t=int(trim_t),
        trim_k=int(trim_k),
        top_n=None,
    )

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
        invalid_points=invalid_points,
        boundary_summary=boundary_summary,
        coordinate_conventions={
            "gatheral": "Analytic path evaluates y = log(K / F(T)) at each fixed strike and maturity.",
            "dupire_call_grid": "Finite-difference path expects call prices on one fixed K grid across all taus.",
            "strike_coordinate_logK": "strike_coordinate='logK' means differentiate along log(strikes); pass positive strikes, not log-strike values.",
            "price_convention_discounted": "price_convention='discounted' means PV call prices with the q*C term in the Dupire numerator.",
        },
    )


def localvol_compare_gatheral_vs_dupire_on_forward_y(
    localvol: LocalVolSurface,
    *,
    expiries: Sequence[float],
    y_grid: np.ndarray,
    market: Any,
    reference_expiry: float | None = None,
    reference_forward: float | None = None,
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
    """Compare Gatheral and call-grid Dupire from a forward-y reference grid.

    The returned strike grid is fixed across maturities. It is anchored by
    ``reference_forward`` or by ``localvol.forward(reference_expiry)``; it is
    not rebuilt per expiry. This is the notebook-safe bridge from an eSSVI
    ``y`` domain to a Dupire call-price grid.
    """

    strikes = fixed_strikes_from_forward_y(
        y_grid,
        forward=localvol.forward,
        reference_expiry=reference_expiry,
        reference_forward=reference_forward,
    )
    report = localvol_compare_gatheral_vs_dupire(
        localvol,
        expiries=expiries,
        strikes=strikes,
        market=market,
        eps_w=eps_w,
        eps_denom=eps_denom,
        price_convention=price_convention,
        strike_coordinate=strike_coordinate,
        trim_t=trim_t,
        trim_k=trim_k,
        eps_rel=eps_rel,
        eps_gamma_rel=eps_gamma_rel,
        top_n=top_n,
        bs_model=bs_model,
    )
    conventions = dict(report.coordinate_conventions)
    conventions["forward_y_reference_grid"] = (
        "Input y_grid is used only to build fixed strikes K = F_ref * exp(y); "
        "per-maturity comparison y is log(K / F(T))."
    )
    if reference_forward is not None:
        conventions["reference_forward"] = str(float(reference_forward))
    elif reference_expiry is not None:
        conventions["reference_expiry"] = str(float(reference_expiry))

    return LocalVolCompareReport(
        expiries=report.expiries,
        strikes=report.strikes,
        forwards=report.forwards,
        y=report.y,
        g_sigma=report.g_sigma,
        g_local_var=report.g_local_var,
        g_denom=report.g_denom,
        g_invalid=report.g_invalid,
        g_reason=report.g_reason,
        dupire=report.dupire,
        diff_sigma=report.diff_sigma,
        diff_local_var=report.diff_local_var,
        invalid_union=report.invalid_union,
        summary=report.summary,
        worst_diffs=report.worst_diffs,
        gatheral_reason_counts=report.gatheral_reason_counts,
        dupire_reason_counts=report.dupire_reason_counts,
        invalid_points=report.invalid_points,
        boundary_summary=report.boundary_summary,
        coordinate_conventions=conventions,
    )
