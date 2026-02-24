"""Diagnostics helpers for implied volatility surfaces.

This module is intentionally **pure** (no plotting) and designed to support
portfolio-style notebooks and demos:

* slice extraction in strike space (K, y, iv, w)
* pricing calls on smile grids (Black-76)
* flattening/summary of no-arbitrage reports
* surfacing SVI calibration diagnostics (when smiles expose them)
* local-vol (Dupire / Gatheral) grid diagnostics summaries

The helpers rely on small Protocol interfaces (``VolSurfaceLike`` / ``SmileSlice``)
so they can work with multiple surface implementations.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from option_pricing.typing import ArrayLike, ScalarFn
from option_pricing.vol.dupire import (
    DupireLVReport,
    GatheralLVReport,
    LVInvalidReason,
    local_vol_from_call_grid_diagnostics,
)
from option_pricing.vol.surface import LocalVolSurface
from option_pricing.vol.vol_types import GridSmileSlice, SmileSlice

# -----------------------------------------------------------------------------
# Protocols: minimal interfaces (diagnostics should be surface-implementation
# agnostic)
# -----------------------------------------------------------------------------


def _smile_grid(smile: SmileSlice, *, n: int = 81) -> tuple[np.ndarray, np.ndarray]:
    """Return a (y, w) grid for a smile.

    * If the smile is grid-based, use its native grid.
    * Otherwise sample it uniformly over [y_min, y_max].
    """

    if isinstance(smile, GridSmileSlice):
        y = np.asarray(smile.y, dtype=float)
        w = np.asarray(smile.w, dtype=float)
        return y, w

    y = np.linspace(float(smile.y_min), float(smile.y_max), int(n), dtype=float)
    w = np.asarray(smile.w_at(y), dtype=float)
    return y, w


class VolSurfaceLike(Protocol):
    @property
    def smiles(self) -> Sequence[SmileSlice]: ...

    def iv(self, K: ArrayLike, T: float) -> np.ndarray: ...


class VolSurfaceClass(Protocol):
    @classmethod
    def from_grid(
        cls,
        rows: list[tuple[float, float, float]],
        *,
        forward: ScalarFn,
    ) -> VolSurfaceLike: ...


class Black76Module(Protocol):
    def black76_call_price_vec(
        self,
        *,
        forward: float,
        strikes: np.ndarray,
        sigma: np.ndarray,
        tau: float,
        df: float,
    ) -> np.ndarray: ...


def _default_black76() -> Black76Module:
    """Lazy-import Black-76 helpers to keep diagnostics lightweight."""

    from option_pricing.models.black_scholes import (
        bs as _bs,  # type: ignore[attr-defined]
    )

    return cast(Black76Module, _bs)


# -----------------------------------------------------------------------------
# Convenience data model
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceSlice:
    """A single smile slice expressed in strike space."""

    T: float
    F: float
    K: np.ndarray
    y: np.ndarray
    iv: np.ndarray
    w: np.ndarray


# -----------------------------------------------------------------------------
# Construction and sampling helpers
# -----------------------------------------------------------------------------


def build_surface_from_iv_function(
    *,
    expiries: Sequence[float],
    x_grid: np.ndarray,
    iv_fn: Callable[[float, np.ndarray], np.ndarray],
    forward: ScalarFn,
    VolSurface_cls: VolSurfaceClass | None = None,
) -> VolSurfaceLike:
    """Build a VolSurface from a synthetic ``iv_fn(T, y)`` function."""

    if VolSurface_cls is None:
        # Import from your real module path if needed; the key is to cast it.
        from option_pricing import (
            VolSurface as _VolSurface,  # type: ignore[attr-defined]
        )

        VolSurface_cls = cast(VolSurfaceClass, _VolSurface)

    exp = np.asarray(list(expiries), dtype=float)
    if exp.ndim != 1 or exp.size == 0:
        raise ValueError("expiries must be a non-empty 1D sequence")

    x_grid = np.asarray(x_grid, dtype=float)
    if x_grid.ndim != 1 or x_grid.size == 0:
        raise ValueError("x_grid must be a non-empty 1D array")

    rows: list[tuple[float, float, float]] = []
    for T in exp:
        T_ = float(T)
        F = float(forward(T_))
        K = F * np.exp(x_grid)

        iv = np.asarray(iv_fn(T_, x_grid), dtype=float)
        if iv.shape != x_grid.shape:
            raise ValueError(
                f"iv_fn must return shape {x_grid.shape}, got {iv.shape} at T={T_}"
            )

        for Ki, ivi in zip(K, iv, strict=True):
            rows.append((T_, float(Ki), float(ivi)))

    return VolSurface_cls.from_grid(rows, forward=forward)


def surface_slices(
    surface: VolSurfaceLike, *, forward: ScalarFn
) -> tuple[SurfaceSlice, ...]:
    """Return all smiles as strike-space slices (K, iv, w) for plotting."""

    out: list[SurfaceSlice] = []
    for s in surface.smiles:
        T = float(s.T)
        F = float(forward(T))
        y, w = _smile_grid(s)
        K = F * np.exp(y)
        iv = np.sqrt(np.maximum(w / T, 0.0))
        out.append(SurfaceSlice(T=T, F=F, K=K, y=y, iv=iv, w=w))
    return tuple(out)


def surface_points_df(surface: VolSurfaceLike, *, forward: ScalarFn) -> pd.DataFrame:
    """Flatten the surface into a tidy DataFrame (T, y, K, iv, w, F)."""

    rows: list[dict[str, float]] = []
    for sl in surface_slices(surface, forward=forward):
        for y, K, iv, w in zip(sl.y, sl.K, sl.iv, sl.w, strict=True):
            rows.append(
                {
                    "T": sl.T,
                    "y": float(y),
                    "K": float(K),
                    "iv": float(iv),
                    "w": float(w),
                    "F": float(sl.F),
                }
            )
    return pd.DataFrame(rows)


def query_iv_curve(surface: VolSurfaceLike, *, K: ArrayLike, T: float) -> np.ndarray:
    """Convenience wrapper around ``surface.iv``."""

    return np.asarray(surface.iv(K, float(T)), dtype=float)


def get_smile_at_T(
    surface: VolSurfaceLike, T: float, *, atol: float = 1e-12
) -> SmileSlice:
    """Find a smile by expiry using ``np.isclose`` matching."""

    T = float(T)
    for s in surface.smiles:
        if np.isclose(float(s.T), T, atol=atol, rtol=0.0):
            return s
    raise KeyError(
        f"No smile found at T={T} (available: {[float(s.T) for s in surface.smiles]})"
    )


# -----------------------------------------------------------------------------
# Call pricing on smile grids
# -----------------------------------------------------------------------------


def call_prices_from_smile(
    surface: VolSurfaceLike,
    *,
    T: float,
    forward: ScalarFn,
    df: ScalarFn,
    bs_model: Black76Module | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute discounted Black-76 call prices on the smile grid for expiry ``T``."""

    if bs_model is None:
        bs_model = _default_black76()

    s = get_smile_at_T(surface, T)
    T = float(s.T)
    F = float(forward(T))
    dfT = float(df(T))

    y, w = _smile_grid(s)
    K = F * np.exp(y)
    iv = np.sqrt(np.maximum(w / T, 0.0))

    C = bs_model.black76_call_price_vec(forward=F, strikes=K, sigma=iv, tau=T, df=dfT)
    return K, np.asarray(C, dtype=float), iv


def call_prices_from_surface_on_strikes(
    surface: VolSurfaceLike,
    *,
    expiries: Sequence[float],
    strikes: np.ndarray,
    forward: ScalarFn,
    df: ScalarFn,
    bs_model: Black76Module | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute discounted Black-76 call prices on a *shared* strike grid.

    This is primarily used for Dupire-from-call-grid diagnostics, which require
    a consistent K-grid across maturities.

    Returns
    -------
    strikes : (nK,)
    taus    : (nT,)
    calls   : (nT, nK)
    iv      : (nT, nK)
    forwards: (nT,)
    """

    if bs_model is None:
        bs_model = _default_black76()

    Ts = np.asarray(list(expiries), dtype=float)
    K = np.asarray(strikes, dtype=float)

    if Ts.ndim != 1 or Ts.size == 0:
        raise ValueError("expiries must be a non-empty 1D sequence")
    if K.ndim != 1 or K.size == 0:
        raise ValueError("strikes must be a non-empty 1D array")
    if not np.all(np.diff(Ts) > 0):
        raise ValueError("expiries must be strictly increasing for Dupire grids")
    if not np.all(np.diff(K) > 0):
        raise ValueError("strikes must be strictly increasing")
    if np.any(K <= 0):
        raise ValueError("strikes must be > 0")

    nT = int(Ts.size)
    nK = int(K.size)
    calls = np.empty((nT, nK), dtype=float)
    iv = np.empty((nT, nK), dtype=float)
    forwards = np.empty((nT,), dtype=float)
    dfs = np.empty((nT,), dtype=float)

    for i, T in enumerate(Ts):
        T_ = float(T)
        F = float(forward(T_))
        dfT = float(df(T_))
        if F <= 0.0:
            raise ValueError(f"forward(T) must be > 0, got {F} at T={T_}")
        if dfT <= 0.0:
            raise ValueError(f"df(T) must be > 0, got {dfT} at T={T_}")

        sigma = np.asarray(surface.iv(K, T_), dtype=float)
        forwards[i] = F
        dfs[i] = dfT
        iv[i, :] = sigma
        calls[i, :] = bs_model.black76_call_price_vec(
            forward=F, strikes=K, sigma=sigma, tau=T_, df=dfT
        )

    return K, Ts, calls, iv, forwards


# -----------------------------------------------------------------------------
# Calendar checks (total variance monotonicity in T)
# -----------------------------------------------------------------------------


def calendar_dW(surface: VolSurfaceLike, *, x_grid: np.ndarray) -> np.ndarray:
    """Compute Δw across maturities on a shared ``x_grid``."""

    xg = np.asarray(x_grid, dtype=float)

    W = np.asarray(
        np.vstack([np.asarray(s.w_at(xg), dtype=float) for s in surface.smiles]),
        dtype=float,
    )
    return np.asarray(W[1:, :] - W[:-1, :], dtype=float)


def first_failing_smile(report: Any) -> tuple[float, Any] | None:
    """Return (T, smile_report) for the first smile that fails strike monotonicity."""

    for T, r in getattr(report, "smile_monotonicity", []):
        if not bool(getattr(r, "ok", True)):
            return float(T), r
    return None


def calendar_dW_from_report(
    surface: VolSurfaceLike, report: Any
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (x_grid, dW) if the calendar check was performed and failed."""

    cal = getattr(report, "calendar_total_variance", None)
    if cal is None:
        return None

    performed = bool(getattr(cal, "performed", False))
    ok = bool(getattr(cal, "ok", True))
    xg = getattr(cal, "x_grid", None)

    if (not performed) or ok or (xg is None):
        return None

    xg_arr = np.asarray(xg, dtype=float)
    return xg_arr, calendar_dW(surface, x_grid=xg_arr)


# -----------------------------------------------------------------------------
# No-arbitrage report helpers
# -----------------------------------------------------------------------------


def noarb_smile_table(report: Any) -> pd.DataFrame:
    """Return a compact per-expiry table from a surface no-arb report.

    Designed for use with :func:`option_pricing.vol.arbitrage.check_surface_noarb`.

    Columns include smile-level monotonicity and convexity checks.
    """

    mono = list(getattr(report, "smile_monotonicity", ()) or ())
    conv = dict(getattr(report, "smile_convexity", ()) or ())

    rows: list[dict[str, Any]] = []
    for T, mrep in mono:
        Trep = float(T)
        crep = conv.get(T, conv.get(Trep))

        rows.append(
            {
                "T": Trep,
                "monotonic_ok": bool(getattr(mrep, "ok", False)),
                "monotonic_bad_count": int(
                    np.asarray(getattr(mrep, "bad_indices", [])).size
                ),
                "monotonic_max_violation": float(
                    getattr(mrep, "max_violation", np.nan)
                ),
                "monotonic_message": str(getattr(mrep, "message", "")),
                "convex_ok": (
                    bool(getattr(crep, "ok", False)) if crep is not None else False
                ),
                "convex_bad_count": (
                    int(np.asarray(getattr(crep, "bad_indices", [])).size)
                    if crep is not None
                    else 0
                ),
                "convex_max_violation": (
                    float(getattr(crep, "max_violation", np.nan))
                    if crep is not None
                    else np.nan
                ),
                "convex_message": (
                    str(getattr(crep, "message", "")) if crep is not None else ""
                ),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("T").reset_index(drop=True)
    return df


def calendar_summary(report: Any) -> dict[str, Any]:
    """Summarize the calendar no-arbitrage portion of a surface report."""

    cal = getattr(report, "calendar_total_variance", None)
    if cal is None:
        return {
            "performed": False,
            "ok": True,
            "n_violations": 0,
            "max_violation": 0.0,
            "message": "Calendar check not present on report.",
        }

    performed = bool(getattr(cal, "performed", False))
    ok = bool(getattr(cal, "ok", True))
    bad_pairs = np.asarray(getattr(cal, "bad_pairs", np.empty((0, 2), dtype=int)))
    n_viol = int(bad_pairs.shape[0]) if bad_pairs.ndim == 2 else int(bad_pairs.size)

    return {
        "performed": performed,
        "ok": ok,
        "n_violations": n_viol,
        "max_violation": float(getattr(cal, "max_violation", 0.0)),
        "message": str(getattr(cal, "message", "")),
        "x_grid": getattr(cal, "x_grid", None),
        "bad_pairs": bad_pairs,
    }


# -----------------------------------------------------------------------------
# Local-vol diagnostics helpers
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalVolGridReport:
    """Local-vol diagnostics sampled on a (T, y) grid.

    This is intended for notebook visualizations: heatmaps, invalid masks,
    denominator stability, etc.
    """

    expiries: np.ndarray  # (nT,)
    y: np.ndarray  # (ny,)
    K: np.ndarray  # (nT, ny)

    local_var: np.ndarray  # (nT, ny)
    sigma: np.ndarray  # (nT, ny)
    denom: np.ndarray  # (nT, ny)

    invalid: np.ndarray  # (nT, ny) bool
    reason: np.ndarray  # (nT, ny) uint32 bitmask

    invalid_count: int
    invalid_frac: float
    reason_counts: dict[str, int]
    worst_points: pd.DataFrame


def _reasons_to_str(mask: int) -> str:
    if mask == 0:
        return ""
    parts: list[str] = []
    for r in LVInvalidReason:
        if int(r) != 0 and (mask & int(r)):
            name = r.name
            if name is not None:
                parts.append(name)
    return "|".join(parts)


def localvol_grid_diagnostics(
    localvol: LocalVolSurface,
    *,
    expiries: Sequence[float],
    y_grid: np.ndarray,
    eps_w: float = 1e-12,
    eps_denom: float = 1e-12,
    top_n: int = 10,
) -> LocalVolGridReport:
    """Sample Gatheral local vol on a grid and return a diagnostics report.

    Parameters
    ----------
    localvol:
        :class:`option_pricing.vol.surface.LocalVolSurface`
    expiries:
        Sequence of maturities (years).
    y_grid:
        Log-moneyness grid y = ln(K/F(T)).

    Notes
    -----
    Uses :meth:`LocalVolSurface.local_var_diagnostics`, which returns a reason-coded
    invalid mask and the Gatheral denominator.
    """

    Ts = np.asarray(list(expiries), dtype=float)
    y = np.asarray(y_grid, dtype=float)
    if Ts.ndim != 1 or Ts.size == 0:
        raise ValueError("expiries must be a non-empty 1D sequence")
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y_grid must be a non-empty 1D array")

    # allocate
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

    # reason breakdown
    reason_counts: dict[str, int] = {}
    for r in LVInvalidReason:
        if int(r) == 0:
            continue
        c = int(np.sum((reason & np.uint32(int(r))) != 0))
        if c:
            name = r.name
            if name is not None:
                reason_counts[name] = c

    # worst points: smallest |denom| among finite denom
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
    """Notebook-friendly summary stats for a :class:`LocalVolGridReport`."""

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


@dataclass(frozen=True)
class LocalVolCompareReport:
    """Compare Gatheral-from-w vs Dupire-from-call-grid local vol on a shared (T, K) grid."""

    expiries: np.ndarray  # (nT,)
    strikes: np.ndarray  # (nK,)
    forwards: np.ndarray  # (nT,)

    # Gatheral diagnostics sampled at each maturity on the shared strike grid
    y: np.ndarray  # (nT, nK)
    g_sigma: np.ndarray  # (nT, nK)
    g_local_var: np.ndarray  # (nT, nK)
    g_denom: np.ndarray  # (nT, nK)
    g_invalid: np.ndarray  # (nT, nK)
    g_reason: np.ndarray  # (nT, nK)

    # Dupire diagnostics on the same call grid
    dupire: DupireLVReport

    # Differences (Dupire - Gatheral)
    diff_sigma: np.ndarray  # (nT, nK)
    diff_local_var: np.ndarray  # (nT, nK)
    invalid_union: np.ndarray  # (nT, nK)

    summary: dict[str, float | int]
    worst_diffs: pd.DataFrame
    gatheral_reason_counts: dict[str, int]
    dupire_reason_counts: dict[str, int]


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


def localvol_compare_gatheral_vs_dupire(
    localvol: LocalVolSurface,
    *,
    expiries: Sequence[float],
    strikes: np.ndarray,
    market: Any,
    # gatheral guardrails
    eps_w: float = 1e-12,
    eps_denom: float = 1e-12,
    # dupire settings
    price_convention: str = "discounted",
    strike_coordinate: str = "logK",
    trim_t: int = 1,
    trim_k: int = 1,
    eps_rel: float = 1e-12,
    eps_gamma_rel: float = 1e-12,
    # outputs
    top_n: int = 10,
    bs_model: Black76Module | None = None,
) -> LocalVolCompareReport:
    """Compare Gatheral local vol against Dupire local vol on a shared (T, K) grid.

    Notes
    -----
    * Gatheral is evaluated via :meth:`LocalVolSurface.local_var_diagnostics` at each
      maturity on the *same strike grid*.
    * Dupire is computed from a call-price grid derived from the **implied** surface
      (Black-76 using the surface implied vols).
    * This is intended as a *diagnostic consistency check*; disagreement can be due
      to finite-difference noise, time interpolation of w(T, y), trimming, or any of
      the Gatheral denominator fragilities.
    """

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

    # Ensure we respect the assumptions of the Dupire finite-difference stencils.
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

    # Build call grid from implied surface on the shared strike grid
    _K, _Ts, calls, _iv, _forwards2 = call_prices_from_surface_on_strikes(
        localvol.implied,
        expiries=Ts.tolist(),
        strikes=K,
        forward=localvol.forward,
        df=localvol.discount,
        bs_model=bs_model,
    )
    # keep forwards from the pricing call (should match the gatheral forwards)
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

    # Summary stats
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

    # Worst differences table
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


# -----------------------------------------------------------------------------
# SVI fit diagnostics (duck-typed)
# -----------------------------------------------------------------------------


def _maybe_get_svi_diagnostics(smile: SmileSlice) -> tuple[Any, Any, Any] | None:
    """Return (diag, checks, solver) if the smile exposes *SVI-like* diagnostics.

    We intentionally keep this duck-typed, but we also try to avoid false positives
    (e.g. a smile with a random ``diagnostics`` attribute).

    A diagnostics object is treated as SVI-like if it has a non-None ``checks``
    attribute (matching :class:`option_pricing.vol.svi.SVIFitDiagnostics`).
    """

    diag = getattr(smile, "diagnostics", None)
    if diag is None:
        return None

    checks = getattr(diag, "checks", None)
    if checks is None:
        return None

    solver = getattr(diag, "solver", None)
    return diag, checks, solver


def svi_fit_table(surface: VolSurfaceLike) -> pd.DataFrame:
    """Flatten available SVI calibration diagnostics into a per-expiry table.

    This helper is **duck-typed**: if a smile slice exposes a ``diagnostics``
    attribute compatible with :class:`option_pricing.vol.svi.SVIFitDiagnostics`,
    its fields are extracted. Otherwise, the slice is included with
    ``has_diagnostics=False``.

    Returns
    -------
    pd.DataFrame
        Sorted by expiry ``T``. Missing values are filled with NaN/False.
    """

    rows: list[dict[str, Any]] = []

    for s in surface.smiles:
        T = float(s.T)
        row: dict[str, Any] = {
            "T": T,
            "has_diagnostics": False,
            "diag_ok": np.nan,
            "failure_reason": "",
        }

        maybe = _maybe_get_svi_diagnostics(s)
        if maybe is None:
            rows.append(row)
            continue

        diag, checks, solver = maybe
        row["has_diagnostics"] = True
        ok_attr = getattr(diag, "ok", None)
        row["diag_ok"] = bool(ok_attr) if ok_attr is not None else np.nan
        row["failure_reason"] = str(getattr(diag, "failure_reason", "") or "")

        # ---- checks (fit + rails) ----
        if checks is not None:
            y_dom = getattr(checks, "y_domain", (np.nan, np.nan))
            try:
                y0, y1 = float(y_dom[0]), float(y_dom[1])
            except Exception:
                y0, y1 = np.nan, np.nan

            row.update(
                {
                    # data fit (total variance space)
                    "rmse_w": float(getattr(checks, "rmse_w", np.nan)),
                    "rmse_unw": float(getattr(checks, "rmse_unw", np.nan)),
                    "mae_w": float(getattr(checks, "mae_w", np.nan)),
                    "max_abs_werr": float(getattr(checks, "max_abs_werr", np.nan)),
                    # domain safety
                    "y_domain_min": y0,
                    "y_domain_max": y1,
                    "min_w_domain": float(getattr(checks, "min_w_domain", np.nan)),
                    "argmin_y_domain": float(
                        getattr(checks, "argmin_y_domain", np.nan)
                    ),
                    "n_domain_viol": int(getattr(checks, "n_violations", 0)),
                    # butterfly proxy (g)
                    "butterfly_ok": bool(getattr(checks, "butterfly_ok", False)),
                    "min_g": float(getattr(checks, "min_g", np.nan)),
                    "argmin_g_y": float(getattr(checks, "argmin_g_y", np.nan)),
                    "g_left_inf": float(getattr(checks, "g_left_inf", np.nan)),
                    "g_right_inf": float(getattr(checks, "g_right_inf", np.nan)),
                    "butterfly_reason": str(
                        getattr(checks, "butterfly_reason", "") or ""
                    ),
                    # wing / Lee
                    "sR": float(getattr(checks, "sR", np.nan)),
                    "sL": float(getattr(checks, "sL", np.nan)),
                    "lee_cap": float(getattr(checks, "lee_cap", np.nan)),
                    "lee_slack_R": float(getattr(checks, "lee_slack_R", np.nan)),
                    "lee_slack_L": float(getattr(checks, "lee_slack_L", np.nan)),
                    # flags
                    "rho_near_pm1": bool(getattr(checks, "rho_near_pm1", False)),
                    "sigma_tiny": bool(getattr(checks, "sigma_tiny", False)),
                    "b_large": bool(getattr(checks, "b_large", False)),
                    "b_blown_up": bool(getattr(checks, "b_blown_up", False)),
                    "m_outside_data": bool(getattr(checks, "m_outside_data", False)),
                    # robust weights summary (if IRLS used)
                    "robust_w_min": float(
                        getattr(checks, "robust_weights_min", np.nan)
                    ),
                    "robust_w_med": float(
                        getattr(checks, "robust_weights_median", np.nan)
                    ),
                    "robust_w_max": float(
                        getattr(checks, "robust_weights_max", np.nan)
                    ),
                    "robust_w_frac_floored": float(
                        getattr(checks, "robust_weights_frac_floored", np.nan)
                    ),
                    "robust_w_entropy": float(
                        getattr(checks, "robust_weights_entropy", np.nan)
                    ),
                }
            )

        # ---- solver metadata ----
        if solver is not None:
            row.update(
                {
                    "termination": str(getattr(solver, "termination", "")),
                    "nfev": int(getattr(solver, "nfev", 0)),
                    "cost": float(getattr(solver, "cost", np.nan)),
                    "optimality": float(getattr(solver, "optimality", np.nan)),
                    "step_norm": float(getattr(solver, "step_norm", np.nan)),
                    "irls_outer_iters": int(
                        getattr(
                            solver, "irls_outer_iters", getattr(solver, "irls_iters", 0)
                        )
                    ),
                }
            )

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("T").reset_index(drop=True)
    return df


def svi_residuals_df(
    surface: VolSurfaceLike,
    quotes_df: pd.DataFrame,
    *,
    T: float,
    forward: ScalarFn | None = None,
    atol: float = 1e-12,
) -> pd.DataFrame:
    """Join quote points to the fitted smile and compute residuals.

    Parameters
    ----------
    surface:
        Calibrated surface.
    quotes_df:
        Quote grid. Must contain either ``y`` or ``K``. Must contain either
        ``iv`` or ``w``. If it contains multiple expiries, a ``T`` column is
        expected and points are filtered using ``np.isclose``.
    T:
        Expiry to slice.
    forward:
        Needed if ``y`` is not present (to convert strikes to log-moneyness).

    Returns
    -------
    pd.DataFrame
        Columns: ``T, y, K, F, iv_mkt, iv_fit, resid_iv, w_mkt, w_fit, resid_w``.
    """

    T = float(T)
    s = get_smile_at_T(surface, T, atol=atol)

    q = quotes_df.copy()
    if "T" in q.columns:
        q = q.loc[np.isclose(q["T"].astype(float), T, atol=atol, rtol=0.0)].copy()

    if q.empty:
        return pd.DataFrame(
            columns=[
                "T",
                "y",
                "K",
                "F",
                "iv_mkt",
                "iv_fit",
                "resid_iv",
                "w_mkt",
                "w_fit",
                "resid_w",
            ]
        )

    # --- y / K ---
    if "y" in q.columns:
        y = q["y"].astype(float).to_numpy()
        if forward is not None:
            F = float(forward(T))
            K = (F * np.exp(y)).astype(float)
        else:
            K = (
                q["K"].astype(float).to_numpy()
                if "K" in q.columns
                else np.full_like(y, np.nan)
            )
    elif "K" in q.columns:
        if forward is None:
            raise ValueError("quotes_df has K but not y; forward(T) is required")
        K = q["K"].astype(float).to_numpy()
        F = float(forward(T))
        y = np.log(K / F)
    else:
        raise ValueError("quotes_df must contain a 'y' or 'K' column")

    # --- market iv / w ---
    if "iv" in q.columns:
        iv_mkt = q["iv"].to_numpy(dtype=np.float64, copy=False)
        w_mkt = (iv_mkt * iv_mkt) * np.float64(T)
    elif "w" in q.columns:
        w_mkt = q["w"].to_numpy(dtype=np.float64, copy=False)
        iv_mkt = np.sqrt(np.maximum(w_mkt / np.float64(T), 0.0)).astype(
            np.float64, copy=False
        )
    else:
        raise ValueError("quotes_df must contain an 'iv' or 'w' column")

    w_fit = np.asarray(s.w_at(y), dtype=float)
    iv_fit = np.asarray(s.iv_at(y), dtype=float)

    out = pd.DataFrame(
        {
            "T": T,
            "y": y.astype(float),
            "K": np.asarray(K, dtype=float),
            "F": float(forward(T)) if forward is not None else np.nan,
            "iv_mkt": iv_mkt.astype(float),
            "iv_fit": iv_fit.astype(float),
            "resid_iv": (iv_fit - iv_mkt).astype(float),
            "w_mkt": w_mkt.astype(float),
            "w_fit": w_fit.astype(float),
            "resid_w": (w_fit - w_mkt).astype(float),
        }
    )

    return out.sort_values("y").reset_index(drop=True)


def surface_domain_report(
    surface: VolSurfaceLike,
    *,
    quotes_df: pd.DataFrame | None = None,
    forward: ScalarFn | None = None,
    atol: float = 1e-12,
) -> pd.DataFrame:
    """Per-expiry domain coverage report (data vs recommended model domain).

    For each expiry ``T`` on the surface, returns:

    * model domain: ``y_model_min/max`` (from the smile slice)
    * optional quote domain: ``y_data_min/max`` (from ``quotes_df``)
    * slack:
        - ``slack_left  = y_data_min - y_model_min``
        - ``slack_right = y_model_max - y_data_max``

    Interpreting slack:
    * positive slack => model recommended domain extends beyond the quote range
    * negative slack => quotes extend beyond the model recommended domain
    """

    rows: list[dict[str, Any]] = []

    q = quotes_df.copy() if quotes_df is not None else None

    for s in surface.smiles:
        T = float(s.T)
        y_model_min = float(s.y_min)
        y_model_max = float(s.y_max)

        row: dict[str, Any] = {
            "T": T,
            "y_model_min": y_model_min,
            "y_model_max": y_model_max,
            "y_data_min": np.nan,
            "y_data_max": np.nan,
            "slack_left": np.nan,
            "slack_right": np.nan,
        }

        if forward is not None:
            row["F"] = float(forward(T))

        if q is not None and not q.empty:
            if "T" in q.columns:
                qq = q.loc[
                    np.isclose(q["T"].astype(float), T, atol=atol, rtol=0.0)
                ].copy()
            else:
                # quotes_df treated as single-expiry; apply to each slice
                qq = q

            if not qq.empty:
                if "y" in qq.columns:
                    y_data = qq["y"].astype(float).to_numpy()
                elif "K" in qq.columns:
                    if forward is None:
                        raise ValueError(
                            "quotes_df has K but not y; forward(T) is required"
                        )
                    F = float(forward(T))
                    K = qq["K"].astype(float).to_numpy()
                    y_data = np.log(K / F)
                else:
                    raise ValueError("quotes_df must contain 'y' or 'K'")

                y0 = float(np.min(y_data))
                y1 = float(np.max(y_data))
                row["y_data_min"] = y0
                row["y_data_max"] = y1
                row["slack_left"] = y0 - y_model_min
                row["slack_right"] = y_model_max - y1

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("T").reset_index(drop=True)
    return df
