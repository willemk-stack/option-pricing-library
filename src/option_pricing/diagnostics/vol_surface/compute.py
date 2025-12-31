from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

ArrayLike = float | np.ndarray
ForwardFn = Callable[[float], float]
DiscountFn = Callable[[float], float]


@dataclass(frozen=True)
class SurfaceSlice:
    """A single smile slice expressed in strike space."""

    T: float
    F: float
    K: np.ndarray
    x: np.ndarray
    iv: np.ndarray
    w: np.ndarray


def build_surface_from_iv_function(
    *,
    expiries: Sequence[float],
    x_grid: np.ndarray,
    iv_fn: Callable[[float, np.ndarray], np.ndarray],
    forward: ForwardFn,
    VolSurface_cls: type | None = None,
) -> Any:
    """Build a VolSurface from a synthetic iv(T, x) function.

    Parameters
    ----------
    expiries:
        Expiries in **years**.
    x_grid:
        Log-moneyness grid x = ln(K / F(T)). This is used to generate K points.
    iv_fn:
        Callable (T, x_grid) -> implied vol array, same length as x_grid.
    forward:
        Callable forward(T) -> forward price F(T).
    VolSurface_cls:
        Optional class to use (defaults to `option_pricing.VolSurface`).

    Returns
    -------
    VolSurface instance created via `VolSurface.from_grid(rows, forward=forward)`.
    """
    if VolSurface_cls is None:
        from option_pricing import VolSurface as VolSurface_cls  # type: ignore

    exp = np.asarray(list(expiries), dtype=float)
    if exp.ndim != 1 or exp.size == 0:
        raise ValueError("expiries must be a non-empty 1D sequence")
    x_grid = np.asarray(x_grid, dtype=float)
    if x_grid.ndim != 1 or x_grid.size == 0:
        raise ValueError("x_grid must be a non-empty 1D array")

    rows: list[tuple[float, float, float]] = []
    for T in exp:
        F = float(forward(float(T)))
        K = F * np.exp(x_grid)
        iv = np.asarray(iv_fn(float(T), x_grid), dtype=float)
        if iv.shape != x_grid.shape:
            raise ValueError(
                f"iv_fn must return shape {x_grid.shape}, got {iv.shape} at T={T}"
            )
        for Ki, ivi in zip(K, iv, strict=True):
            rows.append((float(T), float(Ki), float(ivi)))

    return VolSurface_cls.from_grid(rows, forward=forward)


def surface_slices(
    surface: Any,
    *,
    forward: ForwardFn,
) -> tuple[SurfaceSlice, ...]:
    """Return all smiles as strike-space slices (K, iv, w) for plotting."""
    out: list[SurfaceSlice] = []
    for s in surface.smiles:
        T = float(s.T)
        F = float(forward(T))
        x = np.asarray(s.x, dtype=float)
        w = np.asarray(s.w, dtype=float)
        K = F * np.exp(x)
        iv = np.sqrt(np.maximum(w / T, 0.0))
        out.append(SurfaceSlice(T=T, F=F, K=K, x=x, iv=iv, w=w))
    return tuple(out)


def surface_points_df(
    surface: Any,
    *,
    forward: ForwardFn,
) -> pd.DataFrame:
    """Flatten the surface into a tidy DataFrame (T, x, K, iv, w)."""
    rows = []
    for sl in surface_slices(surface, forward=forward):
        for x, K, iv, w in zip(sl.x, sl.K, sl.iv, sl.w, strict=True):
            rows.append(
                {
                    "T": sl.T,
                    "x": float(x),
                    "K": float(K),
                    "iv": float(iv),
                    "w": float(w),
                    "F": sl.F,
                }
            )
    return pd.DataFrame(rows)


def query_iv_curve(surface: Any, *, K: ArrayLike, T: float) -> np.ndarray:
    """Convenience wrapper around surface.iv."""
    return np.asarray(surface.iv(K, float(T)), dtype=float)


def noarb_smile_table(report: Any) -> pd.DataFrame:
    """Convert `check_surface_noarb` strike-monotonicity results into a DataFrame."""
    rows = []
    for T, r in getattr(report, "smile_monotonicity", []):
        rows.append(
            {
                "T": float(T),
                "ok": bool(r.ok),
                "max_violation": float(r.max_violation),
                "n_bad_intervals": int(getattr(r.bad_indices, "size", 0)),
                "message": str(r.message),
            }
        )
    return (
        pd.DataFrame(rows).sort_values("T", ignore_index=True)
        if rows
        else pd.DataFrame(
            columns=["T", "ok", "max_violation", "n_bad_intervals", "message"]
        )
    )


def calendar_summary(report: Any) -> dict[str, Any]:
    """Extract a small calendar-check summary dict from `check_surface_noarb` report."""
    cal = getattr(report, "calendar_total_variance", None)
    if cal is None:
        return {
            "performed": False,
            "ok": True,
            "message": "calendar_total_variance missing",
        }
    return {
        "performed": bool(getattr(cal, "performed", False)),
        "ok": bool(getattr(cal, "ok", True)),
        "message": str(getattr(cal, "message", "")),
        "x_grid": getattr(cal, "x_grid", None),
    }


def get_smile_at_T(surface: Any, T: float, *, atol: float = 1e-12) -> Any:
    """Find a Smile by expiry (float years) using isclose matching."""
    T = float(T)
    for s in surface.smiles:
        if np.isclose(float(s.T), T, atol=atol, rtol=0.0):
            return s
    raise KeyError(
        f"No smile found at T={T} (available: {[float(s.T) for s in surface.smiles]})"
    )


def call_prices_from_smile(
    surface: Any,
    *,
    T: float,
    forward: ForwardFn,
    df: DiscountFn,
    bs_model: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute discounted Black-76 call prices on the smile grid for a given expiry.

    Returns
    -------
    K, C, iv arrays (same length).
    """
    if bs_model is None:
        from option_pricing.models import bs as bs_model  # type: ignore

    s = get_smile_at_T(surface, T)
    T = float(s.T)
    F = float(forward(T))
    dfT = float(df(T))

    x = np.asarray(s.x, dtype=float)
    w = np.asarray(s.w, dtype=float)
    K = F * np.exp(x)
    iv = np.sqrt(np.maximum(w / T, 0.0))

    C = bs_model.black76_call_price_vec(forward=F, strikes=K, sigma=iv, tau=T, df=dfT)
    return K, np.asarray(C, dtype=float), iv


def calendar_dW(
    surface: Any,
    *,
    x_grid: np.ndarray,
) -> np.ndarray:
    """Compute Δw across maturities on a shared x_grid: dW[i,x] = w(T_{i+1},x) - w(T_i,x)."""
    xg = np.asarray(x_grid, dtype=float)
    W = np.vstack([np.asarray(s.w_at(xg), dtype=float) for s in surface.smiles])
    return W[1:, :] - W[:-1, :]


def first_failing_smile(report: Any) -> tuple[float, Any] | None:
    """Return (T, smile_report) for the first smile that fails strike monotonicity."""
    for T, r in getattr(report, "smile_monotonicity", []):
        if not bool(r.ok):
            return float(T), r
    return None


def calendar_x_grid(report: Any) -> np.ndarray | None:
    """Return the calendar-check x_grid if it was computed."""
    cal = getattr(report, "calendar_total_variance", None)
    if cal is None:
        return None
    xg = getattr(cal, "x_grid", None)
    return None if xg is None else np.asarray(xg, dtype=float)


def calendar_dW_from_report(
    surface: Any, report: Any
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (x_grid, dW) if the calendar check was performed and failed.

    Useful for plotting a heatmap of negative cells.
    """
    cal = getattr(report, "calendar_total_variance", None)
    if cal is None:
        return None
    performed = bool(getattr(cal, "performed", False))
    ok = bool(getattr(cal, "ok", True))
    xg = getattr(cal, "x_grid", None)
    if (not performed) or ok or (xg is None):
        return None
    xg = np.asarray(xg, dtype=float)
    return xg, calendar_dW(surface, x_grid=xg)
