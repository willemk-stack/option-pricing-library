from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

ArrayLike = float | np.ndarray
ForwardFn = Callable[[float], float]
DiscountFn = Callable[[float], float]


# --- Protocols: minimal interfaces to make mypy happy ---


class SmileLike(Protocol):
    T: float
    x: np.ndarray
    w: np.ndarray

    def w_at(self, xq: np.ndarray) -> np.ndarray: ...


class VolSurfaceLike(Protocol):
    smiles: Sequence[SmileLike]

    def iv(self, K: ArrayLike, T: float) -> np.ndarray: ...


class VolSurfaceClass(Protocol):
    @classmethod
    def from_grid(
        cls,
        rows: list[tuple[float, float, float]],
        *,
        forward: ForwardFn,
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
    VolSurface_cls: VolSurfaceClass | None = None,
) -> VolSurfaceLike:
    """Build a VolSurface from a synthetic iv(T, x) function."""
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
    surface: VolSurfaceLike,
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
    surface: VolSurfaceLike,
    *,
    forward: ForwardFn,
) -> pd.DataFrame:
    """Flatten the surface into a tidy DataFrame (T, x, K, iv, w)."""
    rows: list[dict[str, float]] = []
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


def query_iv_curve(surface: VolSurfaceLike, *, K: ArrayLike, T: float) -> np.ndarray:
    """Convenience wrapper around surface.iv."""
    return np.asarray(surface.iv(K, float(T)), dtype=float)


def get_smile_at_T(
    surface: VolSurfaceLike, T: float, *, atol: float = 1e-12
) -> SmileLike:
    """Find a Smile by expiry (float years) using isclose matching."""
    T = float(T)
    for s in surface.smiles:
        if np.isclose(float(s.T), T, atol=atol, rtol=0.0):
            return s
    raise KeyError(
        f"No smile found at T={T} (available: {[float(s.T) for s in surface.smiles]})"
    )


def call_prices_from_smile(
    surface: VolSurfaceLike,
    *,
    T: float,
    forward: ForwardFn,
    df: DiscountFn,
    bs_model: Black76Module | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute discounted Black-76 call prices on the smile grid for a given expiry."""
    if bs_model is None:
        from option_pricing.models import bs as _bs  # type: ignore[attr-defined]

        bs_model = cast(Black76Module, _bs)

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


def calendar_dW(surface: VolSurfaceLike, *, x_grid: np.ndarray) -> np.ndarray:
    """Compute Δw across maturities on a shared x_grid."""
    xg = np.asarray(x_grid, dtype=float)

    W = np.asarray(
        np.vstack([np.asarray(s.w_at(xg), dtype=float) for s in surface.smiles]),
        dtype=float,
    )
    dW = W[1:, :] - W[:-1, :]
    return np.asarray(dW, dtype=float)


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
