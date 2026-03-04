from __future__ import annotations

import numpy as np

from option_pricing.typing import ScalarFn
from option_pricing.vol.vol_types import GridSmileSlice, SmileSlice

from .models import SurfaceSlice


def _smile_grid(smile: SmileSlice, *, n: int = 81) -> tuple[np.ndarray, np.ndarray]:
    """Return a (y, w) grid for a smile.

    If the smile is grid-based, use its native grid. Otherwise sample it
    uniformly over the recommended [y_min, y_max] domain.
    """
    if isinstance(smile, GridSmileSlice):
        y = np.asarray(smile.y, dtype=float)
        w = np.asarray(smile.w, dtype=float)
        return y, w

    y = np.linspace(float(smile.y_min), float(smile.y_max), int(n), dtype=float)
    w = np.asarray(smile.w_at(y), dtype=float)
    return y, w


def surface_slices(surface, *, forward: ScalarFn) -> tuple[SurfaceSlice, ...]:
    out: list[SurfaceSlice] = []
    for s in surface.smiles:
        T = float(s.T)
        F = float(forward(T))
        y, w = _smile_grid(s)
        K = F * np.exp(y)
        iv = np.sqrt(np.maximum(w / T, 0.0))
        out.append(
            SurfaceSlice(
                T=T,
                y=np.asarray(y, dtype=float),
                K=np.asarray(K, dtype=float),
                F=float(F),
                w=np.asarray(w, dtype=float),
                iv=np.asarray(iv, dtype=float),
            )
        )
    return tuple(out)


def surface_points_df(surface, *, forward: ScalarFn):
    import pandas as pd

    rows: list[dict[str, float]] = []
    for sl in surface_slices(surface, forward=forward):
        for y, K, w, iv in zip(sl.y, sl.K, sl.w, sl.iv, strict=True):
            rows.append(
                {
                    "T": float(sl.T),
                    "y": float(y),
                    "K": float(K),
                    "F": float(sl.F),
                    "w": float(w),
                    "iv": float(iv),
                }
            )
    return pd.DataFrame(rows)


def query_iv_curve(surface, *, K, T):
    return np.asarray(surface.iv(K, float(T)), dtype=float)


def get_smile_at_T(surface, T, *, atol: float = 1e-12):
    T = float(T)
    for s in surface.smiles:
        if np.isclose(float(s.T), T, atol=atol, rtol=0.0):
            return s
    avail = [float(s.T) for s in surface.smiles]
    raise KeyError(f"No smile found at T={T} (available: {avail})")
