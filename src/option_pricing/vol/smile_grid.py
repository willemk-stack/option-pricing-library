"""Smile grid interpolation and manipulation helpers.

This module provides the Smile class and supporting functions for interpolating
total-variance grids using adaptive interpolation strategies (Fritsch-Carlson
for monotone regions, piecewise stitching for non-monotone smiles, linear fallback).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..numerics.interpolation import FritschCarlson, linear_interp_factory
from ..typing import ArrayLike, FloatArray


def _is_monotone(y: np.ndarray, *, eps: float = 0.0) -> bool:
    """Check if array is monotone (either all increasing or all decreasing)."""
    dy = np.diff(y)
    return bool(np.all(dy >= -eps) or np.all(dy <= eps))


def _u_split_index(w: np.ndarray, *, eps: float | None = None) -> int | None:
    """
    Return k such that left piece is w[:k+1], right piece is w[k:].
    Detects a U-turn: negative slope region followed by positive slope region,
    with tolerance eps to ignore noise/plateaus.
    If multiple U-turns exist, picks the deepest valley (smallest w[k]).
    """
    w = np.asarray(w, dtype=np.float64)
    if w.size < 3:
        return None

    if eps is None:
        scale = float(max(1.0, np.max(np.abs(w))))
        eps = 1e-12 * scale

    dw = np.diff(w)
    s = np.zeros_like(dw, dtype=np.int8)
    s[dw < -eps] = -1
    s[dw > eps] = +1

    # Find all "neg -> pos" transitions, allowing zeros between them
    candidates: list[int] = []
    seen_neg = False
    for i in range(s.size):
        if s[i] == -1:
            seen_neg = True
            continue
        if seen_neg and s[i] == +1:
            k = i + 1  # slope index i is between w[i] and w[i+1]
            if 0 < k < w.size - 1:
                candidates.append(k)

    if not candidates:
        return None

    # Pick the deepest valley among candidates
    k_best = min(candidates, key=lambda k: w[k])
    return int(k_best)


def _stitch_two(
    xk: float,
    left_interp: Callable[[np.ndarray], np.ndarray],
    right_interp: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    """Stitch two array-capable interpolators at xk, handling scalar xq safely."""

    def pw(xq: np.ndarray) -> np.ndarray:
        xq_in = np.asarray(xq, dtype=np.float64)
        xq_1d = np.atleast_1d(xq_in)

        out = np.empty_like(xq_1d, dtype=np.float64)
        left = xq_1d <= xk
        if np.any(left):
            out[left] = left_interp(xq_1d[left])
        if np.any(~left):
            out[~left] = right_interp(xq_1d[~left])

        if xq_in.ndim == 0:
            return np.asarray(out[0], dtype=np.float64)
        return out.reshape(xq_in.shape)

    return pw


def _make_w_interpolator(
    x: np.ndarray, w: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Pick an interpolator for total variance w(x).

    Priority:
      1) If w is globally monotone: FC on full grid.
      2) Else try split into two monotone pieces (U-shape):
         - robust sign-change split
         - fallback: split at argmin
      3) Else linear with flat extrapolation.
    """
    # Fail-safe: if non-finite, fallback to linear
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(w))):
        return linear_interp_factory(x, w)

    # Fritsch-Carlson needs at least 3 points (finite-diff stencil). For
    # 2-point smiles we still want a sensible interpolator.
    if x.size < 3:
        return linear_interp_factory(x, w)

    # Case 1: monotone overall -> single FC
    if _is_monotone(w):
        w_interp, _ = FritschCarlson(x, w)
        return w_interp

    # Case 2a: robust U-shape split
    k = _u_split_index(w)
    if k is not None:
        xL, wL = x[: k + 1], w[: k + 1]
        xR, wR = x[k:], w[k:]
        if _is_monotone(wL) and _is_monotone(wR):
            if xL.size >= 3:
                pL, _ = FritschCarlson(xL, wL)
            else:
                pL = linear_interp_factory(xL, wL)

            if xR.size >= 3:
                pR, _ = FritschCarlson(xR, wR)
            else:
                pR = linear_interp_factory(xR, wR)
            return _stitch_two(float(x[k]), pL, pR)

    # Case 2b: fallback split at global minimum
    k2 = int(np.argmin(w))
    if 0 < k2 < len(w) - 1:
        xL, wL = x[: k2 + 1], w[: k2 + 1]
        xR, wR = x[k2:], w[k2:]
        if _is_monotone(wL) and _is_monotone(wR):
            if xL.size >= 3:
                pL, _ = FritschCarlson(xL, wL)
            else:
                pL = linear_interp_factory(xL, wL)

            if xR.size >= 3:
                pR, _ = FritschCarlson(xR, wR)
            else:
                pR = linear_interp_factory(xR, wR)
            return _stitch_two(float(x[k2]), pL, pR)

    # Fallback: linear interp with flat extrapolation
    return linear_interp_factory(x, w)


@dataclass(frozen=True, slots=True)
class Smile:
    """Total-variance smile at a single expiry on a log-moneyness grid.

    The smile is represented in terms of total variance:

        w(y) = T * iv(y)^2

    on a strictly increasing log-moneyness grid:

        y = ln(K / F(T)).

    Interpolation:
      - Uses Fritsch-Carlson monotone cubic interpolation when `w(x)` is monotone
        or can be split into two monotone pieces (common U-shape).
      - Falls back to linear interpolation with flat extrapolation otherwise.
    """

    T: float
    y: FloatArray
    w: FloatArray
    _w_interp: Callable[[np.ndarray], np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        y = np.asarray(self.y, dtype=np.float64)
        w = np.asarray(self.w, dtype=np.float64)

        if y.shape != w.shape:
            raise ValueError("Smile.y and Smile.w must have the same shape")
        if y.size < 2:
            raise ValueError("Smile requires at least 2 points")
        if np.any(np.diff(y) <= 0.0):
            raise ValueError("Smile.y must be strictly increasing")

        object.__setattr__(self, "_w_interp", _make_w_interpolator(y, w))

    def w_at(self, xq: ArrayLike) -> FloatArray:
        """Interpolate total variance at given log-moneyness points.

        Parameters
        ----------
        xq
            Log-moneyness point(s) at which to evaluate total variance.

        Returns
        -------
        FloatArray
            Total variance w(x) = T * iv(x)^2 at xq.
        """
        xq_arr = np.asarray(xq, dtype=np.float64)
        out = self._w_interp(xq_arr)
        return np.asarray(out, dtype=np.float64)

    def iv_at(self, xq: ArrayLike) -> FloatArray:
        """Interpolate implied volatility at given log-moneyness points.

        Parameters
        ----------
        xq
            Log-moneyness point(s) at which to evaluate implied volatility.

        Returns
        -------
        FloatArray
            Implied volatility iv(x) = sqrt(w(x) / T) at xq.
        """
        wq = self.w_at(xq)
        out = np.sqrt(np.maximum(wq / np.float64(self.T), np.float64(0.0)))
        return np.asarray(out, dtype=np.float64)

    @property
    def y_min(self) -> float:
        return float(self.y[0])

    @property
    def y_max(self) -> float:
        return float(self.y[-1])
