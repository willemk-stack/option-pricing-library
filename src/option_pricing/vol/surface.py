from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from option_pricing.typing import ArrayLike, FloatArray, ScalarFn

from ..numerics.interpolation import FritschCarlson


def _is_monotone(y: np.ndarray) -> bool:
    dy = np.diff(y)
    return bool(np.all(dy >= 0.0) or np.all(dy <= 0.0))


def _u_split_index(w: np.ndarray, *, eps: float | None = None) -> int | None:
    """Heuristic U-shape split index.

    Returns k such that left piece is w[:k+1], right piece is w[k:].
    Uses a tolerance to reduce sensitivity to small numerical noise / plateaus.
    """
    if w.size < 3:
        return None

    dw = np.diff(w)

    if eps is None:
        scale = float(max(1.0, np.max(np.abs(w))))
        eps = 1e-12 * scale

    neg = dw < -eps
    pos = dw > eps
    if not np.any(neg) or not np.any(pos):
        return None

    # first index where slope becomes meaningfully positive
    k = int(np.argmax(pos)) + 1
    if 0 < k < w.size - 1:
        return k
    return None


def _linear_interp_factory(
    x: np.ndarray, y: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """1D linear interpolation with flat extrapolation."""
    y0 = float(y[0])
    y1 = float(y[-1])

    def plin(xq: np.ndarray) -> np.ndarray:
        xq_in = np.asarray(xq, dtype=np.float64)
        out = np.interp(xq_in, x, y, left=y0, right=y1)
        return np.asarray(out, dtype=np.float64)

    return plin


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
        return _linear_interp_factory(x, w)

    # Case 1: monotone overall -> single FC
    if _is_monotone(w):
        return FritschCarlson(x, w)

    # Case 2a: robust U-shape split
    k = _u_split_index(w)
    if k is not None:
        xL, wL = x[: k + 1], w[: k + 1]
        xR, wR = x[k:], w[k:]
        if _is_monotone(wL) and _is_monotone(wR):
            pL = FritschCarlson(xL, wL)
            pR = FritschCarlson(xR, wR)
            return _stitch_two(float(x[k]), pL, pR)

    # Case 2b: fallback split at global minimum
    k2 = int(np.argmin(w))
    if 0 < k2 < len(w) - 1:
        xL, wL = x[: k2 + 1], w[: k2 + 1]
        xR, wR = x[k2:], w[k2:]
        if _is_monotone(wL) and _is_monotone(wR):
            pL = FritschCarlson(xL, wL)
            pR = FritschCarlson(xR, wR)
            return _stitch_two(float(x[k2]), pL, pR)

    # Fallback: linear interp with flat extrapolation
    return _linear_interp_factory(x, w)


@dataclass(frozen=True, slots=True)
class Smile:
    """Total-variance smile at a single expiry on a log-moneyness grid.

    The smile is represented in terms of total variance:

        w(x) = T * iv(x)^2

    on a strictly increasing log-moneyness grid:

        x = ln(K / F(T)).

    Interpolation:
      - Uses Fritsch–Carlson monotone cubic interpolation when `w(x)` is monotone
        or can be split into two monotone pieces (common U-shape).
      - Falls back to linear interpolation with flat extrapolation otherwise.
    """

    T: float
    x: FloatArray
    w: FloatArray
    _w_interp: Callable[[np.ndarray], np.ndarray] = field(init=False, repr=False)

    def __post_init__(self):
        x = np.asarray(self.x, dtype=np.float64)
        w = np.asarray(self.w, dtype=np.float64)

        if x.shape != w.shape:
            raise ValueError("Smile.x and Smile.w must have the same shape")
        if x.size < 2:
            raise ValueError("Smile requires at least 2 points")
        if np.any(np.diff(x) <= 0.0):
            raise ValueError("Smile.x must be strictly increasing")

        object.__setattr__(self, "_w_interp", _make_w_interpolator(x, w))

    def w_at(self, xq: ArrayLike) -> FloatArray:
        xq_arr = np.asarray(xq, dtype=np.float64)
        out = self._w_interp(xq_arr)
        return np.asarray(out, dtype=np.float64)

    def iv_at(self, xq: ArrayLike) -> FloatArray:
        wq = self.w_at(xq)
        out = np.sqrt(np.maximum(wq / np.float64(self.T), np.float64(0.0)))
        return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class VolSurface:
    """Total-variance volatility surface over expiry.

    Within each expiry slice (Smile), total variance is interpolated using:
      - Fritsch–Carlson on monotone (or split-monotone) data, else linear fallback.

    Across expiry, total variance is linearly interpolated in time between smiles.
    """

    expiries: FloatArray
    smiles: tuple[Smile, ...]
    forward: ScalarFn  # forward(T) -> float

    @classmethod
    def from_grid(
        cls,
        rows: Iterable[tuple[float, float, float]],  # (T, K, iv)
        forward: ScalarFn,
        *,
        expiry_round_decimals: int = 10,
    ) -> VolSurface:
        buckets: dict[float, list[tuple[float, float]]] = {}

        for T_raw, K_raw, iv_raw in rows:
            T_key = round(float(T_raw), expiry_round_decimals)
            buckets.setdefault(T_key, []).append((float(K_raw), float(iv_raw)))

        expiries = np.asarray(sorted(buckets.keys()), dtype=np.float64)
        smiles: list[Smile] = []

        for T_np in expiries:
            pts = buckets[float(T_np)]
            pts.sort(key=lambda p: p[0])  # sort by strike

            K = np.asarray([p[0] for p in pts], dtype=np.float64)
            iv = np.asarray([p[1] for p in pts], dtype=np.float64)

            F = float(forward(float(T_np)))
            if F <= 0.0:
                raise ValueError(f"forward(T) must be > 0, got {F} at T={float(T_np)}")
            if np.any(K <= 0.0):
                raise ValueError(f"All strikes must be > 0 at T={float(T_np)}")

            x = np.log(K / np.float64(F)).astype(np.float64, copy=False)
            w = (np.float64(T_np) * (iv**2)).astype(np.float64, copy=False)

            if np.any(np.diff(x) <= 0.0):
                raise ValueError(f"x grid not strictly increasing at T={float(T_np)}")

            smiles.append(Smile(T=float(T_np), x=x, w=w))

        return cls(expiries=expiries, smiles=tuple(smiles), forward=forward)

    def iv(self, K: ArrayLike, T: float) -> FloatArray:
        T = float(T)
        if T <= 0.0:
            raise ValueError("T must be > 0")

        K_arr = np.asarray(K, dtype=np.float64)
        if np.any(K_arr <= 0.0):
            raise ValueError("Strikes must be > 0 for log-moneyness.")

        xq = np.log(K_arr / np.float64(self.forward(T))).astype(np.float64, copy=False)

        # Clamp outside known range
        if T <= float(self.expiries[0]):
            return self.smiles[0].iv_at(xq)
        if T >= float(self.expiries[-1]):
            return self.smiles[-1].iv_at(xq)

        # Find bracketing expiries
        j = int(np.searchsorted(self.expiries, np.float64(T)))
        i = j - 1

        T0 = float(self.expiries[i])
        T1 = float(self.expiries[j])

        s0 = self.smiles[i]
        s1 = self.smiles[j]

        w0 = s0.w_at(xq)
        w1 = s1.w_at(xq)

        a = (T - T0) / (T1 - T0)
        w = np.float64(1.0 - a) * w0 + np.float64(a) * w1

        out = np.sqrt(np.maximum(w / np.float64(T), np.float64(0.0)))
        return np.asarray(out, dtype=np.float64)
