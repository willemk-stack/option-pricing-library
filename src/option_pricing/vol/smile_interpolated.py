"""Interpolated smile slices for time interpolation between expiries.

This module provides piecewise smile interpolators that blend two endpoint smiles
in time using either no-arbitrage (Gatheral/Jacquier) or linear-in-variance blending.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import erfc
from typing import cast

import numpy as np

from ..typing import ArrayLike, FloatArray
from .vol_types import DifferentiableSmileSlice, SmileSlice


def _norm_cdf(x: ArrayLike) -> np.ndarray:
    """Cumulative standard normal distribution."""
    x = np.asarray(x, dtype=np.float64)
    erfc_vec = np.vectorize(erfc, otypes=[np.float64])
    out = 0.5 * erfc_vec(-x / np.sqrt(2.0))
    return cast(np.ndarray, np.asarray(out, dtype=np.float64))


def _bs_call_fwd_norm(
    y: ArrayLike, w: ArrayLike, *, eps_w: float = 1e-16
) -> np.ndarray:
    """
    Black(-76) forward-normalized call price c = C/F for log-moneyness y=ln(K/F)
    and total variance w = sigma^2 * T.

    Returns c(y,w) = N(d+) - exp(y) N(d-), with:
        d+ = -y/sqrt(w) + 0.5*sqrt(w)
        d- = d+ - sqrt(w)

    For w -> 0: returns intrinsic max(1 - exp(y), 0).
    """
    y_arr = np.asarray(y, dtype=np.float64)
    w_arr = np.asarray(w, dtype=np.float64)

    # Broadcast to common shape
    yb, wb = np.broadcast_arrays(y_arr, w_arr)

    # Intrinsic (F=1)
    ey = np.exp(yb)
    intrinsic = np.maximum(1.0 - ey, 0.0)

    # Clamp negative w to 0 for robustness
    wb_pos = np.maximum(wb, 0.0)
    sqrtw = np.sqrt(wb_pos)

    out = np.empty_like(yb, dtype=np.float64)

    small = sqrtw <= eps_w
    if np.any(small):
        out[small] = intrinsic[small]

    big = ~small
    if np.any(big):
        sw = sqrtw[big]
        yy = yb[big]
        d_plus = (-yy / sw) + 0.5 * sw
        d_minus = d_plus - sw

        Nd1 = _norm_cdf(d_plus)
        Nd2 = _norm_cdf(d_minus)
        out_big = Nd1 - np.exp(yy) * Nd2

        # Numerical safety: enforce no-arb bounds for normalized call
        out[big] = np.clip(out_big, intrinsic[big], 1.0)

    return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class NoArbInterpolatedSmileSlice:
    """
    Gatheral/Jacquier Lemma 5.1 interpolation:
      blend normalized call prices at constant log-moneyness,
      with alpha_t from a monotone theta(t)=w(0,t) interpolation.
    """

    T: float
    s0: SmileSlice
    s1: SmileSlice
    a: float
    theta_interp: Callable[[np.ndarray], np.ndarray]

    _alpha: float = field(init=False, repr=False)
    _inv_sqrtT: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        th0 = float(np.asarray(self.s0.w_at(0.0)))
        th1 = float(np.asarray(self.s1.w_at(0.0)))
        t = np.asarray(self.T)
        thT = float(np.asarray(self.theta_interp(t)))

        s0 = np.sqrt(max(th0, 0.0))
        s1 = np.sqrt(max(th1, 0.0))
        sT = np.sqrt(max(thT, 0.0))

        denom = s1 - s0
        if abs(denom) < 1e-16:
            alpha = 1.0 - float(self.a)  # degenerate theta; any alpha is ok
        else:
            alpha = (s1 - sT) / denom  # eq (5.2) rearranged

        object.__setattr__(self, "_alpha", float(np.clip(alpha, 0.0, 1.0)))
        object.__setattr__(self, "_inv_sqrtT", 1.0 / np.sqrt(float(self.T)))

    def w_at(self, yq: ArrayLike) -> FloatArray:
        from .implied_vol_slice import implied_vol_black76_slice

        y = np.asarray(yq, dtype=np.float64)
        y1d = np.atleast_1d(y)

        # 1) endpoint total variances
        w0 = np.asarray(self.s0.w_at(y1d), dtype=np.float64)
        w1 = np.asarray(self.s1.w_at(y1d), dtype=np.float64)

        # 2) normalized forward call prices (C/F) at fixed log-moneyness y
        c0 = _bs_call_fwd_norm(y1d, w0)
        c1 = _bs_call_fwd_norm(y1d, w1)

        # 3) price blend (eq 5.3)
        alpha = self._alpha
        cT = alpha * c0 + (1.0 - alpha) * c1

        # 4) map to Black76 inputs: choose F=1, df=1, K = exp(y)
        K = np.exp(y1d)

        # guard tiny numerical violations of no-arb bounds:
        intrinsic = np.maximum(1.0 - K, 0.0)  # since F=1, df=1
        cT = np.clip(cT, intrinsic, 1.0)

        # 5) initial guess to reduce iterations (cheap and vectorized)
        # linear-in-w is a *guess only* (still invert using no-arb call blend)
        w_guess = (1.0 - self.a) * w0 + self.a * w1
        sigma0 = np.sqrt(np.maximum(w_guess, 0.0)) * self._inv_sqrtT

        sigma = implied_vol_black76_slice(
            forward=1.0,
            strikes=K,
            tau=float(self.T),
            df=1.0,
            prices=cT,
            is_call=True,
            initial_sigma=sigma0,
            sigma_lo=1e-12,
            sigma_hi=5.0,
            max_iter=50,
            tol=1e-12,
            return_result=False,
        )

        wT = (np.asarray(sigma, dtype=np.float64) ** 2) * np.float64(self.T)

        if np.ndim(yq) == 0:
            return np.asarray(wT[0], dtype=np.float64)
        return wT.reshape(y.shape)

    def iv_at(self, yq: ArrayLike) -> FloatArray:
        wq = self.w_at(yq)
        return np.asarray(
            np.sqrt(np.maximum(wq / np.float64(self.T), 0.0)), dtype=np.float64
        )

    @property
    def y_min(self) -> float:
        return max(float(self.s0.y_min), float(self.s1.y_min))

    @property
    def y_max(self) -> float:
        return min(float(self.s0.y_max), float(self.s1.y_max))


@dataclass(frozen=True, slots=True)
class LinearWInterpolatedSmileSlice:
    """Time interpolation by linear blending in total variance w.

    Derivatives in y are consistent if endpoints provide them.
    Good for LocalVolSurface.
    """

    T: float
    s0: DifferentiableSmileSlice
    s1: DifferentiableSmileSlice
    a: float  # in [0,1]

    def w_at(self, yq: ArrayLike) -> FloatArray:
        y = np.asarray(yq, dtype=np.float64)
        w0 = np.asarray(self.s0.w_at(y), dtype=np.float64)
        w1 = np.asarray(self.s1.w_at(y), dtype=np.float64)
        return np.asarray((1.0 - self.a) * w0 + self.a * w1, dtype=np.float64)

    def iv_at(self, yq: ArrayLike) -> FloatArray:
        wq = self.w_at(yq)
        return np.asarray(np.sqrt(np.maximum(wq / self.T, 0.0)), dtype=np.float64)

    def dw_dy(self, yq: ArrayLike) -> FloatArray:
        y = np.asarray(yq, dtype=np.float64)
        wy0 = np.asarray(self.s0.dw_dy(y), dtype=np.float64)
        wy1 = np.asarray(self.s1.dw_dy(y), dtype=np.float64)
        return np.asarray((1.0 - self.a) * wy0 + self.a * wy1, dtype=np.float64)

    def d2w_dy2(self, yq: ArrayLike) -> FloatArray:
        y = np.asarray(yq, dtype=np.float64)
        wyy0 = np.asarray(self.s0.d2w_dy2(y), dtype=np.float64)
        wyy1 = np.asarray(self.s1.d2w_dy2(y), dtype=np.float64)
        return np.asarray((1.0 - self.a) * wyy0 + self.a * wyy1, dtype=np.float64)

    @property
    def y_min(self) -> float:
        return max(float(self.s0.y_min), float(self.s1.y_min))

    @property
    def y_max(self) -> float:
        return min(float(self.s0.y_max), float(self.s1.y_max))
