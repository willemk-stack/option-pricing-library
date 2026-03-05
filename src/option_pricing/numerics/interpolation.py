from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .fd.diff import diff1_nonuniform

InterpFn = Callable[[np.ndarray], np.ndarray]


def linear_interp_factory(x: np.ndarray, y: np.ndarray) -> InterpFn:
    """1D linear interpolation with flat extrapolation."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if x.size != y.size:
        raise ValueError("x and y must have same length")
    if x.size < 2:
        raise ValueError("Need at least 2 points")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("x must be strictly increasing")

    y0 = float(y[0])
    y1 = float(y[-1])

    def plin(xq: np.ndarray) -> np.ndarray:
        xq_in = np.asarray(xq, dtype=np.float64)
        out = np.interp(xq_in, x, y, left=y0, right=y1)
        return np.asarray(out, dtype=np.float64)

    return plin


def linear_interp_derivative_factory(x: np.ndarray, y: np.ndarray) -> InterpFn:
    """Derivative of 1D linear interpolation with flat extrapolation.

    Inside (x[0], x[-1]) it's piecewise-constant: slope_i on [x[i], x[i+1]).
    Outside it's 0 because the interpolation is flat-extrapolated.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if x.size != y.size:
        raise ValueError("x and y must have same length")
    if x.size < 2:
        raise ValueError("Need at least 2 points")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("x must be strictly increasing")

    slopes = np.diff(y) / np.diff(x)  # shape (n-1,)

    def dplin(xq: np.ndarray) -> np.ndarray:
        xq_in = np.asarray(xq, dtype=np.float64)

        # default: flat extrapolation => derivative 0
        out = np.zeros_like(xq_in, dtype=np.float64)

        # strictly inside the domain (endpoints get 0 to match flat extension convention)
        mid = (xq_in > x[0]) & (xq_in < x[-1])
        if np.any(mid):
            # choose the interval index (right-derivative at knots)
            idx = np.searchsorted(x, xq_in[mid], side="right") - 1
            idx = np.clip(idx, 0, x.size - 2)
            out[mid] = slopes[idx]

        return np.asarray(out, dtype=np.float64)

    return dplin


def FritschCarlson(pi: NDArray, Fn: NDArray) -> tuple[InterpFn, InterpFn]:
    pi = np.asarray(pi, dtype=np.float64)
    Fn = np.asarray(Fn, dtype=np.float64)

    if pi.shape != Fn.shape:
        raise ValueError("pi and Fn must have same shape")
    if pi.size < 2:
        raise ValueError("Need at least 2 points")
    if np.any(np.diff(pi) <= 0.0):
        raise ValueError("pi must be strictly increasing")

    dF = np.diff(Fn)
    eps = 1e-14 * max(1.0, float(np.max(np.abs(Fn))))
    mono_up = np.all(dF >= -eps)
    mono_dn = np.all(dF <= eps)
    if not (mono_up or mono_dn):
        raise ValueError("Fn must be monotone (nondecreasing or nonincreasing)")

    h = np.diff(pi)
    delta = dF / h  # shape (n-1,)

    d = diff1_nonuniform(Fn, pi).astype(np.float64, copy=True)
    if d.shape != Fn.shape:
        raise ValueError("diff1_nonuniform must return an array with same shape as Fn")

    # Step 1: delta==0 => set both adjacent derivatives to 0

    mask_zero = np.abs(delta) <= eps

    d[:-1][mask_zero] = 0.0
    d[1:][mask_zero] = 0.0

    # Sign consistency: sgn(d_i) = sgn(d_{i+1}) = sgn(delta_i)
    d[:-1] = np.where(d[:-1] * delta > 0.0, d[:-1], 0.0)
    d[1:] = np.where(d[1:] * delta > 0.0, d[1:], 0.0)

    # Step 2A: S1 square via ray scaling (tau = 3/max(alpha,beta))
    for i in range(delta.size):
        if delta[i] == 0.0:
            continue
        alpha = d[i] / delta[i]
        beta = d[i + 1] / delta[i]
        m = max(abs(alpha), abs(beta))  # abs() is defensive; optional
        if m > 3.0:
            tau = 3.0 / m
            d[i] *= tau
            d[i + 1] *= tau

    def p(xq: np.ndarray) -> np.ndarray:
        xq_in = np.asarray(xq, dtype=np.float64)
        xq_1d = np.atleast_1d(xq_in)

        out = np.empty_like(xq_1d, dtype=np.float64)

        left = xq_1d <= pi[0]
        right = xq_1d >= pi[-1]
        mid = ~(left | right)

        out[left] = Fn[0]
        out[right] = Fn[-1]

        if np.any(mid):
            xm = xq_1d[mid]
            idx = np.searchsorted(pi, xm) - 1
            idx = np.clip(idx, 0, pi.size - 2)

            x0 = pi[idx]
            x1 = pi[idx + 1]
            y0 = Fn[idx]
            y1 = Fn[idx + 1]
            d0 = d[idx]
            d1 = d[idx + 1]

            hloc = x1 - x0
            t = (xm - x0) / hloc

            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2

            out[mid] = h00 * y0 + h10 * hloc * d0 + h01 * y1 + h11 * hloc * d1

        # restore original shape / scalar-ness
        if xq_in.ndim == 0:
            return np.asarray(out[0], dtype=np.float64)
        return out.reshape(xq_in.shape)

    def dp_dxq(xq: np.ndarray) -> np.ndarray:
        xq_in = np.asarray(xq, dtype=np.float64)
        xq_1d = np.atleast_1d(xq_in)

        out = np.empty_like(xq_1d, dtype=np.float64)

        left = xq_1d <= pi[0]
        right = xq_1d >= pi[-1]
        mid = ~(left | right)

        # clamped constant extrapolation => derivative 0
        out[left] = 0.0
        out[right] = 0.0

        if np.any(mid):
            xm = xq_1d[mid]
            idx = np.searchsorted(pi, xm) - 1
            idx = np.clip(idx, 0, pi.size - 2)

            x0 = pi[idx]
            x1 = pi[idx + 1]
            y0 = Fn[idx]
            y1 = Fn[idx + 1]
            d0 = d[idx]
            d1 = d[idx + 1]

            hloc = x1 - x0
            t = (xm - x0) / hloc

            delta = (y1 - y0) / hloc

            out[mid] = (
                6.0 * t * (1.0 - t) * delta
                + (3.0 * t**2 - 4.0 * t + 1.0) * d0
                + (3.0 * t**2 - 2.0 * t) * d1
            )

        if xq_in.ndim == 0:
            return np.asarray(out[0], dtype=np.float64)
        return out.reshape(xq_in.shape)

    return p, dp_dxq
