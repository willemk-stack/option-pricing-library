from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .fd.diff import diff1_nonuniform


def FritschCarlson(pi: NDArray, Fn: NDArray) -> Callable[[np.ndarray], np.ndarray]:
    pi = np.asarray(pi, dtype=np.float64)
    Fn = np.asarray(Fn, dtype=np.float64)

    if pi.shape != Fn.shape:
        raise ValueError("pi and Fn must have same shape")
    if pi.size < 2:
        raise ValueError("Need at least 2 points")
    if np.any(np.diff(pi) <= 0.0):
        raise ValueError("pi must be strictly increasing")

    dF = np.diff(Fn)
    if not (np.all(dF >= 0.0) or np.all(dF <= 0.0)):
        raise ValueError("Fn must be monotone (nondecreasing or nonincreasing)")

    h = np.diff(pi)
    delta = dF / h  # shape (n-1,)

    d = diff1_nonuniform(Fn, pi).astype(np.float64, copy=True)
    if d.shape != Fn.shape:
        raise ValueError("diff1_nonuniform must return an array with same shape as Fn")

    # Step 1: delta==0 => set both adjacent derivatives to 0
    mask_zero = delta == 0.0
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

    return p
