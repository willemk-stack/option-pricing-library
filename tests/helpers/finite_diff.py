from __future__ import annotations

from collections.abc import Callable

import numpy as np


def central_diff_jac(
    fun: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray | list[float] | tuple[float, ...],
    eps: float = 1.0e-5,
) -> np.ndarray:
    x0 = np.asarray(x, dtype=np.float64).reshape(-1)
    step = float(eps)

    if x0.size == 0:
        raise ValueError("x must contain at least one entry")
    if not np.all(np.isfinite(x0)):
        raise ValueError("x must be finite")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("eps must be a positive finite float")

    base = np.atleast_1d(np.asarray(fun(x0.copy()), dtype=np.float64))
    jac = np.empty((base.size, x0.size), dtype=np.float64)

    for j in range(x0.size):
        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[j] += step
        x_minus[j] -= step

        f_plus = np.atleast_1d(np.asarray(fun(x_plus), dtype=np.float64))
        f_minus = np.atleast_1d(np.asarray(fun(x_minus), dtype=np.float64))

        if f_plus.shape != base.shape:
            raise ValueError(
                "fun output shape changed for positive bump in column "
                f"{j}: expected {base.shape}, got {f_plus.shape}"
            )
        if f_minus.shape != base.shape:
            raise ValueError(
                "fun output shape changed for negative bump in column "
                f"{j}: expected {base.shape}, got {f_minus.shape}"
            )

        jac[:, j] = ((f_plus - f_minus) / (2.0 * step)).reshape(-1)

    return np.asarray(jac, dtype=np.float64)
