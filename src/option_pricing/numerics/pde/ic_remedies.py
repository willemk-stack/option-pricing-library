# src/option_pricing/numerics/pde/ic_remedies.py
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from enum import Enum
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ..grids import Grid
from ..tridiag import Tridiag, solve_tridiag_thomas

FloatArray = NDArray[np.float64]
FloatFn = Callable[[float], float]


class ICRemedy(str, Enum):
    NONE = "none"
    CELL_AVG = "cell_avg"
    L2_PROJ = "l2_proj"


def _midpoints(x: FloatArray) -> FloatArray:
    return cast(FloatArray, 0.5 * (x[:-1] + x[1:]))


def _split_interval(
    a: float, b: float, breaks: Sequence[float]
) -> list[tuple[float, float]]:
    pts = [float(p) for p in breaks if a < float(p) < b]
    pts = [a] + sorted(pts) + [b]
    return [(pts[j], pts[j + 1]) for j in range(len(pts) - 1)]


def _gauss3(f: FloatFn, a: float, b: float) -> float:
    xs = np.array([-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0)], dtype=np.float64)
    ws = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=np.float64)
    m = 0.5 * (a + b)
    h = 0.5 * (b - a)

    vals = np.array([f(m + h * float(xi)) for xi in xs], dtype=np.float64)
    return float(h * float(np.sum(ws * vals)))


def ic_cell_average(
    grid: Grid,
    ic: FloatFn,
    *,
    breakpoints: Sequence[float] = (),
) -> FloatArray:
    x = cast(FloatArray, np.asarray(grid.x, dtype=np.float64))
    Nx = int(x.size)
    if Nx == 0:
        return cast(FloatArray, np.empty((0,), dtype=np.float64))
    if Nx == 1:
        return cast(FloatArray, np.array([ic(float(x[0]))], dtype=np.float64))

    xm = _midpoints(x)

    u = np.empty(Nx, dtype=np.float64)
    for i in range(Nx):
        if i == 0:
            a, b = float(x[0]), float(xm[0])
        elif i == Nx - 1:
            a, b = float(xm[-1]), float(x[-1])
        else:
            a, b = float(xm[i - 1]), float(xm[i])

        pieces = _split_interval(a, b, breakpoints)
        integ = sum(_gauss3(ic, aa, bb) for aa, bb in pieces)
        u[i] = float(integ / (b - a))

    return cast(FloatArray, u)


def ic_l2_projection(
    grid: Grid,
    ic: FloatFn,
    *,
    breakpoints: Sequence[float] = (),
) -> FloatArray:
    x = cast(FloatArray, np.asarray(grid.x, dtype=np.float64))
    Nx = int(x.size)
    if Nx == 0:
        return cast(FloatArray, np.empty((0,), dtype=np.float64))
    if Nx == 1:
        return cast(FloatArray, np.array([ic(float(x[0]))], dtype=np.float64))

    h = cast(FloatArray, np.diff(x))  # length Nx-1

    lower = np.zeros(Nx - 1, dtype=np.float64)
    diag = np.zeros(Nx, dtype=np.float64)
    upper = np.zeros(Nx - 1, dtype=np.float64)

    # boundaries
    diag[0] = float(h[0] / 3.0)
    upper[0] = float(h[0] / 6.0)
    diag[-1] = float(h[-1] / 3.0)
    lower[-1] = float(h[-1] / 6.0)

    # interior
    for i in range(1, Nx - 1):
        diag[i] = float((h[i - 1] + h[i]) / 3.0)
        lower[i - 1] = float(h[i - 1] / 6.0)
        upper[i] = float(h[i] / 6.0)

    F = np.zeros(Nx, dtype=np.float64)

    # i = 0 support: [x0, x1] with hat decreasing
    x0 = float(x[0])
    x1 = float(x[1])
    den01 = x1 - x0
    for a, b in _split_interval(x0, x1, breakpoints):

        def f0(
            xx: float, x1: float = x1, den01: float = den01, ic: FloatFn = ic
        ) -> float:
            return ((x1 - xx) / den01) * ic(xx)

        F[0] += _gauss3(f0, a, b)

    # interior hats
    for i in range(1, Nx - 1):
        xL = float(x[i - 1])
        xM = float(x[i])
        xR = float(x[i + 1])
        denL = xM - xL
        denR = xR - xM

        for a, b in _split_interval(xL, xM, breakpoints):

            def fL(
                xx: float, xL: float = xL, denL: float = denL, ic: FloatFn = ic
            ) -> float:
                return ((xx - xL) / denL) * ic(xx)

            F[i] += _gauss3(fL, a, b)

        for a, b in _split_interval(xM, xR, breakpoints):

            def fR(
                xx: float, xR: float = xR, denR: float = denR, ic: FloatFn = ic
            ) -> float:
                return ((xR - xx) / denR) * ic(xx)

            F[i] += _gauss3(fR, a, b)

    # i = Nx-1 support: [x_{N-2}, x_{N-1}] with hat increasing
    xNm2 = float(x[-2])
    xNm1 = float(x[-1])
    denN = xNm1 - xNm2
    for a, b in _split_interval(xNm2, xNm1, breakpoints):

        def fN(
            xx: float, xNm2: float = xNm2, denN: float = denN, ic: FloatFn = ic
        ) -> float:
            return ((xx - xNm2) / denN) * ic(xx)

        F[-1] += _gauss3(fN, a, b)

    c = solve_tridiag_thomas(Tridiag(lower=lower, diag=diag, upper=upper), F)
    return cast(FloatArray, np.asarray(c, dtype=np.float64))
