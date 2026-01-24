# option_pricing/models/black_scholes/pde.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ...typing import ArrayLike

type Array = np.ndarray
type CoordLike = str | Any  # accept "S"/"LOG_S" or an Enum-like object


CoordKey = Literal["S", "LOG_S"]
CoeffFn = Callable[[ArrayLike, float], Array]
CoordFn = Callable[[ArrayLike], Array]


@dataclass(frozen=True, slots=True)
class BSPDECoeffs1D:
    """
    Coefficients for the forward-in-time (tau) linear parabolic PDE in solver coords x:

        u_tau = a(x,tau) * u_xx + b(x,tau) * u_x + c(x,tau) * u

    Note:
      - For LOG_S, coefficients are constant in x.
      - For S, a and b depend on S.
    """

    a: CoeffFn
    b: CoeffFn
    c: CoeffFn


def _coord_key(coord: CoordLike) -> CoordKey:
    """
    Normalize coordinate spec. Supports:
      - strings: "S", "LOG_S" (case-insensitive; also accepts "log", "log_s")
      - Enum-like objects having `.name` equal to "S" or "LOG_S"
      - objects convertible to str with those tokens
    """
    if hasattr(coord, "name"):
        key = str(coord.name)
    else:
        key = str(coord)

    k = key.strip().upper()
    if k in {"S"}:
        return "S"
    if k in {"LOG_S", "LOG", "LN_S", "LN"}:
        return "LOG_S"

    raise ValueError(f"Unsupported coord: {coord!r} (expected 'S' or 'LOG_S')")


def bs_coord_maps(coord: CoordLike) -> tuple[CoordFn, CoordFn]:
    """
    Return (to_x, to_S) coordinate transforms.

    - coord="S":     x = S,    S = x
    - coord="LOG_S": x = ln S, S = exp x
    """
    ck = _coord_key(coord)

    if ck == "LOG_S":

        def to_x(S: CoordFn):
            return np.log(np.asarray(S, dtype=float))

        def to_S(x: CoordFn):
            return np.exp(np.asarray(x, dtype=float))

        return to_x, to_S

    # ck == "S"
    def to_x(S: CoordFn):
        return np.asarray(S, dtype=float)

    def to_S(x: CoordFn):
        return np.asarray(x, dtype=float)

    return to_x, to_S


def bs_pde_coeffs(
    *,
    coord: CoordLike,
    sigma: float,
    r: float,
    q: float,
) -> BSPDECoeffs1D:
    """
    Black-Scholes PDE coefficients in the chosen solver coordinate.

    For LOG_S (x = ln S):
        a = 0.5 * sigma^2
        b = (r - q - 0.5 * sigma^2)
        c = -r

    For S (x = S):
        a = 0.5 * sigma^2 * S^2
        b = (r - q) * S
        c = -r

    Returned functions are vectorized and broadcast to the shape of the first argument.
    """
    ck = _coord_key(coord)

    sig = float(sigma)
    rr = float(r)
    qq = float(q)

    if ck == "LOG_S":
        a0 = 0.5 * sig * sig
        b0 = rr - qq - 0.5 * sig * sig
        c0 = -rr

        def a(x: ArrayLike, tau: float) -> Array:
            x = np.asarray(x, dtype=float)
            return a0 + 0.0 * x

        def b(x: ArrayLike, tau: float) -> Array:
            x = np.asarray(x, dtype=float)
            return b0 + 0.0 * x

        def c(x: ArrayLike, tau: float) -> Array:
            x = np.asarray(x, dtype=float)
            return c0 + 0.0 * x

        return BSPDECoeffs1D(a=a, b=b, c=c)

    # ck == "S"
    else:

        def a(x: ArrayLike, tau: float) -> Array:
            x = np.asarray(x, dtype=float)  # x is S here
            return 0.5 * sig * sig * (x * x)

        def b(x: ArrayLike, tau: float) -> Array:
            x = np.asarray(x, dtype=float)
            return (rr - qq) * x

        def c(x: ArrayLike, tau: float) -> Array:
            x = np.asarray(x, dtype=float)
            return (-rr) + 0.0 * x

    return BSPDECoeffs1D(a=a, b=b, c=c)


def bs_x0(*, coord: CoordLike, S0: float) -> float:
    """
    Convenience: compute x0 in solver coordinates from spot.
    """
    to_x, _ = bs_coord_maps(coord)
    x0 = to_x(float(S0))
    # ensure scalar float
    return float(np.asarray(x0).reshape(()))
