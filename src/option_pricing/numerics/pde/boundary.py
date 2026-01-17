from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

BoundaryFn = Callable[[float], float]  # t -> scalar


# --- General linear Robin BC: alpha(t) u + beta(t) u_x = gamma(t) ---


@dataclass(frozen=True, slots=True)
class RobinBCSide:
    alpha: BoundaryFn
    beta: BoundaryFn
    gamma: BoundaryFn


@dataclass(frozen=True, slots=True)
class RobinBC:
    left: RobinBCSide
    right: RobinBCSide


# Convenience constructors (still valid in a Robin-only API)


def dirichlet_side(g: BoundaryFn) -> RobinBCSide:
    """Dirichlet u = g(t) as Robin with beta=0."""
    return RobinBCSide(alpha=lambda t: 1.0, beta=lambda t: 0.0, gamma=g)


def neumann_side(q: BoundaryFn) -> RobinBCSide:
    """Neumann u_x = q(t) as Robin with alpha=0, beta=1."""
    return RobinBCSide(alpha=lambda t: 0.0, beta=lambda t: 1.0, gamma=q)


# --- 2nd-order one-sided derivative weights (nonuniform) ---


def _left_dx_weights(h0: float, h1: float) -> tuple[float, float, float]:
    # derivative at x0 using nodes (x0,x1,x2)
    w0 = -(2.0 * h0 + h1) / (h0 * (h0 + h1))
    w1 = (h0 + h1) / (h0 * h1)
    w2 = -h0 / (h1 * (h0 + h1))
    return w0, w1, w2


def _right_dx_weights(h0: float, h1: float) -> tuple[float, float, float]:
    # derivative at xN using nodes (x_{N-3},x_{N-2},x_{N-1})
    w_m3 = h0 / (h1 * (h0 + h1))
    w_m2 = -(h0 + h1) / (h0 * h1)
    w_m1 = (2.0 * h0 + h1) / (h0 * (h0 + h1))
    return w_m3, w_m2, w_m1


# --- Elimination coefficients: express boundary u in terms of interior nodes ---


def elim_left_second_order(
    side: RobinBCSide, *, h0: float, h1: float, t: float
) -> tuple[float, float, float]:
    """
    Return (p1,p2,q) such that u0 = p1*u1 + p2*u2 + q
    for alpha*u0 + beta*u_x(x0) = gamma using a 2nd-order one-sided derivative.
    """
    alpha = float(side.alpha(t))
    beta = float(side.beta(t))
    gamma = float(side.gamma(t))
    w0, w1, w2 = _left_dx_weights(h0, h1)
    denom = alpha + beta * w0
    if abs(denom) < 1e-14:
        raise ValueError("Left Robin BC is singular: alpha + beta*w0 ~ 0.")
    p1 = -(beta * w1) / denom
    p2 = -(beta * w2) / denom
    q = gamma / denom
    return p1, p2, q


def elim_right_second_order(
    side: RobinBCSide, *, h0: float, h1: float, t: float
) -> tuple[float, float, float]:
    """
    Return (p1,p2,q) such that uN = p1*u_{N-2} + p2*u_{N-3} + q
    for alpha*uN + beta*u_x(xN) = gamma using a 2nd-order one-sided derivative.
    """
    alpha = float(side.alpha(t))
    beta = float(side.beta(t))
    gamma = float(side.gamma(t))
    w_m3, w_m2, w_m1 = _right_dx_weights(h0, h1)
    denom = alpha + beta * w_m1
    if abs(denom) < 1e-14:
        raise ValueError("Right Robin BC is singular: alpha + beta*w_m1 ~ 0.")
    p1 = -(beta * w_m2) / denom
    p2 = -(beta * w_m3) / denom
    q = gamma / denom
    return p1, p2, q


def recover_boundaries_second_order(
    *,
    x: NDArray[np.floating],
    u_int: NDArray[np.floating],  # u1..u_{N-2}, length N-2
    bc: RobinBC,
    t: float,
) -> tuple[float, float]:
    """
    Recover (u0,uN) from interior values using the same 2nd-order rule
    used for elimination.
    """
    x = np.asarray(x, dtype=float)
    N = int(x.shape[0])
    if N < 4:
        raise ValueError("Need at least 4 grid points for 2nd-order boundary recovery.")
    if u_int.shape != (N - 2,):
        raise ValueError(f"u_int must have shape {(N-2,)} got {u_int.shape}")

    h0L = float(x[1] - x[0])
    h1L = float(x[2] - x[1])
    p1L, p2L, qL = elim_left_second_order(bc.left, h0=h0L, h1=h1L, t=t)
    u0 = float(p1L * u_int[0] + p2L * u_int[1] + qL)

    h0R = float(x[-1] - x[-2])
    h1R = float(x[-2] - x[-3])
    p1R, p2R, qR = elim_right_second_order(bc.right, h0=h0R, h1=h1R, t=t)
    uN = float(p1R * u_int[-1] + p2R * u_int[-2] + qR)

    return u0, uN


def assemble_full_solution(
    *,
    x: NDArray[np.floating],
    u_int: NDArray[np.floating],
    bc: RobinBC,
    t: float,
) -> NDArray[np.floating]:
    u0, uN = recover_boundaries_second_order(x=x, u_int=u_int, bc=bc, t=t)
    u_full = np.empty_like(x, dtype=float)
    u_full[0] = u0
    u_full[1:-1] = np.asarray(u_int, dtype=float)
    u_full[-1] = uN
    return cast(NDArray[np.floating], u_full)
