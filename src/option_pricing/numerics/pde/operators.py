from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import cast

import numpy as np
from numpy.typing import NDArray

from option_pricing.typing import ScalarFn

from ...numerics.fd.stencils import (
    d1_backward_coeffs,
    d1_central_nonuniform_coeffs,
    d1_forward_coeffs,
    d2_central_nonuniform_coeffs,
)
from ..grids import Grid
from ..tridiag import Tridiag
from .boundary import RobinBC, elim_left_second_order, elim_right_second_order
from .types import CNSystem

XTInput = float | NDArray[np.floating]
XTOutput = float | NDArray[np.floating]

# Coefficient/source functions a(x,t), b(x,t), c(x,t), d(x,t).
# Users may provide scalar-only functions or NumPy-vectorized ones.
ScalarXT = Callable[[XTInput, float], XTOutput]


class AdvectionScheme(str, Enum):
    """Spatial discretization choice for the first-derivative term b(x,t) u_x."""

    CENTRAL = "central"  # 2nd order (on nonuniform: quadratic interpolation)
    UPWIND = "upwind"  # 1st order, monotone-ish for advection-dominated problems


@dataclass(frozen=True, slots=True)
class LinearParabolicPDE1D:
    """1D linear parabolic PDE with Dirichlet boundary conditions.

    PDE form:
        u_t = a(x,t) u_xx + b(x,t) u_x + c(x,t) u + d(x,t)

    Notes
    -----
    - Time marches forward: t in [0, T].
    - Dirichlet boundaries are provided via :class:`~option_pricing.numerics.pde.DirichletBC`.
    - Coefficient functions may be scalar-only or NumPy-vectorized.
    """

    a: ScalarXT
    b: ScalarXT
    c: ScalarXT
    bc: RobinBC
    ic: ScalarFn
    d: ScalarXT | None = None


def _eval_xt(fn: ScalarXT, x: NDArray[np.floating], t: float) -> NDArray[np.floating]:
    """Evaluate fn(x,t) on an array x.

    We attempt a vectorized call first; if that fails or returns an unexpected
    shape, we fall back to scalar evaluation.
    """

    try:
        y = fn(x, t)
        arr = np.asarray(y, dtype=float)
        if arr.shape == x.shape:
            return cast(NDArray[np.floating], arr)
    except Exception:
        pass

    out = np.empty_like(x, dtype=float)
    for i, xi in enumerate(np.asarray(x, dtype=float)):
        out[i] = float(fn(float(xi), float(t)))
    return cast(NDArray[np.floating], out)


def _first_derivative_coeffs(hm, hp, b, scheme):
    """
    Computes first derivative coefficients using Upwind or Central schemes.
    """
    if scheme == AdvectionScheme.CENTRAL:
        return d1_central_nonuniform_coeffs(hm, hp)

    # Initialize tridiagonal components
    dl = np.zeros_like(hm, dtype=float)
    dd = np.zeros_like(hm, dtype=float)
    du = np.zeros_like(hm, dtype=float)

    # Masks for flow direction
    pos = b >= 0.0
    neg = ~pos

    # 1. Handle Backward coefficients (where b >= 0)
    # We only compute these for the indices where 'pos' is True
    bw_l, bw_d, bw_u = d1_backward_coeffs(hm[pos])
    dl[pos] = bw_l
    dd[pos] = bw_d
    du[pos] = bw_u

    # 2. Handle Forward coefficients (where b < 0)
    # We only compute these for the indices where 'neg' is True
    fw_l, fw_d, fw_u = d1_forward_coeffs(hp[neg])
    dl[neg] = fw_l
    dd[neg] = fw_d
    du[neg] = fw_u

    return dl, dd, du


def _second_derivative_coeffs(hm, hp):
    """
    Wrapper for _d2_central_nonuniform_coeffs
    """
    return d2_central_nonuniform_coeffs(hm, hp)


@dataclass(frozen=True, slots=True)
class _L1D:
    """Discrete L operator on interior nodes, plus boundary forcing in L."""

    L: Tridiag
    forcing: NDArray[np.floating]  # add to (L*u_int) as +forcing


def build_L_1d(
    *,
    grid: Grid,
    t: float,
    a_fn: ScalarXT,
    b_fn: ScalarXT,
    c_fn: ScalarXT,
    bc: RobinBC,  # RobinBC
    advection: AdvectionScheme = AdvectionScheme.CENTRAL,
) -> _L1D:
    x = np.asarray(grid.x, dtype=float)
    N = int(x.shape[0])
    if N < 4:
        raise ValueError("Need at least 4 spatial points (Nx>=4)")

    x_int = x[1:-1]
    M = int(x_int.shape[0])

    hm = x[1:-1] - x[0:-2]  # (M,)
    hp = x[2:] - x[1:-1]  # (M,)
    if not np.all(hm > 0.0) or not np.all(hp > 0.0):
        raise ValueError("x grid must be strictly increasing")

    a = _eval_xt(a_fn, x_int, t)
    b = _eval_xt(b_fn, x_int, t)
    c = _eval_xt(c_fn, x_int, t)

    d2l, d2d, d2u = _second_derivative_coeffs(hm, hp)
    d1l, d1d, d1u = _first_derivative_coeffs(hm, hp, b=b, scheme=advection)

    L_lower_full = a * d2l + b * d1l  # coeff for u_{i-1} (includes boundary at i=1)
    L_diag = a * d2d + b * d1d + c
    L_upper_full = a * d2u + b * d1u  # coeff for u_{i+1} (includes boundary at i=N-2)

    forcing = np.zeros(M, dtype=float)

    # --- Left boundary elimination using (u0,u1,u2) ---
    # spacings from boundary:
    h0L = float(x[1] - x[0])
    h1L = float(x[2] - x[1]) if N >= 3 else 0.0
    if N < 3:
        raise ValueError("Need at least 3 points for 2nd-order BC enforcement.")
    p1L, p2L, qL = elim_left_second_order(bc.left, h0L, h1L, t)

    # row for first interior node (global i=1): term L_lower_full[0]*u0
    # substitute u0 = p1L*u1 + p2L*u2 + qL
    L_diag[0] += L_lower_full[0] * p1L
    L_upper_full[0] += L_lower_full[0] * p2L
    forcing[0] += L_lower_full[0] * qL
    L_lower_full[0] = 0.0

    # --- Right boundary elimination using (u_{N-3},u_{N-2},u_{N-1}) ---
    h0R = float(x[-1] - x[-2])
    h1R = float(x[-2] - x[-3])
    p1R, p2R, qR = elim_right_second_order(bc.right, h0R, h1R, t)

    # row for last interior node (global i=N-2): term L_upper_full[-1]*u_{N-1}
    # substitute u_{N-1} = p1R*u_{N-2} + p2R*u_{N-3} + qR
    L_diag[-1] += L_upper_full[-1] * p1R
    L_lower_full[-1] += L_upper_full[-1] * p2R
    forcing[-1] += L_upper_full[-1] * qR
    L_upper_full[-1] = 0.0

    # Interior-interior arrays
    L_lower = np.asarray(L_lower_full[1:], dtype=float)  # (M-1,)
    L_upper = np.asarray(L_upper_full[:-1], dtype=float)  # (M-1,)

    L = Tridiag(lower=L_lower, diag=np.asarray(L_diag, dtype=float), upper=L_upper)
    L.check()
    return _L1D(L=L, forcing=cast(NDArray[np.floating], forcing))


def build_theta_system_1d(
    *,
    problem: LinearParabolicPDE1D,
    grid: Grid,
    t_n: float,
    t_np1: float,
    theta: float,
    advection: AdvectionScheme = AdvectionScheme.CENTRAL,
) -> tuple[CNSystem, NDArray[np.floating] | None]:
    if not (0.0 <= theta <= 1.0):
        raise ValueError("theta must be in [0, 1]")
    dt = float(t_np1 - t_n)
    if dt <= 0.0:
        raise ValueError("Require t_np1 > t_n")

    L_n = build_L_1d(
        grid=grid,
        t=t_n,
        a_fn=problem.a,
        b_fn=problem.b,
        c_fn=problem.c,
        bc=problem.bc,
        advection=advection,
    )
    L_np1 = build_L_1d(
        grid=grid,
        t=t_np1,
        a_fn=problem.a,
        b_fn=problem.b,
        c_fn=problem.c,
        bc=problem.bc,
        advection=advection,
    )

    M = L_n.L.check()
    if L_np1.L.check() != M:
        raise ValueError("Inconsistent operator sizes between t_n and t_np1")

    A = Tridiag(
        lower=-theta * dt * np.asarray(L_np1.L.lower, dtype=float),
        diag=1.0 - theta * dt * np.asarray(L_np1.L.diag, dtype=float),
        upper=-theta * dt * np.asarray(L_np1.L.upper, dtype=float),
    )
    B = Tridiag(
        lower=(1.0 - theta) * dt * np.asarray(L_n.L.lower, dtype=float),
        diag=1.0 + (1.0 - theta) * dt * np.asarray(L_n.L.diag, dtype=float),
        upper=(1.0 - theta) * dt * np.asarray(L_n.L.upper, dtype=float),
    )

    rhs = dt * ((1.0 - theta) * L_n.forcing + theta * L_np1.forcing)

    if problem.d is not None:
        x_int = np.asarray(grid.x[1:-1], dtype=float)
        d_n = _eval_xt(problem.d, x_int, t_n)
        d_np1 = _eval_xt(problem.d, x_int, t_np1)
        rhs = rhs + dt * ((1.0 - theta) * d_n + theta * d_np1)

    # Return None only if truly zero
    if np.allclose(rhs, 0.0):
        rhs_extra = None
    else:
        rhs_extra = cast(NDArray[np.floating], np.asarray(rhs, dtype=float))

    return CNSystem(A=A, B=B), rhs_extra
