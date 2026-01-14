from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ..grids import Grid
from ..tridiag import BoundaryCoupling, Tridiag
from .boundary import DirichletBC
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
    bc: DirichletBC
    ic: Callable[[float], float]
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


def _first_derivative_coeffs(
    hm: NDArray[np.floating],
    hp: NDArray[np.floating],
    b: NDArray[np.floating],
    scheme: AdvectionScheme,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Return (dl, dd, du) for the u_x stencil at interior points.

    hm = x_i - x_{i-1}, hp = x_{i+1} - x_i

    CENTRAL:
      u_x(x_i) ≈ dl*u_{i-1} + dd*u_i + du*u_{i+1}
      (2nd order on nonuniform grids).

    UPWIND:
      if b_i >= 0: backward difference
      if b_i <  0: forward difference
    """

    if scheme == AdvectionScheme.CENTRAL:
        denom = hm * hp * (hm + hp)
        dl = -hp * hp / denom
        dd = (hp * hp - hm * hm) / denom
        du = hm * hm / denom
        return dl, dd, du

    # Upwind (1st order)
    dl = np.zeros_like(hm, dtype=float)
    dd = np.zeros_like(hm, dtype=float)
    du = np.zeros_like(hm, dtype=float)

    pos = b >= 0.0
    neg = ~pos

    # backward: (u_i - u_{i-1}) / hm
    dl[pos] = -1.0 / hm[pos]
    dd[pos] = 1.0 / hm[pos]

    # forward: (u_{i+1} - u_i) / hp
    dd[neg] = -1.0 / hp[neg]
    du[neg] = 1.0 / hp[neg]

    return (
        cast(NDArray[np.floating], dl),
        cast(NDArray[np.floating], dd),
        cast(NDArray[np.floating], du),
    )


def _second_derivative_coeffs(
    hm: NDArray[np.floating], hp: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Return (dl, dd, du) for the u_xx stencil at interior points (2nd order)."""

    dl = 2.0 / (hm * (hm + hp))
    dd = -2.0 / (hm * hp)
    du = 2.0 / (hp * (hm + hp))
    return (
        cast(NDArray[np.floating], dl),
        cast(NDArray[np.floating], dd),
        cast(NDArray[np.floating], du),
    )


@dataclass(frozen=True, slots=True)
class _L1D:
    """Discrete L operator on interior nodes, plus boundary coupling in L."""

    L: Tridiag
    L_left_bc: float
    L_right_bc: float


def build_L_1d(
    *,
    grid: Grid,
    t: float,
    a_fn: ScalarXT,
    b_fn: ScalarXT,
    c_fn: ScalarXT,
    advection: AdvectionScheme = AdvectionScheme.CENTRAL,
) -> _L1D:
    """Build the interior tridiagonal discretization of L at time t.

    Returns
    -------
    _L1D
        Contains L (interior-interior Tridiag) and the coefficients in the
        first and last interior rows that multiply the Dirichlet boundaries.
    """

    x = np.asarray(grid.x, dtype=float)
    N = int(x.shape[0])
    if N < 4:
        raise ValueError("Need at least 4 spatial points (Nx>=4)")

    x_int = x[1:-1]

    hm = x[1:-1] - x[0:-2]  # (M,)
    hp = x[2:] - x[1:-1]  # (M,)
    if not np.all(hm > 0.0) or not np.all(hp > 0.0):
        raise ValueError("x grid must be strictly increasing")

    a = _eval_xt(a_fn, x_int, t)
    b = _eval_xt(b_fn, x_int, t)
    c = _eval_xt(c_fn, x_int, t)

    d2l, d2d, d2u = _second_derivative_coeffs(hm, hp)
    d1l, d1d, d1u = _first_derivative_coeffs(hm, hp, b=b, scheme=advection)

    L_lower_full = a * d2l + b * d1l
    L_diag = a * d2d + b * d1d + c
    L_upper_full = a * d2u + b * d1u

    # Separate the boundary contributions from the interior-interior tridiag.
    L_left_bc = float(L_lower_full[0])
    L_right_bc = float(L_upper_full[-1])

    # Interior-interior arrays
    L_lower = np.asarray(L_lower_full[1:], dtype=float)  # (M-1,)
    L_upper = np.asarray(L_upper_full[:-1], dtype=float)  # (M-1,)

    L = Tridiag(lower=L_lower, diag=np.asarray(L_diag, dtype=float), upper=L_upper)
    L.check()
    return _L1D(L=L, L_left_bc=L_left_bc, L_right_bc=L_right_bc)


def build_theta_system_1d(
    *,
    problem: LinearParabolicPDE1D,
    grid: Grid,
    t_n: float,
    t_np1: float,
    theta: float,
    advection: AdvectionScheme = AdvectionScheme.CENTRAL,
) -> tuple[CNSystem, NDArray[np.floating] | None]:
    """Build (A,B,bc) for the theta-scheme on interior nodes.

    The theta-scheme (with possibly time-dependent coefficients) is:

        (I - theta*dt*L^{n+1}) u^{n+1} = (I + (1-theta)*dt*L^{n}) u^{n}
                                         + dt[(1-theta)d^n + theta d^{n+1}]

    Boundary values enter only in the first and last interior equations; those
    couplings are returned as a :class:`~option_pricing.numerics.BoundaryCoupling`.

    Returns
    -------
    (CNSystem, rhs_extra)
        rhs_extra is the inhomogeneous/source contribution (length M=N-2) that
        should be added to the interior RHS (or None if d is not provided).
    """

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
        advection=advection,
    )
    L_np1 = build_L_1d(
        grid=grid,
        t=t_np1,
        a_fn=problem.a,
        b_fn=problem.b,
        c_fn=problem.c,
        advection=advection,
    )

    M = L_n.L.check()
    if L_np1.L.check() != M:
        raise ValueError("Inconsistent operator sizes between t_n and t_np1")

    # A uses L at t_{n+1}, B uses L at t_n
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

    bc = BoundaryCoupling(
        # Note: stepper adds (B_L*uL_n - A_L*uL_{n+1})
        A_L=-theta * dt * float(L_np1.L_left_bc),
        A_R=-theta * dt * float(L_np1.L_right_bc),
        B_L=(1.0 - theta) * dt * float(L_n.L_left_bc),
        B_R=(1.0 - theta) * dt * float(L_n.L_right_bc),
    )

    rhs_extra: NDArray[np.floating] | None = None
    if problem.d is not None:
        x_int = np.asarray(grid.x[1:-1], dtype=float)
        d_n = _eval_xt(problem.d, x_int, t_n)
        d_np1 = _eval_xt(problem.d, x_int, t_np1)
        rhs_extra = cast(
            NDArray[np.floating],
            dt * ((1.0 - theta) * d_n + theta * d_np1),
        )
        if rhs_extra.shape != (M,):
            raise ValueError("d(x,t) evaluation returned wrong shape")

    return CNSystem(A=A, B=B, bc=bc), rhs_extra
