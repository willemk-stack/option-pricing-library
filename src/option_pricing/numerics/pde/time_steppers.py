# src/option_pricing/numerics/time_steppers.py
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from ..grids import Grid
from ..tridiag import Tridiag, solve_tridiag_thomas
from .boundary import RobinBC, recover_boundaries_second_order

__all__ = ["crank_nicolson_linear_step_robin"]


# --- Robin BC types (Dirichlet/Neumann are special cases) ---

BoundaryFn = Callable[[float], float]  # t -> scalar


def crank_nicolson_linear_step_robin(
    *,
    grid: Grid,
    u_n: NDArray[
        np.floating
    ],  # (N,) includes boundaries; boundaries are ignored in the solve
    t_n: float,
    t_np1: float,
    # Interior-interior operators (M=N-2)
    A: Tridiag,
    B: Tridiag,
    # Robin/Neumann/Dirichlet BC (already eliminated in operator build; used here only to reconstruct u0,uN)
    bc: RobinBC,
    rhs_extra: NDArray[np.floating] | None = None,
    solve_tridiag=solve_tridiag_thomas,
) -> NDArray[np.floating]:
    """
    One CN step (elimination-style BC handling):

        A * u_int^{n+1} = B * u_int^{n} + rhs_extra

    Boundaries are recovered AFTER solving from Robin BC at t_{n+1} using a
    2nd-order one-sided derivative discretization.
    """
    if t_np1 <= t_n:
        raise ValueError("Require t_np1 > t_n")

    x = np.asarray(grid.x, dtype=float)
    N = int(x.shape[0])
    u_n = np.asarray(u_n, dtype=float)
    if u_n.shape != (N,):
        raise ValueError(f"u_n must have shape {(N,)} got {u_n.shape}")

    M = N - 2
    if M < 2:
        raise ValueError("Need at least 4 spatial points so M=N-2 >= 2")

    if A.check() != M or B.check() != M:
        raise ValueError(f"A,B must be sized for M=N-2={M}")

    u_n_int = u_n[1:-1]

    rhs = B.mv(u_n_int)

    if rhs_extra is not None:
        rhs_extra = np.asarray(rhs_extra, dtype=float)
        if rhs_extra.shape != (M,):
            raise ValueError(f"rhs_extra must have shape {(M,)} got {rhs_extra.shape}")
        rhs = rhs + rhs_extra

    u_np1_int = solve_tridiag(A.lower, A.diag, A.upper, rhs)
    if u_np1_int.shape != (M,):
        raise ValueError(
            f"solve_tridiag must return shape {(M,)} got {u_np1_int.shape}"
        )

    u0_np1, uN_np1 = recover_boundaries_second_order(
        x=x, u_int=u_np1_int, bc=bc, t=t_np1
    )

    u_np1 = u_n.copy()
    u_np1[0] = u0_np1
    u_np1[-1] = uN_np1
    u_np1[1:-1] = u_np1_int
    return u_np1
