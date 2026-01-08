# src/option_pricing/numerics/time_steppers.py
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .grids import Grid
from .tridiag import DEFAULT_BC, BoundaryCoupling, Tridiag, solve_tridiag_scipy

__all__ = ["crank_nicolson_linear_step"]


def crank_nicolson_linear_step(
    *,
    grid: Grid,
    u_n: NDArray[np.floating],  # (N,) includes boundaries
    t_n: float,
    t_np1: float,
    # Interior-interior operators
    A: Tridiag,
    B: Tridiag,
    # Time-dependent Dirichlet boundaries
    BC_L: Callable[[float], float],
    BC_R: Callable[[float], float],
    # Boundary coupling coefficients (to add to RHS)
    bc: BoundaryCoupling = DEFAULT_BC,
    solve_tridiag: Callable[
        [
            NDArray[np.floating],
            NDArray[np.floating],
            NDArray[np.floating],
            NDArray[np.floating],
        ],
        NDArray[np.floating],
    ] = solve_tridiag_scipy,
) -> NDArray[np.floating]:
    """
    One CN step: solve interior unknowns and return full u_{n+1}.

    We solve:
        A * u_int^{n+1} = B * u_int^{n} + boundary_terms

    IMPORTANT:
    - A and B are INTERIOR-INTERIOR tridiagonal operators only (length M = N-2).
    - Boundary contributions from Dirichlet values uL(t), uR(t) are injected via `bc`.
    """
    if t_np1 <= t_n:
        raise ValueError("Require t_np1 > t_n")

    x = np.asarray(grid.x)
    N = int(x.shape[0])
    u_n = np.asarray(u_n)
    if u_n.shape != (N,):
        raise ValueError(f"u_n must have shape {(N,)} got {u_n.shape}")

    M = N - 2
    if M < 2:
        raise ValueError("Need at least 4 spatial points so M=N-2 >= 2")

    M_A = A.check()
    M_B = B.check()
    if M_A != M or M_B != M:
        raise ValueError(f"A,B must be sized for M=N-2={M}; got {M_A} and {M_B}")

    uL_n = float(BC_L(t_n))
    uR_n = float(BC_R(t_n))
    uL_np1 = float(BC_L(t_np1))
    uR_np1 = float(BC_R(t_np1))

    u_n_int = u_n[1:-1]

    # rhs = B * u_n_int
    rhs = B.mv(u_n_int)

    # Add boundary contributions (only first and last interior equations)
    rhs[0] += bc.B_L * uL_n - bc.A_L * uL_np1
    rhs[-1] += bc.B_R * uR_n - bc.A_R * uR_np1

    u_np1_int = solve_tridiag(A.lower, A.diag, A.upper, rhs)
    if u_np1_int.shape != (M,):
        raise ValueError(
            f"solve_tridiag must return shape {(M,)} got {u_np1_int.shape}"
        )

    u_np1 = u_n.copy()
    u_np1[0] = uL_np1
    u_np1[-1] = uR_np1
    u_np1[1:-1] = u_np1_int
    return u_np1
