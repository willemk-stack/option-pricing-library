from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from ..grids import Grid, GridConfig, build_grid
from ..tridiag import Tridiag, solve_tridiag_thomas
from .boundary import recover_boundaries_second_order
from .methods import PDEMethod1D, ThetaMethod, resolve_method
from .operators import AdvectionScheme, LinearParabolicPDE1D

ICTransform = Callable[[Grid, Callable[[float], float]], np.ndarray]


@dataclass(frozen=True, slots=True)
class PDESolution1D:
    grid: Grid
    u: NDArray[np.floating]  # (Nt, Nx)
    method: str
    advection: str

    @property
    def u_final(self) -> NDArray[np.floating]:
        return cast(NDArray[np.floating], self.u[-1])


def _solve_tridiag_thomas_arrays(
    lower: NDArray[np.floating],
    diag: NDArray[np.floating],
    upper: NDArray[np.floating],
    rhs: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Adapter: match the (lower,diag,upper,rhs) signature used by steppers."""

    return solve_tridiag_thomas(Tridiag(lower=lower, diag=diag, upper=upper), rhs)


def solve_pde_1d(
    problem: LinearParabolicPDE1D,
    *,
    grid: Grid | None = None,
    grid_cfg: GridConfig | None = None,
    method: str | ThetaMethod | PDEMethod1D = "cn",
    theta: float | None = None,
    advection: AdvectionScheme = AdvectionScheme.CENTRAL,
    store: Literal["all", "final"] = "all",
    solve_tridiag=_solve_tridiag_thomas_arrays,
    ic_transform: ICTransform | None = None,
) -> PDESolution1D:
    if (grid is None) == (grid_cfg is None):
        raise ValueError("Provide exactly one of grid or grid_cfg")

    if grid is None:
        assert grid_cfg is not None
        grid = build_grid(grid_cfg)

    x = np.asarray(grid.x, dtype=float)
    t = np.asarray(grid.t, dtype=float)
    Nx = int(x.shape[0])
    Nt = int(t.shape[0])
    if Nx < 4:
        raise ValueError("Need Nx>=4")
    if Nt < 2:
        raise ValueError("Need Nt>=2")

    method_obj = resolve_method(method=method, theta=theta)
    method_name = method_obj.name

    def _apply_bc(u_full: NDArray[np.floating], tt: float) -> NDArray[np.floating]:
        u_full = np.asarray(u_full, dtype=float)
        if u_full.shape != (Nx,):
            raise ValueError(f"u_full must have shape {(Nx,)} got {u_full.shape}")

        u_int = cast(NDArray[np.floating], u_full[1:-1])
        uL, uR = recover_boundaries_second_order(x=x, u_int=u_int, bc=problem.bc, t=tt)
        u_full[0] = uL
        u_full[-1] = uR
        return cast(NDArray[np.floating], u_full)

    # Initial condition on full grid
    ic_fn = problem.ic
    u0_raw = np.array([float(ic_fn(float(xi))) for xi in x], dtype=float)

    # Apply Initial condition if given
    if ic_transform is not None:
        u0 = np.asarray(ic_transform(grid, ic_fn), dtype=float)
        if u0.shape != u0_raw.shape:
            raise ValueError("ic_transform must return shape (Nx,)")
        if not np.all(np.isfinite(u0)):
            raise ValueError("ic_transform produced non-finite values")

    else:
        u0 = u0_raw

    # Enforce BC at t0 (Dirichlet or Robin/Neumann)
    u0 = _apply_bc(u0, float(t[0]))

    if store == "all":
        U = np.empty((Nt, Nx), dtype=float)
        U[0] = u0
    else:
        U = np.empty((1, Nx), dtype=float)
        U[0] = u0

    u = u0
    for n in range(Nt - 1):
        t_n = float(t[n])
        t_np1 = float(t[n + 1])

        u = method_obj.step(
            problem=problem,
            grid=grid,
            u_n=u,
            t_n=t_n,
            t_np1=t_np1,
            advection=advection,
            solve_tridiag=solve_tridiag,
        )

        # Enforce BC at t_{n+1} (important for Robin/Neumann; safe for Dirichlet)
        u = _apply_bc(u, t_np1)

        if store == "all":
            U[n + 1] = u
        else:
            U[0] = u

    return PDESolution1D(
        grid=grid,
        u=U,
        method=method_name,
        advection=str(advection.value),
    )
