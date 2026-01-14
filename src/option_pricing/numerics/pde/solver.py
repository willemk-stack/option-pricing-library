from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from ..grids import Grid, GridConfig, build_grid
from ..tridiag import Tridiag, solve_tridiag_thomas
from .methods import PDEMethod1D, ThetaMethod, resolve_method
from .operators import AdvectionScheme, LinearParabolicPDE1D


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
) -> PDESolution1D:
    """Solve a 1D linear parabolic PDE on a fixed grid.

    Parameters
    ----------
    problem:
        PDE coefficients + IC/BC.
    grid, grid_cfg:
        Provide either an explicit :class:`~option_pricing.numerics.grids.Grid`
        or a :class:`~option_pricing.numerics.grids.GridConfig`.
    method:
        Either a string alias (e.g. "cn", "implicit", "explicit"), a
        :class:`ThetaMethod`, or a custom object implementing
        :class:`~option_pricing.numerics.pde.methods.PDEMethod1D`.
    theta:
        Optional override for the theta parameter. If provided, this wins over
        `method`.
    advection:
        Spatial scheme for the b(x,t) u_x term.
    store:
        "all" stores the whole time history (Nt x Nx). "final" only stores the
        final slice (1 x Nx) to save memory.
    solve_tridiag:
        Function used to solve the interior tridiagonal system.
    """

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

    # Initial condition + enforce boundary at t0
    u0 = np.array([float(problem.ic(float(xi))) for xi in x], dtype=float)
    u0[0] = float(problem.bc.left(float(t[0])))
    u0[-1] = float(problem.bc.right(float(t[0])))

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
