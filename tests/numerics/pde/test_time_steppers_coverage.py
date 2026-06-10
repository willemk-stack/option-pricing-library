from __future__ import annotations

import numpy as np
import pytest

from option_pricing.numerics.grids import Grid
from option_pricing.numerics.pde.boundary import RobinBC, dirichlet_side
from option_pricing.numerics.pde.time_steppers import crank_nicolson_linear_step_robin
from option_pricing.numerics.tridiag import Tridiag


def _grid(n: int = 4) -> Grid:
    return Grid(t=np.array([0.0, 1.0]), x=np.linspace(0.0, 1.0, n))


def _tri(m: int) -> Tridiag:
    return Tridiag(
        lower=np.zeros(max(m - 1, 0)),
        diag=np.ones(m),
        upper=np.zeros(max(m - 1, 0)),
    )


def _bc() -> RobinBC:
    return RobinBC(
        left=dirichlet_side(lambda _t: 10.0),
        right=dirichlet_side(lambda _t: -3.0),
    )


def _identity_solver(
    _lower: np.ndarray,
    _diag: np.ndarray,
    _upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    return np.asarray(rhs, dtype=float)


def test_crank_nicolson_step_recovers_boundaries_and_adds_rhs_extra() -> None:
    result = crank_nicolson_linear_step_robin(
        grid=_grid(4),
        u_n=np.array([0.0, 1.0, 2.0, 0.0]),
        t_n=0.0,
        t_np1=1.0,
        A=_tri(2),
        B=_tri(2),
        bc=_bc(),
        rhs_extra=np.array([0.5, -0.5]),
        solve_tridiag=_identity_solver,
    )

    np.testing.assert_allclose(result, np.array([10.0, 1.5, 1.5, -3.0]))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"t_n": 1.0, "t_np1": 1.0}, "t_np1"),
        ({"u_n": np.zeros(3)}, "u_n must have shape"),
        (
            {"grid": _grid(3), "u_n": np.zeros(3), "A": _tri(1), "B": _tri(1)},
            "at least 4 spatial points",
        ),
        ({"A": _tri(3)}, "A,B must be sized"),
        ({"rhs_extra": np.zeros(3)}, "rhs_extra must have shape"),
    ],
)
def test_crank_nicolson_step_validates_inputs(
    kwargs: dict[str, object], match: str
) -> None:
    base = dict(
        grid=_grid(4),
        u_n=np.array([0.0, 1.0, 2.0, 0.0]),
        t_n=0.0,
        t_np1=1.0,
        A=_tri(2),
        B=_tri(2),
        bc=_bc(),
        rhs_extra=None,
        solve_tridiag=_identity_solver,
    )
    base.update(kwargs)

    with pytest.raises(ValueError, match=match):
        crank_nicolson_linear_step_robin(**base)


def test_crank_nicolson_step_validates_solver_output_shape() -> None:
    def bad_solver(*_args: object) -> np.ndarray:
        return np.zeros(3)

    with pytest.raises(ValueError, match="solve_tridiag must return shape"):
        crank_nicolson_linear_step_robin(
            grid=_grid(4),
            u_n=np.array([0.0, 1.0, 2.0, 0.0]),
            t_n=0.0,
            t_np1=1.0,
            A=_tri(2),
            B=_tri(2),
            bc=_bc(),
            solve_tridiag=bad_solver,
        )
