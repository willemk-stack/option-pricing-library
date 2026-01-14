# tests/test_tridiag_solvers.py
import numpy as np
import pytest

from option_pricing.numerics.grids import GridConfig, SpacingPolicy, build_grid
from option_pricing.numerics.pde.boundary import DirichletBC
from option_pricing.numerics.pde.operators import (
    AdvectionScheme,
    LinearParabolicPDE1D,
    build_theta_system_1d,
)
from option_pricing.numerics.pde.time_steppers import crank_nicolson_linear_step
from option_pricing.numerics.tridiag import (
    Tridiag,
    solve_tridiag_scipy,
    solve_tridiag_thomas,
    tridiag_mv,
    tridiag_to_dense,
)


def make_diag_dominant_tridiag(
    rng: np.random.Generator, M: int, scale: float = 1.0
) -> Tridiag:
    """Create a random *strictly diagonally dominant* tridiagonal system."""
    if M < 1:
        raise ValueError("M must be >= 1")

    if M == 1:
        lower = np.array([], dtype=float)
        upper = np.array([], dtype=float)
        diag = np.array([1.0 + abs(rng.normal())], dtype=float) * scale
        return Tridiag(lower=lower, diag=diag, upper=upper)

    lower = rng.normal(size=M - 1) * scale
    upper = rng.normal(size=M - 1) * scale

    diag = (1.0 + np.abs(rng.normal(size=M))) * scale
    diag[0] += np.abs(upper[0])
    diag[-1] += np.abs(lower[-1])
    if M > 2:
        diag[1:-1] += np.abs(lower[:-1]) + np.abs(upper[1:])

    return Tridiag(lower=lower, diag=diag, upper=upper)


def solve_thomas_arrays(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Adapter: crank_nicolson_linear_step expects a (lower,diag,upper,rhs) solver."""
    tri = Tridiag(
        lower=np.asarray(lower, dtype=float),
        diag=np.asarray(diag, dtype=float),
        upper=np.asarray(upper, dtype=float),
    )
    return solve_tridiag_thomas(tri, np.asarray(rhs, dtype=float))


# --- Tests: tridiagonal algebra and solvers ---------------------------------


@pytest.mark.parametrize("M", [1, 2, 3, 10, 50, 200])
def test_thomas_matches_scipy_on_diag_dominant_random(M: int) -> None:
    rng = np.random.default_rng(12345 + M)
    A = make_diag_dominant_tridiag(rng, M, scale=1.0)
    rhs = rng.normal(size=M)

    x_thomas = solve_tridiag_thomas(A, rhs)
    x_scipy = solve_tridiag_scipy(A.lower, A.diag, A.upper, rhs)

    np.testing.assert_allclose(x_thomas, x_scipy, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("M", [1, 2, 10, 80])
def test_solutions_match_dense_solve(M: int) -> None:
    rng = np.random.default_rng(777 + M)
    A_tri = make_diag_dominant_tridiag(rng, M, scale=2.0)
    rhs = rng.normal(size=M)

    A = tridiag_to_dense(A_tri)
    x_dense = np.linalg.solve(A, rhs)

    x_thomas = solve_tridiag_thomas(A_tri, rhs)
    x_scipy = solve_tridiag_scipy(A_tri.lower, A_tri.diag, A_tri.upper, rhs)

    np.testing.assert_allclose(x_thomas, x_dense, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(x_scipy, x_dense, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("M", [2, 5, 30])
def test_inputs_not_modified(M: int) -> None:
    rng = np.random.default_rng(999 + M)
    A = make_diag_dominant_tridiag(rng, M, scale=1.0)
    rhs = rng.normal(size=M)

    lower0 = A.lower.copy()
    diag0 = A.diag.copy()
    upper0 = A.upper.copy()
    rhs0 = rhs.copy()

    _ = solve_tridiag_thomas(A, rhs)
    np.testing.assert_array_equal(A.lower, lower0)
    np.testing.assert_array_equal(A.diag, diag0)
    np.testing.assert_array_equal(A.upper, upper0)
    np.testing.assert_array_equal(rhs, rhs0)

    _ = solve_tridiag_scipy(A.lower, A.diag, A.upper, rhs)
    np.testing.assert_array_equal(A.lower, lower0)
    np.testing.assert_array_equal(A.diag, diag0)
    np.testing.assert_array_equal(A.upper, upper0)
    np.testing.assert_array_equal(rhs, rhs0)


def test_shape_errors() -> None:
    rng = np.random.default_rng(0)
    M = 5
    A = make_diag_dominant_tridiag(rng, M)
    rhs = rng.normal(size=M)

    # wrong lower shape: should be (M-1,)
    A_bad_lower = Tridiag(lower=A.lower[:-1], diag=A.diag, upper=A.upper)
    with pytest.raises(ValueError):
        _ = solve_tridiag_thomas(A_bad_lower, rhs)

    # wrong rhs shape
    with pytest.raises(ValueError):
        _ = solve_tridiag_thomas(A, rhs[:-1])

    # SciPy solver shape errors (arrays API)
    with pytest.raises(ValueError):
        _ = solve_tridiag_scipy(A.lower[:-1], A.diag, A.upper, rhs)
    with pytest.raises(ValueError):
        _ = solve_tridiag_scipy(A.lower, A.diag, A.upper, rhs[:-1])


@pytest.mark.parametrize("M", [1, 2, 10, 50])
def test_tridiag_mv_matches_dense(M: int) -> None:
    rng = np.random.default_rng(2024 + M)

    if M == 1:
        Bd = rng.normal(size=1)
        Bl = np.array([], dtype=float)
        Bu = np.array([], dtype=float)
        u = rng.normal(size=1)

        y = tridiag_mv(Bl=Bl, Bd=Bd, Bu=Bu, u=u)
        np.testing.assert_allclose(y, Bd * u)
        return

    Bl = rng.normal(size=M - 1)
    Bd = rng.normal(size=M)
    Bu = rng.normal(size=M - 1)
    u = rng.normal(size=M)

    y = tridiag_mv(Bl=Bl, Bd=Bd, Bu=Bu, u=u)

    T = tridiag_to_dense(Bl, Bd, Bu)
    y_dense = T @ u

    np.testing.assert_allclose(y, y_dense, rtol=1e-12, atol=1e-12)


# --- Test: CN step built via PDE module -------------------------------------


@pytest.mark.parametrize("Nx", [5, 20, 80])
def test_cn_step_thomas_matches_scipy_using_pde_module(Nx: int) -> None:
    """Compare one theta=0.5 step built from the PDE operator builder."""
    rng = np.random.default_rng(4242 + Nx)

    cfg = GridConfig(
        Nx=Nx,
        Nt=3,
        x_lb=-2.0,
        x_ub=2.0,
        T=1.0,
        spacing=SpacingPolicy.UNIFORM,
    )
    grid = build_grid(cfg)

    # Time-dependent Dirichlet boundaries
    def BC_L(t: float) -> float:
        return 0.3 + 0.1 * t

    def BC_R(t: float) -> float:
        return -0.2 + 0.05 * t

    bc = DirichletBC(left=BC_L, right=BC_R)

    # A stable linear parabolic PDE: u_t = a u_xx + b u_x + c u
    # Choose diffusion-dominated coefficients to keep A well-conditioned.
    def a(x, t):
        return 0.5  # diffusion

    def b(x, t):
        return 0.1  # drift

    def c(x, t):
        return 0.05  # reaction

    problem = LinearParabolicPDE1D(
        a=a,
        b=b,
        c=c,
        d=None,
        bc=bc,
        ic=lambda x: 0.0,  # not used directly in this single-step test
    )

    t_n = float(grid.t[0])
    t_np1 = float(grid.t[1])

    system, rhs_extra = build_theta_system_1d(
        problem=problem,
        grid=grid,
        t_n=t_n,
        t_np1=t_np1,
        theta=0.5,  # Crank–Nicolson
        advection=AdvectionScheme.CENTRAL,
    )

    N = int(grid.x.shape[0])
    u_n = rng.normal(size=N)

    # Enforce Dirichlet boundaries at t_n (consistent with the stepper contract)
    u_n[0] = BC_L(t_n)
    u_n[-1] = BC_R(t_n)

    u_np1_thomas = crank_nicolson_linear_step(
        grid=grid,
        u_n=u_n,
        t_n=t_n,
        t_np1=t_np1,
        A=system.A,
        B=system.B,
        BC_L=BC_L,
        BC_R=BC_R,
        bc=system.bc,
        rhs_extra=rhs_extra,
        solve_tridiag=solve_thomas_arrays,
    )

    u_np1_scipy = crank_nicolson_linear_step(
        grid=grid,
        u_n=u_n,
        t_n=t_n,
        t_np1=t_np1,
        A=system.A,
        B=system.B,
        BC_L=BC_L,
        BC_R=BC_R,
        bc=system.bc,
        rhs_extra=rhs_extra,
        solve_tridiag=solve_tridiag_scipy,
    )

    np.testing.assert_allclose(u_np1_thomas, u_np1_scipy, rtol=1e-10, atol=1e-12)
