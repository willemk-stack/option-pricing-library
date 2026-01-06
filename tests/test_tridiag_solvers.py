# tests/test_tridiag_solvers.py
import numpy as np
import pytest

# CHANGE THIS import to match your file/module name
from option_pricing.numerics.pde_fd import (
    BoundaryCoupling,
    GridConfig,
    SpacingPolicy,
    Tridiag,
    _tridiag_mv,
    build_grid,
    crank_nicolson_linear_step,
    solve_tridiag_scipy,
    solve_tridiag_thomas,
    tridiag_to_dense,
)


def make_diag_dominant_tridiag(rng: np.random.Generator, M: int, scale: float = 1.0):
    """
    Create a random *strictly diagonally dominant* tridiagonal system.
    This avoids near-zero pivots and makes comparisons stable.
    """
    if M < 1:
        raise ValueError("M must be >= 1")

    if M == 1:
        lower = np.array([], dtype=float)
        upper = np.array([], dtype=float)
        diag = np.array([1.0 + abs(rng.normal())], dtype=float) * scale
        return lower, diag, upper

    lower = rng.normal(size=M - 1) * scale
    upper = rng.normal(size=M - 1) * scale

    diag = (1.0 + np.abs(rng.normal(size=M))) * scale
    diag[0] += np.abs(upper[0])
    diag[-1] += np.abs(lower[-1])
    if M > 2:
        diag[1:-1] += np.abs(lower[:-1]) + np.abs(upper[1:])

    return lower, diag, upper


@pytest.mark.parametrize("M", [1, 2, 3, 10, 50, 200])
def test_thomas_matches_scipy_on_diag_dominant_random(M):
    rng = np.random.default_rng(12345 + M)
    lower, diag, upper = make_diag_dominant_tridiag(rng, M, scale=1.0)
    rhs = rng.normal(size=M)

    x_thomas = solve_tridiag_thomas(lower, diag, upper, rhs)
    x_scipy = solve_tridiag_scipy(lower, diag, upper, rhs)

    np.testing.assert_allclose(x_thomas, x_scipy, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("M", [1, 2, 10, 80])
def test_solutions_match_dense_solve(M):
    rng = np.random.default_rng(777 + M)
    lower, diag, upper = make_diag_dominant_tridiag(rng, M, scale=2.0)
    rhs = rng.normal(size=M)

    A = tridiag_to_dense(lower, diag, upper)
    x_dense = np.linalg.solve(A, rhs)

    x_thomas = solve_tridiag_thomas(lower, diag, upper, rhs)
    x_scipy = solve_tridiag_scipy(lower, diag, upper, rhs)

    np.testing.assert_allclose(x_thomas, x_dense, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(x_scipy, x_dense, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("M", [2, 5, 30])
def test_inputs_not_modified(M):
    rng = np.random.default_rng(999 + M)
    lower, diag, upper = make_diag_dominant_tridiag(rng, M, scale=1.0)
    rhs = rng.normal(size=M)

    lower0, diag0, upper0, rhs0 = lower.copy(), diag.copy(), upper.copy(), rhs.copy()

    _ = solve_tridiag_thomas(lower, diag, upper, rhs)
    np.testing.assert_array_equal(lower, lower0)
    np.testing.assert_array_equal(diag, diag0)
    np.testing.assert_array_equal(upper, upper0)
    np.testing.assert_array_equal(rhs, rhs0)

    _ = solve_tridiag_scipy(lower, diag, upper, rhs)
    np.testing.assert_array_equal(lower, lower0)
    np.testing.assert_array_equal(diag, diag0)
    np.testing.assert_array_equal(upper, upper0)
    np.testing.assert_array_equal(rhs, rhs0)


def test_shape_errors():
    rng = np.random.default_rng(0)
    M = 5
    lower, diag, upper = make_diag_dominant_tridiag(rng, M)
    rhs = rng.normal(size=M)

    with pytest.raises(ValueError):
        solve_tridiag_thomas(lower[:-1], diag, upper, rhs)  # wrong lower shape
    with pytest.raises(ValueError):
        solve_tridiag_thomas(lower, diag, upper, rhs[:-1])  # wrong rhs shape

    with pytest.raises(ValueError):
        solve_tridiag_scipy(lower[:-1], diag, upper, rhs)
    with pytest.raises(ValueError):
        solve_tridiag_scipy(lower, diag, upper, rhs[:-1])


@pytest.mark.parametrize("M", [1, 2, 10, 50])
def test_tridiag_mv_matches_dense(M):
    rng = np.random.default_rng(2024 + M)

    if M == 1:
        Bd = rng.normal(size=1)
        Bl = np.array([], dtype=float)
        Bu = np.array([], dtype=float)
        u = rng.normal(size=1)

        y = _tridiag_mv(Bl=Bl, Bd=Bd, Bu=Bu, u=u)
        np.testing.assert_allclose(y, Bd * u)
        return

    Bl = rng.normal(size=M - 1)
    Bd = rng.normal(size=M)
    Bu = rng.normal(size=M - 1)
    u = rng.normal(size=M)

    y = _tridiag_mv(Bl=Bl, Bd=Bd, Bu=Bu, u=u)

    T = tridiag_to_dense(Bl, Bd, Bu)
    y_dense = T @ u

    np.testing.assert_allclose(y, y_dense, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("Nx", [5, 20, 80])
def test_crank_nicolson_step_thomas_matches_scipy(Nx):
    """
    Compares one CN step using Thomas vs SciPy solve_banded.
    Uses random diagonals for A and B (not a PDE-derived operator),
    but that's enough to validate wiring & boundary injection.
    """
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

    N = Nx
    M = N - 2
    assert M >= 3

    # Random but stable-ish diagonals for A and B; make A diagonally dominant
    AlA, AdA, AuA = make_diag_dominant_tridiag(rng, M, scale=1.0)
    AlB, AdB, AuB = make_diag_dominant_tridiag(rng, M, scale=0.5)

    A = Tridiag(lower=AlA, diag=AdA, upper=AuA)
    B = Tridiag(lower=AlB, diag=AdB, upper=AuB)

    # Full u at time n (includes boundaries)
    u_n = rng.normal(size=N)

    # Boundary conditions
    def BC_L(t: float) -> float:
        return 0.3 + 0.1 * t

    def BC_R(t: float) -> float:
        return -0.2 + 0.05 * t

    t_n = 0.0
    t_np1 = 0.5

    u_np1_thomas = crank_nicolson_linear_step(
        grid=grid,
        u_n=u_n,
        t_n=t_n,
        t_np1=t_np1,
        A=A,
        B=B,
        BC_L=BC_L,
        BC_R=BC_R,
        bc=BoundaryCoupling(),  # no interior-boundary coupling in this synthetic test
        solve_tridiag=solve_tridiag_thomas,
    )

    u_np1_scipy = crank_nicolson_linear_step(
        grid=grid,
        u_n=u_n,
        t_n=t_n,
        t_np1=t_np1,
        A=A,
        B=B,
        BC_L=BC_L,
        BC_R=BC_R,
        bc=BoundaryCoupling(),
        solve_tridiag=solve_tridiag_scipy,
    )

    np.testing.assert_allclose(u_np1_thomas, u_np1_scipy, rtol=1e-10, atol=1e-12)
