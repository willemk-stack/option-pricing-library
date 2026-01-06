from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import cast

import numpy as np
from numpy.typing import NDArray


class SpacingPolicy(str, Enum):
    UNIFORM = "uniform"
    CLUSTERED = "clustered"


@dataclass(frozen=True, slots=True)
class GridConfig:
    Nx: int
    Nt: int
    x_lb: float
    x_ub: float
    T: float
    spacing: SpacingPolicy = SpacingPolicy.UNIFORM
    # Optional clustering params (only used if clustered)
    x_center: float | None = None  # e.g., log(K) or log(S0)
    cluster_strength: float = 2.0  # bigger -> more clustering near center

    def validate(self) -> None:
        if self.Nx < 3:
            raise ValueError("Nx must be >= 3")
        if self.Nt < 2:
            raise ValueError("Nt must be >= 2")
        if not (self.x_lb < self.x_ub):
            raise ValueError("Need x_lb < x_ub")
        if self.T <= 0:
            raise ValueError("T must be > 0")

        if self.spacing == SpacingPolicy.CLUSTERED:
            if self.x_center is None:
                raise ValueError("x_center required for clustered spacing")
            if not (self.x_lb <= self.x_center <= self.x_ub):
                raise ValueError("x_center must be within [x_lb, x_ub]")
            if self.cluster_strength <= 0:
                raise ValueError("cluster_strength must be > 0")


@dataclass(frozen=True, slots=True)
class Grid:
    t: NDArray[np.floating]
    x: NDArray[np.floating]


def _build_x_grid_validated(cfg: GridConfig) -> NDArray[np.floating]:
    if cfg.spacing == SpacingPolicy.UNIFORM:
        x = np.linspace(cfg.x_lb, cfg.x_ub, cfg.Nx, dtype=float)
    else:
        # clustered: sinh map around center, but keep points within [x_lb, x_ub]
        u = np.linspace(-1.0, 1.0, cfg.Nx, dtype=float)
        b = float(cfg.cluster_strength)

        raw = np.sinh(b * u)
        raw = raw / np.max(np.abs(raw))  # normalize to [-1, 1]

        x = np.empty_like(raw, dtype=float)
        # mypy doesn't know validate() implies non-None; narrow explicitly
        xc_opt = cfg.x_center
        assert xc_opt is not None
        xc = float(xc_opt)

        left_scale = xc - cfg.x_lb
        right_scale = cfg.x_ub - xc

        neg = raw <= 0.0
        pos = ~neg
        x[neg] = xc + raw[neg] * left_scale
        x[pos] = xc + raw[pos] * right_scale

        # enforce endpoints exactly
        x[0] = cfg.x_lb
        x[-1] = cfg.x_ub

    if not np.all(np.diff(x) > 0):
        raise ValueError("x grid must be strictly increasing")
    return x


def _build_time_grid_validated(cfg: GridConfig) -> NDArray[np.floating]:
    return np.linspace(0.0, cfg.T, cfg.Nt, dtype=float)


def build_x_grid(cfg: GridConfig) -> NDArray[np.floating]:
    cfg.validate()
    return _build_x_grid_validated(cfg)


def build_time_grid(cfg: GridConfig) -> NDArray[np.floating]:
    cfg.validate()
    return _build_time_grid_validated(cfg)


def build_grid(cfg: GridConfig) -> Grid:
    cfg.validate()
    return Grid(t=_build_time_grid_validated(cfg), x=_build_x_grid_validated(cfg))


def const_bc(val: float) -> Callable[[float], float]:
    """Convenience for constant-in-time Dirichlet BCs."""
    v = float(val)
    return lambda _t: v


@dataclass(frozen=True, slots=True)
class Tridiag:
    """
    Tridiagonal operator on an interior vector u of length M:
      y[0]   = diag[0]*u[0] + upper[0]*u[1]
      y[j]   = lower[j-1]*u[j-1] + diag[j]*u[j] + upper[j]*u[j+1]
      y[M-1] = lower[M-2]*u[M-2] + diag[M-1]*u[M-1]
    """

    lower: NDArray[np.floating]  # (M-1,)
    diag: NDArray[np.floating]  # (M,)
    upper: NDArray[np.floating]  # (M-1,)

    def check(self) -> int:
        M = int(self.diag.shape[0])
        if self.diag.shape != (M,):
            raise ValueError("diag must be 1D")
        if self.lower.shape != (M - 1,) or self.upper.shape != (M - 1,):
            raise ValueError(f"lower/upper must have shape {(M-1,)}")
        return M

    def mv(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        return _tridiag_mv(Bl=self.lower, Bd=self.diag, Bu=self.upper, u=u)


@dataclass(frozen=True, slots=True)
class BoundaryCoupling:
    """
    Boundary coupling coefficients for CN step, contributing to RHS of interior system:
      rhs[0]  += B_L*uL(t_n)   - A_L*uL(t_{n+1})
      rhs[-1] += B_R*uR(t_n)   - A_R*uR(t_{n+1})
    """

    A_L: float = 0.0
    A_R: float = 0.0
    B_L: float = 0.0
    B_R: float = 0.0


DEFAULT_BC: BoundaryCoupling = BoundaryCoupling()


def _tridiag_mv(
    Bl: NDArray[np.floating],  # (M-1,)
    Bd: NDArray[np.floating],  # (M,)
    Bu: NDArray[np.floating],  # (M-1,)
    u: NDArray[np.floating],  # (M,)
) -> NDArray[np.floating]:
    """
    Compute y = T u where T is tridiagonal with diagonals (Bl,Bd,Bu),
    acting on an interior vector of length M.

    Convention:
      y[0]   = Bd[0]*u[0] + Bu[0]*u[1]
      y[j]   = Bl[j-1]*u[j-1] + Bd[j]*u[j] + Bu[j]*u[j+1]   for 1<=j<=M-2
      y[M-1] = Bl[M-2]*u[M-2] + Bd[M-1]*u[M-1]
    """
    Bd = np.asarray(Bd)
    Bl = np.asarray(Bl)
    Bu = np.asarray(Bu)
    u = np.asarray(u)

    M = Bd.shape[0]
    if u.shape != (M,):
        raise ValueError(f"u must have shape {(M,)} got {u.shape}")
    if Bl.shape != (M - 1,) or Bu.shape != (M - 1,):
        raise ValueError(f"Bl,Bu must have shape {(M-1,)} got {Bl.shape}, {Bu.shape}")

    y = Bd * u
    y[1:] += Bl * u[:-1]
    y[:-1] += Bu * u[1:]
    return y


def solve_tridiag_thomas(
    lower: NDArray[np.floating],
    diag: NDArray[np.floating],
    upper: NDArray[np.floating],
    rhs: NDArray[np.floating],
    *,
    overwrite: bool = False,
) -> NDArray[np.floating]:
    """
    Solve A x = rhs for tridiagonal A with diagonals:
      lower: (M-1,) A[i, i-1] for i=1..M-1
      diag:  (M,)   A[i, i]
      upper: (M-1,) A[i, i+1] for i=0..M-2

    Notes:
    - Uses a Thomas algorithm (no pivoting). Prefer diagonally-dominant systems.
    - Raises np.linalg.LinAlgError on (near-)zero pivots.
    """
    diag = np.asarray(diag)
    M = int(diag.shape[0])

    rhs = np.asarray(rhs)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    if rhs.shape != (M,):
        raise ValueError(f"rhs must have shape {(M,)} got {rhs.shape}")
    if lower.shape != (M - 1,) or upper.shape != (M - 1,):
        raise ValueError(
            f"lower/upper must have shape {(M-1,)} got {lower.shape}, {upper.shape}"
        )

    dtype = np.result_type(lower, diag, upper, rhs, np.float64)
    lower = lower.astype(dtype, copy=not overwrite)
    diag = diag.astype(dtype, copy=not overwrite)
    upper = upper.astype(dtype, copy=not overwrite)
    rhs = rhs.astype(dtype, copy=not overwrite)

    if M == 0:
        return rhs.copy()
    if M == 1:
        denom = diag[0]
        tol = 100.0 * np.finfo(dtype).eps
        if abs(denom) < tol:
            raise np.linalg.LinAlgError("Near-zero pivot at row 0")
        return cast(NDArray[np.floating], rhs / denom)

    tol = 100.0 * np.finfo(dtype).eps

    # Forward sweep (store modified upper and rhs)
    denom = diag[0]
    if abs(denom) < tol:
        raise np.linalg.LinAlgError("Near-zero pivot at row 0")
    upper[0] = upper[0] / denom
    rhs[0] = rhs[0] / denom

    for i in range(1, M - 1):
        denom = diag[i] - lower[i - 1] * upper[i - 1]
        if abs(denom) < tol:
            raise np.linalg.LinAlgError(f"Near-zero pivot at row {i}")
        upper[i] = upper[i] / denom
        rhs[i] = (rhs[i] - lower[i - 1] * rhs[i - 1]) / denom

    denom = diag[M - 1] - lower[M - 2] * upper[M - 2]
    if abs(denom) < tol:
        raise np.linalg.LinAlgError(f"Near-zero pivot at row {M-1}")
    rhs[M - 1] = (rhs[M - 1] - lower[M - 2] * rhs[M - 2]) / denom

    # Back substitution
    x = np.empty(M, dtype=dtype)
    x[M - 1] = rhs[M - 1]
    for i in range(M - 2, -1, -1):
        x[i] = rhs[i] - upper[i] * x[i + 1]
    return x


def solve_tridiag_scipy(
    lower: NDArray[np.floating],
    diag: NDArray[np.floating],
    upper: NDArray[np.floating],
    rhs: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Solve using SciPy banded solver. SciPy is imported lazily.
    """
    from scipy.linalg import (
        solve_banded,  # local import to avoid import-time dependency
    )

    diag = np.asarray(diag)
    M = int(diag.shape[0])
    rhs = np.asarray(rhs)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    if rhs.shape != (M,):
        raise ValueError(f"rhs must have shape {(M,)} got {rhs.shape}")
    if lower.shape != (M - 1,) or upper.shape != (M - 1,):
        raise ValueError(
            f"lower/upper must have shape {(M-1,)} got {lower.shape}, {upper.shape}"
        )

    ab = np.zeros((3, M), dtype=np.result_type(lower, diag, upper, rhs))
    ab[0, 1:] = upper
    ab[1, :] = diag
    ab[2, :-1] = lower

    res = solve_banded((1, 1), ab, rhs)
    # scipy stubs often return Any; cast back to an NDArray
    return cast(NDArray[np.floating], np.asarray(res))


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


def tridiag_to_dense(
    lower: NDArray[np.floating] | Tridiag,
    diag: NDArray[np.floating] | None = None,
    upper: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """
    Convert a tridiagonal to a dense matrix.
    Accepts either (lower, diag, upper) arrays or a Tridiag instance.
    """
    if isinstance(lower, Tridiag):
        tri = lower
        lower = tri.lower
        diag = tri.diag
        upper = tri.upper
    if diag is None or upper is None:
        raise ValueError("Must provide (lower, diag, upper) or a Tridiag")

    diag = np.asarray(diag)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    M = int(diag.shape[0])
    A = np.zeros((M, M), dtype=np.result_type(lower, diag, upper))
    A[np.arange(M), np.arange(M)] = diag
    A[np.arange(1, M), np.arange(M - 1)] = lower
    A[np.arange(M - 1), np.arange(1, M)] = upper
    return A
