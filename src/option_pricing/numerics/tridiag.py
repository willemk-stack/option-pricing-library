# src/option_pricing/numerics/tridiag.py
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Tridiag",
    "BoundaryCoupling",
    "DEFAULT_BC",
    "tridiag_mv",
    "solve_tridiag_thomas",
    "solve_tridiag_scipy",
    "tridiag_to_dense",
]


@dataclass(frozen=True, slots=True)
class Tridiag:
    lower: NDArray[np.floating]
    diag: NDArray[np.floating]
    upper: NDArray[np.floating]

    def check(self) -> int:
        """
        Validate internal shapes and return M (system size).

        Supports M == 0 with empty diagonals:
          diag.shape  == (0,)
          lower.shape == (0,)
          upper.shape == (0,)
        """
        diag = np.asarray(self.diag)
        if diag.ndim != 1:
            raise ValueError("diag must be 1D")

        M = int(diag.shape[0])

        lower = np.asarray(self.lower)
        upper = np.asarray(self.upper)

        if M == 0:
            if lower.shape != (0,) or upper.shape != (0,):
                raise ValueError("For M==0, lower/upper must be empty (shape (0,))")
            return 0

        if lower.shape != (M - 1,) or upper.shape != (M - 1,):
            raise ValueError(f"lower/upper must have shape {(M - 1,)}")
        return M

    def mv(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        M = self.check()  # ensure internal consistency
        u = np.asarray(u)
        if u.shape != (M,):
            raise ValueError(f"u must have shape {(M,)} got {u.shape}")
        return tridiag_mv(Bl=self.lower, Bd=self.diag, Bu=self.upper, u=u)


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


def tridiag_mv(
    Bl: NDArray[np.floating],  # (M-1,) or (0,) if M==0
    Bd: NDArray[np.floating],  # (M,)
    Bu: NDArray[np.floating],  # (M-1,) or (0,) if M==0
    u: NDArray[np.floating],  # (M,)
) -> NDArray[np.floating]:
    """
    Compute y = T u where T is tridiagonal with diagonals (Bl,Bd,Bu),
    acting on a vector u of length M.

    Convention (for M>=2):
      y[0]   = Bd[0]*u[0] + Bu[0]*u[1]
      y[j]   = Bl[j-1]*u[j-1] + Bd[j]*u[j] + Bu[j]*u[j+1]   for 1<=j<=M-2
      y[M-1] = Bl[M-2]*u[M-2] + Bd[M-1]*u[M-1]

    For M==0: returns empty array.
    For M==1: y[0] = Bd[0]*u[0].
    """
    Bd = np.asarray(Bd)
    Bl = np.asarray(Bl)
    Bu = np.asarray(Bu)
    u = np.asarray(u)

    if Bd.ndim != 1:
        raise ValueError("Bd must be 1D")

    M = int(Bd.shape[0])

    if u.shape != (M,):
        raise ValueError(f"u must have shape {(M,)} got {u.shape}")

    if M == 0:
        if Bl.shape != (0,) or Bu.shape != (0,):
            raise ValueError("For M==0, Bl,Bu must be empty (shape (0,))")
        return cast(NDArray[np.floating], Bd * u)  # empty

    if Bl.shape != (M - 1,) or Bu.shape != (M - 1,):
        raise ValueError(f"Bl,Bu must have shape {(M - 1,)} got {Bl.shape}, {Bu.shape}")

    y = Bd * u
    y[1:] += Bl * u[:-1]
    y[:-1] += Bu * u[1:]
    return cast(NDArray[np.floating], y)


def solve_tridiag_thomas(
    A: Tridiag,
    rhs: NDArray[np.floating],
    *,
    overwrite: bool = False,
) -> NDArray[np.floating]:
    """
    Solve A x = rhs for tridiagonal A.

    Notes:
    - Uses a Thomas algorithm (no pivoting). Prefer diagonally-dominant systems.
    - Raises np.linalg.LinAlgError on (near-)zero pivots.
    - If overwrite=True and dtypes already match, this may mutate A.lower/A.diag/A.upper
      and rhs in-place (because it avoids copies).
    """
    M = A.check()  # <-- uses internal check()

    rhs = np.asarray(rhs)
    if rhs.shape != (M,):
        raise ValueError(f"rhs must have shape {(M,)} got {rhs.shape}")

    diag = np.asarray(A.diag)
    lower = np.asarray(A.lower)
    upper = np.asarray(A.upper)

    dtype = np.result_type(lower, diag, upper, rhs, np.float64)
    lower = lower.astype(dtype, copy=not overwrite)
    diag = diag.astype(dtype, copy=not overwrite)
    upper = upper.astype(dtype, copy=not overwrite)
    rhs = rhs.astype(dtype, copy=not overwrite)

    if M == 0:
        return rhs.copy()

    tol = 100.0 * np.finfo(dtype).eps

    if M == 1:
        denom = diag[0]
        if abs(denom) < tol:
            raise np.linalg.LinAlgError("Near-zero pivot at row 0")
        return cast(NDArray[np.floating], rhs / denom)

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
        raise np.linalg.LinAlgError(f"Near-zero pivot at row {M - 1}")
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

    This now validates (lower, diag, upper) via Tridiag.check().
    """
    from scipy.linalg import (
        solve_banded,  # local import to avoid import-time dependency
    )

    # Use internal check() by constructing a Tridiag
    tri = Tridiag(np.asarray(lower), np.asarray(diag), np.asarray(upper))
    M = tri.check()

    rhs = np.asarray(rhs)
    if rhs.shape != (M,):
        raise ValueError(f"rhs must have shape {(M,)} got {rhs.shape}")

    if M == 0:
        return cast(NDArray[np.floating], rhs.copy())

    ab = np.zeros((3, M), dtype=np.result_type(tri.lower, tri.diag, tri.upper, rhs))
    ab[0, 1:] = np.asarray(tri.upper)
    ab[1, :] = np.asarray(tri.diag)
    ab[2, :-1] = np.asarray(tri.lower)

    res = solve_banded((1, 1), ab, rhs)
    # scipy stubs often return Any; cast back to an NDArray
    return cast(NDArray[np.floating], np.asarray(res))


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
        M = tri.check()  # validate shapes early
        lower = np.asarray(tri.lower)
        diag = np.asarray(tri.diag)
        upper = np.asarray(tri.upper)
    else:
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
