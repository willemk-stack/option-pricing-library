"""Type definitions for implied volatility functions."""

from __future__ import annotations

from dataclasses import dataclass

from ..numerics.root_finding import RootResult
from ..typing import ArrayLike


@dataclass(frozen=True, slots=True)
class ImpliedVolResult:
    """Result container for Black-Scholes implied volatility inversion.

    Parameters
    ----------
    vol : float
        Implied volatility corresponding to the root found by the solver.
    root_result : RootResult
        Diagnostics returned by the chosen root-finding method (iterations, status,
        final residual, etc.).
    mkt_price : float
        The input market option price used for inversion.
    bounds : tuple[float, float]
        No-arbitrage bounds ``(lb, ub)`` for the option price.
    tau : float
        Time to expiry used in inversion, ``tau = spec.expiry - t`` in the
        legacy flat-input API where ``spec.expiry`` is the absolute expiry.

    Notes
    -----
    This is returned by :func:`implied_vol_bs_result`.
    """

    vol: float
    root_result: RootResult
    mkt_price: float
    bounds: tuple[float, float]  # (lb, ub)
    tau: float


@dataclass(frozen=True, slots=True)
class ImpliedVolSliceResult:
    vol: ArrayLike
    converged: ArrayLike
    iterations: ArrayLike
    f_at_root: ArrayLike
    bracket_lo: ArrayLike
    bracket_hi: ArrayLike
    status: ArrayLike  # 0=OK, 1=CLIPPED_LOW, 2=CLIPPED_HIGH, 3=INVALID_PRICE, 4=NO_CONVERGENCE
