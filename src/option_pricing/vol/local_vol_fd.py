"""Finite difference computation helpers for local volatility."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..numerics.fd.diff import diff1_nonuniform, diff2_nonuniform

FloatArray = NDArray[np.floating[Any]]


def _fdm_comp(
    prices: FloatArray,  # (nT, nK)
    taus: FloatArray,  # (nT,)
    strikes: FloatArray,  # (nK,)
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute finite differences of call prices on a tau-strike grid.

    Parameters
    ----------
    prices : FloatArray
        Call prices, shape (nT, nK)
    taus : FloatArray
        Time-to-expiry grid, shape (nT,), strictly increasing
    strikes : FloatArray
        Strike grid, shape (nK,), strictly increasing, all > 0

    Returns
    -------
    C_T : FloatArray
        Time derivatives dC/dtau, shape (nT, nK)
    C_K : FloatArray
        Strike derivatives dC/dK, shape (nT, nK)
    C_KK : FloatArray
        Second derivatives d²C/dK², shape (nT, nK)

    Raises
    ------
    ValueError
        If arrays are not 1D/2D or dimensions don't match
    ValueError
        If grids are not strictly increasing or strikes not > 0
    ValueError
        If fewer than 3 points in either dimension (minimum for 3-point stencils)
    """
    prices = np.asarray(prices, dtype=np.float64)
    T = np.asarray(taus, dtype=np.float64)
    K = np.asarray(strikes, dtype=np.float64)

    if prices.ndim != 2:
        raise ValueError(f"prices must be 2D (nT,nK), got shape {prices.shape}")
    nT, nK = prices.shape
    if T.shape != (nT,) or K.shape != (nK,):
        raise ValueError(
            f"Shape mismatch: prices {prices.shape}, taus {T.shape}, strikes {K.shape}"
        )
    if nT < 3 or nK < 3:
        raise ValueError(
            "Need at least 3 maturities and 3 strikes for 3-point stencils."
        )
    if not (np.all(np.diff(T) > 0) and np.all(np.diff(K) > 0)):
        raise ValueError("taus and strikes must be strictly increasing.")
    if np.any(K <= 0):
        raise ValueError("strikes must be > 0.")

    # ---- C_T along tau (axis 0) ----
    C_T = diff1_nonuniform(y=prices, x=taus, axis=0)

    # ---- along K (axis 1) ----
    C_K = diff1_nonuniform(y=prices, x=strikes, axis=1)
    C_KK = diff2_nonuniform(y=prices, x=strikes, axis=1)

    return C_T, C_K, C_KK


def _fdm_comp_logk(
    prices: FloatArray,  # (nT, nK)
    taus: FloatArray,  # (nT,)
    strikes: FloatArray,  # (nK,)
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute finite differences of call prices in log-strike coordinates.

    This is typically more stable than strike-space differentiation for pricing grids.

    Parameters
    ----------
    prices : FloatArray
        Call prices, shape (nT, nK)
    taus : FloatArray
        Time-to-expiry grid, shape (nT,), strictly increasing
    strikes : FloatArray
        Strike grid, shape (nK,), strictly increasing, all > 0

    Returns
    -------
    C_T : FloatArray
        Time derivatives dC/dtau, shape (nT, nK)
    C_x : FloatArray
        Log-strike derivatives dC/dx where x = log(K), shape (nT, nK)
    C_xx : FloatArray
        Second derivatives d²C/dx², shape (nT, nK)

    Raises
    ------
    ValueError
        If strikes are not > 0 (log requires positive strikes)
    ValueError
        If arrays are not 1D/2D or dimensions don't match
    ValueError
        If grids are not strictly increasing
    ValueError
        If fewer than 3 points in either dimension (minimum for 3-point stencils)
    """
    prices = np.asarray(prices, dtype=np.float64)
    T = np.asarray(taus, dtype=np.float64)
    K = np.asarray(strikes, dtype=np.float64)

    if np.any(K <= 0):
        raise ValueError("strikes must be > 0 to use logK coordinate.")
    x = np.log(K)

    if prices.ndim != 2:
        raise ValueError(f"prices must be 2D (nT,nK), got shape {prices.shape}")
    nT, nK = prices.shape
    if T.shape != (nT,) or K.shape != (nK,):
        raise ValueError(
            f"Shape mismatch: prices {prices.shape}, taus {T.shape}, strikes {K.shape}"
        )
    if nT < 3 or nK < 3:
        raise ValueError(
            "Need at least 3 maturities and 3 strikes for 3-point stencils."
        )
    if not (np.all(np.diff(T) > 0) and np.all(np.diff(K) > 0)):
        raise ValueError("taus and strikes must be strictly increasing.")

    # ---- time derivative same as in _fdm_comp ----
    # ---- C_T along tau (axis 0) ----
    C_T = diff1_nonuniform(y=prices, x=taus, axis=0)

    # ---- along log(K) (axis 1) ----
    C_x = diff1_nonuniform(y=prices, x=x, axis=1)
    C_xx = diff2_nonuniform(y=prices, x=x, axis=1)

    return C_T, C_x, C_xx
