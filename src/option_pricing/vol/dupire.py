from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _fd_weights_3pt(
    x0: float, x1: float, x2: float, x_eval: float, deriv: int
) -> tuple[float, float, float]:
    """
    Quadratic Lagrange 3-point finite-difference weights for derivative at x_eval
    using points x0, x1, x2.
    deriv: 1 or 2
    """
    if deriv == 1:
        w0 = (2.0 * x_eval - x1 - x2) / ((x0 - x1) * (x0 - x2))
        w1 = (2.0 * x_eval - x0 - x2) / ((x1 - x0) * (x1 - x2))
        w2 = (2.0 * x_eval - x0 - x1) / ((x2 - x0) * (x2 - x1))
        return w0, w1, w2
    if deriv == 2:
        w0 = 2.0 / ((x0 - x1) * (x0 - x2))
        w1 = 2.0 / ((x1 - x0) * (x1 - x2))
        w2 = 2.0 / ((x2 - x0) * (x2 - x1))
        return w0, w1, w2
    raise ValueError("deriv must be 1 or 2")


def _weights_first_derivative_nonuniform(
    h_m: FloatArray, h_p: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Interior 3-point weights for first derivative at x_i on non-uniform grid:
      y'(x_i) ≈ beta_m y_{i-1} + beta y_i + beta_p y_{i+1}
    where h_m = x_i - x_{i-1}, h_p = x_{i+1} - x_i.
    """
    beta_m = -h_p / (h_m * (h_m + h_p))
    beta = (h_p - h_m) / (h_m * h_p)
    beta_p = h_m / (h_p * (h_m + h_p))
    return beta_m, beta, beta_p


def _weights_second_derivative_nonuniform(
    h_m: FloatArray, h_p: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Interior 3-point weights for second derivative at x_i on non-uniform grid:
      y''(x_i) ≈ alpha_m y_{i-1} + alpha y_i + alpha_p y_{i+1}
    """
    alpha_m = 2.0 / (h_m * (h_m + h_p))
    alpha = -2.0 / (h_m * h_p)
    alpha_p = 2.0 / (h_p * (h_m + h_p))
    return alpha_m, alpha, alpha_p


def _fdm_comp(
    prices: FloatArray,  # (nT, nK)
    maturities: FloatArray,  # (nT,)
    strikes: FloatArray,  # (nK,)
) -> tuple[FloatArray, FloatArray]:
    """
    Returns:
      C_T  : (nT, nK)  dC/dT
      C_KK : (nT, nK)  d²C/dK²
    Uses:
      - interior: nonuniform centered 3-pt stencils
      - boundaries: nonuniform one-sided 3-pt stencils
    """
    prices = np.asarray(prices, dtype=np.float64)
    T = np.asarray(maturities, dtype=np.float64)
    K = np.asarray(strikes, dtype=np.float64)

    if prices.ndim != 2:
        raise ValueError(f"prices must be 2D (nT,nK), got shape {prices.shape}")
    nT, nK = prices.shape
    if T.shape != (nT,) or K.shape != (nK,):
        raise ValueError(
            f"Shape mismatch: prices {prices.shape}, T {T.shape}, K {K.shape}"
        )
    if nT < 3 or nK < 3:
        raise ValueError(
            "Need at least 3 maturities and 3 strikes for 3-point stencils."
        )
    if not (np.all(np.diff(T) > 0) and np.all(np.diff(K) > 0)):
        raise ValueError("maturities and strikes must be strictly increasing.")
    if np.any(K <= 0):
        raise ValueError("strikes must be > 0 (Dupire has K^2 in denominator).")

    C_T = np.empty_like(prices)
    C_KK = np.empty_like(prices)

    # ---------- C_T along T (axis 0) ----------
    # interior i=1..nT-2 (vectorized)
    h_m_T = T[1:-1] - T[:-2]  # (nT-2,)
    h_p_T = T[2:] - T[1:-1]  # (nT-2,)
    beta_m, beta, beta_p = _weights_first_derivative_nonuniform(h_m_T, h_p_T)

    # reshape to (nT-2, 1) so it broadcasts across strikes (nK)
    beta_m = beta_m[:, None]
    beta = beta[:, None]
    beta_p = beta_p[:, None]

    C_T[1:-1, :] = (
        beta_m * prices[:-2, :] + beta * prices[1:-1, :] + beta_p * prices[2:, :]
    )

    # boundaries i=0 and i=nT-1 (one-sided 3pt)
    w0, w1, w2 = _fd_weights_3pt(T[0], T[1], T[2], T[0], deriv=1)
    C_T[0, :] = w0 * prices[0, :] + w1 * prices[1, :] + w2 * prices[2, :]

    w0, w1, w2 = _fd_weights_3pt(T[-3], T[-2], T[-1], T[-1], deriv=1)
    C_T[-1, :] = w0 * prices[-3, :] + w1 * prices[-2, :] + w2 * prices[-1, :]

    # ---------- C_KK along K (axis 1) ----------
    # interior j=1..nK-2 (vectorized)
    h_m_K = K[1:-1] - K[:-2]  # (nK-2,)
    h_p_K = K[2:] - K[1:-1]  # (nK-2,)
    alpha_m, alpha, alpha_p = _weights_second_derivative_nonuniform(h_m_K, h_p_K)
    # these broadcast naturally across rows: (nK-2,) with (nT, nK-2)

    C_KK[:, 1:-1] = (
        alpha_m * prices[:, :-2] + alpha * prices[:, 1:-1] + alpha_p * prices[:, 2:]
    )

    # boundaries j=0 and j=nK-1 (one-sided 3pt)
    w0, w1, w2 = _fd_weights_3pt(K[0], K[1], K[2], K[0], deriv=2)
    C_KK[:, 0] = w0 * prices[:, 0] + w1 * prices[:, 1] + w2 * prices[:, 2]

    w0, w1, w2 = _fd_weights_3pt(K[-3], K[-2], K[-1], K[-1], deriv=2)
    C_KK[:, -1] = w0 * prices[:, -3] + w1 * prices[:, -2] + w2 * prices[:, -1]

    return C_T, C_KK


def local_vol_from_call_grid(
    prices: FloatArray,  # (nT, nK)
    maturities: FloatArray,  # (nT,)
    strikes: FloatArray,  # (nK,)
    *,
    eps: float = 1e-12,
) -> FloatArray:
    """
    Simplified Dupire local vol with r=q=0:
      sigma^2 = 2 C_T / (K^2 C_KK)
      sigma = sqrt(sigma^2)

    Returns (nT, nK) with NaN where invalid.
    """
    C_T, C_KK = _fdm_comp(prices=prices, maturities=maturities, strikes=strikes)

    K = np.asarray(strikes, dtype=np.float64)[None, :]  # (1, nK) for broadcasting
    denom = (K * K) * C_KK

    with np.errstate(divide="ignore", invalid="ignore"):
        local_var = 2.0 * C_T / denom

    invalid = (
        (~np.isfinite(local_var)) | (denom <= eps) | (C_KK <= 0.0) | (local_var < 0.0)
    )
    local_var = np.where(invalid, np.nan, local_var)

    return np.sqrt(local_var)
