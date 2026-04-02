"""
LATER NEEDS FIXED-RULE QUADRATURE
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

from ...typing import FloatArray
from .charfunc import (
    _heston_affine_coeffs,
    _normalize_frequency_grid,
    _restore_frequency_shape,
)
from .params import HestonParams

type ComplexArray = NDArray[np.complex128]
type RealArray = NDArray[np.float64]


def _pj_affine_factor(
    u: float | np.ndarray,
    tau: float,
    params: HestonParams,
    *,
    j: int,
) -> complex | ComplexArray:
    u_arr, scalar_input, original_shape = _normalize_frequency_grid(u)
    C, D = _heston_affine_coeffs(u_arr, tau, params, j=j)
    values = np.exp(C * params.vbar + D * params.v)
    return _restore_frequency_shape(
        values, scalar_input=scalar_input, original_shape=original_shape
    )


def _integrand(
    u: float | FloatArray,
    x: float | FloatArray,
    tau: float,
    params: HestonParams,
    j: int,
) -> float | RealArray:
    u_arr = np.asarray(u, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)

    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x must be finite.")

    affine_factor = np.asarray(
        _pj_affine_factor(u_arr, tau, params, j=j),
        dtype=np.complex128,
    )

    if u_arr.ndim == 0 and x_arr.ndim == 0:
        value = np.real(np.exp(1j * u_arr * x_arr) * affine_factor / (1j * u_arr))
        return float(value)

    if u_arr.ndim == 0:
        values = np.real(np.exp(1j * u_arr * x_arr) * affine_factor / (1j * u_arr))
        return np.asarray(values, dtype=np.float64)

    if x_arr.ndim == 0:
        values = np.real(np.exp(1j * u_arr * x_arr) * affine_factor / (1j * u_arr))
        return np.asarray(values, dtype=np.float64)

    # Full outer-product style broadcasting:
    # u.shape = (n_u, ...)
    # x.shape = (n_x, ...)
    # result.shape = u.shape + x.shape
    u_b = u_arr.reshape(u_arr.shape + (1,) * x_arr.ndim)
    x_b = x_arr.reshape((1,) * u_arr.ndim + x_arr.shape)
    affine_b = affine_factor.reshape(affine_factor.shape + (1,) * x_arr.ndim)

    values = np.real(np.exp(1j * u_b * x_b) * affine_b / (1j * u_b))
    return np.asarray(values, dtype=np.float64)


def _integrand_scalar(
    u: float,
    x: float,
    tau: float,
    params: HestonParams,
    j: int,
) -> float:
    if not np.isfinite(x):
        raise ValueError("x must be finite.")

    u_arr = np.asarray(u, dtype=np.float64)
    affine_factor = np.asarray(
        _pj_affine_factor(u_arr, tau, params, j=j),
        dtype=np.complex128,
    )
    value = np.real(np.exp(1j * u_arr * x) * affine_factor / (1j * u_arr))
    return float(value)


def P_j(
    x: float,
    tau: float,
    params: HestonParams,
    j: int,
) -> float:

    integral, err_est = quad(
        _integrand_scalar,
        a=0,
        b=np.inf,
        args=(x, tau, params, j),
        complex_func=False,
    )
    out = 0.5 + integral / np.pi

    return float(out)
