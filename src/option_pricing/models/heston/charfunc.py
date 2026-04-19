"""Stable Heston characteristic-function building blocks."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ...typing import ArrayLike
from .params import HestonParams

type ComplexArray = NDArray[np.complex128]
type BoolArray = NDArray[np.bool_]
type J = Literal[0, 1]


def _normalize_frequency_grid(
    u: complex | ArrayLike,
) -> tuple[ComplexArray, bool, tuple[int, ...]]:
    u_arr = np.asarray(u, dtype=np.complex128)

    if not np.all(np.isfinite(u_arr.real)) or not np.all(np.isfinite(u_arr.imag)):
        raise ValueError("u must be finite.")

    return (
        np.asarray(u_arr.reshape(-1), dtype=np.complex128),
        u_arr.ndim == 0,
        u_arr.shape,
    )


def _restore_frequency_shape(
    values: np.ndarray, *, scalar_input: bool, original_shape: tuple[int, ...]
) -> complex | ComplexArray:
    if scalar_input:
        return complex(values[0])
    return values.reshape(original_shape)


def _validate_tau(tau: float) -> float:
    tau = float(tau)
    if not np.isfinite(tau):
        raise ValueError("tau must be finite.")
    if tau < 0.0:
        raise ValueError("tau must be nonnegative.")
    return tau


def _validate_probability_index(j: J) -> J:
    if j not in (0, 1):
        raise ValueError("j must be either 0 or 1.")
    return j


def _quadratic_term(u: np.ndarray, *, j: J) -> ComplexArray:
    out: ComplexArray = u * u + 1j * (1 - 2 * j) * u
    return out


def _integrated_variance(params: HestonParams, tau: float) -> float:
    return float(
        params.vbar * tau
        + (params.v - params.vbar) * (1.0 - np.exp(-params.kappa * tau)) / params.kappa
    )


def _deterministic_affine_coeffs(
    u: np.ndarray, tau: float, params: HestonParams, *, j: J
) -> tuple[np.ndarray, np.ndarray]:
    quadratic_term = _quadratic_term(u, j=j)
    mean_reversion_loading = (1.0 - np.exp(-params.kappa * tau)) / params.kappa

    C = -0.5 * quadratic_term * (tau - mean_reversion_loading)
    D = -0.5 * quadratic_term * mean_reversion_loading
    return C, D


def _stable_discriminant(
    beta: ComplexArray,
    quadratic_term: ComplexArray,
    eta: float,
) -> ComplexArray:
    eta2 = eta * eta
    d: ComplexArray = np.asarray(
        np.sqrt(beta * beta + eta2 * quadratic_term),
        dtype=np.complex128,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        g: ComplexArray = np.asarray((beta - d) / (beta + d), dtype=np.complex128)

    flip_sign: BoolArray = (~np.isfinite(g)) | (np.abs(g) > 1.0)

    return np.asarray(np.where(flip_sign, -d, d), dtype=np.complex128)


def _heston_affine_coeffs(
    u: np.ndarray,
    tau: float,
    params: HestonParams,
    *,
    j: J,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stable Heston affine coefficients on a frequency grid.

    Parameters
    ----------
    u : ndarray
        Complex frequency grid, usually already normalized to a flat array.
    tau : float
        Time to expiry in years. Must be finite and nonnegative.
    params : HestonParams
        Heston parameter set.
    j : {0, 1}
        Probability index used by the Lewis/Gatheral inversion formulas.

    Returns
    -------
    tuple of ndarray
        The affine coefficients ``(C, D)`` such that the transform factor is
        ``exp(C * vbar + D * v)``.

    Notes
    -----
    The implementation follows a numerically stable Gatheral-style branch
    selection via :func:`_stable_discriminant`.
    """
    tau = _validate_tau(tau)
    j = _validate_probability_index(j)

    if tau == 0.0 or params.eta == 0.0:  # TODO: Fallback if eta near 0.0
        return _deterministic_affine_coeffs(u, tau, params, j=j)

    eta2 = params.eta * params.eta
    quadratic_term = _quadratic_term(u, j=j)
    beta = params.kappa - params.rho * params.eta * (j + 1j * u)
    d = _stable_discriminant(beta, quadratic_term, params.eta)

    r_minus = -quadratic_term / (beta + d)
    g = -eta2 * quadratic_term / (beta + d) ** 2
    exp_neg_dt = np.exp(-d * tau)
    one_minus_exp_neg_dt = -np.expm1(-d * tau)
    one_minus_g_exp = 1.0 - g * exp_neg_dt

    D = r_minus * (one_minus_exp_neg_dt / one_minus_g_exp)
    C = params.kappa * (
        r_minus * tau - 2.0 * (np.log1p(-g * exp_neg_dt) - np.log1p(-g)) / eta2
    )
    return C, D


def HestonCharFn(
    u: complex | ArrayLike,
    tau: float,
    params: HestonParams,
    *,
    x: float = 0.0,
) -> complex | ComplexArray:
    """Evaluate the Heston characteristic function for log-forward returns.

    Parameters
    ----------
    u : float, complex, or ndarray
        Real or complex frequency, or a frequency grid.
    tau : float
        Time to expiry in years. Must be finite and nonnegative.
    params : HestonParams
        Heston parameter set.
    x : float, default 0.0
        Log-forward return shift added to the exponent. This is typically
        ``log(F_t / F_0)`` or ``log(F / K)`` depending on the calling formula.

    Returns
    -------
    complex or ndarray of complex128
        Characteristic-function value(s) with the same scalar/array shape as
        ``u``.

    Notes
    -----
    The exponent is written in affine Gatheral form,

    ``C(u, tau) * vbar + D(u, tau) * v + i * u * x``,

    using the parameter names from :class:`HestonParams`.
    """
    tau = _validate_tau(tau)
    if not np.isfinite(float(x)):
        raise ValueError("x must be finite.")

    u_arr, scalar_input, original_shape = _normalize_frequency_grid(u)
    C, D = _heston_affine_coeffs(u_arr, tau, params, j=0)
    values = np.exp(C * params.vbar + D * params.v + 1j * u_arr * x)
    return _restore_frequency_shape(
        values, scalar_input=scalar_input, original_shape=original_shape
    )
