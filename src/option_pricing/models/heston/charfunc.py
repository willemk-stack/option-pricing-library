"""Stable Heston characteristic-function building blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ...typing import ArrayLike
from .params import HestonParams

type ComplexArray = NDArray[np.complex128]
type BoolArray = NDArray[np.bool_]
type HestonProbabilityIndex = Literal[0, 1]
type J = HestonProbabilityIndex

HESTON_CHARFUNC_GRADIENT_PARAM_NAMES = ("kappa", "vbar", "eta", "rho", "v")
# Vol-of-vol threshold below which pricing uses deterministic variance.
HESTON_ETA_DETERMINISTIC_THRESHOLD = 1.0e-8


@dataclass(frozen=True, slots=True)
class _CuiStableTerms:
    u: ComplexArray
    xi: ComplexArray
    d: ComplexArray
    A1: ComplexArray
    A2: ComplexArray
    A: ComplexArray
    B: ComplexArray
    D: ComplexArray
    exp_neg_d_tau: ComplexArray


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


def _restore_frequency_param_shape(
    values: ComplexArray, *, scalar_input: bool, original_shape: tuple[int, ...]
) -> ComplexArray:
    if scalar_input:
        return np.asarray(values.reshape(1, -1)[0], dtype=np.complex128)
    return np.asarray(
        values.reshape(original_shape + (values.shape[-1],)), dtype=np.complex128
    )


def _validate_tau(tau: float) -> float:
    tau = float(tau)
    if not np.isfinite(tau):
        raise ValueError("tau must be finite.")
    if tau < 0.0:
        raise ValueError("tau must be nonnegative.")
    return tau


def _validate_probability_index(
    probability_index: HestonProbabilityIndex,
) -> HestonProbabilityIndex:
    if probability_index not in (0, 1):
        raise ValueError("j must be either 0 or 1.")
    return probability_index


def _quadratic_term(u: np.ndarray, *, j: HestonProbabilityIndex) -> ComplexArray:
    out: ComplexArray = u * u + 1j * (1 - 2 * j) * u
    return out


def _integrated_variance(params: HestonParams, tau: float) -> float:
    return float(
        params.vbar * tau
        + (params.v - params.vbar) * (1.0 - np.exp(-params.kappa * tau)) / params.kappa
    )


def _deterministic_affine_coeffs(
    u: np.ndarray,
    tau: float,
    params: HestonParams,
    *,
    j: HestonProbabilityIndex,
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
    j: HestonProbabilityIndex,
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
    selection via :func:`_stable_discriminant`. For ``abs(params.eta) <=
    HESTON_ETA_DETERMINISTIC_THRESHOLD``, pricing uses the deterministic
    variance limit because the stochastic-volatility formula is singular as
    vol-of-vol tends to zero.
    """
    tau = _validate_tau(tau)
    j = _validate_probability_index(j)

    if tau == 0.0 or abs(params.eta) <= HESTON_ETA_DETERMINISTIC_THRESHOLD:
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


def _cui_stable_terms(
    u: np.ndarray,
    tau: float,
    params: HestonParams,
) -> _CuiStableTerms:
    """Return the Cui stable characteristic-function building blocks."""
    tau = _validate_tau(tau)
    u_arr = np.asarray(u, dtype=np.complex128)

    kappa = float(params.kappa)
    eta = float(params.eta)
    rho = float(params.rho)
    iu = 1j * u_arr

    xi = kappa - eta * rho * iu
    quadratic_term = u_arr * u_arr + iu

    # REVIEW: complex sqrt/log branch behavior should be validated against existing stable implementation before using in production pricing.
    d = np.sqrt(xi * xi + eta * eta * quadratic_term)
    half_d_tau = 0.5 * d * tau
    A1 = quadratic_term * np.sinh(half_d_tau)
    A2 = d * np.cosh(half_d_tau) + xi * np.sinh(half_d_tau)
    A = A1 / A2
    B = d * np.exp(0.5 * kappa * tau) / A2
    exp_neg_d_tau = np.exp(-d * tau)
    D = (
        np.log(d)
        + 0.5 * (kappa - d) * tau
        - np.log(0.5 * (d + xi) + 0.5 * (d - xi) * exp_neg_d_tau)
    )

    return _CuiStableTerms(
        u=np.asarray(u_arr, dtype=np.complex128),
        xi=np.asarray(xi, dtype=np.complex128),
        d=np.asarray(d, dtype=np.complex128),
        A1=np.asarray(A1, dtype=np.complex128),
        A2=np.asarray(A2, dtype=np.complex128),
        A=np.asarray(A, dtype=np.complex128),
        B=np.asarray(B, dtype=np.complex128),
        D=np.asarray(D, dtype=np.complex128),
        exp_neg_d_tau=np.asarray(exp_neg_d_tau, dtype=np.complex128),
    )


def _cui_intermediate_derivatives(
    terms: _CuiStableTerms,
    tau: float,
    params: HestonParams,
) -> dict[str, ComplexArray]:
    """Return intermediate Cui derivatives used by the h-vector."""
    tau = _validate_tau(tau)
    eta = float(params.eta)
    rho = float(params.rho)
    u = terms.u
    iu = 1j * u
    quadratic_term = u * u + iu
    half_d_tau = 0.5 * terms.d * tau
    sinh_half_d_tau = np.sinh(half_d_tau)
    cosh_half_d_tau = np.cosh(half_d_tau)

    d_drho = -terms.xi * eta * iu / terms.d
    A2_drho = -(eta * iu * (2.0 + tau * terms.xi) / (2.0 * terms.d)) * (
        terms.xi * cosh_half_d_tau + terms.d * sinh_half_d_tau
    )
    B_drho = np.exp(0.5 * params.kappa * tau) * (
        d_drho / terms.A2 - terms.d * A2_drho / (terms.A2 * terms.A2)
    )
    A1_drho = (
        -iu * quadratic_term * tau * terms.xi * eta / (2.0 * terms.d)
    ) * cosh_half_d_tau
    A_drho = A1_drho / terms.A2 - terms.A * A2_drho / terms.A2

    A_dkappa = 1j / (eta * u) * A_drho
    B_dkappa = 1j / (eta * u) * B_drho + 0.5 * tau * terms.B

    d_deta = (rho / eta - 1.0 / terms.xi) * d_drho + eta * u * u / terms.d
    A1_deta = 0.5 * quadratic_term * tau * d_deta * cosh_half_d_tau
    A2_deta = (
        rho / eta * A2_drho
        - (2.0 + tau * terms.xi) / (iu * tau * terms.xi) * A1_drho
        + 0.5 * eta * tau * terms.A1
    )
    A_deta = A1_deta / terms.A2 - terms.A * A2_deta / terms.A2

    return {
        "d_drho": np.asarray(d_drho, dtype=np.complex128),
        "A2_drho": np.asarray(A2_drho, dtype=np.complex128),
        "B_drho": np.asarray(B_drho, dtype=np.complex128),
        "A1_drho": np.asarray(A1_drho, dtype=np.complex128),
        "A_drho": np.asarray(A_drho, dtype=np.complex128),
        "A_dkappa": np.asarray(A_dkappa, dtype=np.complex128),
        "B_dkappa": np.asarray(B_dkappa, dtype=np.complex128),
        "d_deta": np.asarray(d_deta, dtype=np.complex128),
        "A1_deta": np.asarray(A1_deta, dtype=np.complex128),
        "A2_deta": np.asarray(A2_deta, dtype=np.complex128),
        "A_deta": np.asarray(A_deta, dtype=np.complex128),
    }


def _cui_h_vector(
    terms: _CuiStableTerms,
    derivs: dict[str, ComplexArray],
    tau: float,
    params: HestonParams,
) -> ComplexArray:
    """Return h(u) in repo parameter order: kappa, vbar, eta, rho, v."""
    tau = _validate_tau(tau)
    kappa = float(params.kappa)
    vbar = float(params.vbar)
    eta = float(params.eta)
    rho = float(params.rho)
    v = float(params.v)
    eta2 = eta * eta
    eta3 = eta * eta2
    iu = 1j * terms.u

    h_kappa = (
        v / (eta * iu) * derivs["A_drho"]
        + 2.0 * vbar / eta2 * terms.D
        + 2.0 * kappa * vbar / (eta2 * terms.B) * derivs["B_dkappa"]
        - tau * vbar * rho * iu / eta
    )
    h_vbar = 2.0 * kappa / eta2 * terms.D - tau * kappa * rho * iu / eta
    h_eta = (
        -v * derivs["A_deta"]
        - 4.0 * kappa * vbar / eta3 * terms.D
        + 2.0
        * kappa
        * vbar
        / (eta2 * terms.d)
        * (derivs["d_deta"] - terms.d / terms.A2 * derivs["A2_deta"])
        + tau * kappa * vbar * rho * iu / eta2
    )
    h_rho = (
        -v * derivs["A_drho"]
        + 2.0
        * kappa
        * vbar
        / (eta2 * terms.d)
        * (derivs["d_drho"] - terms.d / terms.A2 * derivs["A2_drho"])
        - tau * kappa * vbar * iu / eta
    )
    h_v = -terms.A

    return np.asarray(
        np.stack([h_kappa, h_vbar, h_eta, h_rho, h_v], axis=-1),
        dtype=np.complex128,
    )


def _cui_char_fn_and_param_grad(
    u: complex | ArrayLike,
    tau: float,
    params: HestonParams,
) -> tuple[complex | ComplexArray, ComplexArray]:
    """Return the zero-shift Cui affine factor and parameter gradient.

    The Fourier integrand owns the log-moneyness phase ``exp(i u x)``. This
    helper returns only the affine transform factor in parameter order
    ``[kappa, vbar, eta, rho, v]``.
    """
    tau = _validate_tau(tau)

    # REVIEW: eta near zero is singular for these formulas; do not silently fallback without a separate design decision.
    if params.eta <= 0.0:
        raise ValueError("analytic Heston gradient requires eta to be positive.")

    u_arr, scalar_input, original_shape = _normalize_frequency_grid(u)

    if tau == 0.0:
        phi = np.ones(u_arr.shape, dtype=np.complex128)
        grad_phi = np.zeros(
            (u_arr.size, len(HESTON_CHARFUNC_GRADIENT_PARAM_NAMES)),
            dtype=np.complex128,
        )
        return (
            _restore_frequency_shape(
                phi, scalar_input=scalar_input, original_shape=original_shape
            ),
            _restore_frequency_param_shape(
                grad_phi, scalar_input=scalar_input, original_shape=original_shape
            ),
        )

    # The analytic formulas are singular at u=0; fixed production nodes exclude it.
    if np.any(u_arr == 0.0):
        raise ValueError("analytic Heston gradient formulas require nonzero u.")

    terms = _cui_stable_terms(u_arr, tau, params)
    derivs = _cui_intermediate_derivatives(terms, tau, params)
    h = _cui_h_vector(terms, derivs, tau, params)

    kappa = float(params.kappa)
    vbar = float(params.vbar)
    eta = float(params.eta)
    rho = float(params.rho)
    v = float(params.v)
    iu = 1j * terms.u

    phi = np.exp(
        -tau * kappa * vbar * rho * iu / eta
        - v * terms.A
        + 2.0 * kappa * vbar / (eta * eta) * terms.D
    )
    grad_phi = phi[:, None] * h

    return (
        _restore_frequency_shape(
            phi, scalar_input=scalar_input, original_shape=original_shape
        ),
        _restore_frequency_param_shape(
            grad_phi, scalar_input=scalar_input, original_shape=original_shape
        ),
    )


def heston_char_fn(
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


# Backward-compatible alias for earlier notebook-facing releases.
HestonCharFn = heston_char_fn
