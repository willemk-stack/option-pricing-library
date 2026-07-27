"""Stable Heston characteristic-function building blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ...typing import ArrayLike
from .numerical import (
    HestonNumericalEvaluationError,
    numerical_error_from_floating_point,
    require_finite,
    require_nonzero,
    require_safe_exp_argument,
)
from .params import HestonParams

type ComplexArray = NDArray[np.complex128]
type BoolArray = NDArray[np.bool_]
type HestonProbabilityIndex = Literal[0, 1]
type J = HestonProbabilityIndex

HESTON_CHARFUNC_GRADIENT_PARAM_NAMES = ("kappa", "vbar", "eta", "rho", "v")
# Vol-of-vol threshold below which pricing uses deterministic variance.
HESTON_ETA_DETERMINISTIC_THRESHOLD = 1.0e-8
# Numerical floor for the Cui analytic-gradient formulas. Price-only Heston
# pricing has a deterministic-variance limit below this region; analytic
# parameter Jacobians do not.
HESTON_ANALYTIC_JAC_ETA_MIN = 1.0e-6


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
        + (params.v - params.vbar) * (-np.expm1(-params.kappa * tau)) / params.kappa
    )


def _deterministic_affine_coeffs(
    u: np.ndarray,
    tau: float,
    params: HestonParams,
    *,
    j: HestonProbabilityIndex,
) -> tuple[np.ndarray, np.ndarray]:
    quadratic_term = _quadratic_term(u, j=j)
    mean_reversion_loading = -np.expm1(-params.kappa * tau) / params.kappa

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

    denominator = beta + d
    singular = denominator == 0
    g = np.zeros_like(d, dtype=np.complex128)
    np.divide(beta - d, denominator, out=g, where=~singular)
    flip_sign: BoolArray = singular | (~np.isfinite(g)) | (np.abs(g) > 1.0)

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
    The production price path follows a numerically stable Gatheral-style
    branch selection via :func:`_stable_discriminant`. For ``abs(params.eta) <=
    HESTON_ETA_DETERMINISTIC_THRESHOLD``, price-only evaluation uses the
    deterministic variance limit because the stochastic-volatility formula is
    singular as vol-of-vol tends to zero.

    Analytic parameter gradients are computed separately with a Cui-style
    expression and are supported only on the production quadrature domain:
    nonzero real fixed-rule frequencies, positive maturities, eta at least
    ``HESTON_ANALYTIC_JAC_ETA_MIN``, and bounded calibration parameters. No
    global complex-plane branch-continuity guarantee is claimed.
    """
    tau = _validate_tau(tau)
    j = _validate_probability_index(j)

    if tau == 0.0 or abs(params.eta) <= HESTON_ETA_DETERMINISTIC_THRESHOLD:
        return _deterministic_affine_coeffs(u, tau, params, j=j)

    parameter_vector = params.as_array()
    stage = "stable_affine_coefficients"
    try:
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            eta2 = params.eta * params.eta
            require_nonzero(
                eta2,
                evaluation_stage=stage,
                parameter_vector=parameter_vector,
                maturity=tau,
                probability_index=j,
                failing_expression="eta**2",
            )
            quadratic_term = _quadratic_term(u, j=j)
            beta = params.kappa - params.rho * params.eta * (j + 1j * u)
            d = _stable_discriminant(beta, quadratic_term, params.eta)

            beta_plus_d = beta + d
            require_nonzero(
                beta_plus_d,
                evaluation_stage=stage,
                parameter_vector=parameter_vector,
                maturity=tau,
                probability_index=j,
                failing_expression="beta + d",
            )
            r_minus = -quadratic_term / beta_plus_d
            g = -eta2 * quadratic_term / beta_plus_d**2
            exp_argument = -d * tau
            require_safe_exp_argument(
                exp_argument,
                evaluation_stage=stage,
                parameter_vector=parameter_vector,
                maturity=tau,
                probability_index=j,
                failing_expression="-d * tau",
            )
            exp_neg_dt = np.exp(exp_argument)
            one_minus_exp_neg_dt = -np.expm1(exp_argument)
            one_minus_g_exp = 1.0 - g * exp_neg_dt
            require_nonzero(
                one_minus_g_exp,
                evaluation_stage=stage,
                parameter_vector=parameter_vector,
                maturity=tau,
                probability_index=j,
                failing_expression="one_minus_g_exp",
            )

            D = r_minus * (one_minus_exp_neg_dt / one_minus_g_exp)
            C = params.kappa * (
                r_minus * tau - 2.0 * (np.log1p(-g * exp_neg_dt) - np.log1p(-g)) / eta2
            )
            require_finite(
                C,
                evaluation_stage=stage,
                parameter_vector=parameter_vector,
                maturity=tau,
                probability_index=j,
                failing_expression="C",
            )
            require_finite(
                D,
                evaluation_stage=stage,
                parameter_vector=parameter_vector,
                maturity=tau,
                probability_index=j,
                failing_expression="D",
            )
            return C, D
    except HestonNumericalEvaluationError:
        raise
    except FloatingPointError as exc:
        raise numerical_error_from_floating_point(
            exc,
            evaluation_stage=stage,
            parameter_vector=parameter_vector,
            maturity=tau,
            probability_index=j,
            failing_expression="stable affine expression",
        ) from exc


def _cui_stable_terms(
    u: np.ndarray,
    tau: float,
    params: HestonParams,
) -> _CuiStableTerms:
    """Return Cui-style characteristic-function gradient building blocks.

    These expressions back the analytic parameter-Jacobian path. They are
    regression-tested against the production stable affine pricing path on
    stressed real-frequency grids used by fixed-rule calibration. They are not
    a general-purpose complex-plane branch-continuity contract.
    """
    tau = _validate_tau(tau)
    u_arr = np.asarray(u, dtype=np.complex128)

    parameter_vector = params.as_array()
    stage = "cui_stable_terms"
    kappa = float(params.kappa)
    eta = float(params.eta)
    rho = float(params.rho)
    iu = 1j * u_arr

    xi = kappa - eta * rho * iu
    quadratic_term = u_arr * u_arr + iu

    # Scale sinh(d*tau/2), cosh(d*tau/2), A1, and A2 by exp(-d*tau/2).
    # Their ratios are unchanged, while the exponentially growing hyperbolic
    # terms disappear. In particular:
    #
    #   A2_scaled = 0.5 * ((d + xi) + (d - xi) * exp(-d*tau))
    #   B = d * exp((kappa - d)*tau/2) / A2_scaled
    #
    # This is algebraically equivalent to the Cui expressions and avoids the
    # former exp(kappa*tau/2) / cosh(d*tau/2) overflow pair.
    require_finite(
        xi,
        evaluation_stage=stage,
        parameter_vector=parameter_vector,
        maturity=tau,
        probability_index=None,
        failing_expression="xi",
    )
    eta2 = eta * eta
    require_nonzero(
        eta2,
        evaluation_stage=stage,
        parameter_vector=parameter_vector,
        maturity=tau,
        probability_index=None,
        failing_expression="eta**2",
    )
    d = np.sqrt(xi * xi + eta2 * quadratic_term)
    require_nonzero(
        d,
        evaluation_stage=stage,
        parameter_vector=parameter_vector,
        maturity=tau,
        probability_index=None,
        failing_expression="d",
    )
    exp_neg_d_tau_argument = -d * tau
    require_safe_exp_argument(
        exp_neg_d_tau_argument,
        evaluation_stage=stage,
        parameter_vector=parameter_vector,
        maturity=tau,
        probability_index=None,
        failing_expression="-d * tau",
    )
    exp_neg_d_tau = np.exp(exp_neg_d_tau_argument)
    scaled_sinh_half_d_tau = -0.5 * np.expm1(exp_neg_d_tau_argument)
    scaled_cosh_half_d_tau = 0.5 * (1.0 + exp_neg_d_tau)
    A1 = quadratic_term * scaled_sinh_half_d_tau
    A2 = d * scaled_cosh_half_d_tau + xi * scaled_sinh_half_d_tau
    require_nonzero(
        A2,
        evaluation_stage=stage,
        parameter_vector=parameter_vector,
        maturity=tau,
        probability_index=None,
        failing_expression="A2_scaled",
    )
    A = A1 / A2
    B_exp_argument = 0.5 * (kappa - d) * tau
    require_safe_exp_argument(
        B_exp_argument,
        evaluation_stage=stage,
        parameter_vector=parameter_vector,
        maturity=tau,
        probability_index=None,
        failing_expression="0.5 * (kappa - d) * tau",
    )
    B = d * np.exp(B_exp_argument) / A2
    require_nonzero(
        B,
        evaluation_stage=stage,
        parameter_vector=parameter_vector,
        maturity=tau,
        probability_index=None,
        failing_expression="B",
    )
    D = np.log(d) + B_exp_argument - np.log(A2)
    for name, value in (("A", A), ("D", D), ("exp(-d*tau)", exp_neg_d_tau)):
        require_finite(
            value,
            evaluation_stage=stage,
            parameter_vector=parameter_vector,
            maturity=tau,
            probability_index=None,
            failing_expression=name,
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
    parameter_vector = params.as_array()
    stage = "cui_intermediate_derivatives"
    eta_u = eta * u
    require_nonzero(
        eta_u,
        evaluation_stage=stage,
        parameter_vector=parameter_vector,
        maturity=tau,
        probability_index=None,
        failing_expression="eta * u",
    )
    require_nonzero(
        terms.xi,
        evaluation_stage=stage,
        parameter_vector=parameter_vector,
        maturity=tau,
        probability_index=None,
        failing_expression="xi",
    )
    require_nonzero(
        terms.d,
        evaluation_stage=stage,
        parameter_vector=parameter_vector,
        maturity=tau,
        probability_index=None,
        failing_expression="d",
    )
    require_nonzero(
        terms.A2,
        evaluation_stage=stage,
        parameter_vector=parameter_vector,
        maturity=tau,
        probability_index=None,
        failing_expression="A2_scaled",
    )
    scaled_sinh_half_d_tau = 0.5 * (1.0 - terms.exp_neg_d_tau)
    scaled_cosh_half_d_tau = 0.5 * (1.0 + terms.exp_neg_d_tau)

    d_drho = -terms.xi * eta * iu / terms.d
    A2_drho = -(eta * iu * (2.0 + tau * terms.xi) / (2.0 * terms.d)) * (
        terms.xi * scaled_cosh_half_d_tau + terms.d * scaled_sinh_half_d_tau
    )
    B_exp_argument = 0.5 * (params.kappa - terms.d) * tau
    require_safe_exp_argument(
        B_exp_argument,
        evaluation_stage=stage,
        parameter_vector=parameter_vector,
        maturity=tau,
        probability_index=None,
        failing_expression="0.5 * (kappa - d) * tau",
    )
    B_drho = np.exp(B_exp_argument) * (
        d_drho / terms.A2 - terms.d * A2_drho / (terms.A2 * terms.A2)
    )
    A1_drho = (
        -iu * quadratic_term * tau * terms.xi * eta / (2.0 * terms.d)
    ) * scaled_cosh_half_d_tau
    A_drho = A1_drho / terms.A2 - terms.A * A2_drho / terms.A2

    A_dkappa = 1j / (eta * u) * A_drho
    B_dkappa = 1j / (eta * u) * B_drho + 0.5 * tau * terms.B

    d_deta = (rho / eta - 1.0 / terms.xi) * d_drho + eta * u * u / terms.d
    A1_deta = 0.5 * quadratic_term * tau * d_deta * scaled_cosh_half_d_tau
    A2_deta = (
        rho / eta * A2_drho
        - (2.0 + tau * terms.xi) / (iu * tau * terms.xi) * A1_drho
        + 0.5 * eta * tau * terms.A1
    )
    A_deta = A1_deta / terms.A2 - terms.A * A2_deta / terms.A2
    derivatives = {
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
    for name, value in derivatives.items():
        require_finite(
            value,
            evaluation_stage=stage,
            parameter_vector=parameter_vector,
            maturity=tau,
            probability_index=None,
            failing_expression=name,
        )
    return derivatives


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
    parameter_vector = params.as_array()
    stage = "cui_h_vector"
    for name, value in (
        ("eta * u", eta * terms.u),
        ("eta**2", eta2),
        ("eta**3", eta3),
        ("d", terms.d),
        ("A2_scaled", terms.A2),
        ("B", terms.B),
    ):
        require_nonzero(
            value,
            evaluation_stage=stage,
            parameter_vector=parameter_vector,
            maturity=tau,
            probability_index=None,
            failing_expression=name,
        )

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

    h = np.asarray(
        np.stack([h_kappa, h_vbar, h_eta, h_rho, h_v], axis=-1),
        dtype=np.complex128,
    )
    require_finite(
        h,
        evaluation_stage=stage,
        parameter_vector=parameter_vector,
        maturity=tau,
        probability_index=None,
        failing_expression="h",
    )
    return h


def _cui_char_fn_and_param_grad(
    u: complex | ArrayLike,
    tau: float,
    params: HestonParams,
) -> tuple[complex | ComplexArray, ComplexArray]:
    """Return the zero-shift Cui affine factor and parameter gradient.

    The Fourier integrand owns the log-moneyness phase ``exp(i u x)``. This
    helper returns only the affine transform factor in parameter order
    ``[kappa, vbar, eta, rho, v]``.

    The ordinary price path uses the stable Gatheral-style affine
    implementation in :func:`_heston_affine_coeffs`. This helper uses a
    Cui-style expression for analytic gradients and is supported only for the
    production fixed Gauss-Legendre quadrature domain: nonzero real quadrature
    nodes, positive maturity, eta at least ``HESTON_ANALYTIC_JAC_ETA_MIN``, and
    bounded calibration parameter ranges. It does not claim global complex
    branch-continuity safety.
    """
    tau = _validate_tau(tau)

    if params.eta < HESTON_ANALYTIC_JAC_ETA_MIN:
        raise ValueError(
            "Analytic Heston gradients require eta >= "
            f"{HESTON_ANALYTIC_JAC_ETA_MIN:g}. Price-only deterministic-limit "
            "pricing near eta=0 is handled separately and is not an analytic "
            "Jacobian validation path."
        )

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

    parameter_vector = params.as_array()
    stage = "cui_characteristic_function_and_jacobian"
    try:
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            terms = _cui_stable_terms(u_arr, tau, params)
            derivs = _cui_intermediate_derivatives(terms, tau, params)
            h = _cui_h_vector(terms, derivs, tau, params)

            kappa = float(params.kappa)
            vbar = float(params.vbar)
            eta = float(params.eta)
            rho = float(params.rho)
            v = float(params.v)
            iu = 1j * terms.u

            characteristic_exponent = (
                -tau * kappa * vbar * rho * iu / eta
                - v * terms.A
                + 2.0 * kappa * vbar / (eta * eta) * terms.D
            )
            require_safe_exp_argument(
                characteristic_exponent,
                evaluation_stage=stage,
                parameter_vector=parameter_vector,
                maturity=tau,
                probability_index=None,
                failing_expression="characteristic_exponent",
            )
            phi = np.exp(characteristic_exponent)
            grad_phi = phi[:, None] * h
            require_finite(
                phi,
                evaluation_stage=stage,
                parameter_vector=parameter_vector,
                maturity=tau,
                probability_index=None,
                failing_expression="phi",
            )
            require_finite(
                grad_phi,
                evaluation_stage=stage,
                parameter_vector=parameter_vector,
                maturity=tau,
                probability_index=None,
                failing_expression="phi * h",
            )

            return (
                _restore_frequency_shape(
                    phi, scalar_input=scalar_input, original_shape=original_shape
                ),
                _restore_frequency_param_shape(
                    grad_phi,
                    scalar_input=scalar_input,
                    original_shape=original_shape,
                ),
            )
    except HestonNumericalEvaluationError:
        raise
    except FloatingPointError as exc:
        raise numerical_error_from_floating_point(
            exc,
            evaluation_stage=stage,
            parameter_vector=parameter_vector,
            maturity=tau,
            failing_expression="Cui characteristic-function/Jacobian expression",
        ) from exc


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

    using the parameter names from :class:`HestonParams`. Analytic parameter
    gradients use the separate Cui-style helper
    :func:`_cui_char_fn_and_param_grad`; that helper is supported on the
    production quadrature domain only and does not claim global complex
    branch-continuity safety.
    """
    tau = _validate_tau(tau)
    if not np.isfinite(float(x)):
        raise ValueError("x must be finite.")

    u_arr, scalar_input, original_shape = _normalize_frequency_grid(u)
    parameter_vector = params.as_array()
    stage = "characteristic_function"
    try:
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            C, D = _heston_affine_coeffs(u_arr, tau, params, j=0)
            characteristic_exponent = C * params.vbar + D * params.v + 1j * u_arr * x
            require_safe_exp_argument(
                characteristic_exponent,
                evaluation_stage=stage,
                parameter_vector=parameter_vector,
                maturity=tau,
                probability_index=0,
                failing_expression="characteristic_exponent",
            )
            values = np.exp(characteristic_exponent)
            require_finite(
                values,
                evaluation_stage=stage,
                parameter_vector=parameter_vector,
                maturity=tau,
                probability_index=0,
                failing_expression="characteristic_function",
            )
            return _restore_frequency_shape(
                values, scalar_input=scalar_input, original_shape=original_shape
            )
    except HestonNumericalEvaluationError:
        raise
    except FloatingPointError as exc:
        raise numerical_error_from_floating_point(
            exc,
            evaluation_stage=stage,
            parameter_vector=parameter_vector,
            maturity=tau,
            probability_index=0,
            failing_expression="characteristic_function",
        ) from exc


# Backward-compatible alias for earlier notebook-facing releases.
HestonCharFn = heston_char_fn
