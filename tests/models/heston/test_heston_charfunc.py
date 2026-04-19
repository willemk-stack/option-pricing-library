from __future__ import annotations

import math

import numpy as np
import pytest

from option_pricing.models.heston.charfunc import HestonCharFn, _heston_affine_coeffs
from option_pricing.models.heston.fourier import _integrand, _pj_affine_factor
from option_pricing.models.heston.params import HestonParams


def _sample_params() -> HestonParams:
    return HestonParams(kappa=1.7, vbar=0.04, eta=0.55, rho=-0.65, v=0.05)


def test_heston_params_round_trip_unconstrained_transform() -> None:
    params = _sample_params()

    raw = params.TransformToUnconstrained()
    rebuilt = HestonParams.TransformToConstrained(raw)

    assert np.allclose(rebuilt.as_array(), params.as_array(), atol=1e-12, rtol=0.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kappa", 0.0),
        ("vbar", -1e-6),
        ("eta", -1e-6),
        ("rho", 1.1),
        ("v", -1e-6),
    ],
)
def test_heston_params_validate_inputs(field: str, value: float) -> None:
    base = dict(kappa=1.5, vbar=0.04, eta=0.5, rho=-0.6, v=0.03)
    base[field] = value

    with pytest.raises(ValueError):
        HestonParams(**base)


def test_heston_charfunc_is_one_at_zero_frequency() -> None:
    params = _sample_params()

    value = HestonCharFn(0.0, 1.25, params)

    assert isinstance(value, complex)
    assert abs(value - (1.0 + 0.0j)) < 1e-12


def test_heston_charfunc_tau_zero_returns_initial_phase() -> None:
    params = _sample_params()
    u = 0.75
    x = 0.2

    value = HestonCharFn(u, 0.0, params, x=x)

    assert abs(value - np.exp(1j * u * x)) < 1e-12


def test_heston_charfunc_supports_frequency_vectors() -> None:
    params = _sample_params()
    u = np.array([0.0, 0.25, 1.0, 2.0], dtype=np.float64)

    values = HestonCharFn(u, 0.75, params)

    assert isinstance(values, np.ndarray)
    assert values.shape == u.shape
    assert np.iscomplexobj(values)
    assert abs(values[0] - (1.0 + 0.0j)) < 1e-12


def test_heston_charfunc_supports_complex_frequency_vectors() -> None:
    params = _sample_params()
    tau = 0.75
    x = 0.3
    u = np.array([0.25 + 0.1j, 1.0 - 0.25j, 2.0 + 0.5j], dtype=np.complex128)

    C, D = _heston_affine_coeffs(u, tau, params, j=0)
    expected = np.exp(C * params.vbar + D * params.v + 1j * u * x)

    values = HestonCharFn(u, tau, params, x=x)

    assert np.allclose(values, expected, atol=1e-12, rtol=0.0)


def test_heston_charfunc_is_built_from_affine_coefficients_and_phase() -> None:
    params = _sample_params()
    tau = 0.75
    x = 0.3
    u = np.array([0.25, 1.0, 2.0], dtype=np.float64)

    C, D = _heston_affine_coeffs(np.asarray(u, dtype=np.complex128), tau, params, j=0)
    expected = np.exp(C * params.vbar + D * params.v + 1j * u * x)

    values = HestonCharFn(u, tau, params, x=x)

    assert np.allclose(values, expected, atol=1e-12, rtol=0.0)


def test_heston_charfunc_matches_deterministic_variance_case_when_eta_is_zero() -> None:
    params = HestonParams(kappa=2.0, vbar=0.04, eta=0.0, rho=-0.3, v=0.09)
    tau = 1.4
    u = np.array([0.2, 0.8, 1.6], dtype=np.float64)

    integrated_var = params.vbar * tau + (
        (params.v - params.vbar) * (1.0 - math.exp(-params.kappa * tau)) / params.kappa
    )
    expected = np.exp(-0.5 * (u * u + 1j * u) * integrated_var)

    values = HestonCharFn(u, tau, params)

    assert np.allclose(values, expected, atol=1e-12, rtol=0.0)


def test_p0_affine_factor_matches_characteristic_function_without_phase() -> None:
    params = _sample_params()
    u = np.array([0.25, 1.0, 2.0], dtype=np.float64)

    kernel = _pj_affine_factor(u, 0.9, params, j=0)
    values = HestonCharFn(u, 0.9, params)

    assert np.allclose(kernel, values, atol=1e-12, rtol=0.0)


def test_p1_affine_factor_matches_deterministic_variance_limit_when_eta_is_zero() -> (
    None
):
    params = HestonParams(kappa=2.0, vbar=0.04, eta=0.0, rho=-0.3, v=0.09)
    tau = 1.4
    u = np.array([0.2, 0.8, 1.6], dtype=np.float64)

    integrated_var = params.vbar * tau + (
        (params.v - params.vbar) * (1.0 - math.exp(-params.kappa * tau)) / params.kappa
    )
    expected = np.exp(-0.5 * (u * u - 1j * u) * integrated_var)

    values = _pj_affine_factor(u, tau, params, j=1)

    assert np.allclose(values, expected, atol=1e-12, rtol=0.0)


def test_p1_affine_factor_is_one_at_zero_frequency_even_when_b1_is_negative() -> None:
    params = HestonParams(kappa=1.0, vbar=0.04, eta=3.0, rho=0.9, v=0.05)

    value = _pj_affine_factor(0.0, 1.0, params, j=1)

    assert isinstance(value, complex)
    assert abs(value - (1.0 + 0.0j)) < 1e-12


def test_heston_integrand_uses_single_phase_factor() -> None:
    params = HestonParams(kappa=2.0, vbar=0.04, eta=0.0, rho=-0.3, v=0.09)
    tau = 1.1
    x = 0.35
    u = np.array([0.25, 1.0, 2.0], dtype=np.float64)

    integrated_var = params.vbar * tau + (
        (params.v - params.vbar) * (1.0 - math.exp(-params.kappa * tau)) / params.kappa
    )
    kernel = np.exp(-0.5 * (u * u + 1j * u) * integrated_var)
    expected = np.real(np.exp(1j * u * x) * kernel / (1j * u))

    values = _integrand(u, x, tau, params=params, j=0)

    assert np.allclose(values, expected, atol=1e-12, rtol=0.0)
