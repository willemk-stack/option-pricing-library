from __future__ import annotations

import numpy as np
import pytest

from option_pricing.models.heston.charfunc import (
    HESTON_ANALYTIC_JAC_ETA_MIN,
    HESTON_CHARFUNC_GRADIENT_PARAM_NAMES,
    _cui_char_fn_and_param_grad,
)
from option_pricing.models.heston.fourier import _pj_affine_factor
from option_pricing.models.heston.params import HestonParams


def _sample_params() -> HestonParams:
    return HestonParams(kappa=1.6, vbar=0.04, eta=0.45, rho=-0.65, v=0.05)


def _params_from_array(values: np.ndarray) -> HestonParams:
    return HestonParams(
        kappa=float(values[0]),
        vbar=float(values[1]),
        eta=float(values[2]),
        rho=float(values[3]),
        v=float(values[4]),
    )


def test_cui_charfunc_gradient_order_is_repo_order() -> None:
    assert HESTON_CHARFUNC_GRADIENT_PARAM_NAMES == (
        "kappa",
        "vbar",
        "eta",
        "rho",
        "v",
    )


def test_cui_charfunc_gradient_kernel_supports_scalar_frequency() -> None:
    phi, grad_phi = _cui_char_fn_and_param_grad(0.7, 0.75, _sample_params())

    assert isinstance(phi, complex)
    assert grad_phi.shape == (5,)
    assert np.iscomplexobj(grad_phi)


def test_cui_charfunc_gradient_kernel_supports_frequency_vectors() -> None:
    params = _sample_params()
    tau = 0.75
    u = np.array([0.2, 0.7, 1.5, 3.0], dtype=np.complex128)

    phi, grad_phi = _cui_char_fn_and_param_grad(u, tau, params)
    expected_phi = _pj_affine_factor(u, tau, params, j=0)

    assert isinstance(phi, np.ndarray)
    assert phi.shape == u.shape
    assert grad_phi.shape == u.shape + (5,)
    np.testing.assert_allclose(phi, expected_phi, atol=1.0e-12, rtol=0.0)


def test_cui_charfunc_gradient_kernel_matches_finite_difference() -> None:
    params = _sample_params()
    tau = 0.75
    u = np.array([0.2, 0.7, 1.5, 3.0], dtype=np.complex128)
    base = params.as_array()
    step = 1.0e-6

    _, grad_phi = _cui_char_fn_and_param_grad(u, tau, params)
    grad_fd = np.empty_like(grad_phi)

    for j in range(base.size):
        plus = base.copy()
        minus = base.copy()
        plus[j] += step
        minus[j] -= step

        phi_plus, _ = _cui_char_fn_and_param_grad(u, tau, _params_from_array(plus))
        phi_minus, _ = _cui_char_fn_and_param_grad(u, tau, _params_from_array(minus))
        grad_fd[:, j] = (np.asarray(phi_plus) - np.asarray(phi_minus)) / (2.0 * step)

    np.testing.assert_allclose(grad_phi, grad_fd, atol=1.0e-8, rtol=2.0e-5)


def test_cui_charfunc_gradient_kernel_rejects_zero_frequency() -> None:
    with pytest.raises(ValueError, match="nonzero u"):
        _cui_char_fn_and_param_grad(0.0, 0.75, _sample_params())


def test_cui_charfunc_gradient_kernel_rejects_array_containing_zero() -> None:
    with pytest.raises(ValueError, match="nonzero u"):
        _cui_char_fn_and_param_grad(
            np.array([0.2, 0.0, 1.0], dtype=np.complex128),
            0.75,
            _sample_params(),
        )


def test_cui_charfunc_gradient_kernel_no_longer_accepts_x_phase() -> None:
    with pytest.raises(TypeError):
        _cui_char_fn_and_param_grad(0.7, 0.75, _sample_params(), x=0.3)


def test_cui_charfunc_gradient_kernel_rejects_eta_below_analytic_floor() -> None:
    params = HestonParams(
        kappa=1.6,
        vbar=0.04,
        eta=0.5 * HESTON_ANALYTIC_JAC_ETA_MIN,
        rho=-0.65,
        v=0.05,
    )

    with pytest.raises(ValueError, match="handled separately"):
        _cui_char_fn_and_param_grad(1.0, 0.75, params)


def test_cui_charfunc_gradient_kernel_accepts_eta_at_analytic_floor() -> None:
    params = HestonParams(
        kappa=1.6,
        vbar=0.04,
        eta=HESTON_ANALYTIC_JAC_ETA_MIN,
        rho=-0.65,
        v=0.05,
    )

    phi, grad = _cui_char_fn_and_param_grad(1.0, 0.75, params)

    assert np.isfinite(complex(phi).real)
    assert np.isfinite(complex(phi).imag)
    assert np.all(np.isfinite(np.asarray(grad).real))
    assert np.all(np.isfinite(np.asarray(grad).imag))


@pytest.mark.parametrize(
    ("name", "params", "tau"),
    [
        ("mild", HestonParams(kappa=1.6, vbar=0.04, eta=0.45, rho=-0.65, v=0.05), 0.75),
        (
            "high_eta",
            HestonParams(kappa=1.1, vbar=0.05, eta=1.40, rho=-0.75, v=0.06),
            1.50,
        ),
        (
            "rho_near_minus_one",
            HestonParams(kappa=2.0, vbar=0.04, eta=0.80, rho=-0.95, v=0.05),
            1.00,
        ),
        (
            "long_maturity",
            HestonParams(kappa=0.7, vbar=0.06, eta=0.60, rho=-0.50, v=0.04),
            5.00,
        ),
        (
            "short_maturity",
            HestonParams(kappa=2.5, vbar=0.03, eta=0.70, rho=-0.40, v=0.05),
            0.02,
        ),
        (
            "low_v",
            HestonParams(kappa=1.8, vbar=0.04, eta=0.50, rho=-0.60, v=1.0e-4),
            1.00,
        ),
        (
            "slow_kappa",
            HestonParams(kappa=0.15, vbar=0.04, eta=0.70, rho=-0.70, v=0.06),
            2.00,
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
@pytest.mark.parametrize("probability_index", [0, 1])
def test_cui_affine_factor_matches_stable_affine_path_on_stressed_grid(
    name: str,
    params: HestonParams,
    tau: float,
    probability_index: int,
) -> None:
    """Validate branch/log continuity against the production stable affine path.

    This checks equality of the zero-shift affine factor over dense and stressed
    real frequencies. It does not prove every complex branch trajectory or all
    parameter gradients are globally stable.
    """
    del name
    u = np.concatenate(
        [
            np.geomspace(1.0e-4, 1.0, 20),
            np.linspace(1.2, 80.0, 80),
        ]
    ).astype(np.complex128)
    gradient_frequency = u if probability_index == 0 else u - 1j

    cui_phi, _ = _cui_char_fn_and_param_grad(gradient_frequency, tau, params)
    stable_phi = _pj_affine_factor(u, tau, params, j=probability_index)

    assert np.all(np.isfinite(np.asarray(cui_phi).real))
    assert np.all(np.isfinite(np.asarray(cui_phi).imag))
    np.testing.assert_allclose(cui_phi, stable_phi, atol=2.0e-13, rtol=2.0e-12)


@pytest.mark.parametrize(
    ("name", "params", "tau"),
    [
        (
            "2023-01-10",
            HestonParams(
                kappa=0.070673914,
                vbar=1.0,
                eta=1.5966849,
                rho=-0.063910313,
                v=0.061253334,
            ),
            2.94247,
        ),
        (
            "2023-01-11",
            HestonParams(
                kappa=2.2732255,
                vbar=0.082174377,
                eta=1.9411882,
                rho=-0.27484765,
                v=0.062892414,
            ),
            2.93973,
        ),
        (
            "2023-01-17",
            HestonParams(
                kappa=0.068556876,
                vbar=0.99370511,
                eta=1.5320165,
                rho=-0.07918461,
                v=0.05172861,
            ),
            2.92329,
        ),
        (
            "2023-01-18",
            HestonParams(
                kappa=0.072233377,
                vbar=1.0,
                eta=1.644752,
                rho=-0.084357932,
                v=0.05661865,
            ),
            2.92055,
        ),
        (
            "2023-01-20",
            HestonParams(
                kappa=0.38742843,
                vbar=0.27180527,
                eta=2.2533346,
                rho=-0.075045179,
                v=0.058373906,
            ),
            1.99452,
        ),
        (
            "2023-01-25",
            HestonParams(
                kappa=0.075741994,
                vbar=1.0,
                eta=1.9439068,
                rho=-0.030359092,
                v=0.053000416,
            ),
            2.90137,
        ),
        (
            "2023-01-26",
            HestonParams(
                kappa=0.081887459,
                vbar=1.0,
                eta=2.2330063,
                rho=0.0011730618,
                v=0.053848472,
            ),
            1.97808,
        ),
    ],
)
def test_cui_gradient_is_finite_for_stage_b_authoritative_jacobian_regimes(
    name: str,
    params: HestonParams,
    tau: float,
) -> None:
    """Regression coverage for January Stage B Fourier-tail failures."""
    del name

    u = np.concatenate(
        [
            np.geomspace(1.0e-4, 1.0, 32),
            np.linspace(1.0, 400.0, 800),
        ]
    ).astype(np.complex128)

    with np.errstate(divide="raise", invalid="raise", over="raise"):
        phi, grad = _cui_char_fn_and_param_grad(u, tau, params)

    phi_array = np.asarray(phi, dtype=np.complex128)
    grad_array = np.asarray(grad, dtype=np.complex128)

    assert phi_array.shape == u.shape
    assert grad_array.shape == u.shape + (5,)
    assert np.all(np.isfinite(phi_array.real))
    assert np.all(np.isfinite(phi_array.imag))
    assert np.all(np.isfinite(grad_array.real))
    assert np.all(np.isfinite(grad_array.imag))
