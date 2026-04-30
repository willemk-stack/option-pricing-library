from __future__ import annotations

import numpy as np
import pytest

from option_pricing.models.heston.charfunc import (
    HESTON_CHARFUNC_GRADIENT_PARAM_NAMES,
    _cui_char_fn_and_param_grad,
    heston_char_fn,
)
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
    phi, grad_phi = _cui_char_fn_and_param_grad(0.7, 0.75, _sample_params(), x=0.3)

    assert isinstance(phi, complex)
    assert grad_phi.shape == (5,)
    assert np.iscomplexobj(grad_phi)


def test_cui_charfunc_gradient_kernel_supports_frequency_vectors() -> None:
    params = _sample_params()
    tau = 0.75
    x = 0.3
    u = np.array([0.2, 0.7, 1.5, 3.0], dtype=np.complex128)

    phi, grad_phi = _cui_char_fn_and_param_grad(u, tau, params, x=x)
    expected_phi = heston_char_fn(u, tau, params, x=x)

    assert isinstance(phi, np.ndarray)
    assert phi.shape == u.shape
    assert grad_phi.shape == u.shape + (5,)
    np.testing.assert_allclose(phi, expected_phi, atol=1.0e-12, rtol=0.0)


def test_cui_charfunc_gradient_kernel_matches_finite_difference() -> None:
    params = _sample_params()
    tau = 0.75
    x = 0.3
    u = np.array([0.2, 0.7, 1.5, 3.0], dtype=np.complex128)
    base = params.as_array()
    step = 1.0e-6

    _, grad_phi = _cui_char_fn_and_param_grad(u, tau, params, x=x)
    grad_fd = np.empty_like(grad_phi)

    for j in range(base.size):
        plus = base.copy()
        minus = base.copy()
        plus[j] += step
        minus[j] -= step

        phi_plus, _ = _cui_char_fn_and_param_grad(u, tau, _params_from_array(plus), x=x)
        phi_minus, _ = _cui_char_fn_and_param_grad(
            u, tau, _params_from_array(minus), x=x
        )
        grad_fd[:, j] = (np.asarray(phi_plus) - np.asarray(phi_minus)) / (2.0 * step)

    # REVIEW: tighten tolerance after validating branch behavior against the existing pricing kernel.
    np.testing.assert_allclose(grad_phi, grad_fd, atol=1.0e-8, rtol=2.0e-5)


def test_cui_charfunc_gradient_kernel_rejects_zero_frequency() -> None:
    with pytest.raises(ValueError, match="nonzero u"):
        _cui_char_fn_and_param_grad(0.0, 0.75, _sample_params())


def test_cui_charfunc_gradient_kernel_rejects_zero_eta() -> None:
    params = HestonParams(kappa=1.6, vbar=0.04, eta=0.0, rho=-0.65, v=0.05)

    with pytest.raises(ValueError, match="eta"):
        _cui_char_fn_and_param_grad(1.0, 0.75, params)
