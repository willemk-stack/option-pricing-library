from __future__ import annotations

import numpy as np
import pytest

from option_pricing.models.heston.charfunc import HESTON_ANALYTIC_JAC_ETA_MIN
from option_pricing.models.heston.fourier import (
    _build_heston_gauss_rule,
    _default_heston_quadrature_config,
    _integrand_and_param_jac,
    _pj_affine_factor_and_param_jac,
    heston_probability,
    heston_probability_and_param_jac,
)
from option_pricing.models.heston.params import HestonParams
from option_pricing.numerics.quadrature import (
    QuadratureConfig,
    build_gauss_legendre_rule,
)

RTOL = 5.0e-4
ATOL = 1.0e-6

BASE_PARAMS = HestonParams(
    kappa=1.6,
    vbar=0.04,
    eta=0.45,
    rho=-0.65,
    v=0.05,
)
QUAD_CFG = QuadratureConfig(
    u_max=120.0,
    n_panels=8,
    nodes_per_panel=8,
)


def central_diff_jac(fun, x, *, eps=1.0e-5):
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(fun(x), dtype=float)
    jac = np.empty(f0.shape + x.shape, dtype=float)

    for j in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[j] += eps
        xm[j] -= eps
        jac[..., j] = (np.asarray(fun(xp)) - np.asarray(fun(xm))) / (2.0 * eps)

    return jac


def params_from_array(a: np.ndarray) -> HestonParams:
    return HestonParams(
        kappa=float(a[0]),
        vbar=float(a[1]),
        eta=float(a[2]),
        rho=float(a[3]),
        v=float(a[4]),
    )


@pytest.mark.parametrize("probability_index", [0, 1])
def test_probability_jac_shape_scalar_x(probability_index: int) -> None:
    prob, jac = heston_probability_and_param_jac(
        x=0.05,
        tau=0.75,
        params=BASE_PARAMS,
        probability_index=probability_index,
        backend="gauss_legendre",
        quad_cfg=QUAD_CFG,
    )

    assert np.ndim(prob) in (0, 1)
    assert jac.shape[-1] == 5
    assert np.all(np.isfinite(jac))


@pytest.mark.parametrize("probability_index", [0, 1])
def test_probability_jac_matches_finite_difference_scalar_x(
    probability_index: int,
) -> None:
    x = 0.05
    tau = 0.75

    def prob_from_theta(theta_array: np.ndarray) -> float:
        return heston_probability(
            x=x,
            tau=tau,
            params=params_from_array(theta_array),
            probability_index=probability_index,
            backend="gauss_legendre",
            quad_cfg=QUAD_CFG,
        )

    _, analytic_jac = heston_probability_and_param_jac(
        x=x,
        tau=tau,
        params=BASE_PARAMS,
        probability_index=probability_index,
        backend="gauss_legendre",
        quad_cfg=QUAD_CFG,
    )
    fd_jac = central_diff_jac(prob_from_theta, BASE_PARAMS.as_array())

    assert analytic_jac.shape == (5,)
    np.testing.assert_allclose(analytic_jac, fd_jac, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("probability_index", [0, 1])
def test_probability_jac_matches_finite_difference_vector_x(
    probability_index: int,
) -> None:
    x = np.array([-0.1, 0.0, 0.1], dtype=float)
    tau = 0.75

    def prob_from_theta(theta_array: np.ndarray) -> np.ndarray:
        return np.asarray(
            heston_probability(
                x=x,
                tau=tau,
                params=params_from_array(theta_array),
                probability_index=probability_index,
                backend="gauss_legendre",
                quad_cfg=QUAD_CFG,
            ),
            dtype=float,
        )

    prob, analytic_jac = heston_probability_and_param_jac(
        x=x,
        tau=tau,
        params=BASE_PARAMS,
        probability_index=probability_index,
        backend="gauss_legendre",
        quad_cfg=QUAD_CFG,
    )
    fd_jac = central_diff_jac(prob_from_theta, BASE_PARAMS.as_array())

    assert np.asarray(prob).shape == x.shape
    assert analytic_jac.shape == x.shape + (5,)
    assert np.all(np.isfinite(analytic_jac))
    np.testing.assert_allclose(analytic_jac, fd_jac, rtol=RTOL, atol=ATOL)


def test_probability_jac_quad_backend_is_not_implemented() -> None:
    with pytest.raises(
        NotImplementedError,
        match="Analytic Heston parameter Jacobians currently support only "
        "backend='gauss_legendre'",
    ):
        heston_probability_and_param_jac(
            x=0.05,
            tau=0.75,
            params=BASE_PARAMS,
            probability_index=0,
            backend="quad",
        )


def test_probability_jac_rejects_eta_below_analytic_floor() -> None:
    params = HestonParams(
        kappa=1.6,
        vbar=0.04,
        eta=0.5 * HESTON_ANALYTIC_JAC_ETA_MIN,
        rho=-0.65,
        v=0.05,
    )

    with pytest.raises(ValueError, match="probability Jacobians require eta"):
        heston_probability_and_param_jac(
            x=0.05,
            tau=0.75,
            params=params,
            probability_index=0,
            backend="gauss_legendre",
            quad_cfg=QUAD_CFG,
        )


def test_integrand_param_jac_rejects_zero_frequency() -> None:
    with pytest.raises(ValueError, match="nonzero frequencies"):
        _integrand_and_param_jac(
            u=0.0,
            x=0.05,
            tau=0.75,
            params=BASE_PARAMS,
            j=0,
        )


def test_integrand_param_jac_rejects_array_containing_zero_frequency() -> None:
    with pytest.raises(ValueError, match="nonzero frequencies"):
        _integrand_and_param_jac(
            u=np.array([0.1, 0.0, 1.0], dtype=float),
            x=0.05,
            tau=0.75,
            params=BASE_PARAMS,
            j=1,
        )


def test_integrand_param_jac_applies_phase_outside_affine_helper() -> None:
    u = np.array([0.25, 0.7, 1.3], dtype=float)
    x = np.array([-0.2, 0.15], dtype=float)
    tau = 0.75

    for probability_index in (0, 1):
        values, jac_values = _integrand_and_param_jac(
            u=u,
            x=x,
            tau=tau,
            params=BASE_PARAMS,
            j=probability_index,
        )
        affine, d_affine = _pj_affine_factor_and_param_jac(
            u,
            tau,
            BASE_PARAMS,
            j=probability_index,
        )
        affine_arr = np.asarray(affine, dtype=np.complex128).reshape(-1)
        d_affine_arr = np.asarray(d_affine, dtype=np.complex128).reshape(-1, 5)
        phase = np.exp(1j * x[:, None] * u[None, :])
        denom = 1j * u[None, :]

        expected_values = np.real(phase * affine_arr[None, :] / denom)
        expected_jac = np.real(
            phase[:, :, None] * d_affine_arr[None, :, :] / denom[:, :, None]
        )

        np.testing.assert_allclose(values, expected_values, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(jac_values, expected_jac, atol=0.0, rtol=0.0)


def test_heston_fixed_gauss_rule_nodes_are_strictly_positive() -> None:
    cfg = _default_heston_quadrature_config()
    generic_rule = build_gauss_legendre_rule(cfg)
    heston_rule = _build_heston_gauss_rule(cfg)

    assert np.all(generic_rule.u_flat > 0.0)
    assert np.all(generic_rule.u_panel > 0.0)
    assert np.all(heston_rule.u_flat > 0.0)
    assert np.all(heston_rule.u_panel > 0.0)
