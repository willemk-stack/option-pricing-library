from __future__ import annotations

import numpy as np
import pytest

from option_pricing.models.heston.fourier import (
    heston_probability,
    heston_probability_and_param_jac,
)
from option_pricing.models.heston.params import HestonParams
from option_pricing.numerics.quadrature import QuadratureConfig

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
    # REVIEW: Probability Jacobian tests currently cover gauss_legendre only
    # because analytic Jacobian support is limited to that backend.
    with pytest.raises(NotImplementedError):
        heston_probability_and_param_jac(
            x=0.05,
            tau=0.75,
            params=BASE_PARAMS,
            probability_index=0,
            backend="quad",
        )
