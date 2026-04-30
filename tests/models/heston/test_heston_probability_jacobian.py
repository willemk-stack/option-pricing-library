from __future__ import annotations

import numpy as np
import pytest

from option_pricing.models.heston.fourier import (
    heston_probability,
    heston_probability_and_param_jac,
)
from option_pricing.models.heston.params import HestonParams
from option_pricing.numerics.quadrature import QuadratureConfig


def _safe_params() -> HestonParams:
    return HestonParams(
        kappa=1.6,
        vbar=0.04,
        eta=0.45,
        rho=-0.65,
        v=0.05,
    )


def _quad_cfg() -> QuadratureConfig:
    return QuadratureConfig(
        u_max=100.0,
        n_panels=12,
        nodes_per_panel=12,
    )


def _params_from_array(values: np.ndarray) -> HestonParams:
    return HestonParams(
        kappa=float(values[0]),
        vbar=float(values[1]),
        eta=float(values[2]),
        rho=float(values[3]),
        v=float(values[4]),
    )


def test_heston_probability_param_jac_scalar_shape() -> None:
    p, jac = heston_probability_and_param_jac(
        x=0.05,
        tau=1.0,
        params=_safe_params(),
        probability_index=0,
        backend="gauss_legendre",
        quad_cfg=_quad_cfg(),
    )

    assert isinstance(p, float)
    assert jac.shape == (5,)


def test_heston_probability_param_jac_vector_shape() -> None:
    x = np.array([-0.1, 0.0, 0.1], dtype=np.float64)

    p, jac = heston_probability_and_param_jac(
        x=x,
        tau=1.0,
        params=_safe_params(),
        probability_index=0,
        backend="gauss_legendre",
        quad_cfg=_quad_cfg(),
    )

    assert p.shape == x.shape
    assert jac.shape == x.shape + (5,)


def test_heston_probability_param_jac_probability_matches_existing_impl() -> None:
    p_old = heston_probability(
        x=0.05,
        tau=1.0,
        params=_safe_params(),
        probability_index=0,
        backend="gauss_legendre",
        quad_cfg=_quad_cfg(),
    )

    p_new, _ = heston_probability_and_param_jac(
        x=0.05,
        tau=1.0,
        params=_safe_params(),
        probability_index=0,
        backend="gauss_legendre",
        quad_cfg=_quad_cfg(),
    )

    np.testing.assert_allclose(p_new, p_old, rtol=1.0e-10, atol=1.0e-12)


def test_heston_probability_param_jac_matches_finite_difference() -> None:
    params = _safe_params()
    quad_cfg = _quad_cfg()
    base = params.as_array()
    steps = np.array([1.0e-5, 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6], dtype=np.float64)

    for probability_index in [0, 1]:
        _, jac = heston_probability_and_param_jac(
            x=0.05,
            tau=1.0,
            params=params,
            probability_index=probability_index,
            backend="gauss_legendre",
            quad_cfg=quad_cfg,
        )
        jac_fd = np.empty_like(jac)

        for col, step in enumerate(steps):
            plus = base.copy()
            minus = base.copy()
            plus[col] += step
            minus[col] -= step

            p_plus = heston_probability(
                x=0.05,
                tau=1.0,
                params=_params_from_array(plus),
                probability_index=probability_index,
                backend="gauss_legendre",
                quad_cfg=quad_cfg,
            )
            p_minus = heston_probability(
                x=0.05,
                tau=1.0,
                params=_params_from_array(minus),
                probability_index=probability_index,
                backend="gauss_legendre",
                quad_cfg=quad_cfg,
            )
            jac_fd[col] = (p_plus - p_minus) / (2.0 * step)

        # REVIEW: tighten tolerance after branch behavior is validated end-to-end against the pricing kernel.
        np.testing.assert_allclose(jac, jac_fd, rtol=5.0e-4, atol=1.0e-6)


def test_heston_probability_param_jac_quad_backend_guard() -> None:
    with pytest.raises(NotImplementedError):
        heston_probability_and_param_jac(
            x=0.05,
            tau=1.0,
            params=_safe_params(),
            probability_index=0,
            backend="quad",
        )
