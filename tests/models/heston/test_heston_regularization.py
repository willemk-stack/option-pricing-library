from __future__ import annotations

from collections.abc import Callable

import numpy as np

from option_pricing.models.heston.calibration.heston_types import HestonRegConfig
from option_pricing.models.heston.calibration.regulate import (
    heston_regularization_jacobian,
    heston_regularization_residuals,
)
from option_pricing.models.heston.params import HestonParams


def _params() -> HestonParams:
    return HestonParams(kappa=1.2, vbar=0.035, eta=0.35, rho=-0.45, v=0.04)


def _reg_config() -> HestonRegConfig:
    return HestonRegConfig(
        feller_penalty_weight=0.70,
        rho_boundary_weight=0.50,
        variance_level_weight=0.80,
        vol_of_vol_weight=1.10,
        rho_abs_soft_limit=0.40,
        vbar_soft_max=0.02,
        v0_soft_max=0.03,
        eta_soft_max=0.30,
    )


def _params_from_array(values: np.ndarray) -> HestonParams:
    return HestonParams(
        kappa=float(values[0]),
        vbar=float(values[1]),
        eta=float(values[2]),
        rho=float(values[3]),
        v=float(values[4]),
    )


def _central_diff_jac(
    fun: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    *,
    eps: float = 1.0e-6,
) -> np.ndarray:
    base = np.asarray(fun(x), dtype=np.float64)
    jac = np.empty((base.size, x.size), dtype=np.float64)
    for j in range(x.size):
        up = x.copy()
        down = x.copy()
        up[j] += eps
        down[j] -= eps
        jac[:, j] = (fun(up) - fun(down)) / (2.0 * eps)
    return jac


def test_heston_regularization_jacobian_matches_finite_difference() -> None:
    params = _params()
    cfg = _reg_config()

    def residual_from_values(values: np.ndarray) -> np.ndarray:
        return heston_regularization_residuals(_params_from_array(values), cfg)

    jac = heston_regularization_jacobian(params, cfg)
    jac_fd = _central_diff_jac(residual_from_values, params.as_array())

    assert jac.shape == (5, 5)
    np.testing.assert_allclose(jac, jac_fd, rtol=1.0e-6, atol=1.0e-8)


def test_heston_regularization_jacobian_matches_residual_row_count() -> None:
    params = _params()
    cfg = _reg_config()

    residual = heston_regularization_residuals(params, cfg)
    jac = heston_regularization_jacobian(params, cfg)

    assert residual.shape == (5,)
    assert jac.shape == (residual.size, 5)
