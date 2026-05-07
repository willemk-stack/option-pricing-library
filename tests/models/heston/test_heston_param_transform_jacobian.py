from __future__ import annotations

import numpy as np

from option_pricing.models.heston.params import HestonParams

RTOL = 5.0e-4
ATOL = 1.0e-6

SEED_PARAMS = HestonParams(
    kappa=1.2,
    vbar=0.035,
    eta=0.35,
    rho=-0.45,
    v=0.04,
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


def test_transform_jacobian_diag_matches_finite_difference() -> None:
    raw = SEED_PARAMS.transform_to_unconstrained()
    diag = HestonParams.transform_jac_diag_from_raw(raw)

    def transform(raw_vec: np.ndarray) -> np.ndarray:
        return HestonParams.transform_to_constrained(raw_vec).as_array()

    fd = central_diff_jac(transform, raw)

    assert fd.shape == (5, 5)
    np.testing.assert_allclose(np.diag(fd), diag, rtol=RTOL, atol=ATOL)

    off_diag = fd - np.diag(np.diag(fd))
    np.testing.assert_allclose(off_diag, 0.0, atol=1.0e-8)


def test_transform_jacobian_diag_shape_and_finiteness() -> None:
    raw = SEED_PARAMS.transform_to_unconstrained()
    diag = HestonParams.transform_jac_diag_from_raw(raw)

    assert diag.shape == (5,)
    assert np.all(np.isfinite(diag))
    assert np.all(diag > 0.0)


def test_heston_param_array_order_is_kappa_vbar_eta_rho_v() -> None:
    np.testing.assert_allclose(
        SEED_PARAMS.as_array(),
        np.array([1.2, 0.035, 0.35, -0.45, 0.04], dtype=float),
        rtol=0.0,
        atol=0.0,
    )
