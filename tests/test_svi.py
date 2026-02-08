from __future__ import annotations

import numpy as np
import pytest

from option_pricing.vol.svi import (
    DomainCheckConfig,
    SVIParams,
    SVITransformLeeCap,
    build_domain_grid,
    calibrate_svi,
    estimate_wing_slopes_one_sided,
    sigmoid,
    softplus,
    softplus_inv,
    svi_total_variance,
)


def _p_to_vec(p: SVIParams) -> np.ndarray:
    return np.asarray([p.a, p.b, p.rho, p.m, p.sigma], dtype=np.float64)


def test_sigmoid_basic_sanity() -> None:
    x = np.array([-1000.0, 0.0, 1000.0], dtype=np.float64)
    y = sigmoid(x)
    assert y.shape == x.shape
    assert float(y[0]) == pytest.approx(0.0, abs=1e-15)
    assert float(y[1]) == pytest.approx(0.5, abs=1e-15)
    assert float(y[2]) == pytest.approx(1.0, abs=1e-15)


def test_softplus_inverse_roundtrip() -> None:
    x = np.array([-8.0, -2.0, -0.5, 0.0, 0.5, 2.0, 8.0], dtype=np.float64)
    y = softplus(x)
    x2 = softplus_inv(y)
    np.testing.assert_allclose(x2, x, rtol=0.0, atol=1e-10)


def test_build_domain_grid_fixed_and_quantile_pad() -> None:
    y = np.array([-0.2, 0.0, 0.4], dtype=np.float64)

    cfg_fixed = DomainCheckConfig(
        mode="fixed_logmoneyness", y_min=-0.5, y_max=0.5, n_grid=11
    )
    dom, grid = build_domain_grid(y, cfg_fixed)
    assert dom == (-0.5, 0.5)
    assert grid.shape == (11,)
    assert float(grid[0]) == pytest.approx(-0.5)
    assert float(grid[-1]) == pytest.approx(0.5)

    # For small samples (<20), quantile_pad falls back to min/max and pads.
    cfg_q = DomainCheckConfig(
        mode="quantile_pad",
        q_lo=0.1,
        q_hi=0.9,
        pad_frac=0.10,
        pad_abs=0.05,
        n_grid=9,
    )
    dom2, grid2 = build_domain_grid(y, cfg_q)
    span = float(y.max() - y.min())
    pad = 0.10 * span + 0.05
    assert dom2[0] == pytest.approx(float(y.min() - pad))
    assert dom2[1] == pytest.approx(float(y.max() + pad))
    assert grid2.shape == (9,)


def test_estimate_wing_slopes_one_sided_piecewise_linear_v_shape() -> None:
    # V-shape: w = c + s*|y| => left slope is -s, right slope is +s
    y = np.linspace(-1.0, 1.0, 101, dtype=np.float64)
    s = 0.20
    w = 0.10 + s * np.abs(y)

    sL, sR = estimate_wing_slopes_one_sided(y=y, w=w, wing_threshold=0.30, q_tail=0.20)
    assert sL is not None and sR is not None
    assert float(sL) == pytest.approx(-s, rel=2e-2, abs=2e-3)
    assert float(sR) == pytest.approx(s, rel=2e-2, abs=2e-3)


def test_svi_transform_encode_decode_roundtrip_matches_total_variance() -> None:
    p = SVIParams(a=0.02, b=0.50, rho=-0.20, m=0.10, sigma=0.30)
    tr = SVITransformLeeCap(slope_cap=1.999)

    u = tr.encode(p)
    p2 = tr.decode(u)

    # Compare in function space (more robust than exact param equality).
    y = np.linspace(-1.0, 1.0, 51, dtype=np.float64)
    w1 = svi_total_variance(y, p)
    w2 = svi_total_variance(y, p2)
    np.testing.assert_allclose(w2, w1, rtol=0.0, atol=5e-10)


def test_svi_transform_dp_du_matches_finite_difference() -> None:
    tr = SVITransformLeeCap(slope_cap=1.999)

    # Pick a u in a safe region (not saturating sigmoid/softplus too hard)
    u = np.array([0.3, -0.1, 0.2, 0.05, -0.2], dtype=np.float64)
    p = tr.decode(u)

    J = tr.dp_du(u, p)  # (5,5)
    assert J.shape == (5, 5)

    eps = 1e-6
    fd = np.zeros((5, 5), dtype=np.float64)

    for j in range(5):
        up = u.copy()
        up[j] += eps
        um = u.copy()
        um[j] -= eps
        pp = tr.decode(up)
        pm = tr.decode(um)
        fd[:, j] = (_p_to_vec(pp) - _p_to_vec(pm)) / (2.0 * eps)

    np.testing.assert_allclose(J, fd, rtol=2e-4, atol=3e-5)


def test_calibrate_svi_fits_synthetic_smile_linear_no_regularization() -> None:
    rng = np.random.default_rng(0)

    y = np.linspace(-0.8, 0.8, 41, dtype=np.float64)
    p_true = SVIParams(a=0.02, b=0.50, rho=0.10, m=-0.05, sigma=0.30)
    w_obs = svi_total_variance(y, p_true)

    # Add tiny noise so we don't accidentally rely on exact parameter recovery.
    w_obs = w_obs + rng.normal(0.0, 1e-10, size=w_obs.shape)

    fit = calibrate_svi(
        y=y,
        w_obs=w_obs,
        loss="linear",
        reg_override={
            "lambda_m": 0.0,
            "lambda_inv_sigma": 0.0,
            "lambda_slope_L": 0.0,
            "lambda_slope_R": 0.0,
        },
        domain_check=DomainCheckConfig(
            mode="fixed_logmoneyness", y_min=-1.0, y_max=1.0, n_grid=51
        ),
    )

    assert fit.diag.ok, fit.diag.summary

    w_fit = svi_total_variance(y, fit.params)
    rmse = float(np.sqrt(np.mean((w_fit - w_obs) ** 2)))
    assert rmse < 5e-6
