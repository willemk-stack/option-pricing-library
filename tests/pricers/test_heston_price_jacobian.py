from __future__ import annotations

import numpy as np

from option_pricing.models.heston import HestonParams
from option_pricing.numerics.quadrature import QuadratureConfig
from option_pricing.pricers.heston import (
    heston_price_call_and_param_jac_from_ctx,
    heston_price_call_from_ctx,
    heston_price_put_and_param_jac_from_ctx,
    heston_price_put_from_ctx,
)
from option_pricing.types import MarketData

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


def test_call_price_jac_shape_and_finiteness() -> None:
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.01)
    ctx = market.to_context()
    tau = 0.75
    strikes = np.array([85.0, 100.0, 115.0], dtype=float)

    prices, jac = heston_price_call_and_param_jac_from_ctx(
        strike=strikes,
        tau=tau,
        ctx=ctx,
        params=BASE_PARAMS,
        backend="gauss_legendre",
        quad_cfg=QUAD_CFG,
    )

    assert np.asarray(prices).shape == strikes.shape
    assert jac.shape == strikes.shape + (5,)
    assert np.all(np.isfinite(prices))
    assert np.all(np.isfinite(jac))


def test_call_price_jac_matches_finite_difference() -> None:
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.01)
    ctx = market.to_context()
    tau = 0.75
    strikes = np.array([85.0, 100.0, 115.0], dtype=float)

    def call_prices_from_theta(theta_array: np.ndarray) -> np.ndarray:
        return np.asarray(
            heston_price_call_from_ctx(
                strike=strikes,
                tau=tau,
                ctx=ctx,
                params=params_from_array(theta_array),
                backend="gauss_legendre",
                quad_cfg=QUAD_CFG,
            ),
            dtype=float,
        )

    _, analytic_jac = heston_price_call_and_param_jac_from_ctx(
        strike=strikes,
        tau=tau,
        ctx=ctx,
        params=BASE_PARAMS,
        backend="gauss_legendre",
        quad_cfg=QUAD_CFG,
    )
    fd_jac = central_diff_jac(call_prices_from_theta, BASE_PARAMS.as_array())

    np.testing.assert_allclose(analytic_jac, fd_jac, rtol=RTOL, atol=ATOL)


def test_put_price_jac_matches_finite_difference() -> None:
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.01)
    ctx = market.to_context()
    tau = 0.75
    strikes = np.array([85.0, 100.0, 115.0], dtype=float)

    def put_prices_from_theta(theta_array: np.ndarray) -> np.ndarray:
        return np.asarray(
            heston_price_put_from_ctx(
                strike=strikes,
                tau=tau,
                ctx=ctx,
                params=params_from_array(theta_array),
                backend="gauss_legendre",
                quad_cfg=QUAD_CFG,
            ),
            dtype=float,
        )

    _, analytic_jac = heston_price_put_and_param_jac_from_ctx(
        strike=strikes,
        tau=tau,
        ctx=ctx,
        params=BASE_PARAMS,
        backend="gauss_legendre",
        quad_cfg=QUAD_CFG,
    )
    fd_jac = central_diff_jac(put_prices_from_theta, BASE_PARAMS.as_array())

    assert analytic_jac.shape == strikes.shape + (5,)
    assert np.all(np.isfinite(analytic_jac))
    np.testing.assert_allclose(analytic_jac, fd_jac, rtol=RTOL, atol=ATOL)


def test_call_and_put_param_jac_match_by_parity() -> None:
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.01)
    ctx = market.to_context()
    tau = 0.75
    strikes = np.array([85.0, 100.0, 115.0], dtype=float)

    _, call_jac = heston_price_call_and_param_jac_from_ctx(
        strike=strikes,
        tau=tau,
        ctx=ctx,
        params=BASE_PARAMS,
        backend="gauss_legendre",
        quad_cfg=QUAD_CFG,
    )
    _, put_jac = heston_price_put_and_param_jac_from_ctx(
        strike=strikes,
        tau=tau,
        ctx=ctx,
        params=BASE_PARAMS,
        backend="gauss_legendre",
        quad_cfg=QUAD_CFG,
    )

    np.testing.assert_allclose(call_jac, put_jac, rtol=RTOL, atol=ATOL)
