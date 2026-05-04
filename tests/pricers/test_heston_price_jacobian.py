from __future__ import annotations

import numpy as np
import pytest

from option_pricing.models.heston import HestonParams
from option_pricing.numerics.quadrature import QuadratureConfig
from option_pricing.pricers.heston import (
    HESTON_PROBABILITY_INDEX_F,
    HESTON_PROBABILITY_INDEX_K,
    heston_price_and_param_jac_from_ctx,
    heston_price_call_and_param_jac_from_ctx,
    heston_price_call_from_ctx,
    heston_price_put_and_param_jac_from_ctx,
    heston_price_put_from_ctx,
)
from option_pricing.types import MarketData, OptionType

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


def test_probability_leg_names_match_public_probability_indices() -> None:
    assert HESTON_PROBABILITY_INDEX_K == 0
    assert HESTON_PROBABILITY_INDEX_F == 1


def test_scalar_call_and_put_price_jac_shapes_are_public_contract() -> None:
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.01)
    ctx = market.to_context()
    tau = 0.75
    strike = 100.0

    call_price, call_jac = heston_price_call_and_param_jac_from_ctx(
        strike=strike,
        tau=tau,
        ctx=ctx,
        params=BASE_PARAMS,
        backend="gauss_legendre",
        quad_cfg=QUAD_CFG,
    )
    put_price, put_jac = heston_price_put_and_param_jac_from_ctx(
        strike=strike,
        tau=tau,
        ctx=ctx,
        params=BASE_PARAMS,
        backend="gauss_legendre",
        quad_cfg=QUAD_CFG,
    )

    assert np.ndim(call_price) == 0
    assert np.ndim(put_price) == 0
    assert call_jac.shape == (5,)
    assert put_jac.shape == (5,)
    assert np.all(np.isfinite(call_jac))
    assert np.all(np.isfinite(put_jac))


def test_1d_call_and_put_price_jac_shapes_are_public_contract() -> None:
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

    assert call_jac.shape == (3, 5)
    assert put_jac.shape == (3, 5)


def test_multidimensional_price_jac_shape_is_strike_shape_plus_parameter_axis() -> None:
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.01)
    ctx = market.to_context()
    tau = 0.75
    strikes = np.array([[85.0, 95.0], [105.0, 115.0]], dtype=float)

    prices, jac = heston_price_and_param_jac_from_ctx(
        kind=OptionType.CALL,
        strike=strikes,
        tau=tau,
        ctx=ctx,
        params=BASE_PARAMS,
        backend="gauss_legendre",
        quad_cfg=QUAD_CFG,
    )

    assert np.asarray(prices).shape == strikes.shape
    assert jac.shape == strikes.shape + (5,)


def test_price_jac_rejects_quad_backend_before_pricing() -> None:
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.01)
    ctx = market.to_context()

    with pytest.raises(
        NotImplementedError,
        match="Analytic Heston parameter Jacobians currently support only "
        "backend='gauss_legendre'",
    ):
        heston_price_call_and_param_jac_from_ctx(
            strike=100.0,
            tau=0.75,
            ctx=ctx,
            params=BASE_PARAMS,
            backend="quad",
        )

    with pytest.raises(
        NotImplementedError,
        match="Analytic Heston parameter Jacobians currently support only "
        "backend='gauss_legendre'",
    ):
        heston_price_put_and_param_jac_from_ctx(
            strike=100.0,
            tau=0.75,
            ctx=ctx,
            params=BASE_PARAMS,
            backend="quad",
        )

    with pytest.raises(
        NotImplementedError,
        match="Analytic Heston parameter Jacobians currently support only "
        "backend='gauss_legendre'",
    ):
        heston_price_and_param_jac_from_ctx(
            kind=OptionType.CALL,
            strike=100.0,
            tau=0.75,
            ctx=ctx,
            params=BASE_PARAMS,
            backend="quad",
        )


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


def test_call_and_put_market_rate_sensitivities_are_not_param_jac_invariant() -> None:
    tau = 0.75
    strike = 100.0
    dr = 1.0e-4

    def call_price_at_rate(rate: float) -> float:
        return float(
            heston_price_call_from_ctx(
                strike=strike,
                tau=tau,
                ctx=MarketData(
                    spot=100.0,
                    rate=float(rate),
                    dividend_yield=0.01,
                ).to_context(),
                params=BASE_PARAMS,
                backend="gauss_legendre",
                quad_cfg=QUAD_CFG,
            )
        )

    def put_price_at_rate(rate: float) -> float:
        return float(
            heston_price_put_from_ctx(
                strike=strike,
                tau=tau,
                ctx=MarketData(
                    spot=100.0,
                    rate=float(rate),
                    dividend_yield=0.01,
                ).to_context(),
                params=BASE_PARAMS,
                backend="gauss_legendre",
                quad_cfg=QUAD_CFG,
            )
        )

    call_rate_fd = (call_price_at_rate(0.02 + dr) - call_price_at_rate(0.02 - dr)) / (
        2.0 * dr
    )
    put_rate_fd = (put_price_at_rate(0.02 + dr) - put_price_at_rate(0.02 - dr)) / (
        2.0 * dr
    )

    assert abs(call_rate_fd - put_rate_fd) > 1.0
