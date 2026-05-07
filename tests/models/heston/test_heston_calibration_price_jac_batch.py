from __future__ import annotations

import numpy as np

from option_pricing.models.heston.calibration.heston_types import HestonQuoteSet
from option_pricing.models.heston.calibration.objective import (
    _price_and_jac_heston_quotes,
    _price_heston_quotes,
)
from option_pricing.models.heston.params import HestonParams
from option_pricing.numerics.quadrature import QuadratureConfig
from option_pricing.pricers.heston import heston_price_and_param_jac_from_ctx
from option_pricing.types import MarketData, OptionType


def _quad_cfg() -> QuadratureConfig:
    return QuadratureConfig(u_max=100.0, n_panels=12, nodes_per_panel=12)


def _params() -> HestonParams:
    return HestonParams(kappa=1.6, vbar=0.04, eta=0.45, rho=-0.65, v=0.05)


def _params_from_array(values: np.ndarray) -> HestonParams:
    return HestonParams(
        kappa=float(values[0]),
        vbar=float(values[1]),
        eta=float(values[2]),
        rho=float(values[3]),
        v=float(values[4]),
    )


def _mixed_quotes() -> HestonQuoteSet:
    strike = np.array([110.0, 90.0, 105.0, 80.0, 120.0, 95.0, 100.0])
    expiry = np.array([1.0, 0.5, 1.0, 0.5, 1.5, 1.0, 0.5])
    is_call = np.array([False, True, True, False, False, True, False])

    return HestonQuoteSet.from_flat_market(
        market=MarketData(spot=100.0, rate=0.015, dividend_yield=0.005),
        strike=strike,
        expiry=expiry,
        is_call=is_call,
        mid=np.zeros(strike.shape, dtype=np.float64),
    )


def test_price_and_jac_heston_quotes_matches_price_helper_and_shapes() -> None:
    quotes = _mixed_quotes()
    params = _params()
    quad_cfg = _quad_cfg()

    prices, dprice_dtheta = _price_and_jac_heston_quotes(
        quotes,
        params,
        backend="gauss_legendre",
        quad_cfg=quad_cfg,
    )
    expected_prices = _price_heston_quotes(
        quotes,
        params,
        backend="gauss_legendre",
        quad_cfg=quad_cfg,
    )

    assert prices.shape == (quotes.n_quotes,)
    assert dprice_dtheta.shape == (quotes.n_quotes, 5)
    np.testing.assert_allclose(prices, expected_prices, rtol=0.0, atol=1.0e-12)


def test_price_and_jac_heston_quotes_preserves_mixed_quote_order() -> None:
    quotes = _mixed_quotes()
    params = _params()
    quad_cfg = _quad_cfg()

    prices, dprice_dtheta = _price_and_jac_heston_quotes(
        quotes,
        params,
        backend="gauss_legendre",
        quad_cfg=quad_cfg,
    )

    expected_prices = np.empty(quotes.n_quotes, dtype=np.float64)
    expected_jac = np.empty((quotes.n_quotes, 5), dtype=np.float64)
    for i in range(quotes.n_quotes):
        price, jac = heston_price_and_param_jac_from_ctx(
            kind=OptionType.CALL if quotes.is_call[i] else OptionType.PUT,
            strike=float(quotes.strike[i]),
            tau=float(quotes.expiry[i]),
            ctx=quotes.ctx,
            params=params,
            backend="gauss_legendre",
            quad_cfg=quad_cfg,
        )
        expected_prices[i] = price
        expected_jac[i, :] = jac

    np.testing.assert_allclose(prices, expected_prices, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(dprice_dtheta, expected_jac, rtol=0.0, atol=1.0e-12)


def test_price_and_jac_heston_quotes_jacobian_matches_finite_difference() -> None:
    quotes = _mixed_quotes()
    params = _params()
    quad_cfg = _quad_cfg()

    _, dprice_dtheta = _price_and_jac_heston_quotes(
        quotes,
        params,
        backend="gauss_legendre",
        quad_cfg=quad_cfg,
    )

    theta = params.as_array()
    eps = 1.0e-5
    jac_fd = np.empty((quotes.n_quotes, 5), dtype=np.float64)
    for j in range(5):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[j] += eps
        theta_minus[j] -= eps

        price_plus = _price_heston_quotes(
            quotes,
            _params_from_array(theta_plus),
            backend="gauss_legendre",
            quad_cfg=quad_cfg,
        )
        price_minus = _price_heston_quotes(
            quotes,
            _params_from_array(theta_minus),
            backend="gauss_legendre",
            quad_cfg=quad_cfg,
        )
        jac_fd[:, j] = (price_plus - price_minus) / (2.0 * eps)

    np.testing.assert_allclose(dprice_dtheta, jac_fd, rtol=5.0e-4, atol=1.0e-6)
