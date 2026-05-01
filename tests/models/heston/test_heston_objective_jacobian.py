from __future__ import annotations

from collections.abc import Callable

import numpy as np

from option_pricing.models.heston.calibration.heston_types import HestonQuoteSet
from option_pricing.models.heston.calibration.objective import (
    HestonObjective,
    _price_heston_quotes,
)
from option_pricing.models.heston.params import HestonParams
from option_pricing.numerics.quadrature import QuadratureConfig
from option_pricing.types import MarketData


def _quad_cfg() -> QuadratureConfig:
    return QuadratureConfig(u_max=100.0, n_panels=12, nodes_per_panel=12)


def _base_params() -> HestonParams:
    return HestonParams(kappa=1.6, vbar=0.04, eta=0.45, rho=-0.65, v=0.05)


def _seed_params() -> HestonParams:
    return HestonParams(kappa=1.2, vbar=0.035, eta=0.35, rho=-0.45, v=0.04)


def _quotes(
    *,
    params: HestonParams,
    quad_cfg: QuadratureConfig,
    strike: np.ndarray | None = None,
    expiry: np.ndarray | None = None,
    is_call: np.ndarray | None = None,
    bs_vega: np.ndarray | None = None,
    sqrt_weights: np.ndarray | None = None,
) -> HestonQuoteSet:
    if strike is None:
        strike = np.array([90.0, 100.0, 110.0, 95.0], dtype=np.float64)
    if expiry is None:
        expiry = np.array([0.5, 0.5, 1.0, 1.0], dtype=np.float64)
    if is_call is None:
        is_call = np.array([True, True, True, True], dtype=np.bool_)
    if bs_vega is None:
        bs_vega = np.linspace(0.7, 1.3, strike.size, dtype=np.float64)

    market = MarketData(spot=100.0, rate=0.015, dividend_yield=0.005)
    empty_quotes = HestonQuoteSet.from_flat_market(
        market=market,
        strike=strike,
        expiry=expiry,
        is_call=is_call,
        mid=np.zeros(strike.shape, dtype=np.float64),
        bs_vega=bs_vega,
        sqrt_weights=sqrt_weights,
    )
    mid = _price_heston_quotes(
        empty_quotes,
        params,
        backend="gauss_legendre",
        quad_cfg=quad_cfg,
    )

    return HestonQuoteSet.from_flat_market(
        market=market,
        strike=strike,
        expiry=expiry,
        is_call=is_call,
        mid=mid,
        bs_vega=bs_vega,
        sqrt_weights=sqrt_weights,
    )


def _central_diff_col(
    fun: Callable[[np.ndarray], np.ndarray],
    u: np.ndarray,
    j: int,
    eps: float,
) -> np.ndarray:
    up = u.copy()
    um = u.copy()
    up[j] += eps
    um[j] -= eps
    return (fun(up) - fun(um)) / (2.0 * eps)


def _central_diff_jac(
    fun: Callable[[np.ndarray], np.ndarray],
    u: np.ndarray,
    eps: float,
) -> np.ndarray:
    base = np.asarray(fun(u), dtype=np.float64)
    jac = np.empty((base.size, u.size), dtype=np.float64)
    for j in range(u.size):
        jac[:, j] = _central_diff_col(fun, u, j, eps)
    return jac


def test_heston_objective_jac_shape() -> None:
    quad_cfg = _quad_cfg()
    quotes = _quotes(params=_base_params(), quad_cfg=quad_cfg)
    objective = HestonObjective(
        quotes=quotes,
        backend="gauss_legendre",
        quad_cfg=quad_cfg,
    )

    u = _seed_params().transform_to_unconstrained()
    jac = objective.jac(u)

    assert jac.shape == (quotes.n_quotes, 5)
    assert np.all(np.isfinite(jac))


def test_heston_objective_jac_matches_finite_difference() -> None:
    quad_cfg = _quad_cfg()
    quotes = _quotes(params=_base_params(), quad_cfg=quad_cfg)
    objective = HestonObjective(
        quotes=quotes,
        backend="gauss_legendre",
        quad_cfg=quad_cfg,
    )

    u = _seed_params().transform_to_unconstrained()
    jac = objective.jac(u)
    jac_fd = _central_diff_jac(objective.residual, u, eps=1.0e-5)

    np.testing.assert_allclose(jac, jac_fd, rtol=5.0e-4, atol=1.0e-6)


def test_heston_objective_jac_includes_residual_scaling() -> None:
    quad_cfg = _quad_cfg()
    base_bs_vega = np.array([0.8, 1.0, 1.2, 1.4], dtype=np.float64)
    base_weights = np.array([1.0, 0.7, 1.4, 0.5], dtype=np.float64)
    scaled_bs_vega = np.array([1.6, 0.5, 2.4, 0.7], dtype=np.float64)
    scaled_weights = np.array([0.5, 1.4, 0.7, 2.0], dtype=np.float64)

    quotes = _quotes(
        params=_base_params(),
        quad_cfg=quad_cfg,
        bs_vega=base_bs_vega,
        sqrt_weights=base_weights,
    )
    scaled_quotes = _quotes(
        params=_base_params(),
        quad_cfg=quad_cfg,
        bs_vega=scaled_bs_vega,
        sqrt_weights=scaled_weights,
    )
    objective = HestonObjective(
        quotes=quotes,
        vega_floor=1.0e-8,
        backend="gauss_legendre",
        quad_cfg=quad_cfg,
    )
    scaled_objective = HestonObjective(
        quotes=scaled_quotes,
        vega_floor=1.0e-8,
        backend="gauss_legendre",
        quad_cfg=quad_cfg,
    )

    u = _seed_params().transform_to_unconstrained()
    scale_ratio = (
        scaled_weights / scaled_quotes.vega_price_scales(scaled_objective.vega_floor)
    ) / (base_weights / quotes.vega_price_scales(objective.vega_floor))

    np.testing.assert_allclose(
        scaled_objective.residual(u),
        scale_ratio * objective.residual(u),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        scaled_objective.jac(u),
        scale_ratio[:, None] * objective.jac(u),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_heston_objective_jac_preserves_quote_order() -> None:
    quad_cfg = _quad_cfg()
    quotes = _quotes(
        params=_base_params(),
        quad_cfg=quad_cfg,
        strike=np.array([110.0, 90.0, 105.0, 80.0, 120.0, 95.0], dtype=np.float64),
        expiry=np.array([1.0, 0.5, 1.0, 0.5, 1.5, 1.0], dtype=np.float64),
        is_call=np.array([False, True, True, False, False, True], dtype=np.bool_),
        bs_vega=np.array([1.1, 0.8, 1.3, 0.9, 1.5, 1.0], dtype=np.float64),
        sqrt_weights=np.array([1.0, 1.2, 0.7, 1.5, 0.6, 1.3], dtype=np.float64),
    )
    objective = HestonObjective(
        quotes=quotes,
        backend="gauss_legendre",
        quad_cfg=quad_cfg,
    )

    u = _seed_params().transform_to_unconstrained()
    jac = objective.jac(u)
    jac_fd = _central_diff_jac(objective.residual, u, eps=1.0e-5)

    np.testing.assert_allclose(jac, jac_fd, rtol=5.0e-4, atol=1.0e-6)
