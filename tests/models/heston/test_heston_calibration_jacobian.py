from __future__ import annotations

import numpy as np

from option_pricing.models.heston.calibration.heston_types import HestonQuoteSet
from option_pricing.models.heston.calibration.objective import HestonObjective
from option_pricing.models.heston.params import HestonParams
from option_pricing.numerics.quadrature import QuadratureConfig
from option_pricing.pricers.heston import heston_price_from_ctx
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
SEED_PARAMS = HestonParams(
    kappa=1.2,
    vbar=0.035,
    eta=0.35,
    rho=-0.45,
    v=0.04,
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


def synthetic_heston_quotes(
    *,
    params: HestonParams = BASE_PARAMS,
    sqrt_weights: np.ndarray | None = None,
) -> HestonQuoteSet:
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.01)
    ctx = market.to_context()

    strike = np.array([105.0, 90.0, 100.0, 110.0, 95.0, 115.0], dtype=float)
    expiry = np.array([1.00, 0.50, 1.00, 0.50, 1.50, 1.00], dtype=float)
    is_call = np.array([False, True, True, False, False, True], dtype=bool)
    bs_vega = np.array([1.10, 0.80, 1.30, 0.90, 1.50, 1.00], dtype=float)
    labels = (
        "put-1y-105",
        "call-6m-90",
        "call-1y-100",
        "put-6m-110",
        "put-18m-95",
        "call-1y-115",
    )

    mid = np.empty(strike.shape, dtype=float)
    for i in range(strike.size):
        mid[i] = float(
            heston_price_from_ctx(
                kind=OptionType.CALL if is_call[i] else OptionType.PUT,
                strike=float(strike[i]),
                tau=float(expiry[i]),
                ctx=ctx,
                params=params,
                backend="gauss_legendre",
                quad_cfg=QUAD_CFG,
            )
        )

    return HestonQuoteSet.from_flat_market(
        market=market,
        strike=strike,
        expiry=expiry,
        is_call=is_call,
        mid=mid,
        bs_vega=bs_vega,
        sqrt_weights=sqrt_weights,
        labels=labels,
    )


def test_heston_objective_jac_shape_and_finiteness() -> None:
    quotes = synthetic_heston_quotes()
    objective = HestonObjective(
        quotes=quotes,
        backend="gauss_legendre",
        quad_cfg=QUAD_CFG,
    )

    u = SEED_PARAMS.transform_to_unconstrained()
    jac = objective.jac(u)

    assert jac.shape == (quotes.n_quotes, 5)
    assert np.all(np.isfinite(jac))


def test_heston_objective_jac_matches_finite_difference() -> None:
    quotes = synthetic_heston_quotes()
    objective = HestonObjective(
        quotes=quotes,
        backend="gauss_legendre",
        quad_cfg=QUAD_CFG,
    )

    u = SEED_PARAMS.transform_to_unconstrained()
    jac = objective.jac(u)
    jac_fd = central_diff_jac(objective.residual, u, eps=1.0e-5)

    np.testing.assert_allclose(jac, jac_fd, rtol=RTOL, atol=ATOL)


def test_heston_objective_jac_includes_residual_scaling() -> None:
    n_quotes = synthetic_heston_quotes().n_quotes
    quotes_a = synthetic_heston_quotes(sqrt_weights=np.ones(n_quotes, dtype=float))
    quotes_b = synthetic_heston_quotes(
        sqrt_weights=2.0 * np.ones(n_quotes, dtype=float)
    )
    objective_a = HestonObjective(
        quotes=quotes_a,
        backend="gauss_legendre",
        quad_cfg=QUAD_CFG,
    )
    objective_b = HestonObjective(
        quotes=quotes_b,
        backend="gauss_legendre",
        quad_cfg=QUAD_CFG,
    )

    u = SEED_PARAMS.transform_to_unconstrained()
    jac_a = objective_a.jac(u)
    jac_b = objective_b.jac(u)

    np.testing.assert_allclose(jac_b, 2.0 * jac_a, rtol=RTOL, atol=ATOL)


def test_heston_objective_jac_preserves_quote_order() -> None:
    quotes = synthetic_heston_quotes(
        sqrt_weights=np.array([1.0, 1.2, 0.7, 1.5, 0.6, 1.3], dtype=float)
    )
    objective = HestonObjective(
        quotes=quotes,
        backend="gauss_legendre",
        quad_cfg=QUAD_CFG,
    )

    u = SEED_PARAMS.transform_to_unconstrained()
    jac = objective.jac(u)
    jac_fd = central_diff_jac(objective.residual, u, eps=1.0e-5)

    assert quotes.labels == (
        "put-1y-105",
        "call-6m-90",
        "call-1y-100",
        "put-6m-110",
        "put-18m-95",
        "call-1y-115",
    )
    np.testing.assert_allclose(jac, jac_fd, rtol=RTOL, atol=ATOL)
