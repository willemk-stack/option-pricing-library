from __future__ import annotations

import numpy as np
import pytest

from option_pricing.models.heston.calibration.heston_types import HestonQuoteSet
from option_pricing.models.heston.calibration.objective import HestonObjective
from option_pricing.models.heston.params import HestonParams
from option_pricing.numerics.quadrature import QuadratureConfig
from option_pricing.pricers.black_scholes import bs_call_greeks_from_ctx
from option_pricing.pricers.heston import heston_price_from_ctx
from option_pricing.types import MarketData, OptionType
from option_pricing.vol.implied_vol_slice import implied_vol_black76_slice
from tests.helpers.finite_diff import central_diff_jac


def _tiny_heston_quad_cfg() -> QuadratureConfig:
    return QuadratureConfig(
        u_max=100.0,
        n_panels=12,
        nodes_per_panel=12,
    )


def _synthetic_heston_quotes(
    *,
    params: HestonParams,
    quad_cfg: QuadratureConfig,
) -> HestonQuoteSet:
    market = MarketData(spot=100.0, rate=0.01, dividend_yield=0.00)
    ctx = market.to_context()

    strike = np.array([90.0, 100.0, 110.0, 90.0, 100.0, 110.0], dtype=np.float64)
    expiry = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
    is_call = np.ones(strike.shape, dtype=np.bool_)

    mid = np.empty_like(strike)
    bs_vega = np.empty_like(strike)

    for tau in np.unique(expiry):
        idx = expiry == tau
        strikes_tau = strike[idx]
        tau_float = float(tau)

        prices_tau = np.asarray(
            heston_price_from_ctx(
                kind=OptionType.CALL,
                strike=strikes_tau,
                tau=tau_float,
                ctx=ctx,
                params=params,
                backend="gauss_legendre",
                quad_cfg=quad_cfg,
            ),
            dtype=np.float64,
        )

        implied_vols_tau = np.asarray(
            implied_vol_black76_slice(
                forward=ctx.fwd(tau_float),
                strikes=strikes_tau,
                tau=tau_float,
                df=ctx.df(tau_float),
                prices=prices_tau,
                is_call=True,
            ),
            dtype=np.float64,
        )

        if not np.all(np.isfinite(implied_vols_tau)):
            raise AssertionError("Synthetic Heston prices must map to finite IVs")

        vegas_tau = np.array(
            [
                bs_call_greeks_from_ctx(
                    strike=float(strike_i),
                    sigma=float(sigma_i),
                    tau=tau_float,
                    ctx=ctx,
                )["vega"]
                for strike_i, sigma_i in zip(
                    strikes_tau,
                    implied_vols_tau,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(vegas_tau)) or np.any(vegas_tau <= 0.0):
            raise AssertionError("Synthetic quote vegas must be finite and positive")

        mid[idx] = prices_tau
        bs_vega[idx] = vegas_tau

    return HestonQuoteSet.from_flat_market(
        market=market,
        strike=strike,
        expiry=expiry,
        is_call=is_call,
        mid=mid,
        bs_vega=bs_vega,
    )


def test_heston_param_transform_jacobian_matches_finite_difference() -> None:
    raw = np.array([0.2, -3.0, -0.7, -0.4, -2.8], dtype=np.float64)

    def transform(raw_params: np.ndarray) -> np.ndarray:
        return HestonParams.transform_to_constrained(raw_params).as_array()

    jac_fd = central_diff_jac(transform, raw)
    off_diagonal = jac_fd - np.diag(np.diag(jac_fd))

    assert jac_fd.shape == (5, 5)
    assert np.all(np.isfinite(jac_fd))
    np.testing.assert_allclose(off_diagonal, 0.0, atol=1.0e-10, rtol=0.0)

    diagonal = np.diag(jac_fd)
    assert np.all(diagonal[[0, 1, 2, 4]] > 0.0)
    assert np.isfinite(diagonal[3])
    assert diagonal[3] > 0.0

    # TODO: Compare the finite-difference diagonal to the future analytic
    # raw-to-constrained transform Jacobian once that helper is public.


def test_heston_objective_residual_finite_difference_jacobian_is_well_formed() -> None:
    quad_cfg = _tiny_heston_quad_cfg()
    true_params = HestonParams(kappa=1.7, vbar=0.04, eta=0.55, rho=-0.65, v=0.05)
    seed_params = HestonParams(kappa=1.1, vbar=0.055, eta=0.40, rho=-0.30, v=0.035)

    quotes = _synthetic_heston_quotes(params=true_params, quad_cfg=quad_cfg)
    objective = HestonObjective(
        quotes=quotes,
        backend="gauss_legendre",
        quad_cfg=quad_cfg,
    )

    u0 = seed_params.transform_to_unconstrained()
    jac_fd = central_diff_jac(objective.residual, u0)
    seed_residual = objective.residual(u0)
    true_residual = objective.residual(true_params.transform_to_unconstrained())

    assert jac_fd.shape == (quotes.n_quotes, 5)
    assert np.all(np.isfinite(jac_fd))
    assert np.all(np.max(np.abs(jac_fd), axis=0) > 1.0e-10)
    assert np.linalg.norm(true_residual) < np.linalg.norm(seed_residual)


@pytest.mark.skip(reason="Analytic probability Jacobian not implemented yet")
def test_heston_probability_param_jac_matches_finite_difference() -> None:
    # TODO: evaluate the future analytic Heston probability Jacobian at a
    # representative (x, tau, params) point and compare against finite
    # differences in constrained parameter space.
    # J_analytic = ...
    # J_fd = central_diff_jac(...)
    # np.testing.assert_allclose(J_analytic, J_fd, rtol=5e-4, atol=1e-6)
    ...


@pytest.mark.skip(reason="Analytic price Jacobian not implemented yet")
def test_heston_price_param_jac_matches_finite_difference() -> None:
    # TODO: compare the future analytic call-price parameter Jacobian against a
    # finite-difference price scaffold on a small strike slice.
    # J_analytic = ...
    # J_fd = central_diff_jac(...)
    # np.testing.assert_allclose(J_analytic, J_fd, rtol=5e-4, atol=1e-6)
    ...


@pytest.mark.skip(reason="Analytic objective Jacobian not implemented yet")
def test_heston_objective_jac_matches_finite_difference() -> None:
    # TODO: compare objective.jac(u) to a finite-difference Jacobian of
    # objective.residual(u) in unconstrained raw calibration space.
    # J_analytic = ...
    # J_fd = central_diff_jac(...)
    # np.testing.assert_allclose(J_analytic, J_fd, rtol=5e-4, atol=1e-6)
    ...
