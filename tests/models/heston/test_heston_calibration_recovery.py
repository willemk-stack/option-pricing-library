from __future__ import annotations

import numpy as np

from option_pricing.diagnostics.heston import build_synthetic_heston_quote_set
from option_pricing.models.heston.calibration import (
    calibrate_heston_multistart,
    default_heston_seed,
)
from option_pricing.models.heston.calibration.bounds import HestonCalibrationBounds
from option_pricing.models.heston.calibration.heston_types import HestonQuoteSet
from option_pricing.models.heston.calibration.objective import HestonObjective
from option_pricing.models.heston.params import HestonParams
from option_pricing.numerics.quadrature import QuadratureConfig
from option_pricing.pricers.heston import heston_price_from_ctx
from option_pricing.types import OptionType
from option_pricing.vol.implied_vol_slice import implied_vol_black76_slice


def _true_params() -> HestonParams:
    return HestonParams(kappa=1.7, vbar=0.04, eta=0.55, rho=-0.55, v=0.045)


def _quad_cfg() -> QuadratureConfig:
    # NOTE: Repricing-recovery tests use the same modest fixed-rule quadrature for
    # synthetic generation and calibration to keep the suite fast and focused.
    return QuadratureConfig(u_max=50.0, n_panels=6, nodes_per_panel=6)


def _synthetic_quotes(*, noise_vol_bps: float = 0.0) -> HestonQuoteSet:
    # NOTE: The compact 3x5 surface is broad enough to exercise skew and term
    # structure without turning routine unit tests into a benchmark.
    return build_synthetic_heston_quote_set(
        market=None,
        true_params=_true_params(),
        expiries=np.array([0.5, 1.0, 2.0], dtype=np.float64),
        log_moneyness=np.array([-0.12, -0.06, 0.0, 0.06, 0.12], dtype=np.float64),
        backend="gauss_legendre",
        quad_cfg=_quad_cfg(),
        random_seed=123,
        noise_vol_bps=float(noise_vol_bps),
    )


def _objective_cost(
    quotes: HestonQuoteSet,
    params: HestonParams,
    bounds: HestonCalibrationBounds,
) -> float:
    objective = HestonObjective(
        quotes=quotes,
        objective_type="vega_scaled_price",
        backend="gauss_legendre",
        quad_cfg=_quad_cfg(),
        parameter_transform="bounded",
        bounds=bounds,
    )
    raw = params.transform_to_bounded_unconstrained(bounds)
    residual = objective.residual(raw)
    return float(0.5 * np.sum(residual * residual))


def _model_prices(quotes: HestonQuoteSet, params: HestonParams) -> np.ndarray:
    prices = np.empty(quotes.n_quotes, dtype=np.float64)
    for tau in np.unique(quotes.expiry):
        idx = np.flatnonzero(quotes.expiry == tau)
        prices[idx] = np.asarray(
            heston_price_from_ctx(
                kind=OptionType.CALL,
                strike=quotes.strike[idx],
                tau=float(tau),
                ctx=quotes.ctx,
                params=params,
                backend="gauss_legendre",
                quad_cfg=_quad_cfg(),
            ),
            dtype=np.float64,
        )
    return prices


def _model_iv_residual_bps(
    quotes: HestonQuoteSet,
    params: HestonParams,
) -> np.ndarray:
    assert quotes.iv_mid is not None
    prices = _model_prices(quotes, params)
    model_iv = np.empty(quotes.n_quotes, dtype=np.float64)
    for tau in np.unique(quotes.expiry):
        idx = np.flatnonzero(quotes.expiry == tau)
        model_iv[idx] = np.asarray(
            implied_vol_black76_slice(
                forward=quotes.ctx.fwd(float(tau)),
                strikes=quotes.strike[idx],
                tau=float(tau),
                df=quotes.ctx.df(float(tau)),
                prices=prices[idx],
                is_call=True,
                initial_sigma=quotes.iv_mid[idx],
            ),
            dtype=np.float64,
        )
    return np.asarray((model_iv - quotes.iv_mid) * 1.0e4, dtype=np.float64)


def _assert_inside_bounds(
    params: HestonParams,
    bounds: HestonCalibrationBounds,
) -> None:
    values = params.as_array()
    assert np.all(values >= bounds.lower_array())
    assert np.all(values <= bounds.upper_array())


def test_clean_synthetic_surface_reprices_heston_generated_quotes() -> None:
    quotes = _synthetic_quotes()
    bounds = HestonCalibrationBounds()
    default_seed = default_heston_seed(quotes, bounds=bounds)
    default_cost = _objective_cost(quotes, default_seed, bounds)

    result = calibrate_heston_multistart(
        quotes=quotes,
        objective_type="vega_scaled_price",
        bounds=bounds,
        quad_cfg=_quad_cfg(),
        max_seeds=6,
        parameter_transform="bounded",
        loss="linear",
        max_nfev=80,
        ftol=1.0e-9,
        xtol=1.0e-9,
        gtol=1.0e-9,
    )

    assert result.success_count >= 1
    _assert_inside_bounds(result.best_params, bounds)
    assert np.all(np.isfinite(result.best_params.as_array()))
    assert result.best_run.cost < default_cost * 1.0e-6

    price_residual = _model_prices(quotes, result.best_params) - quotes.mid
    assert float(np.max(np.abs(price_residual))) < 1.0e-8
    assert float(np.sqrt(np.mean(price_residual * price_residual))) < 1.0e-9


def test_noisy_synthetic_surface_prioritizes_repricing_stability() -> None:
    # NOTE: Two vol basis points is a small deterministic quote perturbation
    # for a stability smoke test, not a universal market-data noise model.
    quotes = _synthetic_quotes(noise_vol_bps=2.0)
    bounds = HestonCalibrationBounds()
    default_seed = default_heston_seed(quotes, bounds=bounds)
    default_cost = _objective_cost(quotes, default_seed, bounds)

    result = calibrate_heston_multistart(
        quotes=quotes,
        objective_type="vega_scaled_price",
        bounds=bounds,
        quad_cfg=_quad_cfg(),
        max_seeds=6,
        parameter_transform="bounded",
        loss="linear",
        max_nfev=100,
        ftol=1.0e-9,
        xtol=1.0e-9,
        gtol=1.0e-9,
    )

    assert result.success_count >= 1
    _assert_inside_bounds(result.best_params, bounds)
    assert result.best_run.cost < default_cost * 1.0e-2

    # NOTE: With noisy quotes, repricing/IV error and bounded stability are
    # the primary assertions; closeness to the generating parameters is
    # deliberately not required.
    iv_residual_bps = _model_iv_residual_bps(quotes, result.best_params)
    assert float(np.sqrt(np.mean(iv_residual_bps * iv_residual_bps))) < 3.0
    assert float(np.max(np.abs(iv_residual_bps))) < 5.0
