from __future__ import annotations

import numpy as np
import pytest

import option_pricing.models.heston.calibration.calibrate as calibrate_module
import option_pricing.models.heston.calibration.objective as objective_module
from option_pricing.models.heston.calibration.calibrate import calibrate_heston
from option_pricing.models.heston.calibration.heston_types import HestonQuoteSet
from option_pricing.models.heston.calibration.objective import HestonObjective
from option_pricing.models.heston.params import HestonParams
from option_pricing.types import MarketData


def _sample_quotes(*, bs_vega: np.ndarray | None = None) -> HestonQuoteSet:
    return HestonQuoteSet.from_flat_market(
        market=MarketData(spot=100.0, rate=0.02, dividend_yield=0.01),
        strike=np.array([90.0, 110.0], dtype=np.float64),
        expiry=np.array([0.5, 1.0], dtype=np.float64),
        is_call=np.array([True, False], dtype=np.bool_),
        mid=np.array([12.0, 11.0], dtype=np.float64),
        bs_vega=bs_vega,
    )


def _sample_seed() -> HestonParams:
    return HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05)


_SYNTHETIC_HESTON_MARKET = MarketData(spot=100.0, rate=0.02, dividend_yield=0.01)
_SYNTHETIC_HESTON_TRUE_PARAMS = HestonParams(
    kappa=1.5,
    vbar=0.04,
    eta=0.45,
    rho=-0.55,
    v=0.04,
)
_SYNTHETIC_HESTON_STRIKES = np.array(
    [85.0, 95.0, 100.0, 105.0, 115.0],
    dtype=np.float64,
)
_SYNTHETIC_HESTON_EXPIRIES = np.array([0.5, 1.0, 1.5], dtype=np.float64)
_SYNTHETIC_HESTON_NOISE = np.array(
    [0.02, -0.01, 0.0, 0.015, -0.02] * 3,
    dtype=np.float64,
)
_SYNTHETIC_HESTON_SEEDS = (
    HestonParams(kappa=0.9, vbar=0.03, eta=0.30, rho=-0.25, v=0.03),
    HestonParams(kappa=2.5, vbar=0.06, eta=0.80, rho=-0.80, v=0.06),
    HestonParams(kappa=1.1, vbar=0.05, eta=0.55, rho=-0.45, v=0.05),
)
MAX_CLEAN_REPRICE_RMSE = 1.0e-8
MAX_NOISY_REPRICE_RMSE = 2.0e-2
NOISY_REPRICE_RMSE_BUFFER = 5.0e-3


def _synthetic_heston_quotes(
    *,
    params: HestonParams = _SYNTHETIC_HESTON_TRUE_PARAMS,
    price_noise: np.ndarray | None = None,
) -> HestonQuoteSet:
    strike_grid, expiry_grid = np.meshgrid(
        _SYNTHETIC_HESTON_STRIKES,
        _SYNTHETIC_HESTON_EXPIRIES,
        indexing="xy",
    )
    strike = strike_grid.reshape(-1)
    expiry = expiry_grid.reshape(-1)
    is_call = strike >= _SYNTHETIC_HESTON_MARKET.spot
    base_quotes = HestonQuoteSet.from_flat_market(
        market=_SYNTHETIC_HESTON_MARKET,
        strike=strike,
        expiry=expiry,
        is_call=is_call,
        mid=np.zeros_like(strike),
    )
    mid = objective_module._price_heston_quotes(base_quotes, params)

    if price_noise is not None:
        mid = np.maximum(mid + np.asarray(price_noise, dtype=np.float64), 1.0e-6)

    return HestonQuoteSet.from_flat_market(
        market=_SYNTHETIC_HESTON_MARKET,
        strike=strike,
        expiry=expiry,
        is_call=is_call,
        mid=mid,
    )


def _reprice_rmse(quotes: HestonQuoteSet, params: HestonParams) -> float:
    repriced = objective_module._price_heston_quotes(quotes, params)
    return float(np.sqrt(np.mean((repriced - quotes.mid) ** 2)))


def _calibrate_synthetic_quotes(quotes: HestonQuoteSet):
    return calibrate_module.calibrate_heston_multistart(
        quotes=quotes,
        objective_type="price_rmse",
        seeds=list(_SYNTHETIC_HESTON_SEEDS),
        include_default_seed=False,
        parameter_transform="bounded",
        max_seeds=len(_SYNTHETIC_HESTON_SEEDS),
        max_nfev=100,
    )


def test_heston_quote_set_from_flat_market_derives_context_arrays() -> None:
    quotes = _sample_quotes()

    np.testing.assert_allclose(
        quotes.discount,
        np.exp(-0.02 * quotes.expiry),
        atol=1.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        quotes.forward,
        100.0 * np.exp(0.01 * quotes.expiry),
        atol=1.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        quotes.log_moneyness,
        np.log(quotes.strike / quotes.forward),
        atol=1.0e-12,
        rtol=0.0,
    )
    assert quotes.bs_vega is None


def test_heston_objective_residual_uses_override_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quotes = _sample_quotes(bs_vega=np.array([0.1, 0.5], dtype=np.float64))
    weights = np.array([2.0, 0.5], dtype=np.float64)
    seed = HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05)
    model_prices = quotes.mid + np.array([0.4, -0.2], dtype=np.float64)
    seen: dict[str, object] = {}

    def fake_price_heston_quotes(
        quotes_arg: HestonQuoteSet,
        params_arg: HestonParams,
        *,
        backend: str = "gauss_legendre",
        quad_cfg: object = None,
    ) -> np.ndarray:
        seen["backend"] = backend
        seen["quad_cfg"] = quad_cfg
        assert quotes_arg is quotes
        np.testing.assert_allclose(
            params_arg.as_array(),
            seed.as_array(),
            atol=2.0e-12,
            rtol=0.0,
        )
        return model_prices

    monkeypatch.setattr(
        objective_module, "_price_heston_quotes", fake_price_heston_quotes
    )

    objective = HestonObjective(
        quotes=quotes,
        sqrt_weights=weights,
        vega_floor=0.2,
        backend="quad",
    )
    residual = objective.residual(seed.transform_to_unconstrained())

    np.testing.assert_allclose(
        residual,
        weights * (model_prices - quotes.mid) / np.array([0.2, 0.5], dtype=np.float64),
        atol=1.0e-12,
        rtol=0.0,
    )
    assert seen["backend"] == "quad"
    assert seen["quad_cfg"] is None


def test_calibrate_heston_without_initial_guess_requires_iv_mid() -> None:
    quotes = _sample_quotes(bs_vega=np.array([0.3, 0.4], dtype=np.float64))

    with pytest.raises(ValueError, match="default_heston_seed requires quotes.iv_mid"):
        calibrate_heston(quotes)


def test_heston_calibration_rejects_empty_quote_set_before_optimization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quotes = HestonQuoteSet.from_flat_market(
        market=MarketData(spot=100.0, rate=0.02, dividend_yield=0.01),
        strike=np.array([], dtype=np.float64),
        expiry=np.array([], dtype=np.float64),
        is_call=np.array([], dtype=np.bool_),
        mid=np.array([], dtype=np.float64),
    )

    def fail_calibration(**_kwargs: object) -> None:
        raise AssertionError("calibration should not run for empty quotes")

    monkeypatch.setattr(calibrate_module, "calibrate_heston", fail_calibration)

    with pytest.raises(ValueError, match="at least one quote"):
        calibrate_module.calibrate_heston_multistart(
            quotes=quotes,
            objective_type="price_rmse",
            seeds=[_sample_seed()],
        )


@pytest.mark.parametrize("invalid_mid", [np.nan, np.inf, -np.inf])
def test_heston_calibration_rejects_nonfinite_prices_before_optimization(
    invalid_mid: float,
) -> None:
    with pytest.raises(ValueError, match="mid.*finite"):
        HestonQuoteSet.from_flat_market(
            market=MarketData(spot=100.0, rate=0.02, dividend_yield=0.01),
            strike=np.array([90.0, 110.0], dtype=np.float64),
            expiry=np.array([0.5, 1.0], dtype=np.float64),
            is_call=np.array([True, False], dtype=np.bool_),
            mid=np.array([12.0, invalid_mid], dtype=np.float64),
        )


def test_heston_calibration_rejects_negative_prices_before_optimization() -> None:
    with pytest.raises(ValueError, match="mid prices must be nonnegative"):
        HestonQuoteSet.from_flat_market(
            market=MarketData(spot=100.0, rate=0.02, dividend_yield=0.01),
            strike=np.array([90.0, 110.0], dtype=np.float64),
            expiry=np.array([0.5, 1.0], dtype=np.float64),
            is_call=np.array([True, False], dtype=np.bool_),
            mid=np.array([12.0, -0.01], dtype=np.float64),
        )


@pytest.mark.parametrize("invalid_expiry", [0.0, -0.5])
def test_heston_calibration_rejects_nonpositive_maturities_before_optimization(
    invalid_expiry: float,
) -> None:
    with pytest.raises(ValueError, match="expiry.*positive"):
        HestonQuoteSet.from_flat_market(
            market=MarketData(spot=100.0, rate=0.02, dividend_yield=0.01),
            strike=np.array([90.0, 110.0], dtype=np.float64),
            expiry=np.array([0.5, invalid_expiry], dtype=np.float64),
            is_call=np.array([True, False], dtype=np.bool_),
            mid=np.array([12.0, 11.0], dtype=np.float64),
        )


@pytest.mark.parametrize(
    ("bid", "ask", "message"),
    [
        (
            np.array([-0.01, 0.50], dtype=np.float64),
            np.array([0.10, 0.70], dtype=np.float64),
            "bid prices must be nonnegative",
        ),
        (
            np.array([0.01, 0.50], dtype=np.float64),
            np.array([-0.10, 0.70], dtype=np.float64),
            "ask prices must be nonnegative",
        ),
        (
            np.array([0.15, 0.50], dtype=np.float64),
            np.array([0.10, 0.70], dtype=np.float64),
            "ask must be >= bid",
        ),
    ],
)
def test_heston_calibration_rejects_invalid_bid_ask_fields_before_optimization(
    bid: np.ndarray,
    ask: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        HestonQuoteSet.from_flat_market(
            market=MarketData(spot=100.0, rate=0.02, dividend_yield=0.01),
            strike=np.array([90.0, 110.0], dtype=np.float64),
            expiry=np.array([0.5, 1.0], dtype=np.float64),
            is_call=np.array([True, False], dtype=np.bool_),
            mid=np.array([12.0, 11.0], dtype=np.float64),
            bid=bid,
            ask=ask,
        )


def test_heston_calibration_rejects_zero_bid_ask_spread_before_optimization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quotes = HestonQuoteSet.from_flat_market(
        market=MarketData(spot=100.0, rate=0.02, dividend_yield=0.01),
        strike=np.array([90.0, 110.0], dtype=np.float64),
        expiry=np.array([0.5, 1.0], dtype=np.float64),
        is_call=np.array([True, False], dtype=np.bool_),
        mid=np.array([12.0, 11.0], dtype=np.float64),
        bid=np.array([11.8, 10.5], dtype=np.float64),
        ask=np.array([12.2, 10.5], dtype=np.float64),
    )

    def fail_optimizer(*args: object, **kwargs: object) -> None:
        raise AssertionError("optimizer should not run for invalid bid/ask spreads")

    monkeypatch.setattr(calibrate_module, "least_squares", fail_optimizer)

    with pytest.raises(ValueError, match="strictly positive"):
        calibrate_heston(
            quotes,
            objective_type="bid_ask_normalized",
            x0_params=_sample_seed(),
        )


def test_heston_calibration_recovers_synthetic_prices_with_small_repricing_rmse() -> (
    None
):
    quotes = _synthetic_heston_quotes()

    result = _calibrate_synthetic_quotes(quotes)

    # Heston vanilla calibration can be weakly identifiable; this checks
    # repricing recovery and valid fitted parameters rather than uniqueness.
    assert result.success_count == len(_SYNTHETIC_HESTON_SEEDS)
    assert result.failure_count == 0
    assert np.all(np.isfinite(result.best_params.as_array()))
    assert _reprice_rmse(quotes, result.best_params) <= MAX_CLEAN_REPRICE_RMSE


def test_heston_calibration_noisy_synthetic_fixture_returns_valid_fit() -> None:
    clean_quotes = _synthetic_heston_quotes()
    noisy_quotes = _synthetic_heston_quotes(price_noise=_SYNTHETIC_HESTON_NOISE)

    result = _calibrate_synthetic_quotes(noisy_quotes)
    fitted_params = result.best_params
    repricing_rmse = _reprice_rmse(noisy_quotes, fitted_params)
    noise_rmse = float(np.sqrt(np.mean((noisy_quotes.mid - clean_quotes.mid) ** 2)))

    # Heston vanilla calibration can be weakly identifiable; this checks
    # repricing recovery and valid fitted parameters rather than uniqueness.
    assert result.success_count == len(_SYNTHETIC_HESTON_SEEDS)
    assert result.failure_count == 0
    assert np.all(np.isfinite(fitted_params.as_array()))
    assert repricing_rmse <= MAX_NOISY_REPRICE_RMSE
    assert repricing_rmse <= noise_rmse + NOISY_REPRICE_RMSE_BUFFER
