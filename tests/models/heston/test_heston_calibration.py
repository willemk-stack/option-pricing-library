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
