from __future__ import annotations

import numpy as np
import pytest

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


def test_calibrate_heston_requires_initial_guess() -> None:
    quotes = _sample_quotes(bs_vega=np.array([0.3, 0.4], dtype=np.float64))

    with pytest.raises(ValueError, match="x0_params must be provided"):
        calibrate_heston(quotes)
