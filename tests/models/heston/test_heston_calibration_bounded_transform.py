from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import option_pricing.models.heston.calibration.calibrate as calibrate_module
import option_pricing.models.heston.calibration.objective as objective_module
from option_pricing.models.heston.calibration.bounds import (
    HestonCalibrationBounds,
    bounded_transform_jac_diag_from_raw,
)
from option_pricing.models.heston.calibration.calibrate import calibrate_heston
from option_pricing.models.heston.calibration.heston_types import HestonQuoteSet
from option_pricing.models.heston.calibration.objective import HestonObjective
from option_pricing.models.heston.params import HestonParams
from option_pricing.types import MarketData

DPRICE_DTHETA = np.array(
    [
        [0.10, -0.20, 0.30, -0.40, 0.50],
        [0.25, 0.15, -0.35, 0.45, -0.55],
        [-0.12, 0.22, 0.32, -0.42, 0.52],
    ],
    dtype=np.float64,
)


def _seed_params() -> HestonParams:
    return HestonParams(kappa=1.2, vbar=0.035, eta=0.35, rho=-0.45, v=0.04)


def _quotes() -> HestonQuoteSet:
    strike = np.array([90.0, 100.0, 110.0], dtype=np.float64)
    return HestonQuoteSet.from_flat_market(
        market=MarketData(spot=100.0, rate=0.02, dividend_yield=0.01),
        strike=strike,
        expiry=np.array([0.5, 1.0, 1.5], dtype=np.float64),
        is_call=np.array([True, True, False], dtype=np.bool_),
        mid=np.array([10.0, 5.0, 4.0], dtype=np.float64),
    )


def test_calibrate_heston_bounded_passes_bounded_raw_to_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_least_squares(**kwargs: object) -> OptimizeResult:
        captured.update(kwargs)
        return OptimizeResult(
            x=np.asarray(kwargs["x0"], dtype=np.float64),
            success=True,
            message="ok",
        )

    monkeypatch.setattr(calibrate_module, "least_squares", fake_least_squares)
    seed = _seed_params()
    bounds = HestonCalibrationBounds()

    fit = calibrate_module.calibrate_heston(
        quotes=_quotes(),
        objective_type="price_rmse",
        x0_params=seed,
        parameter_transform="bounded",
        bounds=bounds,
        use_analytic_jac=False,
        max_nfev=1,
    )

    expected_x0 = seed.transform_to_bounded_unconstrained(bounds)
    np.testing.assert_allclose(captured["x0"], expected_x0, rtol=0.0, atol=0.0)
    assert "bounds" not in captured
    assert captured["jac"] == "2-point"
    objective = captured["fun"].__self__
    assert objective.parameter_transform == "bounded"
    assert objective.bounds is bounds
    np.testing.assert_allclose(fit.as_array(), seed.as_array(), rtol=1.0e-12)


def test_calibrate_heston_bounded_result_stays_inside_default_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    raw_fit = np.array([-20.0, 20.0, 0.0, -8.0, 8.0], dtype=np.float64)

    def fake_least_squares(**kwargs: object) -> OptimizeResult:
        captured.update(kwargs)
        return OptimizeResult(x=raw_fit, success=True, message="ok")

    monkeypatch.setattr(calibrate_module, "least_squares", fake_least_squares)

    fit = calibrate_heston(
        quotes=_quotes(),
        objective_type="price_rmse",
        x0_params=_seed_params(),
        parameter_transform="bounded",
        use_analytic_jac=False,
        max_nfev=1,
    )

    bounds = HestonCalibrationBounds()
    values = fit.as_array()
    assert "bounds" not in captured
    assert np.all(values > bounds.lower_array())
    assert np.all(values < bounds.upper_array())


def test_heston_objective_residual_works_in_bounded_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quotes = _quotes()
    seed = _seed_params()
    bounds = HestonCalibrationBounds()
    price_error = np.array([0.40, -0.20, 0.10], dtype=np.float64)
    model_prices = quotes.mid + price_error

    def fake_price_heston_quotes(
        quotes_arg: HestonQuoteSet,
        params_arg: HestonParams,
        *,
        backend: str = "gauss_legendre",
        quad_cfg: object = None,
    ) -> np.ndarray:
        assert quotes_arg is quotes
        np.testing.assert_allclose(params_arg.as_array(), seed.as_array(), rtol=1.0e-12)
        return model_prices

    monkeypatch.setattr(
        objective_module,
        "_price_heston_quotes",
        fake_price_heston_quotes,
    )
    objective = HestonObjective(
        quotes=quotes,
        objective_type="price_rmse",
        parameter_transform="bounded",
        bounds=bounds,
    )

    raw = seed.transform_to_bounded_unconstrained(bounds)
    np.testing.assert_allclose(
        objective.residual(raw), price_error, rtol=0.0, atol=1e-12
    )


def test_heston_objective_jac_uses_bounded_chain_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quotes = _quotes()
    seed = _seed_params()
    bounds = HestonCalibrationBounds()

    def fake_price_and_jac_heston_quotes(
        quotes_arg: HestonQuoteSet,
        params_arg: HestonParams,
        *,
        backend: str = "gauss_legendre",
        quad_cfg: object = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert quotes_arg is quotes
        np.testing.assert_allclose(params_arg.as_array(), seed.as_array(), rtol=1.0e-12)
        return quotes.mid, DPRICE_DTHETA

    monkeypatch.setattr(
        objective_module,
        "_price_and_jac_heston_quotes",
        fake_price_and_jac_heston_quotes,
    )
    objective = HestonObjective(
        quotes=quotes,
        objective_type="price_rmse",
        parameter_transform="bounded",
        bounds=bounds,
    )

    raw = seed.transform_to_bounded_unconstrained(bounds)
    jac = objective.jac(raw)
    expected = (
        DPRICE_DTHETA
        * bounded_transform_jac_diag_from_raw(raw, bounds)[
            None,
            :,
        ]
    )

    assert jac.shape == (quotes.n_quotes, 5)
    np.testing.assert_allclose(jac, expected, rtol=0.0, atol=1.0e-12)


def test_calibrate_heston_unconstrained_rejects_bounds() -> None:
    with pytest.raises(ValueError, match="bounds.*parameter_transform='bounded'"):
        calibrate_heston(
            quotes=_quotes(),
            objective_type="price_rmse",
            x0_params=_seed_params(),
            parameter_transform="unconstrained",
            bounds=HestonCalibrationBounds(),
            max_nfev=1,
        )
