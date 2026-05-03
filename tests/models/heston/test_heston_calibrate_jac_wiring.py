from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import option_pricing.models.heston.calibration.calibrate as calibrate_module
from option_pricing.models.heston.calibration.calibrate import calibrate_heston
from option_pricing.models.heston.calibration.heston_types import HestonQuoteSet
from option_pricing.models.heston.calibration.objective import HestonObjective
from option_pricing.models.heston.params import HestonParams
from option_pricing.numerics.quadrature import QuadratureConfig
from option_pricing.types import MarketData


def _quad_cfg() -> QuadratureConfig:
    return QuadratureConfig(u_max=20.0, n_panels=2, nodes_per_panel=4)


def _seed_params() -> HestonParams:
    return HestonParams(kappa=1.2, vbar=0.035, eta=0.35, rho=-0.45, v=0.04)


def _quotes() -> HestonQuoteSet:
    strike = np.array([90.0, 95.0, 100.0, 105.0, 110.0], dtype=np.float64)
    expiry = np.array([0.5, 0.5, 0.75, 1.0, 1.0], dtype=np.float64)

    return HestonQuoteSet.from_flat_market(
        market=MarketData(spot=100.0, rate=0.015, dividend_yield=0.005),
        strike=strike,
        expiry=expiry,
        is_call=np.ones(strike.shape, dtype=np.bool_),
        mid=np.array([12.0, 9.0, 7.0, 4.5, 2.5], dtype=np.float64),
        bs_vega=np.ones(strike.shape, dtype=np.float64),
    )


def _patch_zero_objective(
    monkeypatch: pytest.MonkeyPatch,
    jac_calls: dict[str, int] | None = None,
) -> None:
    def residual(self: HestonObjective, u: np.ndarray) -> np.ndarray:
        return np.zeros(self.quotes.n_quotes, dtype=np.float64)

    def jac(self: HestonObjective, u: np.ndarray) -> np.ndarray:
        if jac_calls is not None:
            jac_calls["count"] += 1
        return np.zeros((self.quotes.n_quotes, 5), dtype=np.float64)

    monkeypatch.setattr(HestonObjective, "residual", residual)
    monkeypatch.setattr(HestonObjective, "jac", jac)


def test_calibrate_heston_uses_analytic_jac_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jac_calls = {"count": 0}
    _patch_zero_objective(monkeypatch, jac_calls)

    calibrate_heston(
        quotes=_quotes(),
        x0_params=_seed_params(),
        backend="gauss_legendre",
        quad_cfg=_quad_cfg(),
        max_nfev=1,
    )

    assert jac_calls["count"] > 0


def test_calibrate_heston_can_disable_analytic_jac(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jac_calls = {"count": 0}
    _patch_zero_objective(monkeypatch, jac_calls)

    calibrate_heston(
        quotes=_quotes(),
        x0_params=_seed_params(),
        backend="gauss_legendre",
        quad_cfg=_quad_cfg(),
        use_analytic_jac=False,
        max_nfev=1,
    )

    assert jac_calls["count"] == 0


def test_calibrate_heston_lm_rejects_non_linear_loss() -> None:
    with pytest.raises(ValueError, match="method='lm'.*loss='linear'"):
        calibrate_heston(
            quotes=_quotes(),
            x0_params=_seed_params(),
            method="lm",
            loss="soft_l1",
        )


def test_calibrate_heston_lm_accepts_linear_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_zero_objective(monkeypatch)

    fit = calibrate_heston(
        quotes=_quotes(),
        x0_params=_seed_params(),
        method="lm",
        loss="linear",
        use_analytic_jac=False,
        max_nfev=1,
    )

    assert isinstance(fit, HestonParams)


def test_calibrate_heston_return_result_false_preserves_old_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_zero_objective(monkeypatch)

    fit = calibrate_heston(
        quotes=_quotes(),
        x0_params=_seed_params(),
        quad_cfg=_quad_cfg(),
        max_nfev=1,
    )

    assert isinstance(fit, HestonParams)


def test_calibrate_heston_return_result_true_returns_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_zero_objective(monkeypatch)

    fit, result = calibrate_heston(
        quotes=_quotes(),
        x0_params=_seed_params(),
        quad_cfg=_quad_cfg(),
        max_nfev=1,
        return_result=True,
    )

    assert isinstance(fit, HestonParams)
    assert hasattr(result, "x")
    assert hasattr(result, "cost")


def test_calibrate_heston_forwards_optimizer_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_least_squares(**kwargs: object) -> OptimizeResult:
        captured.update(kwargs)
        return OptimizeResult(
            x=np.asarray(kwargs["x0"], dtype=np.float64),
            success=True,
            message="ok",
            cost=0.0,
        )

    monkeypatch.setattr(calibrate_module, "least_squares", fake_least_squares)
    x_scale = np.array([1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float64)

    calibrate_module.calibrate_heston(
        quotes=_quotes(),
        objective_type="relative_price_rmse",
        x0_params=_seed_params(),
        loss="huber",
        x_scale=x_scale,
        method="dogbox",
        max_nfev=7,
        ftol=1.0e-9,
        xtol=2.0e-9,
        gtol=3.0e-9,
        use_analytic_jac=False,
    )

    assert captured["method"] == "dogbox"
    assert captured["max_nfev"] == 7
    assert captured["ftol"] == 1.0e-9
    assert captured["xtol"] == 2.0e-9
    assert captured["gtol"] == 3.0e-9
    np.testing.assert_allclose(captured["x_scale"], x_scale, rtol=0.0, atol=0.0)
    assert captured["loss"] == "huber"
    assert captured["jac"] == "2-point"
    assert captured["fun"].__self__.objective_type == "relative_price_rmse"
