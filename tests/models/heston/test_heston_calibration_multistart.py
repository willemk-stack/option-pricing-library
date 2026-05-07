from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import option_pricing.models.heston.calibration.calibrate as calibrate_module
from option_pricing.exceptions import NoConvergenceError
from option_pricing.models.heston.calibration import (
    HestonCalibrationRun as ExportedHestonCalibrationRun,
)
from option_pricing.models.heston.calibration import (
    HestonMultistartResult as ExportedHestonMultistartResult,
)
from option_pricing.models.heston.calibration.bounds import HestonCalibrationBounds
from option_pricing.models.heston.calibration.heston_types import (
    HestonCalibrationRun,
    HestonMultistartResult,
    HestonQuoteSet,
)
from option_pricing.models.heston.params import HestonParams
from option_pricing.types import MarketData


def _quotes() -> HestonQuoteSet:
    strike = np.array([90.0, 100.0, 110.0], dtype=np.float64)
    return HestonQuoteSet.from_flat_market(
        market=MarketData(spot=100.0, rate=0.02, dividend_yield=0.01),
        strike=strike,
        expiry=np.array([0.5, 1.0, 1.5], dtype=np.float64),
        is_call=np.array([True, True, False], dtype=np.bool_),
        mid=np.array([10.0, 5.0, 4.0], dtype=np.float64),
    )


def _seed(
    *,
    kappa: float = 1.0,
    vbar: float = 0.04,
    eta: float = 0.50,
    rho: float = -0.40,
    v: float = 0.04,
) -> HestonParams:
    return HestonParams(kappa=kappa, vbar=vbar, eta=eta, rho=rho, v=v)


def _fake_result(
    *,
    cost: float,
    x: np.ndarray | None = None,
    nfev: int = 3,
    njev: int = 2,
    status: int = 1,
    message: str = "ok",
) -> OptimizeResult:
    return OptimizeResult(
        success=True,
        cost=float(cost),
        optimality=1.0e-8,
        nfev=nfev,
        njev=njev,
        status=status,
        message=message,
        x=np.zeros(5, dtype=np.float64) if x is None else x,
    )


def test_result_types_have_one_canonical_public_identity() -> None:
    assert ExportedHestonCalibrationRun is HestonCalibrationRun
    assert ExportedHestonMultistartResult is HestonMultistartResult
    assert (
        importlib.util.find_spec("option_pricing.models.heston.calibration.results")
        is None
    )


def test_explicit_seeds_call_calibrator_once_per_unique_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed1 = _seed(kappa=1.0)
    seed2 = _seed(kappa=2.0)
    fitted = [_seed(kappa=1.1), _seed(kappa=2.1)]
    costs = [4.0, 1.5]
    calls: list[HestonParams] = []

    def fake_calibrate_heston(**kwargs: object) -> tuple[HestonParams, OptimizeResult]:
        calls.append(kwargs["x0_params"])  # type: ignore[arg-type]
        idx = len(calls) - 1
        assert kwargs["return_result"] is True
        return fitted[idx], _fake_result(cost=costs[idx])

    monkeypatch.setattr(calibrate_module, "calibrate_heston", fake_calibrate_heston)

    result = calibrate_module.calibrate_heston_multistart(
        quotes=_quotes(),
        objective_type="price_rmse",
        seeds=[seed1, seed2],
    )

    assert calls == [seed1, seed2]
    assert result.success_count == 2
    assert result.failure_count == 0
    assert result.best_run.cost == costs[1]
    assert result.best_params == fitted[1]


def test_failed_seed_is_retained_and_successful_seed_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed1 = _seed(kappa=1.0)
    seed2 = _seed(kappa=2.0)
    fitted = _seed(kappa=2.1)
    calls = {"count": 0}

    def fake_calibrate_heston(**_kwargs: object) -> tuple[HestonParams, OptimizeResult]:
        calls["count"] += 1
        if calls["count"] == 1:
            raise NoConvergenceError("bad start")
        return fitted, _fake_result(cost=0.25)

    monkeypatch.setattr(calibrate_module, "calibrate_heston", fake_calibrate_heston)

    result = calibrate_module.calibrate_heston_multistart(
        quotes=_quotes(),
        objective_type="price_rmse",
        seeds=[seed1, seed2],
    )

    assert len(result.runs) == 2
    assert result.success_count == 1
    assert result.failure_count == 1
    assert len(result.failed_runs) == 1
    assert result.failed_runs[0].fitted_params is None
    assert result.failed_runs[0].raw_x is None
    assert np.isinf(result.failed_runs[0].cost)
    assert "NoConvergenceError" in result.failed_runs[0].message
    assert "bad start" in result.failed_runs[0].message
    assert result.best_run.success
    assert result.best_params == fitted


def test_runs_are_sorted_success_cost_then_seed_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds = [
        _seed(kappa=1.0),
        _seed(kappa=2.0),
        _seed(kappa=3.0),
        _seed(kappa=4.0),
    ]
    costs_by_kappa = {
        1.0: 1.0,
        2.0: np.inf,
        3.0: 0.5,
        4.0: 0.5,
    }

    def fake_calibrate_heston(**kwargs: object) -> tuple[HestonParams, OptimizeResult]:
        seed = kwargs["x0_params"]
        assert isinstance(seed, HestonParams)
        cost = costs_by_kappa[seed.kappa]
        if not np.isfinite(cost):
            raise NoConvergenceError("seed diverged")
        return seed, _fake_result(cost=cost)

    monkeypatch.setattr(calibrate_module, "calibrate_heston", fake_calibrate_heston)

    result = calibrate_module.calibrate_heston_multistart(
        quotes=_quotes(),
        objective_type="price_rmse",
        seeds=seeds,
    )

    assert [run.seed_index for run in result.runs] == [2, 3, 0, 1]
    assert [run.seed_index for run in result.successful_runs] == [2, 3, 0]
    assert [run.seed_index for run in result.failed_runs] == [1]
    assert result.best_run.seed_index == 2


def test_result_metadata_fields_are_populated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds = [_seed(kappa=1.0), _seed(kappa=2.0)]

    def fake_calibrate_heston(**kwargs: object) -> tuple[HestonParams, OptimizeResult]:
        seed = kwargs["x0_params"]
        assert isinstance(seed, HestonParams)
        if seed.kappa == 1.0:
            raise RuntimeError("synthetic optimizer failure")
        return seed, _fake_result(cost=0.125, status=2, message="metadata ok")

    monkeypatch.setattr(calibrate_module, "calibrate_heston", fake_calibrate_heston)

    quotes = _quotes()
    result = calibrate_module.calibrate_heston_multistart(
        quotes=quotes,
        objective_type="relative_price_rmse",
        seeds=seeds,
        parameter_transform="bounded",
        backend="quad",
    )

    assert result.objective_type == "relative_price_rmse"
    assert result.parameter_transform == "bounded"
    assert result.backend == "quad"
    assert result.quote_count == quotes.n_quotes
    assert result.success_count == 1
    assert result.failure_count == 1
    assert result.best_run.status == 2
    assert result.best_run.message == "metadata ok"


def test_all_seeds_fail_raises_no_convergence_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_calibrate_heston(**_kwargs: object) -> tuple[HestonParams, OptimizeResult]:
        raise NoConvergenceError("still no fit")

    monkeypatch.setattr(calibrate_module, "calibrate_heston", fake_calibrate_heston)

    with pytest.raises(NoConvergenceError) as excinfo:
        calibrate_module.calibrate_heston_multistart(
            quotes=_quotes(),
            objective_type="price_rmse",
            seeds=[
                _seed(kappa=1.0),
                _seed(kappa=2.0),
                _seed(kappa=3.0),
                _seed(kappa=4.0),
            ],
        )

    message = str(excinfo.value)
    assert "all 4 seed" in message
    assert "seed 0" in message
    assert "seed 1" in message
    assert "seed 2" in message
    assert "1 more failure" in message
    assert "still no fit" in message


def test_x0_params_is_prepended_before_explicit_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x0_params = _seed(kappa=3.0)
    seed1 = _seed(kappa=1.0)
    seed2 = _seed(kappa=2.0)
    calls: list[HestonParams] = []

    def fake_calibrate_heston(**kwargs: object) -> tuple[HestonParams, OptimizeResult]:
        seed = kwargs["x0_params"]
        calls.append(seed)  # type: ignore[arg-type]
        return seed, _fake_result(cost=float(len(calls)))

    monkeypatch.setattr(calibrate_module, "calibrate_heston", fake_calibrate_heston)

    calibrate_module.calibrate_heston_multistart(
        quotes=_quotes(),
        objective_type="price_rmse",
        seeds=[seed1, seed2],
        x0_params=x0_params,
    )

    assert calls == [x0_params, seed1, seed2]


def test_duplicate_seeds_are_deduped(monkeypatch: pytest.MonkeyPatch) -> None:
    seed = _seed(kappa=1.0)
    duplicate = _seed(kappa=1.0)
    calls: list[HestonParams] = []

    def fake_calibrate_heston(**kwargs: object) -> tuple[HestonParams, OptimizeResult]:
        calls.append(kwargs["x0_params"])  # type: ignore[arg-type]
        return seed, _fake_result(cost=0.0)

    monkeypatch.setattr(calibrate_module, "calibrate_heston", fake_calibrate_heston)

    result = calibrate_module.calibrate_heston_multistart(
        quotes=_quotes(),
        objective_type="price_rmse",
        seeds=[seed, duplicate, seed],
    )

    assert calls == [seed]
    assert result.success_count == 1


@pytest.mark.parametrize("max_seeds", [0, -1])
def test_invalid_max_seeds_raises_value_error(max_seeds: int) -> None:
    with pytest.raises(ValueError, match="max_seeds"):
        calibrate_module.calibrate_heston_multistart(
            quotes=_quotes(),
            objective_type="price_rmse",
            seeds=[_seed()],
            max_seeds=max_seeds,
        )


def test_generated_seed_path_uses_heston_seed_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed1 = _seed(kappa=1.0)
    seed2 = _seed(kappa=2.0)
    bounds = HestonCalibrationBounds()
    grid_call: dict[str, object] = {}
    calibrator_calls: list[HestonParams] = []

    def fake_heston_seed_grid(
        quotes: HestonQuoteSet,
        *,
        bounds: HestonCalibrationBounds | None = None,
        max_seeds: int | None = None,
        include_default: bool = True,
    ) -> tuple[HestonParams, ...]:
        grid_call["quotes"] = quotes
        grid_call["bounds"] = bounds
        grid_call["max_seeds"] = max_seeds
        grid_call["include_default"] = include_default
        return (seed1, seed2)

    def fake_calibrate_heston(**kwargs: object) -> tuple[HestonParams, OptimizeResult]:
        seed = kwargs["x0_params"]
        calibrator_calls.append(seed)  # type: ignore[arg-type]
        return seed, _fake_result(cost=float(len(calibrator_calls)))

    monkeypatch.setattr(calibrate_module, "heston_seed_grid", fake_heston_seed_grid)
    monkeypatch.setattr(calibrate_module, "calibrate_heston", fake_calibrate_heston)

    quotes = _quotes()
    calibrate_module.calibrate_heston_multistart(
        quotes=quotes,
        objective_type="price_rmse",
        seeds=None,
        bounds=bounds,
        max_seeds=5,
        include_default_seed=False,
    )

    assert grid_call == {
        "quotes": quotes,
        "bounds": bounds,
        "max_seeds": 5,
        "include_default": False,
    }
    assert calibrator_calls == [seed1, seed2]
