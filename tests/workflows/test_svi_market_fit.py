from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import option_pricing.workflows as workflows
import option_pricing.workflows.surface_fit as surface_fit
from option_pricing.marketdata.surface_ready import (
    PreparedSVIMarketFit,
    SurfaceReadyStats,
)
from option_pricing.types import MarketData
from option_pricing.vol.svi import SVIParams, svi_total_variance
from option_pricing.workflows import (
    HestonCalibrationConfig,
    HestonMarketFitResult,
    SVIMarketFitConfig,
    SVIMarketFitResult,
    SVISliceMarketFitResult,
    fit_heston_market,
    fit_svi_from_bundle,
    fit_svi_market,
)

ASOF = "2026-05-22T00:00:00Z"


def _market_data() -> MarketData:
    return MarketData(spot=100.0, rate=0.02, dividend_yield=0.01)


def _expiry_from_asof(days: float) -> str:
    timestamp = pd.Timestamp(ASOF).tz_convert("UTC").normalize()
    return (timestamp + pd.Timedelta(days=days)).strftime("%Y-%m-%d")


def _selected_svi_points(
    *,
    expiry_point_counts: dict[float, int] | None = None,
) -> pd.DataFrame:
    market_data = _market_data()
    records: list[dict[str, object]] = []
    point_counts = expiry_point_counts or {30.0: 7, 60.0: 7}

    for expiry_days, point_count in point_counts.items():
        expiry_years = expiry_days / 365.0
        forward = market_data.forward(expiry_years)
        discount = market_data.df(expiry_years)
        y_grid = np.linspace(-0.24, 0.24, point_count, dtype=np.float64)
        params = SVIParams(
            a=0.018 + 0.004 * expiry_years,
            b=0.12 + 0.01 * expiry_years,
            rho=-0.20,
            m=0.02,
            sigma=0.28,
        )
        w_grid = svi_total_variance(y_grid, params)

        for index, (y_value, total_variance) in enumerate(
            zip(y_grid, w_grid, strict=True)
        ):
            records.append(
                {
                    "underlying": "SYNTH",
                    "quote_id": f"surface-{int(expiry_days)}-{index}",
                    "asof": ASOF,
                    "expiry": _expiry_from_asof(expiry_days),
                    "expiry_years": expiry_years,
                    "strike": forward * math.exp(float(y_value)),
                    "right": "call" if index % 2 == 0 else "put",
                    "mid": 1.0 + 0.05 * index,
                    "iv": math.sqrt(float(total_variance) / expiry_years),
                    "source": "unit_test",
                    "cleaning_policy": "quote_cleaning_policy.v1",
                    "expiry_days": expiry_days,
                    "forward": forward,
                    "discount": discount,
                    "log_moneyness": float(y_value),
                    "total_variance": float(total_variance),
                    "sqrt_weight": 1.0,
                    "option_type": "call" if index % 2 == 0 else "put",
                    "is_call": index % 2 == 0,
                }
            )

    return pd.DataFrame(records)


def _rejected_points() -> pd.DataFrame:
    rejected = _selected_svi_points(expiry_point_counts={30.0: 1})
    rejected = rejected.copy(deep=True)
    rejected["reject_reasons"] = [("synthetic_rejection",)]
    return rejected


def _prepared(
    selected_points: pd.DataFrame | None = None,
    *,
    rejected_points: pd.DataFrame | None = None,
    status: str = "ready",
    warnings: tuple[str, ...] = (),
) -> PreparedSVIMarketFit:
    selected = (
        _selected_svi_points() if selected_points is None else selected_points
    ).reset_index(drop=True)
    rejected = (
        pd.DataFrame() if rejected_points is None else rejected_points
    ).reset_index(drop=True)
    expiry_days = (
        pd.to_numeric(selected["expiry_days"], errors="coerce").dropna()
        if "expiry_days" in selected
        else pd.Series(dtype=float)
    )
    stats = SurfaceReadyStats(
        input_point_count=int(len(selected) + len(rejected)),
        selected_point_count=int(len(selected)),
        rejected_point_count=int(len(rejected)),
        rejection_counts={},
        expiry_count=(
            int(selected["expiry_years"].nunique()) if "expiry_years" in selected else 0
        ),
        min_expiry_days=None if expiry_days.empty else float(expiry_days.min()),
        max_expiry_days=None if expiry_days.empty else float(expiry_days.max()),
        warnings=warnings,
    )
    return PreparedSVIMarketFit(
        model_name="svi",
        market_data=_market_data(),
        selected_points=selected,
        rejected_points=rejected,
        stats=stats,
        status=status,
        warnings=warnings,
    )


def test_fit_svi_market_happy_path_with_synthetic_prepared_points() -> None:
    rejected = _rejected_points()
    prepared = _prepared(rejected_points=rejected)

    result = fit_svi_market(
        prepared,
        fit_config=SVIMarketFitConfig(loss="linear"),
    )

    assert isinstance(result, SVIMarketFitResult)
    assert result.status == "ok"
    assert result.surface is not None
    assert len(result.slice_results) == 2
    assert all(
        isinstance(item, SVISliceMarketFitResult) for item in result.slice_results
    )
    assert all(item.params is not None for item in result.slice_results)
    assert all(item.diagnostics is not None for item in result.slice_results)
    assert result.summary["model_name"] == "svi"
    assert result.summary["calibration_status"] == "ok"
    assert result.summary["selected_point_count"] == len(prepared.selected_points)
    assert result.summary["rejected_point_count"] == len(rejected)
    assert result.summary["fitted_expiry_count"] == 2
    assert result.summary["failed_expiry_count"] == 0
    pd.testing.assert_frame_equal(result.selected_points, prepared.selected_points)
    pd.testing.assert_frame_equal(result.rejected_points, prepared.rejected_points)


def test_fit_svi_from_bundle_calls_loader_and_preparation_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared(_selected_svi_points(expiry_point_counts={45.0: 7}))
    bundle = SimpleNamespace(name="loaded-bundle")
    calls: list[tuple[str, object, dict[str, object]]] = []

    def fake_load_model_validation_bundle(path: object) -> object:
        calls.append(("load", path, {}))
        return bundle

    def fake_prepare_svi_market_fit(
        loaded: object, **kwargs: object
    ) -> PreparedSVIMarketFit:
        calls.append(("prepare", loaded, kwargs))
        return prepared

    monkeypatch.setattr(
        surface_fit,
        "load_model_validation_bundle",
        fake_load_model_validation_bundle,
    )
    monkeypatch.setattr(
        surface_fit,
        "prepare_svi_market_fit",
        fake_prepare_svi_market_fit,
    )

    result = fit_svi_from_bundle(
        "bundle-root",
        min_expiry_days=3.0,
        min_points_per_expiry=4,
        fit_config=SVIMarketFitConfig(loss="linear"),
    )

    assert result.status == "ok"
    assert calls[0] == ("load", "bundle-root", {})
    assert calls[1][0] == "prepare"
    assert calls[1][1] is bundle
    assert calls[1][2]["min_expiry_days"] == 3.0
    assert calls[1][2]["min_points_per_expiry"] == 4


def test_empty_prepared_result_returns_empty_without_fitting() -> None:
    columns = _selected_svi_points().columns
    prepared = _prepared(pd.DataFrame(columns=columns), status="empty")

    result = fit_svi_market(prepared)

    assert result.status == "empty"
    assert result.slice_results == ()
    assert result.parameter_table.empty
    assert result.surface is None
    assert result.summary["calibration_status"] == "empty"
    assert result.summary["fitted_expiry_count"] == 0
    assert result.errors


def test_sparse_expiry_produces_partial_status_without_crashing() -> None:
    prepared = _prepared(_selected_svi_points(expiry_point_counts={30.0: 7, 60.0: 4}))

    result = fit_svi_market(
        prepared,
        fit_config=SVIMarketFitConfig(loss="linear"),
    )

    assert result.status == "partial"
    assert result.summary["fitted_expiry_count"] == 1
    assert result.summary["failed_expiry_count"] == 1
    assert result.surface is not None
    assert result.errors
    assert "requires at least 5 selected points" in result.errors[0]


def test_parameter_table_has_expected_columns() -> None:
    prepared = _prepared(_selected_svi_points(expiry_point_counts={30.0: 7}))

    result = fit_svi_market(
        prepared,
        fit_config=SVIMarketFitConfig(loss="linear"),
    )

    assert list(result.parameter_table.columns) == [
        "expiry_years",
        "expiry_days",
        "status",
        "point_count",
        "a",
        "b",
        "rho",
        "m",
        "sigma",
        "diagnostics_ok",
        "failure_reason",
        "rmse_w",
        "rmse_unw",
        "max_abs_werr",
        "solver_cost",
        "solver_nfev",
        "error",
    ]
    assert len(result.parameter_table) == 1
    assert result.parameter_table.loc[0, "status"] == "ok"


def test_public_workflow_exports_expose_svi_helpers() -> None:
    assert "SVIMarketFitConfig" in workflows.__all__
    assert "SVISliceMarketFitResult" in workflows.__all__
    assert "SVIMarketFitResult" in workflows.__all__
    assert "fit_svi_market" in workflows.__all__
    assert "fit_svi_from_bundle" in workflows.__all__
    assert workflows.SVIMarketFitConfig is SVIMarketFitConfig
    assert workflows.SVISliceMarketFitResult is SVISliceMarketFitResult
    assert workflows.SVIMarketFitResult is SVIMarketFitResult
    assert workflows.fit_svi_market is fit_svi_market
    assert workflows.fit_svi_from_bundle is fit_svi_from_bundle
    assert not hasattr(workflows, "fit_market_model")


def test_heston_workflow_imports_still_work() -> None:
    assert workflows.HestonCalibrationConfig is HestonCalibrationConfig
    assert workflows.HestonMarketFitResult is HestonMarketFitResult
    assert workflows.fit_heston_market is fit_heston_market
