from __future__ import annotations

import math
from dataclasses import fields
from types import SimpleNamespace

import pandas as pd
import pytest

import option_pricing.marketdata as marketdata
import option_pricing.marketdata.surface_ready as surface_ready
from option_pricing.marketdata import prepare_svi_market_fit
from option_pricing.marketdata.schemas import SURFACE_INPUTS_COLUMNS
from option_pricing.marketdata.surface_ready import (
    PreparedSVIMarketFit,
    SurfaceReadyStats,
)
from option_pricing.types import MarketData

ASOF = "2026-05-22T00:00:00Z"


def _market_data() -> MarketData:
    return MarketData(spot=100.0, rate=0.02, dividend_yield=0.01)


def _expiry_from_asof(days: float) -> str:
    timestamp = pd.Timestamp(ASOF).tz_convert("UTC").normalize()
    return (timestamp + pd.Timedelta(days=days)).strftime("%Y-%m-%d")


def _surface_inputs(
    row_overrides: dict[int, dict[str, object]] | None = None,
    *,
    point_count: int = 5,
    expiry_days: float = 30.0,
) -> pd.DataFrame:
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0, 95.0, 105.0]
    records: list[dict[str, object]] = []
    for index in range(point_count):
        records.append(
            {
                "underlying": "SYNTH",
                "quote_id": f"surface-quote-{index}",
                "asof": ASOF,
                "expiry": _expiry_from_asof(expiry_days),
                "strike": strikes[index],
                "right": "call" if index % 2 == 0 else "put",
                "mid": 2.0 + index * 0.1,
                "iv": 0.20 + index * 0.01,
                "source": "unit_test",
                "cleaning_policy": "quote_cleaning_policy.v1",
            }
        )

    for row_index, overrides in (row_overrides or {}).items():
        records[row_index].update(overrides)
    return pd.DataFrame(records)


def _empty_surface_inputs() -> pd.DataFrame:
    return pd.DataFrame(columns=SURFACE_INPUTS_COLUMNS)


def _bundle(surface_inputs: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(
        surface_inputs=surface_inputs,
        market_data=_market_data(),
    )


def test_public_result_dataclasses_have_expected_fields() -> None:
    assert tuple(field.name for field in fields(SurfaceReadyStats)) == (
        "input_point_count",
        "selected_point_count",
        "rejected_point_count",
        "rejection_counts",
        "expiry_count",
        "min_expiry_days",
        "max_expiry_days",
        "warnings",
    )
    assert tuple(field.name for field in fields(PreparedSVIMarketFit)) == (
        "model_name",
        "market_data",
        "selected_points",
        "rejected_points",
        "stats",
        "status",
        "warnings",
    )


def test_surface_expiry_fallback_preserves_non_midnight_timestamp() -> None:
    result = surface_ready._compute_expiry_years(
        "2024-03-11T20:00:00Z",
        "2024-03-08T16:00:00-05:00",
    )

    assert result == pytest.approx((71.0 * 3600.0) / (365.0 * 86400.0))


def test_prepare_svi_market_fit_success_from_bundle_like_object() -> None:
    surface_inputs = _surface_inputs(point_count=5)
    original = surface_inputs.copy(deep=True)

    prepared = prepare_svi_market_fit(_bundle(surface_inputs))

    assert isinstance(prepared, PreparedSVIMarketFit)
    assert prepared.status == "ready"
    assert prepared.model_name == "svi"
    assert set(SURFACE_INPUTS_COLUMNS) <= set(prepared.selected_points.columns)
    assert prepared.rejected_points.empty
    assert prepared.stats.input_point_count == len(surface_inputs)
    assert prepared.stats.selected_point_count == len(surface_inputs)
    assert prepared.stats.rejected_point_count == 0
    assert prepared.stats.rejection_counts == {}
    assert prepared.stats.expiry_count == 1
    assert prepared.stats.min_expiry_days == pytest.approx(30.0)
    assert prepared.stats.max_expiry_days == pytest.approx(30.0)
    pd.testing.assert_frame_equal(surface_inputs, original)


def test_missing_surface_inputs_raises_guided_error() -> None:
    bundle = SimpleNamespace(market_data=_market_data())

    with pytest.raises(TypeError) as exc_info:
        prepare_svi_market_fit(bundle)

    message = str(exc_info.value)
    assert "bundle.surface_inputs must be a pandas DataFrame" in message
    assert "prepare_svi_market_fit(bundle)" in message


def test_missing_and_invalid_iv_are_rejected_with_reasons() -> None:
    surface_inputs = _surface_inputs(
        {
            0: {"iv": pd.NA},
            1: {"iv": 0.0},
        },
        point_count=7,
    )

    prepared = prepare_svi_market_fit(_bundle(surface_inputs))

    assert prepared.status == "ready"
    assert len(prepared.selected_points) == 5
    assert prepared.stats.rejection_counts == {
        "missing_iv": 1,
        "nonpositive_iv": 1,
    }
    assert prepared.rejected_points.loc[0, "reject_reasons"] == ("missing_iv",)
    assert prepared.rejected_points.loc[1, "reject_reasons"] == ("nonpositive_iv",)


def test_short_expiry_is_rejected_with_reason() -> None:
    surface_inputs = _surface_inputs(
        {
            0: {
                "expiry": _expiry_from_asof(3.0),
                "expiry_years": 3.0 / 365.0,
            }
        },
        point_count=6,
    )

    prepared = prepare_svi_market_fit(_bundle(surface_inputs))

    assert prepared.status == "ready"
    assert len(prepared.selected_points) == 5
    assert prepared.stats.rejection_counts == {"short_expiry": 1}
    assert prepared.rejected_points.loc[0, "reject_reasons"] == ("short_expiry",)


def test_sparse_expiry_rejects_otherwise_selected_points() -> None:
    surface_inputs = _surface_inputs(point_count=4)

    prepared = prepare_svi_market_fit(_bundle(surface_inputs))

    assert prepared.status == "empty"
    assert prepared.selected_points.empty
    assert len(prepared.rejected_points) == 4
    assert prepared.stats.rejection_counts == {"sparse_expiry": 4}
    assert all(
        reasons == ("sparse_expiry",)
        for reasons in prepared.rejected_points["reject_reasons"]
    )


def test_selected_points_include_svi_surface_fields() -> None:
    prepared = prepare_svi_market_fit(_bundle(_surface_inputs(point_count=5)))

    selected = prepared.selected_points
    assert {
        "expiry_years",
        "expiry_days",
        "forward",
        "discount",
        "log_moneyness",
        "total_variance",
        "sqrt_weight",
        "option_type",
        "is_call",
    } <= set(selected.columns)

    first = selected.iloc[0]
    tau = 30.0 / 365.0
    forward = _market_data().forward(tau)
    assert float(first["expiry_years"]) == pytest.approx(tau)
    assert float(first["forward"]) == pytest.approx(forward)
    assert float(first["discount"]) == pytest.approx(_market_data().df(tau))
    assert float(first["log_moneyness"]) == pytest.approx(
        math.log(float(first["strike"]) / forward)
    )
    assert float(first["total_variance"]) == pytest.approx(tau * 0.20**2)
    assert float(first["sqrt_weight"]) == pytest.approx(1.0)


def test_empty_surface_universe_returns_empty_status() -> None:
    prepared = prepare_svi_market_fit(_bundle(_empty_surface_inputs()))

    assert prepared.status == "empty"
    assert prepared.selected_points.empty
    assert prepared.rejected_points.empty
    assert prepared.stats.input_point_count == 0
    assert prepared.stats.selected_point_count == 0
    assert prepared.stats.rejected_point_count == 0
    assert prepared.stats.min_expiry_days is None
    assert prepared.stats.max_expiry_days is None


def test_public_lazy_export_from_marketdata_package_works() -> None:
    assert "PreparedSVIMarketFit" in marketdata.__all__
    assert "SurfaceReadyStats" in marketdata.__all__
    assert "prepare_svi_market_fit" in marketdata.__all__
    assert marketdata.PreparedSVIMarketFit is PreparedSVIMarketFit
    assert marketdata.SurfaceReadyStats is SurfaceReadyStats
    assert marketdata.prepare_svi_market_fit is prepare_svi_market_fit
