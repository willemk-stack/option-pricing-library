from __future__ import annotations

import math
from dataclasses import fields
from types import SimpleNamespace

import pandas as pd
import pytest

import option_pricing.marketdata as marketdata
from option_pricing.marketdata import prepare_essvi_market_fit
from option_pricing.marketdata.schemas import SURFACE_INPUTS_COLUMNS
from option_pricing.marketdata.surface_ready import (
    PreparedESSVIMarketFit,
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
    expiry_specs: tuple[tuple[float, int], ...] = (
        (30.0, 5),
        (60.0, 5),
        (90.0, 5),
    ),
    row_overrides: dict[int, dict[str, object]] | None = None,
) -> pd.DataFrame:
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0, 95.0]
    records: list[dict[str, object]] = []
    for expiry_index, (expiry_days, point_count) in enumerate(expiry_specs):
        for point_index in range(point_count):
            row_index = len(records)
            records.append(
                {
                    "underlying": "SYNTH",
                    "quote_id": f"essvi-quote-{row_index}",
                    "asof": ASOF,
                    "expiry": _expiry_from_asof(expiry_days),
                    "strike": strikes[point_index],
                    "right": "call" if point_index % 2 == 0 else "put",
                    "mid": 2.0 + 0.2 * expiry_index + 0.05 * point_index,
                    "iv": 0.20 + 0.01 * expiry_index + 0.005 * point_index,
                    "source": "unit_test",
                    "cleaning_policy": "quote_cleaning_policy.v1",
                }
            )

    for row_index, overrides in (row_overrides or {}).items():
        records[row_index].update(overrides)
    return pd.DataFrame(records, columns=SURFACE_INPUTS_COLUMNS)


def _bundle(surface_inputs: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(
        surface_inputs=surface_inputs,
        market_data=_market_data(),
    )


def test_public_result_dataclass_has_expected_fields() -> None:
    assert tuple(field.name for field in fields(PreparedESSVIMarketFit)) == (
        "model_name",
        "market_data",
        "selected_points",
        "rejected_points",
        "stats",
        "status",
        "warnings",
    )


def test_prepare_essvi_market_fit_success_from_three_expiries() -> None:
    surface_inputs = _surface_inputs()
    original = surface_inputs.copy(deep=True)

    prepared = prepare_essvi_market_fit(_bundle(surface_inputs))

    assert isinstance(prepared, PreparedESSVIMarketFit)
    assert prepared.status == "ready"
    assert prepared.model_name == "essvi"
    assert set(SURFACE_INPUTS_COLUMNS) <= set(prepared.selected_points.columns)
    assert prepared.rejected_points.empty
    assert prepared.stats.input_point_count == len(surface_inputs)
    assert prepared.stats.selected_point_count == len(surface_inputs)
    assert prepared.stats.rejected_point_count == 0
    assert prepared.stats.rejection_counts == {}
    assert prepared.stats.expiry_count == 3
    assert prepared.stats.min_expiry_days == pytest.approx(30.0)
    assert prepared.stats.max_expiry_days == pytest.approx(90.0)
    assert prepared.warnings == ()
    pd.testing.assert_frame_equal(surface_inputs, original)


def test_fewer_than_min_expiry_count_returns_blocked_with_warning() -> None:
    prepared = prepare_essvi_market_fit(
        _bundle(_surface_inputs(((30.0, 5), (60.0, 5)))),
    )

    assert prepared.status == "blocked"
    assert len(prepared.selected_points) == 10
    assert prepared.rejected_points.empty
    assert prepared.stats.expiry_count == 2
    assert prepared.warnings == (
        "eSSVI global calibration requires at least 3 expiries; selected 2.",
    )
    assert prepared.stats.warnings == prepared.warnings


def test_missing_mid_is_rejected_when_required() -> None:
    surface_inputs = _surface_inputs(
        ((30.0, 6), (60.0, 5), (90.0, 5)),
        {0: {"mid": pd.NA}},
    )

    prepared = prepare_essvi_market_fit(_bundle(surface_inputs))

    assert prepared.status == "ready"
    assert len(prepared.selected_points) == 15
    assert prepared.stats.rejection_counts == {"missing_mid": 1}
    assert prepared.rejected_points.loc[0, "reject_reasons"] == ("missing_mid",)


def test_missing_iv_is_allowed_by_default() -> None:
    surface_inputs = _surface_inputs(row_overrides={0: {"iv": pd.NA}})

    prepared = prepare_essvi_market_fit(_bundle(surface_inputs))

    assert prepared.status == "ready"
    assert len(prepared.selected_points) == len(surface_inputs)
    assert prepared.rejected_points.empty
    first = prepared.selected_points.iloc[0]
    assert pd.isna(first["implied_vol"])
    assert pd.isna(first["total_variance"])


def test_sparse_expiry_rejects_only_that_expiry_when_surface_remains_wide() -> None:
    surface_inputs = _surface_inputs(
        ((30.0, 4), (60.0, 5), (90.0, 5), (120.0, 5)),
    )

    prepared = prepare_essvi_market_fit(_bundle(surface_inputs))

    assert prepared.status == "ready"
    assert len(prepared.selected_points) == 15
    assert len(prepared.rejected_points) == 4
    assert prepared.stats.expiry_count == 3
    assert prepared.stats.rejection_counts == {"sparse_expiry": 4}
    assert all(
        reasons == ("sparse_expiry",)
        for reasons in prepared.rejected_points["reject_reasons"]
    )


def test_selected_points_include_essvi_global_calibration_fields() -> None:
    prepared = prepare_essvi_market_fit(_bundle(_surface_inputs()))

    selected = prepared.selected_points
    assert {
        "y",
        "T",
        "price_mkt",
        "is_call",
        "sqrt_weight",
        "strike",
        "forward",
        "discount",
        "implied_vol",
        "total_variance",
    } <= set(selected.columns)

    first = selected.iloc[0]
    tau = 30.0 / 365.0
    forward = _market_data().forward(tau)
    assert float(first["T"]) == pytest.approx(tau)
    assert float(first["price_mkt"]) == pytest.approx(2.0)
    assert bool(first["is_call"]) is True
    assert float(first["forward"]) == pytest.approx(forward)
    assert float(first["discount"]) == pytest.approx(_market_data().df(tau))
    assert float(first["y"]) == pytest.approx(
        math.log(float(first["strike"]) / forward)
    )
    assert float(first["sqrt_weight"]) == pytest.approx(1.0)


def test_public_lazy_export_from_marketdata_package_works() -> None:
    assert "PreparedESSVIMarketFit" in marketdata.__all__
    assert "SurfaceReadyStats" in marketdata.__all__
    assert "prepare_essvi_market_fit" in marketdata.__all__
    assert marketdata.PreparedESSVIMarketFit is PreparedESSVIMarketFit
    assert marketdata.SurfaceReadyStats is SurfaceReadyStats
    assert marketdata.prepare_essvi_market_fit is prepare_essvi_market_fit
