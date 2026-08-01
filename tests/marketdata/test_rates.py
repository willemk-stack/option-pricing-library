from __future__ import annotations

import math

import pandas as pd
import pytest

from option_pricing.marketdata.normalize import normalize_fred_observations
from option_pricing.marketdata.providers.fred import FredRateUnavailableError
from option_pricing.marketdata.rates import (
    build_fred_treasury_zero_proxy_curve,
    fred_percent_to_continuous_decimal,
    representative_time_to_expiry_years,
    resolve_fred_treasury_zero_proxy_rate,
    select_latest_fred_rate_at_or_before_asof,
)


def _fred_frame(
    values: list[tuple[str, str]],
    *,
    series_id: str = "DGS3MO",
) -> pd.DataFrame:
    return normalize_fred_observations(
        {
            "observations": [
                {
                    "realtime_start": "2026-01-01",
                    "realtime_end": "2026-12-31",
                    "date": observation_date,
                    "value": value,
                }
                for observation_date, value in values
            ]
        },
        series_id=series_id,
        asof="2026-01-05T15:30:00Z",
    )


def test_fred_percent_to_continuous_decimal_uses_log1p() -> None:
    assert fred_percent_to_continuous_decimal(4.25) == pytest.approx(
        math.log1p(4.25 / 100.0)
    )


def test_representative_expiry_preserves_non_midnight_timestamp() -> None:
    result = representative_time_to_expiry_years(
        ["2024-03-11T20:00:00Z"],
        asof="2024-03-08T16:00:00-05:00",
    )

    assert result == pytest.approx((71.0 * 3600.0) / (365.0 * 86400.0))


def test_select_latest_fred_rate_at_or_before_asof() -> None:
    frame = _fred_frame(
        [
            ("2026-01-01", "4.00"),
            ("2026-01-03", "4.25"),
            ("2026-01-04", "4.50"),
        ]
    )

    selection = select_latest_fred_rate_at_or_before_asof(
        frame,
        series_id="DGS3MO",
        asof="2026-01-03T18:00:00Z",
    )

    assert selection.series_id == "DGS3MO"
    assert selection.observation_date == pd.Timestamp("2026-01-03")
    assert selection.value_percent == pytest.approx(4.25)
    assert selection.rate == pytest.approx(math.log1p(4.25 / 100.0))
    assert selection.continuous_decimal == selection.rate
    assert selection.rate_compounding == "continuous"
    assert selection.source == "fred"


def test_select_latest_fred_rate_skips_missing_values() -> None:
    frame = _fred_frame(
        [
            ("2026-01-01", "4.00"),
            ("2026-01-02", "4.10"),
            ("2026-01-03", "."),
        ]
    )

    selection = select_latest_fred_rate_at_or_before_asof(
        frame,
        series_id="DGS3MO",
        asof="2026-01-03T18:00:00Z",
    )

    assert selection.observation_date == pd.Timestamp("2026-01-02")
    assert selection.value_percent == pytest.approx(4.10)


def test_select_latest_fred_rate_rejects_future_only_data() -> None:
    frame = _fred_frame(
        [
            ("2026-01-04", "4.00"),
            ("2026-01-05", "4.10"),
        ]
    )

    with pytest.raises(FredRateUnavailableError, match="at or before 2026-01-03"):
        select_latest_fred_rate_at_or_before_asof(
            frame,
            series_id="DGS3MO",
            asof="2026-01-03T18:00:00Z",
        )


def test_select_latest_fred_rate_fails_clearly_when_no_valid_rate_exists() -> None:
    frame = _fred_frame(
        [
            ("2026-01-01", "."),
            ("2026-01-02", "."),
        ]
    )

    with pytest.raises(FredRateUnavailableError, match="No usable FRED rate"):
        select_latest_fred_rate_at_or_before_asof(
            frame,
            series_id="DGS3MO",
            asof="2026-01-03T18:00:00Z",
        )


def test_build_fred_treasury_proxy_curve_converts_percent_to_continuous_rates() -> None:
    curve = build_fred_treasury_zero_proxy_curve(
        {
            "DGS1MO": _fred_frame([("2026-01-02", "4.00")], series_id="DGS1MO"),
            "DGS3MO": _fred_frame([("2026-01-02", "5.00")], series_id="DGS3MO"),
        },
        asof="2026-01-05T15:30:00Z",
    )

    assert curve["series_id"].astype(str).tolist() == ["DGS1MO", "DGS3MO"]
    assert curve["tenor_years"].astype(float).tolist() == pytest.approx(
        [1.0 / 12.0, 3.0 / 12.0]
    )
    assert curve["raw_percent_rate"].astype(float).tolist() == pytest.approx([4.0, 5.0])
    assert curve["decimal_rate"].astype(float).tolist() == pytest.approx([0.04, 0.05])
    assert curve["continuous_rate"].astype(float).tolist() == pytest.approx(
        [math.log1p(0.04), math.log1p(0.05)]
    )


def test_fred_treasury_proxy_rate_interpolates_linearly_on_continuous_rates() -> None:
    curve = build_fred_treasury_zero_proxy_curve(
        {
            "DGS1MO": _fred_frame([("2026-01-02", "4.00")], series_id="DGS1MO"),
            "DGS3MO": _fred_frame([("2026-01-02", "5.00")], series_id="DGS3MO"),
        },
        asof="2026-01-05T15:30:00Z",
    )

    resolution = resolve_fred_treasury_zero_proxy_rate(
        curve,
        time_to_expiry_years=2.0 / 12.0,
        selected_rate_fallback=0.03,
    )

    expected = (math.log1p(0.04) + math.log1p(0.05)) / 2.0
    assert resolution.rate == pytest.approx(expected)
    assert resolution.selected_rate_fallback_used is False
    assert resolution.warnings == ()


def test_fred_treasury_proxy_rate_clamps_short_expiry_with_warning() -> None:
    curve = build_fred_treasury_zero_proxy_curve(
        {
            "DGS1MO": _fred_frame([("2026-01-02", "4.00")], series_id="DGS1MO"),
            "DGS3MO": _fred_frame([("2026-01-02", "5.00")], series_id="DGS3MO"),
        },
        asof="2026-01-05T15:30:00Z",
    )

    resolution = resolve_fred_treasury_zero_proxy_rate(
        curve,
        time_to_expiry_years=0.01,
        selected_rate_fallback=0.03,
    )

    assert resolution.rate == pytest.approx(math.log1p(0.04))
    assert resolution.warnings == (
        "rate_curve_clamped: expiry_shorter_than_shortest_tenor; "
        "time_to_expiry_years=0.01, tenor_years=0.0833333333333",
    )


def test_fred_treasury_proxy_rate_clamps_long_expiry_with_warning() -> None:
    curve = build_fred_treasury_zero_proxy_curve(
        {
            "DGS1MO": _fred_frame([("2026-01-02", "4.00")], series_id="DGS1MO"),
            "DGS2": _fred_frame([("2026-01-02", "6.00")], series_id="DGS2"),
        },
        asof="2026-01-05T15:30:00Z",
    )

    resolution = resolve_fred_treasury_zero_proxy_rate(
        curve,
        time_to_expiry_years=3.0,
        selected_rate_fallback=0.03,
    )

    assert resolution.rate == pytest.approx(math.log1p(0.06))
    assert resolution.warnings == (
        "rate_curve_clamped: expiry_longer_than_longest_tenor; "
        "time_to_expiry_years=3, tenor_years=2",
    )


def test_fred_treasury_proxy_rate_uses_flat_fallback_when_curve_is_insufficient() -> (
    None
):
    curve = build_fred_treasury_zero_proxy_curve(
        {"DGS1MO": _fred_frame([("2026-01-02", "4.00")], series_id="DGS1MO")},
        asof="2026-01-05T15:30:00Z",
    )

    resolution = resolve_fred_treasury_zero_proxy_rate(
        curve,
        time_to_expiry_years=1.0,
        selected_rate_fallback=0.031,
    )

    assert resolution.rate == pytest.approx(0.031)
    assert resolution.selected_rate_fallback_used is True
    assert resolution.metadata["rate_is_bootstrapped"] is False
    assert resolution.metadata["selected_rate_fallback_used"] is True
    assert resolution.metadata["rate_warnings"] == [
        "rate_curve_fallback: fewer_than_two_valid_curve_points; "
        "selected_flat_rate_used"
    ]
