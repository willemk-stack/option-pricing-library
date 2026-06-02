from __future__ import annotations

import math

import pandas as pd
import pytest

from option_pricing.marketdata.normalize import normalize_fred_observations
from option_pricing.marketdata.providers.fred import FredRateUnavailableError
from option_pricing.marketdata.rates import (
    fred_percent_to_continuous_decimal,
    select_latest_fred_rate_at_or_before_asof,
)


def _fred_frame(values: list[tuple[str, str]]) -> pd.DataFrame:
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
        series_id="DGS3MO",
        asof="2026-01-05T15:30:00Z",
    )


def test_fred_percent_to_continuous_decimal_uses_log1p() -> None:
    assert fred_percent_to_continuous_decimal(4.25) == pytest.approx(
        math.log1p(4.25 / 100.0)
    )


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
