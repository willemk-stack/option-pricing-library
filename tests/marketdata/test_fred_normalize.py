from __future__ import annotations

import pandas as pd
import pytest

from option_pricing.marketdata.normalize import normalize_fred_observations
from option_pricing.marketdata.schemas import FRED_SERIES_COLUMNS, FRED_SERIES_DTYPES
from option_pricing.marketdata.validation import validate_dtypes


def _fred_payload() -> dict[str, object]:
    return {
        "observations": [
            {
                "realtime_start": "2026-05-01",
                "realtime_end": "2026-05-31",
                "date": "2026-05-20",
                "value": "4.25",
                "extra": "discarded",
            },
            {
                "realtime_start": "2026-05-01",
                "realtime_end": "2026-05-31",
                "date": "2026-05-21",
                "value": ".",
            },
        ]
    }


def test_normalize_fred_observations_outputs_canonical_schema_order() -> None:
    normalized = normalize_fred_observations(
        _fred_payload(),
        series_id="DGS3MO",
        asof="2026-05-22T15:30:00Z",
    )

    assert tuple(normalized.columns) == FRED_SERIES_COLUMNS
    assert {column: str(normalized[column].dtype) for column in normalized} == (
        FRED_SERIES_DTYPES
    )
    validate_dtypes(normalized, "fred_series", allow_extra=False)


def test_normalize_fred_observations_converts_dot_to_nullable_missing() -> None:
    normalized = normalize_fred_observations(
        _fred_payload(),
        series_id="DGS3MO",
        asof="2026-05-22T15:30:00Z",
    )

    assert float(normalized.loc[0, "value"]) == pytest.approx(4.25)
    assert pd.isna(normalized.loc[1, "value"])
    assert str(normalized["value"].dtype) == "Float64"


def test_normalize_fred_observations_coerces_dates_source_and_asof() -> None:
    normalized = normalize_fred_observations(
        _fred_payload(),
        series_id="DGS3MO",
        asof=pd.Timestamp("2026-05-22T15:30:00-04:00"),
    )

    assert normalized.loc[0, "series_id"] == "DGS3MO"
    assert normalized.loc[0, "observation_date"] == pd.Timestamp("2026-05-20")
    assert normalized.loc[0, "realtime_start"] == pd.Timestamp("2026-05-01")
    assert normalized.loc[0, "realtime_end"] == pd.Timestamp("2026-05-31")
    assert normalized["source"].astype(str).tolist() == ["fred", "fred"]
    assert normalized.loc[0, "asof"] == pd.Timestamp("2026-05-22T19:30:00Z")


def test_normalize_fred_observations_requires_observation_list() -> None:
    with pytest.raises(ValueError, match="observations list"):
        normalize_fred_observations({}, series_id="DGS3MO", asof="2026-05-22")
