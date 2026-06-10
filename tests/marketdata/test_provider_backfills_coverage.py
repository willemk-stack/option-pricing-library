from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd
import pytest

import option_pricing.marketdata.provider_backfills as backfills
from option_pricing.marketdata.storage import LocalStorage


@dataclass
class _BarObject:
    symbol: str
    timestamp: str = "2026-01-01T00:00:00Z"
    open: float = 1.0
    high: float = 2.0
    low: float = 0.5
    close: float = 1.5
    volume: int = 100


def test_backfill_cleaners_validate_text_values() -> None:
    assert backfills._clean_fred_series_ids([" gdp ", "cpi"]) == ("GDP", "CPI")
    assert backfills._clean_alpaca_symbols(" spy ") == ("SPY",)

    with pytest.raises(ValueError, match="at least one"):
        backfills._clean_alpaca_symbols([])

    with pytest.raises(TypeError, match="only strings"):
        backfills._clean_fred_series_ids(["GDP", 123])  # type: ignore[list-item]

    with pytest.raises(TypeError, match="string or sequence"):
        backfills._clean_fred_series_ids(123)  # type: ignore[arg-type]


def test_backfill_date_and_timestamp_coercion() -> None:
    assert backfills._coerce_backfill_date("2026-01-02", "start") == date(2026, 1, 2)
    assert backfills._coerce_backfill_date(
        datetime(2026, 1, 2, 23, 0, tzinfo=UTC),
        "start",
    ) == date(2026, 1, 2)

    ts = backfills._coerce_backfill_timestamp("2026-01-02", "start")
    assert ts.tzinfo is not None
    assert ts.tz_convert(UTC).date() == date(2026, 1, 2)

    with pytest.raises(ValueError, match="must not be missing"):
        backfills._coerce_backfill_timestamp(pd.NaT, "start")


def test_backfill_target_preflight_detects_existing_files(tmp_path) -> None:
    storage = LocalStorage(tmp_path)
    kwargs = dict(
        series_id="GDP",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        run_id="run-1",
    )
    target = backfills._expected_fred_backfill_target_paths(storage, **kwargs)[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        backfills._preflight_fred_backfill_targets(
            storage,
            series_ids=("GDP",),
            start_date=kwargs["start_date"],
            end_date=kwargs["end_date"],
            run_id=kwargs["run_id"],
            overwrite=False,
        )

    backfills._preflight_fred_backfill_targets(
        storage,
        series_ids=("GDP",),
        start_date=kwargs["start_date"],
        end_date=kwargs["end_date"],
        run_id=kwargs["run_id"],
        overwrite=True,
    )


def test_backfill_manifests_select_schema_and_sanitize_request_metadata() -> None:
    metadata = backfills._backfill_metadata(
        run_id="run-1",
        prefix="fred",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        library_commit="abc123",
    )
    fred = backfills._fred_backfill_manifest(
        metadata,
        series_id="GDP",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        request_metadata={"api_key": "secret", "series_id": "GDP"},
        raw_rows=2,
        normalized_rows=1,
        layer="silver",
        artifacts={"silver": "fred_series.parquet"},
        warnings=("trimmed",),
        library_commit="abc123",
    )
    bars = backfills._bars_backfill_manifest(
        metadata,
        symbol="SPY",
        start=pd.Timestamp("2026-01-01T00:00:00Z"),
        end=pd.Timestamp("2026-01-02T00:00:00Z"),
        timeframe="1Day",
        feed="iex",
        request_metadata={"secret_key": "secret", "symbol": "SPY"},
        raw_rows=3,
        normalized_rows=2,
        layer="bronze",
        artifacts={"bronze": "bars.json"},
        warnings=(),
        library_commit=None,
    )

    assert fred["schema_version"] == backfills.FRED_BACKFILL_SILVER_SCHEMA_VERSION
    assert (
        bars["schema_version"] == backfills.EQUITY_BARS_BACKFILL_BRONZE_SCHEMA_VERSION
    )
    assert fred["request_metadata"]["api_key"] != "secret"
    assert bars["request_metadata"]["secret_key"] != "secret"


def test_alpaca_bar_counting_handles_mapping_sequence_objects_and_raw_data() -> None:
    payload_mapping = {"bars": {"spy": [{"close": 1.0}, {"close": 2.0}]}}
    assert backfills._count_alpaca_bars_for_symbol(payload_mapping, "SPY") == 2

    payload_sequence = {"symbols": ["SPY"], "bars": [{"close": 1.0}, {"close": 2.0}]}
    assert backfills._count_alpaca_bars_for_symbol(payload_sequence, "SPY") == 2

    payload_objects = {"bars": [_BarObject("SPY"), _BarObject("MSFT")]}
    assert backfills._count_alpaca_bars_for_symbol(payload_objects, "SPY") == 1

    raw_data_payload = {"bars": {"SPY": {"raw_data": {"t": "2026-01-01", "c": 1.0}}}}
    assert backfills._count_alpaca_bars_for_symbol(raw_data_payload, "SPY") == 1
    assert backfills._count_alpaca_bars_for_symbol({"bars": {}}, "SPY") == 0


def test_alpaca_bars_payload_for_symbol_filters_mapping_payload() -> None:
    out = backfills._alpaca_bars_payload_for_symbol(
        {"symbols": ["SPY", "MSFT"], "bars": {"SPY": [1], "MSFT": [2]}},
        "SPY",
    )

    assert out["symbols"] == ["SPY"]
    assert out["bars"] == {"SPY": [1]}
