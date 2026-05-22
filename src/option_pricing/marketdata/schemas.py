"""
Implement:
- config dataclasses
- run metadata dataclasses
- typed result objects
- canonical dataset_name names
- optional small enums:
    - Right
    - DatasetLayer
    - ProviderName

Definition of done:
every pipeline function returns a typed result, not loose dicts
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

import pandas as pd

type PandasSchemaDtype = Literal[
    "datetime64[ns, UTC]",
    "datetime64[ns]",
    "Float64",
    "Int64",
    "string",
]


def _require_aware_utc(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    if value.tzinfo != UTC and value.astimezone(UTC) != value:
        # optional: drop this branch if you only care that it is aware, not UTC
        pass


@dataclass(frozen=True, slots=True)
class AlpacaConfig:
    api_key_env: str = "ALPACA_API_KEY"
    secret_key_env: str = "ALPACA_SECRET_KEY"
    feed: str = "indicative"
    sandbox: bool = False


@dataclass(frozen=True, slots=True)
class FredConfig:
    api_key_env: str = "FRED_API_KEY"
    base_url: str = "https://api.stlouisfed.org/fred"


@dataclass(frozen=True, slots=True)
class StorageConfig:
    root: Path
    compression: str = "zstd"

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise TypeError("storage.root must be a pathlib.Path")


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    alpaca: AlpacaConfig
    fred: FredConfig
    storage: StorageConfig


@dataclass(frozen=True, slots=True)
class RunMetadata:
    run_id: str
    asof: datetime
    started_at: datetime
    git_sha: str | None = None

    def __post_init__(self) -> None:
        _require_aware_utc("asof", self.asof)
        _require_aware_utc("started_at", self.started_at)


#########
# Constants
#########

EQUITY_QUOTES_COLUMNS = (
    "symbol",
    "quote_ts",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "mid",
    "source",
    "asof",
)

EQUITY_QUOTES_DTYPES: dict[str, PandasSchemaDtype] = {
    "symbol": "string",
    "quote_ts": "datetime64[ns, UTC]",
    "bid": "Float64",
    "ask": "Float64",
    "bid_size": "Int64",
    "ask_size": "Int64",
    "mid": "Float64",
    "source": "string",
    "asof": "datetime64[ns, UTC]",
}

EQUITY_BARS_COLUMNS = (
    "symbol",
    "bar_ts",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
    "source",
    "asof",
)

EQUITY_BARS_DTYPES: dict[str, PandasSchemaDtype] = {
    "symbol": "string",
    "bar_ts": "datetime64[ns, UTC]",
    "timeframe": "string",
    "open": "Float64",
    "high": "Float64",
    "low": "Float64",
    "close": "Float64",
    "volume": "Int64",
    "trade_count": "Int64",
    "vwap": "Float64",
    "source": "string",
    "asof": "datetime64[ns, UTC]",
}

OPTION_CHAIN_COLUMNS = (
    "underlying",
    "contract_symbol",
    "quote_ts",
    "expiry",
    "strike",
    "right",
    "bid",
    "ask",
    "mid",
    "last",
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "open_interest",
    "source",
    "asof",
)

OPTION_CHAIN_DTYPES: dict[str, PandasSchemaDtype] = {
    "underlying": "string",
    "contract_symbol": "string",
    "quote_ts": "datetime64[ns, UTC]",
    "expiry": "datetime64[ns]",
    "strike": "Float64",
    "right": "string",  # can later narrow to Right enum at code boundary
    "bid": "Float64",
    "ask": "Float64",
    "mid": "Float64",
    "last": "Float64",
    "iv": "Float64",
    "delta": "Float64",
    "gamma": "Float64",
    "theta": "Float64",
    "vega": "Float64",
    "rho": "Float64",
    "open_interest": "Int64",
    "source": "string",
    "asof": "datetime64[ns, UTC]",
}

FRED_SERIES_COLUMNS = (
    "series_id",
    "observation_date",
    "value",
    "realtime_start",
    "realtime_end",
    "source",
    "asof",
)

FRED_SERIES_DTYPES: dict[str, PandasSchemaDtype] = {
    "series_id": "string",
    "observation_date": "datetime64[ns]",
    "value": "Float64",
    "realtime_start": "datetime64[ns]",
    "realtime_end": "datetime64[ns]",
    "source": "string",
    "asof": "datetime64[ns, UTC]",
}

MARKET_SNAPSHOT_COLUMNS = (
    "underlying",
    "asof",
    "spot",
    "spot_source",
    "rate",
    "rate_source",
    "rate_observation_date",
    "dividend_yield",
    "dividend_source",
    "option_contract_count",
)

MARKET_SNAPSHOT_DTYPES: dict[str, PandasSchemaDtype] = {
    "underlying": "string",
    "asof": "datetime64[ns, UTC]",
    "spot": "Float64",
    "spot_source": "string",
    "rate": "Float64",
    "rate_source": "string",
    "rate_observation_date": "datetime64[ns]",
    "dividend_yield": "Float64",
    "dividend_source": "string",
    "option_contract_count": "Int64",
}


class DatasetName(StrEnum):
    """
    Enum object for safe internal handling of dataset_name calling
    """

    EQUITY_QUOTES = "equity_quotes"
    EQUITY_BARS = "equity_bars"
    OPTION_CHAIN = "option_chain"
    FRED_SERIES = "fred_series"
    MARKET_SNAPSHOT = "market_snapshot"


DATASET_COLUMNS: dict[DatasetName, tuple[str, ...]] = {
    DatasetName.EQUITY_QUOTES: EQUITY_QUOTES_COLUMNS,
    DatasetName.EQUITY_BARS: EQUITY_BARS_COLUMNS,
    DatasetName.OPTION_CHAIN: OPTION_CHAIN_COLUMNS,
    DatasetName.FRED_SERIES: FRED_SERIES_COLUMNS,
    DatasetName.MARKET_SNAPSHOT: MARKET_SNAPSHOT_COLUMNS,
}

DATASET_DTYPES: dict[DatasetName, dict[str, PandasSchemaDtype]] = {
    DatasetName.EQUITY_QUOTES: EQUITY_QUOTES_DTYPES,
    DatasetName.EQUITY_BARS: EQUITY_BARS_DTYPES,
    DatasetName.OPTION_CHAIN: OPTION_CHAIN_DTYPES,
    DatasetName.FRED_SERIES: FRED_SERIES_DTYPES,
    DatasetName.MARKET_SNAPSHOT: MARKET_SNAPSHOT_DTYPES,
}


###########
# Validation helpers
###########


def parse_dataset_name(dataset_name: DatasetName | str) -> DatasetName:
    """Normalize a dataset name into a DatasetName enum."""

    if isinstance(dataset_name, DatasetName):
        return dataset_name

    if not isinstance(dataset_name, str):
        raise TypeError(
            f"dataset_name must be a DatasetName or str, got {type(dataset_name).__name__}"
        )

    try:
        return DatasetName(dataset_name.strip().lower())
    except ValueError as exc:
        known = ", ".join(item.value for item in DatasetName)
        raise ValueError(
            f"Unknown marketdata dataset_name {dataset_name!r}. "
            f"Expected one of: {known}"
        ) from exc


def dataset_columns(dataset_name: DatasetName | str) -> tuple[str, ...]:
    """Return required canonical columns for a marketdata dataset."""

    parsed_name = parse_dataset_name(dataset_name)
    return DATASET_COLUMNS[parsed_name]


def dataset_dtypes(dataset_name: DatasetName | str) -> dict[str, PandasSchemaDtype]:
    """Return expected pandas dtypes for a marketdata dataset."""

    parsed_name = parse_dataset_name(dataset_name)
    return DATASET_DTYPES[parsed_name]


def validate_columns(
    frame: pd.DataFrame,
    dataset_name: DatasetName | str,
    *,
    allow_extra: bool = True,
) -> None:
    """Validate that a DataFrame has the required columns for a dataset."""

    parsed_name = parse_dataset_name(dataset_name)
    required = dataset_columns(parsed_name)
    actual = tuple(frame.columns)

    missing = [column for column in required if column not in actual]

    if missing:
        raise ValueError(f"{parsed_name.value} is missing required columns: {missing}")

    if not allow_extra:
        extra = [column for column in actual if column not in required]

        if extra:
            raise ValueError(
                f"{parsed_name.value} has unexpected extra columns: {extra}"
            )


def validate_dtypes(
    frame: pd.DataFrame,
    dataset_name: DatasetName | str,
    *,
    allow_extra: bool = True,
) -> None:
    """Validate that a DataFrame has the expected pandas dtypes for a dataset."""

    parsed_name = parse_dataset_name(dataset_name)

    # First make sure the required columns are present.
    validate_columns(frame, parsed_name, allow_extra=allow_extra)

    expected_dtypes = dataset_dtypes(parsed_name)

    mismatches: dict[str, tuple[str, str]] = {}

    for column, expected_dtype in expected_dtypes.items():
        actual_dtype = str(frame[column].dtype)

        if actual_dtype != expected_dtype:
            mismatches[column] = (actual_dtype, expected_dtype)

    if mismatches:
        details = ", ".join(
            f"{column}: actual={actual!r}, expected={expected!r}"
            for column, (actual, expected) in mismatches.items()
        )

        raise TypeError(f"{parsed_name.value} has dtype mismatches: {details}")


def order_columns(frame: pd.DataFrame, dataset_name: DatasetName | str) -> pd.DataFrame:
    """Return frame with canonical columns first and extras after."""

    required = dataset_columns(dataset_name)
    extra = [column for column in frame.columns if column not in required]
    return frame.loc[:, [*required, *extra]]


def coerce_frame(
    frame: pd.DataFrame,
    dataset_name: DatasetName | str,
    *,
    allow_extra: bool = True,
) -> pd.DataFrame:
    """Return a copy of frame coerced to the expected pandas dtypes."""

    parsed_name = parse_dataset_name(dataset_name)
    validate_columns(frame, parsed_name, allow_extra=allow_extra)

    out = frame.copy()
    expected_dtypes = dataset_dtypes(parsed_name)

    for column, dtype in expected_dtypes.items():
        try:
            if dtype == "datetime64[ns, UTC]":
                out[column] = pd.to_datetime(out[column], utc=True)

            elif dtype == "datetime64[ns]":
                out[column] = pd.to_datetime(out[column]).dt.tz_localize(None)

            elif dtype == "Float64":
                out[column] = pd.to_numeric(out[column], errors="raise").astype(
                    pd.Float64Dtype()
                )

            elif dtype == "Int64":
                out[column] = pd.to_numeric(out[column], errors="raise").astype(
                    pd.Int64Dtype()
                )

            else:
                out[column] = out[column].astype(dtype)

        except Exception as exc:
            raise TypeError(
                f"Could not coerce column {column!r} in dataset "
                f"{parsed_name.value!r} to dtype {dtype!r}"
            ) from exc

    validate_dtypes(out, parsed_name, allow_extra=allow_extra)

    return out
