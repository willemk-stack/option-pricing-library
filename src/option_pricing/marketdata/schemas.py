"""Marketdata schema contracts and validation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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


@dataclass(frozen=True, slots=True)
class ResultStats:
    rows_in: int = 0
    rows_out: int = 0
    files_written: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PipelineResult[T]:
    value: T
    metadata: RunMetadata
    stats: ResultStats = field(default_factory=ResultStats)


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    frame: pd.DataFrame
    metadata: RunMetadata
    stats: ResultStats = field(default_factory=ResultStats)


@dataclass(frozen=True, slots=True)
class ModelValidationBundleResult:
    manifest: Mapping[str, object]
    manifest_path: Path
    metadata: RunMetadata
    artifact_paths: tuple[Path, ...] = ()
    stats: ResultStats = field(default_factory=ResultStats)


@dataclass(frozen=True, slots=True)
class ResearchBundleResult:
    root_path: Path
    metadata: RunMetadata
    artifact_paths: tuple[Path, ...] = ()
    stats: ResultStats = field(default_factory=ResultStats)


@dataclass(frozen=True, slots=True)
class BackfillResult:
    metadata: RunMetadata
    run_ids: tuple[str, ...] = ()
    artifact_paths: tuple[Path, ...] = ()
    stats: ResultStats = field(default_factory=ResultStats)


#########
# Constants
#########

MARKET_INPUTS_SCHEMA_VERSION = "market_inputs.v1"
OPTION_CHAIN_SCHEMA_VERSION = "option_chain.v1"
CLEANED_QUOTES_SCHEMA_VERSION = "cleaned_quotes.v1"
REJECTED_QUOTES_SCHEMA_VERSION = "rejected_quotes.v1"
HESTON_QUOTES_SCHEMA_VERSION = "heston_quotes.v1"
SURFACE_INPUTS_SCHEMA_VERSION = "surface_inputs.v1"
MODEL_VALIDATION_BUNDLE_VERSION = "model_validation_bundle.v1"

MODEL_VALIDATION_MANIFEST_REQUIRED_FIELDS = (
    "artifact_schema_version",
    "run_id",
    "created_at_utc",
    "library_commit",
    "underlying",
    "valuation_timestamp_utc",
    "spot_source",
    "rate_source",
    "rate_compounding",
    "dividend_yield_source",
    "day_count",
    "quote_cleaning_policy",
    "rows",
    "warnings",
    "artifacts",
)

_SECRET_MANIFEST_KEY_TERMS = frozenset(
    {
        "api_key",
        "secret_key",
        "token",
        "password",
        "alpaca_api_key",
        "alpaca_secret_key",
        "fred_api_key",
    }
)

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

CLEANED_QUOTES_COLUMNS = (
    "underlying",
    "contract_symbol",
    "quote_id",
    "quote_ts",
    "asof",
    "expiry",
    "expiry_years",
    "strike",
    "right",
    "bid",
    "ask",
    "mid",
    "iv",
    "vega",
    "delta",
    "gamma",
    "theta",
    "rho",
    "open_interest",
    "moneyness",
    "source",
    "cleaning_policy",
)

CLEANED_QUOTES_DTYPES: dict[str, PandasSchemaDtype] = {
    "underlying": "string",
    "contract_symbol": "string",
    "quote_id": "string",
    "quote_ts": "datetime64[ns, UTC]",
    "asof": "datetime64[ns, UTC]",
    "expiry": "datetime64[ns]",
    "expiry_years": "Float64",
    "strike": "Float64",
    "right": "string",
    "bid": "Float64",
    "ask": "Float64",
    "mid": "Float64",
    "iv": "Float64",
    "vega": "Float64",
    "delta": "Float64",
    "gamma": "Float64",
    "theta": "Float64",
    "rho": "Float64",
    "open_interest": "Int64",
    "moneyness": "Float64",
    "source": "string",
    "cleaning_policy": "string",
}

REJECTED_QUOTES_COLUMNS = (
    "underlying",
    "contract_symbol",
    "quote_id",
    "quote_ts",
    "asof",
    "expiry",
    "strike",
    "right",
    "bid",
    "ask",
    "mid",
    "iv",
    "vega",
    "source",
    "rejection_reason",
    "rejection_detail",
    "cleaning_policy",
)

REJECTED_QUOTES_DTYPES: dict[str, PandasSchemaDtype] = {
    "underlying": "string",
    "contract_symbol": "string",
    "quote_id": "string",
    "quote_ts": "datetime64[ns, UTC]",
    "asof": "datetime64[ns, UTC]",
    "expiry": "datetime64[ns]",
    "strike": "Float64",
    "right": "string",
    "bid": "Float64",
    "ask": "Float64",
    "mid": "Float64",
    "iv": "Float64",
    "vega": "Float64",
    "source": "string",
    "rejection_reason": "string",
    "rejection_detail": "string",
    "cleaning_policy": "string",
}

HESTON_QUOTES_COLUMNS = (
    "underlying",
    "contract_symbol",
    "quote_id",
    "asof",
    "expiry",
    "expiry_years",
    "strike",
    "right",
    "mid",
    "bid",
    "ask",
    "iv",
    "vega",
    "option_type",
    "label",
    "source",
    "cleaning_policy",
)

HESTON_QUOTES_DTYPES: dict[str, PandasSchemaDtype] = {
    "underlying": "string",
    "contract_symbol": "string",
    "quote_id": "string",
    "asof": "datetime64[ns, UTC]",
    "expiry": "datetime64[ns]",
    "expiry_years": "Float64",
    "strike": "Float64",
    "right": "string",
    "mid": "Float64",
    "bid": "Float64",
    "ask": "Float64",
    "iv": "Float64",
    "vega": "Float64",
    "option_type": "string",
    "label": "string",
    "source": "string",
    "cleaning_policy": "string",
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

MARKET_INPUTS_COLUMNS = (
    "underlying",
    "asof",
    "spot",
    "spot_source",
    "rate",
    "rate_source",
    "rate_observation_date",
    "rate_compounding",
    "dividend_yield",
    "dividend_yield_source",
    "day_count",
)

MARKET_INPUTS_DTYPES: dict[str, PandasSchemaDtype] = {
    "underlying": "string",
    "asof": "datetime64[ns, UTC]",
    "spot": "Float64",
    "spot_source": "string",
    "rate": "Float64",
    "rate_source": "string",
    "rate_observation_date": "datetime64[ns]",
    "rate_compounding": "string",
    "dividend_yield": "Float64",
    "dividend_yield_source": "string",
    "day_count": "string",
}

SURFACE_INPUTS_COLUMNS = (
    "underlying",
    "quote_id",
    "asof",
    "expiry",
    "expiry_years",
    "strike",
    "right",
    "mid",
    "iv",
    "source",
    "cleaning_policy",
)

SURFACE_INPUTS_DTYPES: dict[str, PandasSchemaDtype] = {
    "underlying": "string",
    "quote_id": "string",
    "asof": "datetime64[ns, UTC]",
    "expiry": "datetime64[ns]",
    "expiry_years": "Float64",
    "strike": "Float64",
    "right": "string",
    "mid": "Float64",
    "iv": "Float64",
    "source": "string",
    "cleaning_policy": "string",
}

MODEL_VALIDATION_BUNDLE_COLUMNS: tuple[str, ...] = ()
MODEL_VALIDATION_BUNDLE_DTYPES: dict[str, PandasSchemaDtype] = {}


class DatasetName(StrEnum):
    """
    Enum object for safe internal handling of dataset_name calling
    """

    EQUITY_QUOTES = "equity_quotes"
    EQUITY_BARS = "equity_bars"
    OPTION_CHAIN = "option_chain"
    CLEANED_QUOTES = "cleaned_quotes"
    REJECTED_QUOTES = "rejected_quotes"
    HESTON_QUOTES = "heston_quotes"
    FRED_SERIES = "fred_series"
    MARKET_SNAPSHOT = "market_snapshot"
    MARKET_INPUTS = "market_inputs"
    SURFACE_INPUTS = "surface_inputs"
    MODEL_VALIDATION_BUNDLE = "model_validation_bundle"


DATASET_COLUMNS: dict[DatasetName, tuple[str, ...]] = {
    DatasetName.EQUITY_QUOTES: EQUITY_QUOTES_COLUMNS,
    DatasetName.EQUITY_BARS: EQUITY_BARS_COLUMNS,
    DatasetName.OPTION_CHAIN: OPTION_CHAIN_COLUMNS,
    DatasetName.CLEANED_QUOTES: CLEANED_QUOTES_COLUMNS,
    DatasetName.REJECTED_QUOTES: REJECTED_QUOTES_COLUMNS,
    DatasetName.HESTON_QUOTES: HESTON_QUOTES_COLUMNS,
    DatasetName.FRED_SERIES: FRED_SERIES_COLUMNS,
    DatasetName.MARKET_SNAPSHOT: MARKET_SNAPSHOT_COLUMNS,
    DatasetName.MARKET_INPUTS: MARKET_INPUTS_COLUMNS,
    DatasetName.SURFACE_INPUTS: SURFACE_INPUTS_COLUMNS,
    DatasetName.MODEL_VALIDATION_BUNDLE: MODEL_VALIDATION_BUNDLE_COLUMNS,
}

DATASET_DTYPES: dict[DatasetName, dict[str, PandasSchemaDtype]] = {
    DatasetName.EQUITY_QUOTES: EQUITY_QUOTES_DTYPES,
    DatasetName.EQUITY_BARS: EQUITY_BARS_DTYPES,
    DatasetName.OPTION_CHAIN: OPTION_CHAIN_DTYPES,
    DatasetName.CLEANED_QUOTES: CLEANED_QUOTES_DTYPES,
    DatasetName.REJECTED_QUOTES: REJECTED_QUOTES_DTYPES,
    DatasetName.HESTON_QUOTES: HESTON_QUOTES_DTYPES,
    DatasetName.FRED_SERIES: FRED_SERIES_DTYPES,
    DatasetName.MARKET_SNAPSHOT: MARKET_SNAPSHOT_DTYPES,
    DatasetName.MARKET_INPUTS: MARKET_INPUTS_DTYPES,
    DatasetName.SURFACE_INPUTS: SURFACE_INPUTS_DTYPES,
    DatasetName.MODEL_VALIDATION_BUNDLE: MODEL_VALIDATION_BUNDLE_DTYPES,
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


def _is_secret_manifest_key(key: str) -> bool:
    normalized = key.strip().lower()
    return (
        normalized in _SECRET_MANIFEST_KEY_TERMS
        or "api_key" in normalized
        or "secret_key" in normalized
        or normalized.endswith("token")
        or "password" in normalized
    )


def _find_secret_manifest_keys(value: object, path: str = "") -> list[str]:
    secret_keys: list[str] = []

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            key_path = f"{path}.{key_text}" if path else key_text

            if _is_secret_manifest_key(key_text):
                secret_keys.append(key_path)

            secret_keys.extend(_find_secret_manifest_keys(item, key_path))

    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]" if path else f"[{index}]"
            secret_keys.extend(_find_secret_manifest_keys(item, item_path))

    return secret_keys


def validate_model_validation_manifest(manifest: Mapping[str, object]) -> None:
    """Validate the minimum model-validation bundle manifest contract."""

    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")

    missing = [
        field
        for field in MODEL_VALIDATION_MANIFEST_REQUIRED_FIELDS
        if field not in manifest
    ]

    if missing:
        raise ValueError(
            "model_validation_bundle manifest is missing required fields: " f"{missing}"
        )

    artifact_schema_version = manifest["artifact_schema_version"]
    if artifact_schema_version != MODEL_VALIDATION_BUNDLE_VERSION:
        raise ValueError(
            "model_validation_bundle manifest has artifact_schema_version "
            f"{artifact_schema_version!r}; expected "
            f"{MODEL_VALIDATION_BUNDLE_VERSION!r}"
        )

    secret_keys = _find_secret_manifest_keys(manifest)
    if secret_keys:
        raise ValueError(
            "model_validation_bundle manifest contains secret-looking keys: "
            f"{secret_keys}"
        )


def validate_manifest(
    manifest: Mapping[str, object],
    dataset_name: DatasetName | str = DatasetName.MODEL_VALIDATION_BUNDLE,
) -> None:
    """Validate a dataset manifest when a manifest-level contract exists."""

    parsed_name = parse_dataset_name(dataset_name)
    if parsed_name != DatasetName.MODEL_VALIDATION_BUNDLE:
        raise ValueError(f"No manifest validator is defined for {parsed_name.value!r}")

    validate_model_validation_manifest(manifest)
