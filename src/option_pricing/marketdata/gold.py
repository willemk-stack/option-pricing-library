"""Gold conversion helpers for library-ready marketdata objects."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from option_pricing.marketdata.schemas import (
    HESTON_QUOTES_COLUMNS,
    HESTON_QUOTES_SCHEMA_VERSION,
    DatasetName,
)
from option_pricing.marketdata.storage import LocalStorage, PartitionValue
from option_pricing.marketdata.validation import validate_dtypes
from option_pricing.types import MarketData

if TYPE_CHECKING:
    from option_pricing.models.heston.calibration.heston_types import HestonQuoteSet

GOLD_MARKET_DATA_SCHEMA_VERSION = "gold_market_data.v1"
GOLD_CONVERSION_MANIFEST_VERSION = "gold_conversion_manifest.v1"
_OPTION_RIGHTS = frozenset({"call", "put"})


@dataclass(frozen=True, slots=True)
class GoldMarketDataSnapshot:
    """MarketData plus the metadata needed to serialize a Gold snapshot."""

    market_data: MarketData
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class GoldHestonQuotesResult:
    """Result contract for Gold Heston quote conversion."""

    heston_quotes: pd.DataFrame
    quote_count: int
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GoldConversionPaths:
    """Filesystem paths for a persisted Gold conversion output set."""

    market_data: Path
    market_manifest: Path
    heston_quotes: Path
    heston_manifest: Path


def build_market_data_snapshot(
    market_inputs: pd.DataFrame,
    *,
    run_id: str,
    snapshot_id: str,
    cleaning_policy: str,
    library_commit: str | None = None,
) -> GoldMarketDataSnapshot:
    """Convert one normalized ``market_inputs`` row into a Gold MarketData snapshot."""

    _require_market_inputs_frame(market_inputs)
    _require_single_row(market_inputs)
    run_id = _required_text_value("run_id", run_id)
    snapshot_id = _required_text_value("snapshot_id", snapshot_id)
    cleaning_policy = _required_text_value("cleaning_policy", cleaning_policy)

    row = market_inputs.iloc[0]
    spot = _required_finite_float(row, "spot")
    if spot <= 0.0:
        raise ValueError("market_inputs spot must be finite and > 0")

    rate = _required_finite_float(row, "rate")
    dividend_yield = _required_finite_float(row, "dividend_yield")
    rate_compounding = _required_row_text(row, "rate_compounding")
    if rate_compounding != "continuous":
        raise ValueError(
            "market_inputs rate_compounding must be 'continuous'; "
            f"got {rate_compounding!r}"
        )

    day_count = _required_row_text(row, "day_count")
    if day_count != "ACT/365":
        raise ValueError(
            f"market_inputs day_count must be 'ACT/365'; got {day_count!r}"
        )

    metadata: dict[str, object] = {
        "schema_version": GOLD_MARKET_DATA_SCHEMA_VERSION,
        "underlying": _required_row_text(row, "underlying"),
        "valuation_timestamp_utc": _utc_isoformat(row["asof"]),
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "sources": {
            "spot_source": _required_row_text(row, "spot_source"),
            "rate_source": _required_row_text(row, "rate_source"),
            "dividend_yield_source": _required_row_text(row, "dividend_yield_source"),
        },
        "rate_compounding": rate_compounding,
        "day_count": day_count,
        "quote_cleaning_policy": cleaning_policy,
        "library_commit": _optional_text("library_commit", library_commit),
    }
    return GoldMarketDataSnapshot(
        market_data=MarketData(
            spot=spot,
            rate=rate,
            dividend_yield=dividend_yield,
        ),
        metadata=metadata,
    )


def build_heston_quotes(cleaned_quotes: pd.DataFrame) -> GoldHestonQuotesResult:
    """Convert schema-cleaned A3 quotes into Gold Heston quote artifacts."""

    _require_cleaned_quotes_frame(cleaned_quotes)
    _require_non_empty_frame(cleaned_quotes, "cleaned_quotes")
    _require_call_put_values(cleaned_quotes, "right", "cleaned_quotes")

    source = cleaned_quotes.reset_index(drop=True)
    heston_quotes = pd.DataFrame(
        {
            "underlying": source["underlying"],
            "contract_symbol": source["contract_symbol"],
            "quote_id": source["quote_id"],
            "asof": source["asof"],
            "expiry": source["expiry"],
            "expiry_years": source["expiry_years"],
            "strike": source["strike"],
            "right": source["right"],
            "mid": source["mid"],
            "bid": source["bid"],
            "ask": source["ask"],
            "iv": source["iv"],
            "vega": source["vega"],
            "option_type": source["right"],
            "label": source["contract_symbol"],
            "source": source["source"],
            "cleaning_policy": source["cleaning_policy"],
        },
        columns=HESTON_QUOTES_COLUMNS,
    )

    validate_dtypes(heston_quotes, DatasetName.HESTON_QUOTES, allow_extra=False)
    warnings = _optional_heston_quote_warnings(heston_quotes)
    return GoldHestonQuotesResult(
        heston_quotes=heston_quotes,
        quote_count=int(len(heston_quotes)),
        warnings=warnings,
    )


def heston_quote_set_from_frame(
    heston_quotes: pd.DataFrame,
    market_data: MarketData,
) -> HestonQuoteSet:
    """Reconstruct a HestonQuoteSet from a Gold Heston quote artifact."""

    from option_pricing.models.heston.calibration.heston_types import HestonQuoteSet

    _require_heston_quotes_frame(heston_quotes)
    _require_non_empty_frame(heston_quotes, "heston_quotes")
    _require_market_data(market_data)
    _require_call_put_values(heston_quotes, "right", "heston_quotes")
    _require_call_put_values(heston_quotes, "option_type", "heston_quotes")
    _require_matching_option_type(heston_quotes)

    iv_mid = _optional_heston_float_array(
        heston_quotes,
        "iv",
        strictly_positive=True,
    )
    bs_vega = _optional_heston_float_array(
        heston_quotes,
        "vega",
        strictly_positive=False,
    )
    warnings = _optional_heston_quote_warnings(heston_quotes)
    metadata: dict[str, object] = {
        "schema_version": HESTON_QUOTES_SCHEMA_VERSION,
        "quote_count": int(len(heston_quotes)),
        "underlying": _single_text_metadata_value(heston_quotes, "underlying"),
        "asof": _single_timestamp_metadata_value(heston_quotes, "asof"),
        "cleaning_policy": _single_text_metadata_value(
            heston_quotes,
            "cleaning_policy",
        ),
        "iv_mid_included": iv_mid is not None,
        "bs_vega_included": bs_vega is not None,
    }
    if warnings:
        metadata["optional_data_warnings"] = warnings

    return HestonQuoteSet.from_flat_market(
        market=market_data,
        strike=_float_array(heston_quotes, "strike"),
        expiry=_float_array(heston_quotes, "expiry_years"),
        is_call=(heston_quotes["right"] == "call").to_numpy(dtype=np.bool_),
        mid=_float_array(heston_quotes, "mid"),
        bid=_float_array(heston_quotes, "bid"),
        ask=_float_array(heston_quotes, "ask"),
        iv_mid=iv_mid,
        bs_vega=bs_vega,
        labels=tuple(heston_quotes["label"].astype(str)),
        metadata=metadata,
    )


def market_data_snapshot_to_json(
    snapshot: GoldMarketDataSnapshot,
) -> dict[str, object]:
    """Return the JSON-serializable Gold ``market_data.json`` payload."""

    if not isinstance(snapshot, GoldMarketDataSnapshot):
        raise TypeError("snapshot must be a GoldMarketDataSnapshot")

    schema_version = snapshot.metadata.get("schema_version")
    if schema_version != GOLD_MARKET_DATA_SCHEMA_VERSION:
        raise ValueError(
            "Gold market data snapshot metadata schema_version must be "
            f"{GOLD_MARKET_DATA_SCHEMA_VERSION!r}"
        )

    payload = dict(snapshot.metadata)
    payload["market_data"] = {
        "spot": float(snapshot.market_data.spot),
        "rate": float(snapshot.market_data.rate),
        "dividend_yield": float(snapshot.market_data.dividend_yield),
    }
    return payload


def market_data_snapshot_from_json(
    payload: Mapping[str, object],
) -> GoldMarketDataSnapshot:
    """Rehydrate a Gold ``market_data.json`` payload into a MarketData snapshot."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")

    schema_version = payload.get("schema_version")
    if schema_version != GOLD_MARKET_DATA_SCHEMA_VERSION:
        raise ValueError(
            "market_data.json schema_version must be "
            f"{GOLD_MARKET_DATA_SCHEMA_VERSION!r}; got {schema_version!r}"
        )

    market_data_payload = payload.get("market_data")
    if not isinstance(market_data_payload, Mapping):
        raise ValueError("market_data.json must contain a market_data object")

    spot = _payload_finite_float(market_data_payload, "spot")
    if spot <= 0.0:
        raise ValueError("market_data spot must be finite and > 0")
    rate = _payload_finite_float(market_data_payload, "rate")
    dividend_yield = _payload_finite_float(market_data_payload, "dividend_yield")

    metadata = dict(payload)
    metadata.pop("market_data", None)
    _validate_loaded_metadata(metadata)
    return GoldMarketDataSnapshot(
        market_data=MarketData(
            spot=spot,
            rate=rate,
            dividend_yield=dividend_yield,
        ),
        metadata=metadata,
    )


def write_market_data_gold(
    storage: LocalStorage,
    *,
    snapshot: GoldMarketDataSnapshot,
    partitions: Mapping[str, PartitionValue],
    overwrite: bool = False,
) -> Path:
    """Write a Gold ``market_data.json`` snapshot to local storage."""

    return storage.write_json(
        market_data_snapshot_to_json(snapshot),
        layer="gold",
        dataset=DatasetName.MARKET_SNAPSHOT.value,
        partitions=partitions,
        filename="market_data.json",
        overwrite=overwrite,
    )


def _require_market_inputs_frame(market_inputs: pd.DataFrame) -> None:
    if not isinstance(market_inputs, pd.DataFrame):
        raise TypeError(
            "market_inputs must be a pandas DataFrame, "
            f"got {type(market_inputs).__name__}"
        )
    validate_dtypes(market_inputs, DatasetName.MARKET_INPUTS, allow_extra=False)


def _require_cleaned_quotes_frame(cleaned_quotes: pd.DataFrame) -> None:
    if not isinstance(cleaned_quotes, pd.DataFrame):
        raise TypeError(
            "cleaned_quotes must be a pandas DataFrame, "
            f"got {type(cleaned_quotes).__name__}"
        )
    validate_dtypes(cleaned_quotes, DatasetName.CLEANED_QUOTES, allow_extra=False)


def _require_heston_quotes_frame(heston_quotes: pd.DataFrame) -> None:
    if not isinstance(heston_quotes, pd.DataFrame):
        raise TypeError(
            "heston_quotes must be a pandas DataFrame, "
            f"got {type(heston_quotes).__name__}"
        )
    validate_dtypes(heston_quotes, DatasetName.HESTON_QUOTES, allow_extra=False)


def _require_non_empty_frame(frame: pd.DataFrame, name: str) -> None:
    if frame.empty:
        raise ValueError(f"{name} must contain at least one quote")


def _require_single_row(market_inputs: pd.DataFrame) -> None:
    if len(market_inputs) != 1:
        raise ValueError(
            "market_inputs must contain exactly one row for Gold MarketData "
            f"conversion; found {len(market_inputs)}"
        )


def _required_finite_float(row: pd.Series, column: str) -> float:
    value = row[column]
    if pd.isna(value):
        raise ValueError(f"market_inputs {column} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"market_inputs {column} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"market_inputs {column} must be finite")
    return number


def _payload_finite_float(payload: Mapping[str, object], key: str) -> float:
    if key not in payload:
        raise ValueError(f"market_data must contain {key!r}")
    try:
        number = float(cast(Any, payload[key]))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"market_data {key} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"market_data {key} must be finite")
    return number


def _float_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame[column].to_numpy(dtype=np.float64, na_value=np.nan)


def _optional_heston_float_array(
    frame: pd.DataFrame,
    column: str,
    *,
    strictly_positive: bool,
) -> np.ndarray | None:
    values = _float_array(frame, column)
    if not np.all(np.isfinite(values)):
        return None
    if strictly_positive:
        if np.any(values <= 0.0):
            return None
    elif np.any(values < 0.0):
        return None
    return values


def _optional_heston_quote_warnings(frame: pd.DataFrame) -> tuple[str, ...]:
    warnings: list[str] = []
    if _optional_heston_float_array(frame, "iv", strictly_positive=True) is None:
        warnings.append(
            "optional IV values are incomplete or invalid for HestonQuoteSet "
            "reconstruction; iv_mid will be omitted unless every IV is finite "
            "and > 0"
        )
    if _optional_heston_float_array(frame, "vega", strictly_positive=False) is None:
        warnings.append(
            "optional vega values are incomplete or invalid for HestonQuoteSet "
            "reconstruction; bs_vega will be omitted unless every vega is finite "
            "and >= 0"
        )
    return tuple(warnings)


def _require_call_put_values(
    frame: pd.DataFrame,
    column: str,
    dataset_name: str,
) -> None:
    values = frame[column].astype("string")
    invalid = values[~values.isin(_OPTION_RIGHTS)]
    if invalid.empty:
        return

    seen: list[str] = []
    for value in invalid:
        rendered = "<NA>" if pd.isna(value) else str(value)
        if rendered not in seen:
            seen.append(rendered)
    joined = ", ".join(repr(value) for value in seen)
    raise ValueError(
        f"{dataset_name} {column} must contain only 'call' or 'put'; found {joined}"
    )


def _require_matching_option_type(heston_quotes: pd.DataFrame) -> None:
    mismatch = heston_quotes["option_type"].astype("string") != heston_quotes[
        "right"
    ].astype("string")
    if bool(mismatch.any()):
        raise ValueError("heston_quotes option_type must match right for every quote")


def _require_market_data(market_data: MarketData) -> None:
    if not isinstance(market_data, MarketData):
        raise TypeError(
            "market_data must be a MarketData instance, "
            f"got {type(market_data).__name__}"
        )

    spot = _finite_market_data_value(market_data, "spot")
    if spot <= 0.0:
        raise ValueError("market_data spot must be finite and > 0")
    _finite_market_data_value(market_data, "rate")
    _finite_market_data_value(market_data, "dividend_yield")


def _finite_market_data_value(market_data: MarketData, field_name: str) -> float:
    try:
        value = float(getattr(market_data, field_name))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"market_data {field_name} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"market_data {field_name} must be finite")
    return value


def _single_text_metadata_value(frame: pd.DataFrame, column: str) -> str:
    values: list[str] = []
    for value in frame[column]:
        if pd.isna(value):
            rendered = ""
        else:
            rendered = str(value).strip()
        if rendered and rendered not in values:
            values.append(rendered)

    if len(values) != 1:
        raise ValueError(f"heston_quotes {column} must contain exactly one value")
    return values[0]


def _single_timestamp_metadata_value(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"heston_quotes {column} must contain exactly one value")
    return _utc_isoformat(values.iloc[0])


def _required_row_text(row: pd.Series, column: str) -> str:
    value = row[column]
    if pd.isna(value):
        raise ValueError(f"market_inputs {column} must not be missing")
    return _required_text_value(column, value)


def _required_text_value(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string")
    return text


def _optional_text(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    return _required_text_value(name, value)


def _utc_isoformat(value: object) -> str:
    timestamp = pd.Timestamp(cast(Any, value))
    if pd.isna(timestamp):
        raise ValueError("market_inputs asof must not be missing")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(UTC)
    else:
        timestamp = timestamp.tz_convert(UTC)
    return timestamp.to_pydatetime().isoformat().replace("+00:00", "Z")


def _validate_loaded_metadata(metadata: Mapping[str, object]) -> None:
    for key in (
        "underlying",
        "valuation_timestamp_utc",
        "run_id",
        "snapshot_id",
        "rate_compounding",
        "day_count",
        "quote_cleaning_policy",
    ):
        _required_mapping_text(metadata, key)

    if metadata["rate_compounding"] != "continuous":
        raise ValueError("market_data.json rate_compounding must be 'continuous'")
    if metadata["day_count"] != "ACT/365":
        raise ValueError("market_data.json day_count must be 'ACT/365'")

    sources = metadata.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("market_data.json must contain a sources object")
    for key in ("spot_source", "rate_source", "dividend_yield_source"):
        _required_mapping_text(sources, key)


def _required_mapping_text(payload: Mapping[str, object], key: str) -> str:
    if key not in payload:
        raise ValueError(f"market_data.json must contain {key!r}")
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"market_data.json {key} must be a non-empty string")
    return value.strip()


__all__ = [
    "GOLD_CONVERSION_MANIFEST_VERSION",
    "GOLD_MARKET_DATA_SCHEMA_VERSION",
    "GoldConversionPaths",
    "GoldHestonQuotesResult",
    "GoldMarketDataSnapshot",
    "build_heston_quotes",
    "build_market_data_snapshot",
    "heston_quote_set_from_frame",
    "market_data_snapshot_from_json",
    "market_data_snapshot_to_json",
    "write_market_data_gold",
]
