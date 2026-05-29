"""Gold conversion helpers for library-ready marketdata objects."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

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
_DEFAULT_CLEANING_POLICY_ID = "quote_cleaning_policy.v1"
_OPTION_RIGHTS = frozenset({"call", "put"})


class _GoldLocalSnapshot(Protocol):
    fixture_name: str
    snapshot_id: str
    run_id: str | None
    underlying: str
    asof: pd.Timestamp


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
    _require_heston_quote_conventions(heston_quotes)

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

    _require_market_data(snapshot.market_data)
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

    _require_local_storage(storage)
    return storage.write_json(
        market_data_snapshot_to_json(snapshot),
        layer="gold",
        dataset=DatasetName.MARKET_SNAPSHOT.value,
        partitions=partitions,
        filename="market_data.json",
        overwrite=overwrite,
    )


def write_heston_quotes_gold(
    storage: LocalStorage,
    *,
    heston_quotes: pd.DataFrame,
    partitions: Mapping[str, PartitionValue],
    overwrite: bool = False,
) -> Path:
    """Write a Gold Heston quote artifact to local storage."""

    _require_local_storage(storage)
    _require_heston_quotes_frame(heston_quotes)
    _require_heston_quote_conventions(heston_quotes)
    return storage.write_frame(
        heston_quotes,
        layer="gold",
        dataset=DatasetName.HESTON_QUOTES.value,
        partitions=partitions,
        filename="heston_quotes.parquet",
        overwrite=overwrite,
    )


def write_gold_artifacts(
    storage: LocalStorage,
    *,
    local_snapshot: _GoldLocalSnapshot,
    market_inputs: pd.DataFrame,
    cleaned_quotes: pd.DataFrame,
    rejected_quotes: pd.DataFrame,
    reason_counts: Mapping[str, int],
    warnings: Sequence[str],
    overwrite: bool = False,
    library_commit: str | None = None,
) -> GoldConversionPaths:
    """Write the A4 Gold MarketData and Heston quote artifacts."""

    _require_local_storage(storage)
    run_id = _required_run_id(local_snapshot.run_id)
    underlying = _required_text_value(
        "local_snapshot.underlying", local_snapshot.underlying
    )
    snapshot_id = _required_text_value(
        "local_snapshot.snapshot_id", local_snapshot.snapshot_id
    )
    fixture_name = _required_text_value(
        "local_snapshot.fixture_name", local_snapshot.fixture_name
    )
    library_commit = _optional_text("library_commit", library_commit)

    _require_market_inputs_frame(market_inputs)
    _require_single_row(market_inputs)
    _require_cleaned_quotes_frame(cleaned_quotes)
    _require_rejected_quotes_frame(rejected_quotes)

    valuation_timestamp = _utc_timestamp(
        local_snapshot.asof,
        field_name="local_snapshot.asof",
    )
    market_asof = _utc_timestamp(
        market_inputs.iloc[0]["asof"],
        field_name="market_inputs asof",
    )
    if market_asof != valuation_timestamp:
        raise ValueError(
            "local_snapshot.asof and market_inputs asof must match in UTC; "
            f"got local_snapshot.asof={_utc_isoformat(valuation_timestamp)} and "
            f"market_inputs asof={_utc_isoformat(market_asof)}"
        )

    _require_matching_underlying(
        market_inputs,
        frame_name="market_inputs",
        expected=underlying,
    )
    _require_matching_underlying(
        cleaned_quotes,
        frame_name="cleaned_quotes",
        expected=underlying,
    )
    _require_matching_underlying(
        rejected_quotes,
        frame_name="rejected_quotes",
        expected=underlying,
    )

    cleaning_policy = _cleaning_policy_from_cleaned_quotes(cleaned_quotes)
    partitions: dict[str, PartitionValue] = {
        "underlying": underlying,
        "date": valuation_timestamp.date(),
        "run_id": run_id,
    }
    expected_paths = _expected_gold_paths(storage, partitions)
    _ensure_gold_targets_available(expected_paths, overwrite=overwrite)

    snapshot = build_market_data_snapshot(
        market_inputs,
        run_id=run_id,
        snapshot_id=snapshot_id,
        cleaning_policy=cleaning_policy,
        library_commit=library_commit,
    )
    market_payload = market_data_snapshot_to_json(snapshot)
    heston_result = build_heston_quotes(cleaned_quotes)

    market_manifest = _market_snapshot_manifest(
        local_snapshot,
        run_id=run_id,
        snapshot_id=snapshot_id,
        fixture_name=fixture_name,
        valuation_timestamp=valuation_timestamp,
        market_payload=market_payload,
        market_inputs=market_inputs,
        cleaned_quotes=cleaned_quotes,
        rejected_quotes=rejected_quotes,
        reason_counts=reason_counts,
        warnings=warnings,
        library_commit=library_commit,
    )
    heston_manifest = _heston_quotes_manifest(
        local_snapshot,
        run_id=run_id,
        snapshot_id=snapshot_id,
        fixture_name=fixture_name,
        valuation_timestamp=valuation_timestamp,
        cleaned_quotes=cleaned_quotes,
        rejected_quotes=rejected_quotes,
        heston_result=heston_result,
        reason_counts=reason_counts,
        warnings=warnings,
        library_commit=library_commit,
    )

    market_data_path = write_market_data_gold(
        storage,
        snapshot=snapshot,
        partitions=partitions,
        overwrite=overwrite,
    )
    heston_quotes_path = write_heston_quotes_gold(
        storage,
        heston_quotes=heston_result.heston_quotes,
        partitions=partitions,
        overwrite=overwrite,
    )
    market_manifest_path = storage.write_manifest(
        market_manifest,
        layer="gold",
        dataset=DatasetName.MARKET_SNAPSHOT.value,
        partitions=partitions,
        filename="manifest.json",
        overwrite=overwrite,
    )
    heston_manifest_path = storage.write_manifest(
        heston_manifest,
        layer="gold",
        dataset=DatasetName.HESTON_QUOTES.value,
        partitions=partitions,
        filename="manifest.json",
        overwrite=overwrite,
    )

    return GoldConversionPaths(
        market_data=market_data_path,
        market_manifest=market_manifest_path,
        heston_quotes=heston_quotes_path,
        heston_manifest=heston_manifest_path,
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


def _require_rejected_quotes_frame(rejected_quotes: pd.DataFrame) -> None:
    if not isinstance(rejected_quotes, pd.DataFrame):
        raise TypeError(
            "rejected_quotes must be a pandas DataFrame, "
            f"got {type(rejected_quotes).__name__}"
        )
    validate_dtypes(rejected_quotes, DatasetName.REJECTED_QUOTES, allow_extra=False)


def _require_heston_quotes_frame(heston_quotes: pd.DataFrame) -> None:
    if not isinstance(heston_quotes, pd.DataFrame):
        raise TypeError(
            "heston_quotes must be a pandas DataFrame, "
            f"got {type(heston_quotes).__name__}"
        )
    validate_dtypes(heston_quotes, DatasetName.HESTON_QUOTES, allow_extra=False)


def _require_local_storage(storage: LocalStorage) -> None:
    if not isinstance(storage, LocalStorage):
        raise TypeError(
            "storage must be a LocalStorage instance, " f"got {type(storage).__name__}"
        )


def _require_non_empty_frame(frame: pd.DataFrame, name: str) -> None:
    if frame.empty:
        raise ValueError(f"{name} must contain at least one quote")


def _require_single_row(market_inputs: pd.DataFrame) -> None:
    if len(market_inputs) != 1:
        raise ValueError(
            "market_inputs must contain exactly one row for Gold MarketData "
            f"conversion; found {len(market_inputs)}"
        )


def _required_run_id(value: str | None) -> str:
    if value is None:
        raise ValueError("local_snapshot.run_id is required to write Gold artifacts")
    run_id = value.strip()
    if not run_id:
        raise ValueError("local_snapshot.run_id is required to write Gold artifacts")
    return run_id


def _expected_gold_paths(
    storage: LocalStorage,
    partitions: Mapping[str, PartitionValue],
) -> GoldConversionPaths:
    return GoldConversionPaths(
        market_data=_gold_target_path(
            storage,
            dataset=DatasetName.MARKET_SNAPSHOT.value,
            partitions=partitions,
            filename="market_data.json",
        ),
        market_manifest=_gold_target_path(
            storage,
            dataset=DatasetName.MARKET_SNAPSHOT.value,
            partitions=partitions,
            filename="manifest.json",
        ),
        heston_quotes=_gold_target_path(
            storage,
            dataset=DatasetName.HESTON_QUOTES.value,
            partitions=partitions,
            filename="heston_quotes.parquet",
        ),
        heston_manifest=_gold_target_path(
            storage,
            dataset=DatasetName.HESTON_QUOTES.value,
            partitions=partitions,
            filename="manifest.json",
        ),
    )


def _gold_target_path(
    storage: LocalStorage,
    *,
    dataset: str,
    partitions: Mapping[str, PartitionValue],
    filename: str,
) -> Path:
    ordered_partitions = storage._ordered_partitions(
        layer="gold",
        dataset=dataset,
        partitions=partitions,
    )
    return (
        storage._dataset_dir(
            layer="gold",
            dataset=dataset,
            ordered_partitions=ordered_partitions,
        )
        / filename
    )


def _ensure_gold_targets_available(
    paths: GoldConversionPaths,
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    for path in (
        paths.market_data,
        paths.market_manifest,
        paths.heston_quotes,
        paths.heston_manifest,
    ):
        if path.exists():
            raise FileExistsError(
                f"{path} already exists; pass overwrite=True to replace it"
            )


def _cleaning_policy_from_cleaned_quotes(cleaned_quotes: pd.DataFrame) -> str:
    if cleaned_quotes.empty:
        return _DEFAULT_CLEANING_POLICY_ID

    policies: list[str] = []
    for value in cleaned_quotes["cleaning_policy"]:
        if pd.isna(value):
            rendered = ""
        else:
            rendered = str(value).strip()
        if rendered and rendered not in policies:
            policies.append(rendered)

    if len(policies) != 1:
        raise ValueError(
            "cleaned_quotes cleaning_policy must contain exactly one non-empty value"
        )
    return policies[0]


def _market_snapshot_manifest(
    local_snapshot: _GoldLocalSnapshot,
    *,
    run_id: str,
    snapshot_id: str,
    fixture_name: str,
    valuation_timestamp: pd.Timestamp,
    market_payload: Mapping[str, object],
    market_inputs: pd.DataFrame,
    cleaned_quotes: pd.DataFrame,
    rejected_quotes: pd.DataFrame,
    reason_counts: Mapping[str, int],
    warnings: Sequence[str],
    library_commit: str | None,
) -> dict[str, object]:
    market_data = market_payload.get("market_data")
    if not isinstance(market_data, Mapping):
        raise ValueError("market_data payload must contain market_data object")

    return {
        "conversion_manifest_version": GOLD_CONVERSION_MANIFEST_VERSION,
        "artifact": "market_data",
        "artifact_schema_version": GOLD_MARKET_DATA_SCHEMA_VERSION,
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "underlying": _required_text_value(
            "local_snapshot.underlying",
            local_snapshot.underlying,
        ),
        "valuation_timestamp_utc": _utc_isoformat(valuation_timestamp),
        "library_commit": library_commit,
        "quote_cleaning_policy": _required_mapping_text(
            market_payload,
            "quote_cleaning_policy",
        ),
        "rate_compounding": _required_mapping_text(
            market_payload,
            "rate_compounding",
        ),
        "day_count": _required_mapping_text(market_payload, "day_count"),
        "spot": _payload_finite_float(market_data, "spot"),
        "rate": _payload_finite_float(market_data, "rate"),
        "dividend_yield": _payload_finite_float(market_data, "dividend_yield"),
        "sources": _market_payload_sources(market_payload),
        "row_counts": {
            "market_inputs": int(len(market_inputs)),
            "cleaned_quotes": int(len(cleaned_quotes)),
            "rejected_quotes": int(len(rejected_quotes)),
        },
        "reason_counts": _reason_counts_payload(reason_counts),
        "warnings": _warnings_payload(warnings),
        "artifacts": {
            "market_data": "market_data.json",
        },
        "source": {
            "source_type": "local_fixture",
            "fixture_name": fixture_name,
        },
    }


def _heston_quotes_manifest(
    local_snapshot: _GoldLocalSnapshot,
    *,
    run_id: str,
    snapshot_id: str,
    fixture_name: str,
    valuation_timestamp: pd.Timestamp,
    cleaned_quotes: pd.DataFrame,
    rejected_quotes: pd.DataFrame,
    heston_result: GoldHestonQuotesResult,
    reason_counts: Mapping[str, int],
    warnings: Sequence[str],
    library_commit: str | None,
) -> dict[str, object]:
    return {
        "conversion_manifest_version": GOLD_CONVERSION_MANIFEST_VERSION,
        "artifact": "heston_quotes",
        "artifact_schema_version": HESTON_QUOTES_SCHEMA_VERSION,
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "underlying": _required_text_value(
            "local_snapshot.underlying",
            local_snapshot.underlying,
        ),
        "valuation_timestamp_utc": _utc_isoformat(valuation_timestamp),
        "library_commit": library_commit,
        "quote_cleaning_policy": _cleaning_policy_from_cleaned_quotes(cleaned_quotes),
        "row_counts": {
            "cleaned_quotes": int(len(cleaned_quotes)),
            "rejected_quotes": int(len(rejected_quotes)),
            "heston_quotes": int(heston_result.quote_count),
        },
        "reason_counts": _reason_counts_payload(reason_counts),
        "warnings": _warnings_payload(warnings),
        "optional_data_warnings": list(heston_result.warnings),
        "iv_mid_policy": (
            "included for HestonQuoteSet reconstruction only when every IV is "
            "finite and > 0"
        ),
        "bs_vega_policy": (
            "included for HestonQuoteSet reconstruction only when every vega is "
            "finite and >= 0"
        ),
        "artifacts": {
            "heston_quotes": "heston_quotes.parquet",
        },
        "source": {
            "source_type": "local_fixture",
            "fixture_name": fixture_name,
        },
    }


def _market_payload_sources(
    market_payload: Mapping[str, object],
) -> dict[str, str]:
    sources = market_payload.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("market_data payload must contain sources object")
    return {
        "spot_source": _required_mapping_text(sources, "spot_source"),
        "rate_source": _required_mapping_text(sources, "rate_source"),
        "dividend_yield_source": _required_mapping_text(
            sources,
            "dividend_yield_source",
        ),
    }


def _reason_counts_payload(reason_counts: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(reason_counts, Mapping):
        raise TypeError("reason_counts must be a mapping")
    return {
        str(reason): int(count)
        for reason, count in sorted(
            reason_counts.items(), key=lambda item: str(item[0])
        )
    }


def _warnings_payload(warnings: Sequence[str]) -> list[str]:
    if isinstance(warnings, (str, bytes, bytearray)):
        raise TypeError("warnings must be a sequence of strings")
    return [str(warning) for warning in warnings]


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


def _require_heston_quote_conventions(heston_quotes: pd.DataFrame) -> None:
    _require_call_put_values(heston_quotes, "right", "heston_quotes")
    _require_call_put_values(heston_quotes, "option_type", "heston_quotes")
    _require_matching_option_type(heston_quotes)
    _require_matching_label(heston_quotes)


def _require_matching_label(heston_quotes: pd.DataFrame) -> None:
    mismatch = heston_quotes["label"].astype("string") != heston_quotes[
        "contract_symbol"
    ].astype("string")
    if bool(mismatch.any()):
        raise ValueError(
            "heston_quotes label must match contract_symbol for every quote"
        )


def _require_matching_underlying(
    frame: pd.DataFrame,
    *,
    frame_name: str,
    expected: str,
) -> None:
    if frame.empty:
        return

    values: list[str] = []
    for value in frame["underlying"]:
        if pd.isna(value):
            rendered = ""
        else:
            rendered = str(value).strip()
        if rendered and rendered not in values:
            values.append(rendered)

    if values == [expected]:
        return

    found = ", ".join(repr(value) for value in values) or "<missing>"
    raise ValueError(
        f"{frame_name} underlying must match local_snapshot.underlying "
        f"{expected!r}; found {found}"
    )


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


def _utc_timestamp(value: object, *, field_name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(cast(Any, value))
    if pd.isna(timestamp):
        raise ValueError(f"{field_name} must not be missing")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(UTC)
    return timestamp.tz_convert(UTC)


def _utc_isoformat(value: object) -> str:
    timestamp = _utc_timestamp(value, field_name="market_inputs asof")
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
    "write_gold_artifacts",
    "write_heston_quotes_gold",
    "write_market_data_gold",
]
