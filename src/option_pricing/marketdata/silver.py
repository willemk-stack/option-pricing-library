"""Silver storage helpers for local-first normalized marketdata outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date
from pathlib import Path
from typing import Any

import pandas as pd

from option_pricing.marketdata.cleaning import QuoteCleaningResult
from option_pricing.marketdata.providers.local import LocalSnapshotResult
from option_pricing.marketdata.schemas import (
    CLEANED_QUOTES_SCHEMA_VERSION,
    MARKET_INPUTS_SCHEMA_VERSION,
    REJECTED_QUOTES_SCHEMA_VERSION,
    DatasetName,
)
from option_pricing.marketdata.storage import LocalStorage
from option_pricing.marketdata.validation import validate_dtypes

SILVER_CLEANING_SCHEMA_VERSION = "a3_silver_cleaning.v1"
_CLEANING_POLICY_ID = "quote_cleaning_policy.v1"


@dataclass(frozen=True, slots=True)
class SilverCleaningPaths:
    """Filesystem paths for one Silver quote-cleaning output set."""

    market_inputs: Path
    cleaned_quotes: Path
    rejected_quotes: Path
    manifest: Path


def write_cleaned_quotes_silver(
    storage: LocalStorage,
    *,
    local_snapshot: LocalSnapshotResult,
    market_inputs: pd.DataFrame,
    result: QuoteCleaningResult,
    overwrite: bool = False,
    library_commit: str | None = None,
) -> SilverCleaningPaths:
    """Write normalized market inputs and quote-cleaning outputs to Silver."""

    run_id = _required_run_id(local_snapshot.run_id)
    valuation_timestamp = _utc_timestamp(local_snapshot.asof)
    partitions: dict[str, str | date] = {
        "underlying": local_snapshot.underlying,
        "date": valuation_timestamp.date(),
        "run_id": run_id,
    }

    _validate_frame(market_inputs, DatasetName.MARKET_INPUTS)
    _validate_market_inputs_row_count(market_inputs)
    _validate_frame(result.cleaned_quotes, DatasetName.CLEANED_QUOTES)
    _validate_frame(result.rejected_quotes, DatasetName.REJECTED_QUOTES)
    manifest = _silver_cleaning_manifest(
        local_snapshot,
        market_inputs=market_inputs,
        result=result,
        run_id=run_id,
        valuation_timestamp=valuation_timestamp,
        library_commit=library_commit,
    )

    expected_paths = _expected_paths(storage, partitions)
    _ensure_targets_available(expected_paths, overwrite=overwrite)

    market_inputs_path = storage.write_frame(
        market_inputs,
        layer="silver",
        dataset=DatasetName.MARKET_INPUTS.value,
        partitions=partitions,
        filename="market_inputs.parquet",
        overwrite=overwrite,
    )
    cleaned_quotes_path = storage.write_frame(
        result.cleaned_quotes,
        layer="silver",
        dataset=DatasetName.CLEANED_QUOTES.value,
        partitions=partitions,
        filename="cleaned_quotes.parquet",
        overwrite=overwrite,
    )
    rejected_quotes_path = storage.write_frame(
        result.rejected_quotes,
        layer="silver",
        dataset=DatasetName.REJECTED_QUOTES.value,
        partitions=partitions,
        filename="rejected_quotes.parquet",
        overwrite=overwrite,
    )
    manifest_path = storage.write_manifest(
        manifest,
        layer="silver",
        dataset=DatasetName.CLEANED_QUOTES.value,
        partitions=partitions,
        filename="manifest.json",
        overwrite=overwrite,
    )

    return SilverCleaningPaths(
        market_inputs=market_inputs_path,
        cleaned_quotes=cleaned_quotes_path,
        rejected_quotes=rejected_quotes_path,
        manifest=manifest_path,
    )


def _required_run_id(value: str | None) -> str:
    if value is None:
        raise ValueError("run_id is required to write Silver quote-cleaning outputs")
    run_id = value.strip()
    if not run_id:
        raise ValueError("run_id is required to write Silver quote-cleaning outputs")
    return run_id


def _utc_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(UTC)
    return timestamp.tz_convert(UTC)


def _utc_isoformat(value: pd.Timestamp) -> str:
    return value.to_pydatetime().astimezone(UTC).isoformat().replace("+00:00", "Z")


def _validate_frame(frame: pd.DataFrame, dataset_name: DatasetName) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"{dataset_name.value} must be a pandas DataFrame, "
            f"got {type(frame).__name__}"
        )
    validate_dtypes(frame, dataset_name, allow_extra=False)


def _validate_market_inputs_row_count(market_inputs: pd.DataFrame) -> None:
    if len(market_inputs) != 1:
        raise ValueError(
            "market_inputs must contain exactly one row for Silver cleaning "
            f"manifest fields; found {len(market_inputs)}"
        )


def _expected_paths(
    storage: LocalStorage,
    partitions: Mapping[str, str | date],
) -> SilverCleaningPaths:
    return SilverCleaningPaths(
        market_inputs=_target_path(
            storage,
            dataset=DatasetName.MARKET_INPUTS.value,
            partitions=partitions,
            filename="market_inputs.parquet",
        ),
        cleaned_quotes=_target_path(
            storage,
            dataset=DatasetName.CLEANED_QUOTES.value,
            partitions=partitions,
            filename="cleaned_quotes.parquet",
        ),
        rejected_quotes=_target_path(
            storage,
            dataset=DatasetName.REJECTED_QUOTES.value,
            partitions=partitions,
            filename="rejected_quotes.parquet",
        ),
        manifest=_target_path(
            storage,
            dataset=DatasetName.CLEANED_QUOTES.value,
            partitions=partitions,
            filename="manifest.json",
        ),
    )


def _target_path(
    storage: LocalStorage,
    *,
    dataset: str,
    partitions: Mapping[str, str | date],
    filename: str,
) -> Path:
    # Mirror the local Bronze writer's preflight path checks so Silver writes
    # fail before any artifact is created when overwrite is disabled.
    ordered_partitions = storage._ordered_partitions(
        layer="silver",
        dataset=dataset,
        partitions=partitions,
    )
    return (
        storage._dataset_dir(
            layer="silver",
            dataset=dataset,
            ordered_partitions=ordered_partitions,
        )
        / filename
    )


def _ensure_targets_available(
    paths: SilverCleaningPaths,
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    for path in (
        paths.market_inputs,
        paths.cleaned_quotes,
        paths.rejected_quotes,
        paths.manifest,
    ):
        if path.exists():
            raise FileExistsError(
                f"{path} already exists; pass overwrite=True to replace it"
            )


def _silver_cleaning_manifest(
    local_snapshot: LocalSnapshotResult,
    *,
    market_inputs: pd.DataFrame,
    result: QuoteCleaningResult,
    run_id: str,
    valuation_timestamp: pd.Timestamp,
    library_commit: str | None,
) -> dict[str, Any]:
    market_row = market_inputs.iloc[0]
    return {
        "silver_schema_version": SILVER_CLEANING_SCHEMA_VERSION,
        "cleaning_policy": _CLEANING_POLICY_ID,
        "market_inputs_schema_version": MARKET_INPUTS_SCHEMA_VERSION,
        "cleaned_quotes_schema_version": CLEANED_QUOTES_SCHEMA_VERSION,
        "rejected_quotes_schema_version": REJECTED_QUOTES_SCHEMA_VERSION,
        "fixture_name": local_snapshot.fixture_name,
        "snapshot_id": local_snapshot.snapshot_id,
        "run_id": run_id,
        "source_type": "local_fixture",
        "underlying": local_snapshot.underlying,
        "valuation_timestamp_utc": _utc_isoformat(valuation_timestamp),
        "spot": _float_field(market_row, "spot"),
        "rate": _float_field(market_row, "rate"),
        "dividend_yield": _float_field(market_row, "dividend_yield"),
        "day_count": _text_field(market_row, "day_count"),
        "rows": {
            "market_inputs": int(len(market_inputs)),
            "option_chain_input": int(len(local_snapshot.option_chain_raw)),
            "accepted": int(len(result.cleaned_quotes)),
            "rejected": int(len(result.rejected_quotes)),
        },
        "reason_counts": {
            str(reason): int(count)
            for reason, count in sorted(result.reason_counts.items())
        },
        "warnings": list(result.warnings),
        "artifacts": {
            "market_inputs": "market_inputs.parquet",
            "cleaned_quotes": "cleaned_quotes.parquet",
            "rejected_quotes": "rejected_quotes.parquet",
        },
        "library_commit": _optional_text(library_commit),
    }


def _float_field(row: pd.Series, column: str) -> float | None:
    value = row[column]
    if pd.isna(value):
        return None
    return float(value)


def _text_field(row: pd.Series, column: str) -> str | None:
    value = row[column]
    if pd.isna(value):
        return None
    return str(value)


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("library_commit must be a non-empty string when provided")
    return cleaned


__all__ = [
    "SILVER_CLEANING_SCHEMA_VERSION",
    "SilverCleaningPaths",
    "write_cleaned_quotes_silver",
]
