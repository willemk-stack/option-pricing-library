from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from option_pricing.marketdata.contracts import RunMetadata
from option_pricing.marketdata.provider_diagnostics import _diagnostics_payload
from option_pricing.marketdata.provider_results import ProviderCallDiagnostic
from option_pricing.marketdata.provider_serialization import (
    _jsonable_provider_value,
    _sanitized_request_metadata,
)
from option_pricing.marketdata.schemas import DatasetName
from option_pricing.marketdata.storage import LocalStorage, PartitionValue

FRED_BACKFILL_BRONZE_SCHEMA_VERSION = "fred_backfill_bronze.v1"
FRED_BACKFILL_SILVER_SCHEMA_VERSION = "fred_backfill_silver.v1"
EQUITY_BARS_BACKFILL_BRONZE_SCHEMA_VERSION = "equity_bars_backfill_bronze.v1"
EQUITY_BARS_BACKFILL_SILVER_SCHEMA_VERSION = "equity_bars_backfill_silver.v1"


def _backfill_metadata(
    *,
    run_id: str | None,
    prefix: str,
    started_at: datetime,
    library_commit: str | None,
) -> RunMetadata:
    cleaned_run_id = _optional_run_id(run_id)
    cleaned_library_commit = _optional_text(library_commit, "library_commit")
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    return RunMetadata(
        run_id=cleaned_run_id or f"{prefix}-{timestamp}-{uuid4().hex[:8]}",
        asof=started_at,
        started_at=started_at,
        git_sha=cleaned_library_commit,
    )


def _clean_fred_series_ids(series_ids: str | Sequence[str]) -> tuple[str, ...]:
    return tuple(
        _required_text(value, "series_id").upper()
        for value in _one_or_more_text_values(series_ids, "series_ids")
    )


def _clean_alpaca_symbols(symbols: str | Sequence[str]) -> tuple[str, ...]:
    return tuple(
        _required_text(value, "symbol").upper()
        for value in _one_or_more_text_values(symbols, "symbols")
    )


def _one_or_more_text_values(
    values: str | Sequence[str],
    field_name: str,
) -> tuple[str, ...]:
    raw_values: tuple[str, ...]
    if isinstance(values, str):
        raw_values = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        raw_values = tuple(values)
    else:
        raise TypeError(f"{field_name} must be a string or sequence of strings")

    if not raw_values:
        raise ValueError(f"{field_name} must contain at least one value")
    for value in raw_values:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must contain only strings")
    return raw_values


def _coerce_backfill_date(value: date | datetime | str, field_name: str) -> date:
    if isinstance(value, datetime):
        timestamp = pd.Timestamp(value)
    elif isinstance(value, date):
        return value
    else:
        timestamp = pd.Timestamp(value)

    if pd.isna(timestamp):
        raise ValueError(f"{field_name} must not be missing")
    if timestamp.tzinfo is None:
        return timestamp.date()
    return timestamp.tz_convert(UTC).date()


def _coerce_backfill_timestamp(
    value: date | datetime | str,
    field_name: str,
) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"{field_name} must not be missing")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(UTC)
    return timestamp.tz_convert(UTC)


def _preflight_fred_backfill_targets(
    storage: LocalStorage,
    *,
    series_ids: Sequence[str],
    start_date: date,
    end_date: date,
    run_id: str,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    for series_id in series_ids:
        for path in _expected_fred_backfill_target_paths(
            storage,
            series_id=series_id,
            start_date=start_date,
            end_date=end_date,
            run_id=run_id,
        ):
            if path.exists():
                raise FileExistsError(
                    f"{path} already exists; pass overwrite=True to replace it"
                )


def _preflight_bars_backfill_targets(
    storage: LocalStorage,
    *,
    symbols: Sequence[str],
    timeframe: str,
    start_date: date,
    end_date: date,
    run_id: str,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    for symbol in symbols:
        for path in _expected_bars_backfill_target_paths(
            storage,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            run_id=run_id,
        ):
            if path.exists():
                raise FileExistsError(
                    f"{path} already exists; pass overwrite=True to replace it"
                )


def _expected_fred_backfill_target_paths(
    storage: LocalStorage,
    *,
    series_id: str,
    start_date: date,
    end_date: date,
    run_id: str,
) -> tuple[Path, ...]:
    partitions = _fred_backfill_partitions(
        series_id=series_id,
        start_date=start_date,
        end_date=end_date,
        run_id=run_id,
    )
    return (
        _target_path(
            storage,
            layer="bronze",
            dataset=DatasetName.FRED_SERIES.value,
            partitions=partitions,
            filename="observations.json",
        ),
        _target_path(
            storage,
            layer="bronze",
            dataset=DatasetName.FRED_SERIES.value,
            partitions=partitions,
            filename="manifest.json",
        ),
        _target_path(
            storage,
            layer="silver",
            dataset=DatasetName.FRED_SERIES.value,
            partitions=partitions,
            filename="fred_series.parquet",
        ),
        _target_path(
            storage,
            layer="silver",
            dataset=DatasetName.FRED_SERIES.value,
            partitions=partitions,
            filename="manifest.json",
        ),
    )


def _expected_bars_backfill_target_paths(
    storage: LocalStorage,
    *,
    symbol: str,
    timeframe: str,
    start_date: date,
    end_date: date,
    run_id: str,
) -> tuple[Path, ...]:
    partitions = _bars_backfill_partitions(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        run_id=run_id,
    )
    return (
        _target_path(
            storage,
            layer="bronze",
            dataset=DatasetName.EQUITY_BARS.value,
            partitions=partitions,
            filename="bars.json",
        ),
        _target_path(
            storage,
            layer="bronze",
            dataset=DatasetName.EQUITY_BARS.value,
            partitions=partitions,
            filename="manifest.json",
        ),
        _target_path(
            storage,
            layer="silver",
            dataset=DatasetName.EQUITY_BARS.value,
            partitions=partitions,
            filename="equity_bars.parquet",
        ),
        _target_path(
            storage,
            layer="silver",
            dataset=DatasetName.EQUITY_BARS.value,
            partitions=partitions,
            filename="manifest.json",
        ),
    )


def _fred_backfill_partitions(
    *,
    series_id: str,
    start_date: date,
    end_date: date,
    run_id: str,
) -> dict[str, PartitionValue]:
    return {
        "series_id": series_id,
        "start_date": start_date,
        "end_date": end_date,
        "run_id": run_id,
    }


def _bars_backfill_partitions(
    *,
    symbol: str,
    timeframe: str,
    start_date: date,
    end_date: date,
    run_id: str,
) -> dict[str, PartitionValue]:
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "run_id": run_id,
    }


def _provider_backfill_payload_document(
    payload: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "request": _sanitized_request_metadata(request),
        "payload": _jsonable_provider_value(payload),
    }


def _fred_backfill_manifest(
    metadata: RunMetadata,
    *,
    series_id: str,
    start_date: date,
    end_date: date,
    request_metadata: Mapping[str, Any],
    raw_rows: int,
    normalized_rows: int,
    layer: str,
    artifacts: Mapping[str, str],
    warnings: Sequence[str],
    library_commit: str | None,
    diagnostics: Sequence[ProviderCallDiagnostic] = (),
) -> dict[str, object]:
    return {
        "schema_version": (
            FRED_BACKFILL_BRONZE_SCHEMA_VERSION
            if layer == "bronze"
            else FRED_BACKFILL_SILVER_SCHEMA_VERSION
        ),
        "operation": "backfill_fred",
        "run_id": metadata.run_id,
        "source_type": "provider_backfill",
        "provider": "fred",
        "series_id": series_id,
        "start_date": start_date,
        "end_date": end_date,
        "request_metadata": _sanitized_request_metadata(request_metadata),
        "rows": {"raw": raw_rows, "normalized": normalized_rows},
        "provider_operation_diagnostics": _diagnostics_payload(diagnostics),
        "warnings": list(warnings),
        "artifacts": dict(artifacts),
        "library_commit": library_commit,
    }


def _bars_backfill_manifest(
    metadata: RunMetadata,
    *,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeframe: str,
    feed: str,
    request_metadata: Mapping[str, Any],
    raw_rows: int,
    normalized_rows: int,
    layer: str,
    artifacts: Mapping[str, str],
    warnings: Sequence[str],
    library_commit: str | None,
    diagnostics: Sequence[ProviderCallDiagnostic] = (),
) -> dict[str, object]:
    return {
        "schema_version": (
            EQUITY_BARS_BACKFILL_BRONZE_SCHEMA_VERSION
            if layer == "bronze"
            else EQUITY_BARS_BACKFILL_SILVER_SCHEMA_VERSION
        ),
        "operation": "backfill_bars",
        "run_id": metadata.run_id,
        "source_type": "provider_backfill",
        "provider": "alpaca",
        "symbol": symbol,
        "start": start,
        "end": end,
        "timeframe": timeframe,
        "feed": feed,
        "request_metadata": _sanitized_request_metadata(request_metadata),
        "rows": {"raw": raw_rows, "normalized": normalized_rows},
        "provider_operation_diagnostics": _diagnostics_payload(diagnostics),
        "warnings": list(warnings),
        "artifacts": dict(artifacts),
        "library_commit": library_commit,
    }


def _backfill_run_details(
    *,
    operation: str,
    provider: str,
    targets: Sequence[str],
    start: date | datetime | pd.Timestamp,
    end: date | datetime | pd.Timestamp,
    rows_in: int,
    rows_out: int,
    requests: Sequence[Mapping[str, object]],
    warnings: Sequence[str],
    library_commit: str | None,
    diagnostics: Sequence[ProviderCallDiagnostic] = (),
) -> dict[str, object]:
    return {
        "operation": operation,
        "provider": provider,
        "targets": list(targets),
        "start": start,
        "end": end,
        "rows": {"raw": rows_in, "normalized": rows_out},
        "requests": [_sanitized_request_metadata(request) for request in requests],
        "provider_operation_diagnostics": _diagnostics_payload(diagnostics),
        "warnings": list(warnings),
        "library_commit": library_commit,
    }


def _count_fred_observations(payload: Mapping[str, Any]) -> int:
    observations = payload.get("observations")
    return len(observations) if isinstance(observations, list) else 0


def _bars_frame_for_symbol(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    matches = frame["symbol"].astype("string").str.upper() == symbol
    return frame.loc[matches].reset_index(drop=True)


def _count_alpaca_bars_for_symbol(payload: Mapping[str, Any], symbol: str) -> int:
    bars_payload = _alpaca_bars_payload(payload)
    if isinstance(bars_payload, Mapping):
        records = _symbol_mapping_value(bars_payload, symbol)
        return 0 if records is None else _count_bar_records(records)
    if isinstance(bars_payload, Sequence) and not isinstance(
        bars_payload,
        (str, bytes, bytearray),
    ):
        return sum(
            1
            for record in bars_payload
            if _record_symbol_matches(record, symbol)
            or _payload_targets_single_symbol(payload, symbol)
        )
    return 1 if _record_symbol_matches(bars_payload, symbol) else 0


def _alpaca_bars_payload_for_symbol(
    payload: Mapping[str, Any],
    symbol: str,
) -> dict[str, object]:
    out = dict(payload)
    out["symbols"] = [symbol]
    bars_payload = _alpaca_bars_payload(payload)
    if isinstance(bars_payload, Mapping):
        records = _symbol_mapping_value(bars_payload, symbol)
        out["bars"] = {symbol: [] if records is None else records}
    else:
        out["bars"] = bars_payload
    return out


def _alpaca_bars_payload(payload: Mapping[str, Any]) -> Any:
    bars_payload = payload.get("bars", payload.get("bar", payload))
    return getattr(bars_payload, "data", bars_payload)


def _symbol_mapping_value(mapping: Mapping[Any, Any], symbol: str) -> Any | None:
    for key, value in mapping.items():
        if str(key).strip().upper() == symbol:
            return value
    return None


def _count_bar_records(records: Any) -> int:
    records = getattr(records, "data", records)
    if _is_bar_record_like(records):
        return 1
    if isinstance(records, Sequence) and not isinstance(
        records,
        (str, bytes, bytearray),
    ):
        return len(records)
    return 0


def _record_symbol_matches(record: Any, symbol: str) -> bool:
    record_symbol = _provider_record_value(record, ("symbol", "S", "s"))
    if record_symbol is None:
        return False
    return str(record_symbol).strip().upper() == symbol


def _payload_targets_single_symbol(payload: Mapping[str, Any], symbol: str) -> bool:
    payload_symbols = payload.get("symbols", payload.get("symbol"))
    if isinstance(payload_symbols, str):
        return payload_symbols.strip().upper() == symbol
    if isinstance(payload_symbols, Sequence) and not isinstance(
        payload_symbols,
        (bytes, bytearray),
    ):
        cleaned = [str(value).strip().upper() for value in payload_symbols]
        return cleaned == [symbol]
    return False


def _is_bar_record_like(value: Any) -> bool:
    if isinstance(value, Mapping):
        raw_data = value.get("raw_data")
        if isinstance(raw_data, Mapping) and _is_bar_record_like(raw_data):
            return True
        return any(
            key in value
            for key in (
                "timestamp",
                "t",
                "open",
                "o",
                "high",
                "h",
                "low",
                "l",
                "close",
                "c",
                "volume",
                "v",
            )
        )
    return any(
        hasattr(value, field_name)
        for field_name in ("timestamp", "open", "high", "low", "close", "volume")
    )


def _provider_record_value(record: Any, aliases: Sequence[str]) -> Any | None:
    if isinstance(record, Mapping):
        for alias in aliases:
            if alias in record:
                return record[alias]
        raw_data = record.get("raw_data")
        if isinstance(raw_data, Mapping):
            return _provider_record_value(raw_data, aliases)
        return None

    for alias in aliases:
        try:
            return getattr(record, alias)
        except AttributeError:
            continue
    raw_data = getattr(record, "raw_data", None)
    if isinstance(raw_data, Mapping):
        return _provider_record_value(raw_data, aliases)
    return None


def _required_run_id(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("run_id must be a string")
    run_id = value.strip()
    if not run_id:
        raise ValueError("run_id is required")
    return run_id


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    return cleaned


def _optional_text(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _optional_run_id(value: str | None) -> str | None:
    if value is None:
        return None
    return _required_run_id(value)


def _target_path(
    storage: LocalStorage,
    *,
    layer: str,
    dataset: str,
    partitions: dict[str, PartitionValue],
    filename: str,
) -> Path:
    return (
        storage.dataset_dir(
            layer=layer,
            dataset=dataset,
            partitions=partitions,
        )
        / filename
    )
