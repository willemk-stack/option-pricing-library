from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

_SECRET_KEY_PARTS = frozenset(
    {"api_key", "apikey", "secret", "token", "authorization", "password"}
)


def _coerce_asof(value: str | pd.Timestamp | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz=UTC)
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("asof must not be missing")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(UTC)
    return timestamp.tz_convert(UTC)


def _utc_isoformat(value: pd.Timestamp) -> str:
    return value.to_pydatetime().astimezone(UTC).isoformat().replace("+00:00", "Z")


def _utc_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(UTC)
    return timestamp.tz_convert(UTC)


def _provider_payload_document(payload: Mapping[str, Any]) -> dict[str, object]:
    return {"payload": _jsonable_provider_value(payload)}


def _jsonable_provider_value(value: Any) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable_provider_value(asdict(value))
    if isinstance(value, pd.Timestamp):
        return _utc_isoformat(_coerce_asof(value))
    if isinstance(value, datetime):
        timestamp = pd.Timestamp(value)
        return _utc_isoformat(_coerce_asof(timestamp))
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        out: dict[str, object] = {}
        for key, item in value.items():
            text_key = str(key)
            if _is_secret_payload_key(text_key):
                out[text_key] = "<redacted>"
            else:
                out[text_key] = _jsonable_provider_value(item)
        return out
    raw_data = getattr(value, "raw_data", None)
    if isinstance(raw_data, Mapping):
        return _jsonable_provider_value(raw_data)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable_provider_value(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return {"type": type(value).__name__}


def _is_secret_payload_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(part in lowered for part in _SECRET_KEY_PARTS)


def _sanitized_request_metadata(request: Mapping[str, Any]) -> dict[str, object]:
    sanitized = _jsonable_provider_value(dict(request))
    if not isinstance(sanitized, dict):
        raise TypeError("request metadata must serialize to a JSON object")
    return sanitized


def _relative_artifact_references(
    storage_root: Path,
    artifact_paths: Sequence[Path],
) -> list[str]:
    references: list[str] = []
    for path in artifact_paths:
        try:
            references.append(path.relative_to(storage_root).as_posix())
        except ValueError:
            references.append(path.as_posix())
    return references
