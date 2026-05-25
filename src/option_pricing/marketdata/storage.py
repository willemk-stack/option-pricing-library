"""Local filesystem-backed storage for the market-data pipeline.

This module owns filesystem layout and lightweight metadata only. It does not
define schemas; callers are expected to pass frames that already satisfy the
contracts from :mod:`option_pricing.marketdata.schemas`.

The implementation stays intentionally small:

- frames are written as Parquet files under predictable partition directories
- manifests are JSON files stored alongside the data they describe
- runs and artifacts are appended to JSONL registries
- checkpoints are kept in a human-readable JSON document

DuckDB was part of the original sketch, but the project does not currently ship
with DuckDB as a dependency. JSONL + JSON keep the interface usable today
without forcing a heavier storage stack.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from .schemas import RunMetadata, StorageConfig, validate_model_validation_manifest

if TYPE_CHECKING:
    from pandas import DataFrame
else:
    DataFrame = Any

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
type PartitionValue = str | int | float | bool | date | datetime

_LAYERS = frozenset({"bronze", "silver", "gold"})
_DATASET_ALIASES = {
    "market_snapshots": "market_snapshot",
}
_PARTITION_ORDERS: dict[tuple[str, str], tuple[str, ...]] = {
    ("bronze", "equity_quotes"): ("date",),
    ("bronze", "equity_bars"): ("symbol", "timeframe", "date"),
    ("bronze", "option_chain"): ("underlying", "asof_date"),
    ("bronze", "fred_series"): ("series_id", "date"),
    ("silver", "equity_quotes"): ("date",),
    ("silver", "equity_bars"): ("symbol", "timeframe", "date"),
    ("silver", "option_chain"): ("underlying", "asof_date"),
    ("silver", "fred_series"): ("series_id", "date"),
    ("silver", "market_inputs"): ("underlying", "date", "run_id"),
    ("silver", "cleaned_quotes"): ("underlying", "date", "run_id"),
    ("silver", "rejected_quotes"): ("underlying", "date", "run_id"),
    ("gold", "market_snapshot"): ("underlying", "date"),
    ("gold", "curves"): ("date",),
    ("gold", "vol_inputs"): ("underlying", "date"),
    ("gold", "heston_quotes"): ("underlying", "date", "run_id"),
    ("gold", "surface_inputs"): ("underlying", "date", "run_id"),
    ("gold", "model_validation_bundle"): ("underlying", "date", "run_id"),
}


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised only in minimal installs
        raise ImportError(
            "LocalStorage requires pandas. Install `option_pricing[dev]` or add "
            "`pandas` to your environment."
        ) from exc
    return pd


def _parquet_runtime_error(action: str, exc: Exception) -> RuntimeError:
    del exc
    return RuntimeError(
        f"Unable to {action} Parquet data. Install `pyarrow` or `fastparquet` "
        "to use LocalStorage frame IO."
    )


def _normalise_layer(layer: str) -> str:
    normalised = layer.strip().lower()
    if normalised not in _LAYERS:
        raise ValueError(f"layer must be one of {sorted(_LAYERS)}, received {layer!r}")
    return normalised


def _normalise_dataset(dataset: str) -> str:
    normalised = dataset.strip()
    if not normalised:
        raise ValueError("dataset must be a non-empty string")
    return _DATASET_ALIASES.get(normalised, normalised)


def _format_partition_value(value: PartitionValue) -> str:
    if isinstance(value, datetime):
        dt = value.astimezone(UTC) if value.tzinfo is not None else value
        text = dt.strftime("%Y-%m-%dT%H-%M-%SZ")
    elif isinstance(value, date):
        text = value.isoformat()
    elif isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)

    return (
        text.replace("\\", "-")
        .replace("/", "-")
        .replace(":", "-")
        .replace("*", "-")
        .replace("?", "-")
    )


def _jsonify(value: Any) -> JsonValue:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonify(asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        dt = value.astimezone(UTC) if value.tzinfo is not None else value
        return dt.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonify(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")


def _jsonify_object(value: Mapping[str, Any]) -> dict[str, JsonValue]:
    payload = _jsonify(dict(value))
    if not isinstance(payload, dict):
        raise TypeError("Expected a JSON object")
    return payload


def _relative_posix(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


class LocalStorage:
    """Persist market-data artifacts on the local filesystem."""

    def __init__(self, config: StorageConfig | Path) -> None:
        self.config = (
            config if isinstance(config, StorageConfig) else StorageConfig(config)
        )
        self.root = self.config.root
        self.root.mkdir(parents=True, exist_ok=True)

        self.meta_root = self.root / "_meta"
        self.meta_root.mkdir(parents=True, exist_ok=True)

        self.runs_path = self.meta_root / "runs.jsonl"
        self.artifacts_path = self.meta_root / "artifacts.jsonl"
        self.checkpoints_path = self.meta_root / "checkpoints.json"

    def write_frame(
        self,
        frame: DataFrame,
        *,
        dataset: str,
        layer: str = "bronze",
        partitions: Mapping[str, PartitionValue] | None = None,
        filename: str | None = None,
        compression: str | None = None,
        index: bool = False,
        overwrite: bool = False,
    ) -> Path:
        """Write a DataFrame to a partitioned Parquet file."""

        pd = _require_pandas()
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas.DataFrame")

        layer_name = _normalise_layer(layer)
        dataset_name = _normalise_dataset(dataset)
        ordered_partitions = self._ordered_partitions(
            layer=layer_name,
            dataset=dataset_name,
            partitions=partitions,
        )
        target_dir = self._dataset_dir(
            layer=layer_name,
            dataset=dataset_name,
            ordered_partitions=ordered_partitions,
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / self._frame_filename(filename)
        parquet_compression = compression or self.config.compression
        if target_path.exists() and not overwrite:
            raise FileExistsError(
                f"{target_path} already exists; pass overwrite=True to replace it"
            )

        try:
            frame.to_parquet(
                target_path,
                compression=parquet_compression,
                index=index,
            )
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            raise _parquet_runtime_error("write", exc) from exc

        self._append_jsonl(
            self.artifacts_path,
            {
                "artifact_type": "frame",
                "dataset": dataset_name,
                "layer": layer_name,
                "partitions": self._serialise_partitions(ordered_partitions),
                "path": _relative_posix(self.root, target_path),
                "rows": int(len(frame)),
                "columns": [str(column) for column in frame.columns],
                "written_at": _utcnow().isoformat(),
            },
        )
        return target_path

    def read_frame(
        self,
        *,
        dataset: str,
        layer: str = "bronze",
        partitions: Mapping[str, PartitionValue] | None = None,
        columns: Sequence[str] | None = None,
    ) -> DataFrame:
        """Read one logical dataset slice back into a DataFrame."""

        pd = _require_pandas()
        layer_name = _normalise_layer(layer)
        dataset_name = _normalise_dataset(dataset)
        ordered_partitions = self._ordered_partitions(
            layer=layer_name,
            dataset=dataset_name,
            partitions=partitions,
        )
        dataset_root = self._dataset_root(layer=layer_name, dataset=dataset_name)
        target_dir = self._dataset_dir(
            layer=layer_name,
            dataset=dataset_name,
            ordered_partitions=ordered_partitions,
        )

        if not target_dir.exists():
            raise FileNotFoundError(f"No dataset found at {target_dir}")

        files = sorted(target_dir.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No Parquet files found under {target_dir}")

        selected_columns = list(columns) if columns is not None else None
        requested_columns = set(columns) if columns is not None else None

        frames: list[DataFrame] = []
        for path in files:
            try:
                chunk = cast(DataFrame, pd.read_parquet(path, columns=selected_columns))
            except (ImportError, ModuleNotFoundError, ValueError) as exc:
                raise _parquet_runtime_error("read", exc) from exc

            for key, value in self._extract_partition_values(
                file_path=path,
                dataset_root=dataset_root,
            ).items():
                if key in chunk.columns:
                    continue
                if requested_columns is not None and key not in requested_columns:
                    continue
                chunk[key] = value
            frames.append(chunk)

        if len(frames) == 1:
            return frames[0]
        return cast(DataFrame, pd.concat(frames, ignore_index=True, sort=False))

    def write_manifest(
        self,
        manifest: Mapping[str, Any],
        *,
        dataset: str,
        layer: str = "bronze",
        partitions: Mapping[str, PartitionValue] | None = None,
        filename: str = "manifest.json",
        overwrite: bool = False,
    ) -> Path:
        """Write a JSON manifest next to a dataset partition."""

        layer_name = _normalise_layer(layer)
        dataset_name = _normalise_dataset(dataset)
        ordered_partitions = self._ordered_partitions(
            layer=layer_name,
            dataset=dataset_name,
            partitions=partitions,
        )
        target_dir = self._dataset_dir(
            layer=layer_name,
            dataset=dataset_name,
            ordered_partitions=ordered_partitions,
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = target_dir / self._json_filename(filename)
        if manifest_path.exists() and not overwrite:
            raise FileExistsError(
                f"{manifest_path} already exists; pass overwrite=True to replace it"
            )

        payload = _jsonify_object(manifest)
        payload.setdefault("dataset", dataset_name)
        payload.setdefault("layer", layer_name)
        payload.setdefault("partitions", self._serialise_partitions(ordered_partitions))
        payload.setdefault("written_at", _utcnow().isoformat())

        if dataset_name == "model_validation_bundle":
            validate_model_validation_manifest(payload)

        self._write_json_atomic(manifest_path, payload)
        self._append_jsonl(
            self.artifacts_path,
            {
                "artifact_type": "manifest",
                "dataset": dataset_name,
                "layer": layer_name,
                "partitions": self._serialise_partitions(ordered_partitions),
                "path": _relative_posix(self.root, manifest_path),
                "written_at": payload["written_at"],
            },
        )
        return manifest_path

    def record_run(
        self,
        metadata: RunMetadata,
        *,
        artifacts: Sequence[str | Path] = (),
        details: Mapping[str, Any] | None = None,
    ) -> Path:
        """Append one pipeline run entry to the run registry."""

        if not isinstance(metadata, RunMetadata):
            raise TypeError("metadata must be a RunMetadata instance")

        payload = _jsonify_object(asdict(metadata))

        payload["recorded_at"] = _utcnow().isoformat()
        payload["artifacts"] = [self._artifact_reference(item) for item in artifacts]
        if details is not None:
            payload["details"] = _jsonify(dict(details))

        self._append_jsonl(self.runs_path, payload)
        return self.runs_path

    def get_checkpoint(
        self,
        name: str,
        default: JsonValue | None = None,
    ) -> JsonValue | None:
        """Return the stored checkpoint value for a logical key."""

        document = self._read_checkpoints_document()
        entry = document["checkpoints"].get(name)
        if entry is None:
            return default
        if isinstance(entry, Mapping) and "value" in entry:
            value = entry["value"]
            if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                return value
            return default
        if isinstance(entry, (dict, list, str, int, float, bool)) or entry is None:
            return entry
        return default

    def set_checkpoint(self, name: str, value: JsonValue) -> Path:
        """Persist the latest checkpoint or dedupe marker for a logical key."""

        checkpoint_name = name.strip()
        if not checkpoint_name:
            raise ValueError("checkpoint name must be a non-empty string")

        document = self._read_checkpoints_document()
        now = _utcnow().isoformat()
        document["updated_at"] = now
        document["checkpoints"][checkpoint_name] = {
            "value": _jsonify(value),
            "updated_at": now,
        }
        self._write_json_atomic(self.checkpoints_path, document)
        return self.checkpoints_path

    def _artifact_reference(self, artifact: str | Path) -> str:
        if isinstance(artifact, Path):
            return _relative_posix(self.root, artifact)
        return artifact.replace("\\", "/")

    def _dataset_root(self, *, layer: str, dataset: str) -> Path:
        return self.root / layer / dataset

    def _dataset_dir(
        self,
        *,
        layer: str,
        dataset: str,
        ordered_partitions: Sequence[tuple[str, PartitionValue]],
    ) -> Path:
        path = self._dataset_root(layer=layer, dataset=dataset)
        for key, value in ordered_partitions:
            path /= f"{key}={_format_partition_value(value)}"
        return path

    def _ordered_partitions(
        self,
        *,
        layer: str,
        dataset: str,
        partitions: Mapping[str, PartitionValue] | None,
    ) -> tuple[tuple[str, PartitionValue], ...]:
        if not partitions:
            return ()

        cleaned: dict[str, PartitionValue] = {}
        for key, value in partitions.items():
            partition_key = str(key).strip()
            if not partition_key:
                raise ValueError("partition keys must be non-empty strings")
            cleaned[partition_key] = value

        preferred_order = _PARTITION_ORDERS.get((layer, dataset), ())
        ordered: list[tuple[str, PartitionValue]] = []
        seen: set[str] = set()

        for key in preferred_order:
            if key in cleaned:
                ordered.append((key, cleaned[key]))
                seen.add(key)

        for key, value in cleaned.items():
            if key in seen:
                continue
            ordered.append((key, value))

        return tuple(ordered)

    def _serialise_partitions(
        self, ordered_partitions: Sequence[tuple[str, PartitionValue]]
    ) -> dict[str, JsonValue]:
        return {
            key: _format_partition_value(value) for key, value in ordered_partitions
        }

    def _extract_partition_values(
        self,
        *,
        file_path: Path,
        dataset_root: Path,
    ) -> dict[str, str]:
        relative = file_path.relative_to(dataset_root)
        partitions: dict[str, str] = {}
        for part in relative.parts[:-1]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            partitions[key] = value
        return partitions

    def _frame_filename(self, filename: str | None) -> str:
        if filename is None:
            timestamp = _utcnow().strftime("%Y%m%dT%H%M%SZ")
            return f"part-{timestamp}-{uuid4().hex[:8]}.parquet"
        if filename.endswith(".parquet"):
            return filename
        return f"{filename}.parquet"

    def _json_filename(self, filename: str) -> str:
        return filename if filename.endswith(".json") else f"{filename}.json"

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")

    def _write_json_atomic(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(f"{path.suffix}.{uuid4().hex}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temp_path.replace(path)

    def _read_checkpoints_document(self) -> dict[str, Any]:
        if not self.checkpoints_path.exists():
            return {"updated_at": None, "checkpoints": {}}

        with self.checkpoints_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, dict) and isinstance(payload.get("checkpoints"), dict):
            return payload

        if isinstance(payload, dict):
            return {"updated_at": None, "checkpoints": payload}

        raise ValueError(
            f"Checkpoint file at {self.checkpoints_path} does not contain a JSON object"
        )
