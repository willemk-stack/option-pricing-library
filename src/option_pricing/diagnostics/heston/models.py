"""Typed Step 1 report and artifact models for Heston diagnostics.

These containers are intentionally lightweight:

- runtime tables are ``pandas.DataFrame``
- runtime arrays are ``numpy.ndarray`` or nested dicts of array-like values
- serialization is explicit and JSON-friendly

The Step 1 contract freezes report topology and notebook-facing table naming,
without adding new pricing logic or hidden recomputation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    HESTON_REQUIRED_REPORT_TABLES,
    HESTON_SEVERITY_ORDER,
    HESTON_SLICE_TABLE_COLUMNS,
    HESTON_SUMMARY_TABLE_COLUMNS,
)

type SerializedTable = dict[str, Any]
type SerializedArtifact = dict[str, Any]
type SerializedReport = dict[str, Any]


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    elif isinstance(value, np.bool_):
        value = bool(value)

    if isinstance(value, float) and not np.isfinite(value):
        return None

    if value is pd.NA:
        return None

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    return value


def _serialize_array_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _serialize_array_value(val) for key, val in value.items()}

    if isinstance(value, np.ndarray):
        return _serialize_array_value(value.tolist())

    if isinstance(value, (list, tuple)):
        return [_serialize_array_value(item) for item in value]

    return _json_scalar(value)


def _normalize_array_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_array_value(val) for key, val in value.items()}

    if isinstance(value, np.ndarray):
        return np.asarray(value)

    if isinstance(value, tuple):
        value = list(value)

    if isinstance(value, list):
        if any(isinstance(item, Mapping) for item in value):
            return [_normalize_array_value(item) for item in value]
        return np.asarray(value)

    return _json_scalar(value)


def _normalize_meta(meta: Mapping[str, Any] | None) -> dict[str, Any]:
    if meta is None:
        return {}
    return {str(key): value for key, value in meta.items()}


def _normalize_arrays(arrays: Mapping[str, Any] | None) -> dict[str, Any]:
    if arrays is None:
        return {}
    return {str(key): _normalize_array_value(value) for key, value in arrays.items()}


def _as_dataframe(table: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(table, pd.DataFrame):
        raise TypeError("table must be a pandas.DataFrame.")
    return table.copy()


def _ensure_columns(
    table: pd.DataFrame,
    *,
    required_columns: tuple[str, ...],
    label: str,
) -> pd.DataFrame:
    missing = [column for column in required_columns if column not in table.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required column(s): {', '.join(missing)}."
        )

    ordered_columns = [
        *required_columns,
        *[column for column in table.columns if column not in required_columns],
    ]
    return table.loc[:, ordered_columns].copy()


def _validate_severity_column(
    table: pd.DataFrame,
    *,
    column: str,
    allow_null: bool,
    label: str,
) -> None:
    if column not in table.columns or table.empty:
        return

    values = table[column]
    if not allow_null and values.isna().any():
        raise ValueError(f"{label}.{column} may not contain null values.")

    normalized = values.dropna().astype(str)
    invalid = sorted(set(normalized) - set(HESTON_SEVERITY_ORDER))
    if invalid:
        raise ValueError(
            f"{label}.{column} contains unsupported severity values: "
            f"{', '.join(invalid)}."
        )


def _serialize_table(table: pd.DataFrame) -> SerializedTable:
    frame = _as_dataframe(table)
    frame = frame.replace({np.nan: None, pd.NA: None})

    serialized: SerializedTable = {
        "columns": [str(column) for column in frame.columns],
        "records": [
            {str(key): _serialize_array_value(value) for key, value in row.items()}
            for row in frame.to_dict(orient="records")
        ],
    }

    if not isinstance(frame.index, pd.RangeIndex):
        serialized["index"] = [_serialize_array_value(value) for value in frame.index]

    if frame.attrs:
        serialized["metadata"] = {
            str(key): _serialize_array_value(value)
            for key, value in frame.attrs.items()
        }

    return serialized


def _deserialize_table(payload: Mapping[str, Any]) -> pd.DataFrame:
    columns = payload.get("columns")
    records = payload.get("records")
    if not isinstance(columns, list) or not isinstance(records, list):
        raise TypeError("Serialized tables require 'columns' and 'records'.")

    frame = pd.DataFrame.from_records(records, columns=columns)

    index_values = payload.get("index")
    if index_values is not None:
        frame.index = pd.Index(index_values)

    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        frame.attrs.update(dict(metadata))

    return frame


def _serialize_tables(tables: Mapping[str, pd.DataFrame]) -> dict[str, SerializedTable]:
    return {str(name): _serialize_table(table) for name, table in tables.items()}


def _deserialize_tables(payload: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    return {
        str(name): _deserialize_table(table_payload)
        for name, table_payload in payload.items()
    }


@dataclass(frozen=True, slots=True)
class _HestonTableArtifact:
    """Shared runtime container for notebook-facing diagnostics artifacts."""

    meta: dict[str, Any]
    table: pd.DataFrame
    arrays: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "meta", _normalize_meta(self.meta))
        object.__setattr__(self, "table", _as_dataframe(self.table))
        object.__setattr__(self, "arrays", _normalize_arrays(self.arrays))

        if self.table.shape[1] == 0:
            raise ValueError("artifact tables must expose at least one column.")

    def to_dict(self) -> SerializedArtifact:
        return {
            "meta": dict(self.meta),
            "table": _serialize_table(self.table),
            "arrays": _serialize_array_value(self.arrays),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


@dataclass(frozen=True, slots=True)
class HestonProbabilitySliceDiagnostics(_HestonTableArtifact):
    """Notebook-facing artifact for a packaged probability slice."""

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> HestonProbabilitySliceDiagnostics:
        return cls(
            meta=_normalize_meta(payload.get("meta")),
            table=_deserialize_table(payload["table"]),
            arrays=_normalize_arrays(payload.get("arrays")),
        )

    @classmethod
    def from_json(cls, payload: str) -> HestonProbabilitySliceDiagnostics:
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True, slots=True)
class HestonPriceSliceDiagnostics(_HestonTableArtifact):
    """Notebook-facing artifact for the canonical slice diagnostics table."""

    def __post_init__(self) -> None:
        _HestonTableArtifact.__post_init__(self)
        normalized = _ensure_columns(
            self.table,
            required_columns=HESTON_SLICE_TABLE_COLUMNS,
            label="HestonPriceSliceDiagnostics.table",
        )
        _validate_severity_column(
            normalized,
            column="severity",
            allow_null=False,
            label="HestonPriceSliceDiagnostics.table",
        )
        object.__setattr__(self, "table", normalized)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> HestonPriceSliceDiagnostics:
        return cls(
            meta=_normalize_meta(payload.get("meta")),
            table=_deserialize_table(payload["table"]),
            arrays=_normalize_arrays(payload.get("arrays")),
        )

    @classmethod
    def from_json(cls, payload: str) -> HestonPriceSliceDiagnostics:
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True, slots=True)
class HestonBackendComparisonDiagnostics(_HestonTableArtifact):
    """Notebook-facing artifact for backend-comparison diagnostics."""

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> HestonBackendComparisonDiagnostics:
        return cls(
            meta=_normalize_meta(payload.get("meta")),
            table=_deserialize_table(payload["table"]),
            arrays=_normalize_arrays(payload.get("arrays")),
        )

    @classmethod
    def from_json(cls, payload: str) -> HestonBackendComparisonDiagnostics:
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True, slots=True)
class HestonIntegrationDiagnosticsBundle:
    """Readable Step 2 integration diagnostics bundle.

    The bundle keeps the Step 1 packaged probability artifact intact while
    exposing the additional notebook-facing tables needed for downstream review.
    """

    meta: dict[str, Any]
    probability: HestonProbabilitySliceDiagnostics
    tables: dict[str, pd.DataFrame]
    arrays: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "meta", _normalize_meta(self.meta))
        object.__setattr__(self, "arrays", _normalize_arrays(self.arrays))
        object.__setattr__(
            self,
            "tables",
            {str(name): _as_dataframe(table) for name, table in self.tables.items()},
        )

        required_tables = ("panels", "warning_summary", "worst_panels", "reason_counts")
        missing_tables = [name for name in required_tables if name not in self.tables]
        if missing_tables:
            raise ValueError(
                "HestonIntegrationDiagnosticsBundle.tables is missing required "
                f"table(s): {', '.join(missing_tables)}."
            )


@dataclass(frozen=True, slots=True)
class HestonDiagnosticsReport:
    """Full Step 1 notebook-facing Heston diagnostics report.

    The report topology is frozen to three top-level keys:

    - ``meta``
    - ``tables``
    - ``arrays``

    Runtime tables stay as ``pandas.DataFrame`` objects until explicitly
    serialized via :meth:`to_dict`.
    """

    meta: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    arrays: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "meta", _normalize_meta(self.meta))
        object.__setattr__(
            self,
            "tables",
            {str(name): _as_dataframe(table) for name, table in self.tables.items()},
        )
        object.__setattr__(self, "arrays", _normalize_arrays(self.arrays))

        missing_tables = [
            name for name in HESTON_REQUIRED_REPORT_TABLES if name not in self.tables
        ]
        if missing_tables:
            raise ValueError(
                "HestonDiagnosticsReport.tables is missing required table(s): "
                f"{', '.join(missing_tables)}."
            )

        summary_table = _ensure_columns(
            self.tables["summary"],
            required_columns=HESTON_SUMMARY_TABLE_COLUMNS,
            label="HestonDiagnosticsReport.tables['summary']",
        )
        _validate_severity_column(
            summary_table,
            column="severity",
            allow_null=True,
            label="HestonDiagnosticsReport.tables['summary']",
        )

        slice_table = _ensure_columns(
            self.tables["slice"],
            required_columns=HESTON_SLICE_TABLE_COLUMNS,
            label="HestonDiagnosticsReport.tables['slice']",
        )
        _validate_severity_column(
            slice_table,
            column="severity",
            allow_null=False,
            label="HestonDiagnosticsReport.tables['slice']",
        )

        normalized_tables = dict(self.tables)
        normalized_tables["summary"] = summary_table
        normalized_tables["slice"] = slice_table
        object.__setattr__(self, "tables", normalized_tables)

    def to_dict(self) -> SerializedReport:
        """Serialize the report into the Step 1 JSON-friendly shape."""
        return {
            "meta": dict(self.meta),
            "tables": _serialize_tables(self.tables),
            "arrays": _serialize_array_value(self.arrays),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> HestonDiagnosticsReport:
        keys = set(payload.keys())
        expected = {"meta", "tables", "arrays"}
        if keys != expected:
            raise ValueError(
                "Serialized HestonDiagnosticsReport payload must contain exactly "
                "'meta', 'tables', and 'arrays'."
            )

        tables_payload = payload["tables"]
        if not isinstance(tables_payload, Mapping):
            raise TypeError("Serialized report 'tables' must be a mapping.")

        return cls(
            meta=_normalize_meta(payload.get("meta")),
            tables=_deserialize_tables(tables_payload),
            arrays=_normalize_arrays(payload.get("arrays")),
        )

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, payload: str) -> HestonDiagnosticsReport:
        return cls.from_dict(json.loads(payload))
