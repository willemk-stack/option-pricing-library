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
    HESTON_CALIBRATION_BENCHMARK_PARAMETER_COLUMNS,
    HESTON_CALIBRATION_BENCHMARK_REQUIRED_TABLES,
    HESTON_CALIBRATION_BENCHMARK_RESIDUAL_COLUMNS,
    HESTON_CALIBRATION_BENCHMARK_RUN_COLUMNS,
    HESTON_CALIBRATION_BENCHMARK_SUMMARY_COLUMNS,
    HESTON_CALIBRATION_FIT_IV_GRID_COLUMNS,
    HESTON_CALIBRATION_FIT_REQUIRED_TABLES,
    HESTON_CALIBRATION_FIT_RESIDUAL_COLUMNS,
    HESTON_CALIBRATION_FIT_SMILE_COLUMNS,
    HESTON_MODEL_COMPARISON_ERROR_SUMMARY_COLUMNS,
    HESTON_MODEL_COMPARISON_FIT_ERROR_COLUMNS,
    HESTON_MODEL_COMPARISON_REQUIRED_TABLES,
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


def _is_matplotlib_object(value: Any) -> bool:
    module = type(value).__module__
    return module.startswith("matplotlib")


def _reject_plot_objects(value: Any, *, label: str) -> None:
    if _is_matplotlib_object(value):
        raise TypeError(
            f"{label} may not include matplotlib figure, axes, or artist objects."
        )

    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_plot_objects(item, label=f"{label}.{key}")
        return

    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _reject_plot_objects(item, label=f"{label}[{idx}]")


@dataclass(frozen=True, slots=True)
class _HestonTableArtifact:
    """Shared runtime container for notebook-facing diagnostics artifacts.

    Subclasses keep three plain-data pieces together:

    - ``meta`` for lightweight context
    - ``table`` for the main notebook-facing ``pandas.DataFrame``
    - ``arrays`` for optional dense arrays or nested array groups

    The container normalizes inputs eagerly and does not perform any hidden
    recomputation.
    """

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
    """Notebook-facing artifact for one packaged ``P0`` or ``P1`` slice.

    The table is the readable per-point view that notebooks inspect directly.
    When created from the Fourier diagnostics wrappers, ``arrays`` can also
    carry panel-local detail such as ``panel_contribs`` or ``panel_reason``.

    This artifact surfaces numerical behavior for review; it does not prove the
    underlying probabilities are correct.
    """

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
    """Notebook-facing artifact for the canonical strike-slice diagnostics table.

    The required front columns follow the frozen Step 1 contract so notebooks
    can rely on a stable view of price, implied-vol, warning, and
    continuity-related signals. Extra downstream review columns are preserved
    after that canonical prefix.

    The artifact is descriptive only: it packages already-computed diagnostics
    and should not be interpreted as proof of pricing correctness.
    """

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
    """Notebook-facing artifact for backend-comparison diagnostics.

    This keeps the primary/comparison backend table and any aligned arrays in a
    plain runtime container so notebooks can inspect disagreement without
    re-running the pricing engine.
    """

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
    exposing the additional notebook-facing tables needed for downstream review,
    especially warning summaries, worst panels, and panel-reason counts.
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
class HestonCalibrationBenchmarkDiagnostics:
    """Notebook-facing Heston calibration benchmark diagnostics report.

    The report keeps the same lightweight topology used by the Heston pricing
    diagnostics:

    - ``meta`` for scenario, environment, and provenance fields
    - ``tables`` for readable ``pandas.DataFrame`` outputs
    - ``arrays`` for optional dense quote/residual/parameter payloads

    Required tables include ``runs``, ``summary``, ``parameter_recovery``, and
    ``residuals``. Runtime tables stay as DataFrames until serialization.

    REVIEW: The required tables and canonical column prefixes are a public
    diagnostics contract for the benchmark surface.
    """

    meta: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    arrays: dict[str, Any]

    def __post_init__(self) -> None:
        _reject_plot_objects(self.meta, label="meta")
        _reject_plot_objects(self.arrays, label="arrays")

        object.__setattr__(self, "meta", _normalize_meta(self.meta))
        object.__setattr__(self, "arrays", _normalize_arrays(self.arrays))
        object.__setattr__(
            self,
            "tables",
            {str(name): _as_dataframe(table) for name, table in self.tables.items()},
        )

        missing_tables = [
            name
            for name in HESTON_CALIBRATION_BENCHMARK_REQUIRED_TABLES
            if name not in self.tables
        ]
        if missing_tables:
            raise ValueError(
                "HestonCalibrationBenchmarkDiagnostics.tables is missing required "
                f"table(s): {', '.join(missing_tables)}."
            )

        column_contracts = {
            "runs": HESTON_CALIBRATION_BENCHMARK_RUN_COLUMNS,
            "summary": HESTON_CALIBRATION_BENCHMARK_SUMMARY_COLUMNS,
            "parameter_recovery": HESTON_CALIBRATION_BENCHMARK_PARAMETER_COLUMNS,
            "residuals": HESTON_CALIBRATION_BENCHMARK_RESIDUAL_COLUMNS,
        }
        normalized_tables = dict(self.tables)
        for table_name, columns in column_contracts.items():
            normalized_tables[table_name] = _ensure_columns(
                normalized_tables[table_name],
                required_columns=columns,
                label=(
                    "HestonCalibrationBenchmarkDiagnostics.tables" f"[{table_name!r}]"
                ),
            )

        object.__setattr__(self, "tables", normalized_tables)

    def to_dict(self) -> SerializedReport:
        """Serialize the report into a JSON-friendly plain-data shape."""
        return {
            "meta": dict(self.meta),
            "tables": _serialize_tables(self.tables),
            "arrays": _serialize_array_value(self.arrays),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> HestonCalibrationBenchmarkDiagnostics:
        keys = set(payload.keys())
        expected = {"meta", "tables", "arrays"}
        if keys != expected:
            raise ValueError(
                "Serialized HestonCalibrationBenchmarkDiagnostics payload must "
                "contain exactly 'meta', 'tables', and 'arrays'."
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
    def from_json(cls, payload: str) -> HestonCalibrationBenchmarkDiagnostics:
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True, slots=True)
class HestonCalibrationFitDiagnostics:
    """Notebook-facing Heston calibration fit diagnostics report.

    The report packages final calibration evidence as plain data:

    - quote-level price and implied-vol residuals
    - smile overlay data
    - parameter recovery or fitted-parameter summary
    - multistart run metadata when available
    - held-out error summaries when a mask is supplied
    - lightweight objective slices around the fitted point

    REVIEW: The tables are descriptive calibration evidence. They do not prove
    parameter uniqueness or economic validity of the fitted Heston dynamics.
    """

    meta: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    arrays: dict[str, Any]

    def __post_init__(self) -> None:
        _reject_plot_objects(self.meta, label="meta")
        _reject_plot_objects(self.arrays, label="arrays")

        object.__setattr__(self, "meta", _normalize_meta(self.meta))
        object.__setattr__(self, "arrays", _normalize_arrays(self.arrays))
        object.__setattr__(
            self,
            "tables",
            {str(name): _as_dataframe(table) for name, table in self.tables.items()},
        )

        missing_tables = [
            name
            for name in HESTON_CALIBRATION_FIT_REQUIRED_TABLES
            if name not in self.tables
        ]
        if missing_tables:
            raise ValueError(
                "HestonCalibrationFitDiagnostics.tables is missing required "
                f"table(s): {', '.join(missing_tables)}."
            )

        column_contracts = {
            "residuals": HESTON_CALIBRATION_FIT_RESIDUAL_COLUMNS,
            "smile_fit": HESTON_CALIBRATION_FIT_SMILE_COLUMNS,
            "iv_residual_grid": HESTON_CALIBRATION_FIT_IV_GRID_COLUMNS,
        }
        normalized_tables = dict(self.tables)
        for table_name, columns in column_contracts.items():
            normalized_tables[table_name] = _ensure_columns(
                normalized_tables[table_name],
                required_columns=columns,
                label=("HestonCalibrationFitDiagnostics.tables" f"[{table_name!r}]"),
            )

        object.__setattr__(self, "tables", normalized_tables)

    def to_dict(self) -> SerializedReport:
        return {
            "meta": dict(self.meta),
            "tables": _serialize_tables(self.tables),
            "arrays": _serialize_array_value(self.arrays),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HestonCalibrationFitDiagnostics:
        keys = set(payload.keys())
        expected = {"meta", "tables", "arrays"}
        if keys != expected:
            raise ValueError(
                "Serialized HestonCalibrationFitDiagnostics payload must contain "
                "exactly 'meta', 'tables', and 'arrays'."
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
    def from_json(cls, payload: str) -> HestonCalibrationFitDiagnostics:
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True, slots=True)
class HestonModelComparisonDiagnostics:
    """Notebook-facing Heston-vs-local-vol comparison diagnostics report.

    The comparison is intentionally table-first. It keeps model fit errors,
    bucketed summaries, and qualitative tradeoff notes together without
    embedding plotting objects or rerunning pricing during serialization.

    REVIEW: Conclusions depend on the chosen target surface and local-vol proxy.
    """

    meta: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    arrays: dict[str, Any]

    def __post_init__(self) -> None:
        _reject_plot_objects(self.meta, label="meta")
        _reject_plot_objects(self.arrays, label="arrays")

        object.__setattr__(self, "meta", _normalize_meta(self.meta))
        object.__setattr__(self, "arrays", _normalize_arrays(self.arrays))
        object.__setattr__(
            self,
            "tables",
            {str(name): _as_dataframe(table) for name, table in self.tables.items()},
        )

        missing_tables = [
            name
            for name in HESTON_MODEL_COMPARISON_REQUIRED_TABLES
            if name not in self.tables
        ]
        if missing_tables:
            raise ValueError(
                "HestonModelComparisonDiagnostics.tables is missing required "
                f"table(s): {', '.join(missing_tables)}."
            )

        column_contracts = {
            "fit_errors": HESTON_MODEL_COMPARISON_FIT_ERROR_COLUMNS,
            "error_summary": HESTON_MODEL_COMPARISON_ERROR_SUMMARY_COLUMNS,
        }
        normalized_tables = dict(self.tables)
        for table_name, columns in column_contracts.items():
            normalized_tables[table_name] = _ensure_columns(
                normalized_tables[table_name],
                required_columns=columns,
                label=("HestonModelComparisonDiagnostics.tables" f"[{table_name!r}]"),
            )

        object.__setattr__(self, "tables", normalized_tables)

    def to_dict(self) -> SerializedReport:
        return {
            "meta": dict(self.meta),
            "tables": _serialize_tables(self.tables),
            "arrays": _serialize_array_value(self.arrays),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HestonModelComparisonDiagnostics:
        keys = set(payload.keys())
        expected = {"meta", "tables", "arrays"}
        if keys != expected:
            raise ValueError(
                "Serialized HestonModelComparisonDiagnostics payload must contain "
                "exactly 'meta', 'tables', and 'arrays'."
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
    def from_json(cls, payload: str) -> HestonModelComparisonDiagnostics:
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True, slots=True)
class HestonDiagnosticsReport:
    """Full Step 1 notebook-facing Heston diagnostics report.

    The report topology is frozen to three top-level keys:

    - ``meta``
    - ``tables``
    - ``arrays``

    Runtime tables stay as ``pandas.DataFrame`` objects until explicitly
    serialized via :meth:`to_dict`.

    Required tables include ``summary``, ``slice``, ``worst_strikes``,
    ``backend_compare``, ``config_sweep``, ``worst_panels_p0``, and
    ``worst_panels_p1``. Optional arrays may carry probability-detail payloads
    and convergence/config-sweep artifacts for plotting.

    The report is a review surface over existing diagnostics logic. It does not
    prove price correctness, economic validity of the smile, or final
    suspiciousness policy.
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
