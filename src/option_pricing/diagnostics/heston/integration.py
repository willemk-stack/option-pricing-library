"""Readable Step 2 wrappers over the existing Heston Fourier diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntFlag
from typing import Any, Literal

import numpy as np
import pandas as pd

from ...models.heston.fourier import (
    HestonIntegralBatchDiagnostics,
    HestonIntegralDiagnostics,
    HestonIntegralWarning,
    HestonPanelReason,
    P_j_batch_with_diagnostics,
    P_j_with_diagnostics,
)
from ...models.heston.params import HestonParams
from ...numerics.quadrature import CompositeRule, QuadratureConfig
from .contracts import HESTON_SEVERITY_ORDER, HestonDiagnosticsBackend
from .models import (
    HestonIntegrationDiagnosticsBundle,
    HestonProbabilitySliceDiagnostics,
)

type ProbabilityDiagnosticsInput = (
    HestonIntegralBatchDiagnostics | HestonIntegralDiagnostics
)
type IntegrationSweepCase = Mapping[str, Any]
type SeverityLabel = Literal["ok", "info", "warning", "severe", "critical"]

_SEVERITY_RANK: dict[str, int] = {
    str(severity): rank for rank, severity in enumerate(HESTON_SEVERITY_ORDER, start=0)
}

_WARNING_LABELS: dict[HestonIntegralWarning, str] = {
    HestonIntegralWarning.NONFINITE_TOTAL: "non-finite total integral",
    HestonIntegralWarning.NONFINITE_PROBABILITY: "non-finite probability",
    HestonIntegralWarning.PROBABILITY_OUT_OF_RANGE: "probability outside [0, 1]",
    HestonIntegralWarning.LARGE_TAIL_FRACTION: "large tail fraction",
    HestonIntegralWarning.EXCESSIVE_CANCELLATION: "excessive cancellation",
    HestonIntegralWarning.TOO_MANY_BAD_PANELS: "too many bad panels",
    HestonIntegralWarning.QUAD_ERROR_LARGE: "large quad error estimate",
}

_WARNING_SEVERITY: dict[HestonIntegralWarning, str] = {
    HestonIntegralWarning.NONFINITE_TOTAL: "critical",
    HestonIntegralWarning.NONFINITE_PROBABILITY: "critical",
    HestonIntegralWarning.PROBABILITY_OUT_OF_RANGE: "critical",
    HestonIntegralWarning.LARGE_TAIL_FRACTION: "warning",
    HestonIntegralWarning.EXCESSIVE_CANCELLATION: "severe",
    HestonIntegralWarning.TOO_MANY_BAD_PANELS: "severe",
    HestonIntegralWarning.QUAD_ERROR_LARGE: "warning",
}

_PANEL_REASON_LABELS: dict[HestonPanelReason, str] = {
    HestonPanelReason.NONFINITE_PANEL_CONTRIB: "non-finite panel contribution",
    HestonPanelReason.NONFINITE_INTEGRAND: "non-finite integrand values",
    HestonPanelReason.UNDERRESOLVED_NEAR_ORIGIN: "underresolved near origin",
    HestonPanelReason.UNDERRESOLVED_TAIL: "underresolved tail",
    HestonPanelReason.TAIL_TOO_LARGE: "tail panel too large",
    HestonPanelReason.OSCILLATION_SPIKE: "oscillation spike",
}

_PANEL_REASON_SEVERITY: dict[HestonPanelReason, str] = {
    HestonPanelReason.NONFINITE_PANEL_CONTRIB: "critical",
    HestonPanelReason.NONFINITE_INTEGRAND: "critical",
    HestonPanelReason.UNDERRESOLVED_NEAR_ORIGIN: "warning",
    HestonPanelReason.UNDERRESOLVED_TAIL: "warning",
    HestonPanelReason.TAIL_TOO_LARGE: "severe",
    HestonPanelReason.OSCILLATION_SPIKE: "severe",
}


def _severity_rank(value: str) -> int:
    key = str(value)
    if key in _SEVERITY_RANK:
        return _SEVERITY_RANK[key]
    return -1


def _worst_severity(values: Sequence[str]) -> str:
    if not values:
        return "ok"
    return max(values, key=_severity_rank)


def _flag_members[FlagT: IntFlag](
    flag_value: int, flag_type: type[FlagT]
) -> list[FlagT]:
    active: list[FlagT] = []
    for member in flag_type:
        if int(member) == 0:
            continue
        if int(flag_value) & int(member):
            active.append(member)
    return active


def _member_name(member: IntFlag) -> str:
    name = member.name
    if name is None:
        return str(int(member))
    return name.lower()


def _warning_names(flag_value: int) -> list[str]:
    members = _flag_members(flag_value, HestonIntegralWarning)
    return [_member_name(member) for member in members]


def _warning_labels(flag_value: int) -> list[str]:
    members = _flag_members(flag_value, HestonIntegralWarning)
    return [_WARNING_LABELS.get(member, _member_name(member)) for member in members]


def _warning_severity(flag_value: int) -> str:
    members = _flag_members(flag_value, HestonIntegralWarning)
    if not members:
        return "ok"
    return _worst_severity(
        [_WARNING_SEVERITY.get(member, "warning") for member in members]
    )


def _panel_reason_names(reason_value: int) -> list[str]:
    members = _flag_members(reason_value, HestonPanelReason)
    return [_member_name(member) for member in members]


def _panel_reason_labels(reason_value: int) -> list[str]:
    members = _flag_members(reason_value, HestonPanelReason)
    return [
        _PANEL_REASON_LABELS.get(member, _member_name(member)) for member in members
    ]


def _panel_reason_severity(reason_value: int) -> str:
    members = _flag_members(reason_value, HestonPanelReason)
    if not members:
        return "ok"
    return _worst_severity(
        [_PANEL_REASON_SEVERITY.get(member, "warning") for member in members]
    )


def _join_labels(labels: Sequence[str]) -> str:
    if not labels:
        return "none"
    return "; ".join(str(label) for label in labels)


def _format_index_list(values: Sequence[int]) -> str:
    if not values:
        return ""
    return ", ".join(str(int(value)) for value in values)


def _coerce_columns(columns: Mapping[str, Any], *, label: str) -> pd.DataFrame:
    if not columns:
        raise ValueError(f"{label} must contain at least one column.")
    return pd.DataFrame(
        {
            str(name): np.atleast_1d(np.asarray(values))
            for name, values in columns.items()
        }
    )


def _coerce_optional_strike(
    strike: Any,
    *,
    size: int,
) -> np.ndarray | None:
    if strike is None:
        return None

    strike_arr = np.asarray(strike, dtype=np.float64).reshape(-1)
    if strike_arr.size == 1:
        strike_arr = np.repeat(strike_arr, size)
    if strike_arr.size != size:
        raise ValueError(
            "strike must be scalar or have the same flattened size as the "
            "probability diagnostics slice."
        )
    return strike_arr


def _warning_count(flags: Any, *, size: int) -> np.ndarray:
    if flags is None:
        return np.zeros(size, dtype=np.int64)

    flags_arr = np.asarray(flags, dtype=np.uint32).reshape(-1)
    return np.asarray(
        [int(int(flag).bit_count()) for flag in flags_arr],
        dtype=np.int64,
    )


def _augment_probability_table(table: pd.DataFrame) -> pd.DataFrame:
    enriched = table.copy()
    if "warning_flags" not in enriched.columns:
        return enriched

    flags = (
        enriched["warning_flags"]
        .fillna(0)
        .map(lambda value: int(np.uint32(value)))
        .to_numpy(dtype=np.uint32)
    )
    if "warning_count" not in enriched.columns:
        enriched["warning_count"] = _warning_count(flags, size=len(enriched))

    warning_names = [_warning_names(int(flag)) for flag in flags]
    warning_labels = [_warning_labels(int(flag)) for flag in flags]
    severities = [_warning_severity(int(flag)) for flag in flags]

    if "warning_names" not in enriched.columns:
        enriched["warning_names"] = [_join_labels(names) for names in warning_names]
    if "warning_labels" not in enriched.columns:
        enriched["warning_labels"] = [_join_labels(labels) for labels in warning_labels]
    if "severity" not in enriched.columns:
        enriched["severity"] = severities
    if "severity_rank" not in enriched.columns:
        enriched["severity_rank"] = np.asarray(
            [_severity_rank(severity) for severity in severities],
            dtype=np.int64,
        )

    return enriched


def _probability_diag_to_table(
    diagnostics: ProbabilityDiagnosticsInput,
    *,
    strike: Any = None,
) -> pd.DataFrame:
    x = np.asarray(diagnostics.x, dtype=np.float64).reshape(-1)
    size = x.size

    warning_flags = getattr(diagnostics, "warning_flags", None)
    warning_flags_arr = (
        None
        if warning_flags is None
        else np.asarray(warning_flags, dtype=np.uint32).reshape(-1)
    )

    data: dict[str, Any] = {
        "log_moneyness": x,
        "probability": np.asarray(diagnostics.probability, dtype=np.float64).reshape(
            -1
        ),
        "warning_count": _warning_count(warning_flags_arr, size=size),
        "tail_fraction": (
            None
            if diagnostics.tail_abs_fraction is None
            else np.asarray(diagnostics.tail_abs_fraction, dtype=np.float64).reshape(-1)
        ),
        "cancellation_ratio": (
            None
            if diagnostics.cancellation_ratio is None
            else np.asarray(diagnostics.cancellation_ratio, dtype=np.float64).reshape(
                -1
            )
        ),
        "warning_flags": warning_flags_arr,
        "backend": np.full(size, diagnostics.backend, dtype=object),
        "probability_index": np.full(size, diagnostics.j, dtype=np.int64),
    }

    strike_arr = _coerce_optional_strike(strike, size=size)
    if strike_arr is not None:
        data["strike"] = strike_arr

    quad_error = getattr(diagnostics, "quad_error_estimate", None)
    if quad_error is not None:
        data["quad_error_estimate"] = np.asarray(quad_error, dtype=np.float64).reshape(
            -1
        )

    return _augment_probability_table(pd.DataFrame(data))


def _probability_diag_to_arrays(
    diagnostics: ProbabilityDiagnosticsInput,
) -> dict[str, Any]:
    arrays: dict[str, Any] = {
        "log_moneyness": np.asarray(diagnostics.x, dtype=np.float64),
        "probability": np.asarray(diagnostics.probability, dtype=np.float64),
        "total_integral": np.asarray(diagnostics.total_integral, dtype=np.float64),
    }

    if diagnostics.panel_edges is not None:
        arrays["panel_edges"] = np.asarray(diagnostics.panel_edges, dtype=np.float64)
    if diagnostics.panel_contribs is not None:
        arrays["panel_contribs"] = np.asarray(
            diagnostics.panel_contribs,
            dtype=np.float64,
        )
    if diagnostics.panel_invalid is not None:
        arrays["panel_invalid"] = np.asarray(diagnostics.panel_invalid, dtype=np.bool_)
    if diagnostics.panel_reason is not None:
        arrays["panel_reason"] = np.asarray(diagnostics.panel_reason, dtype=np.uint32)
    if diagnostics.quad_error_estimate is not None:
        arrays["quad_error_estimate"] = np.asarray(
            diagnostics.quad_error_estimate,
            dtype=np.float64,
        )
    if diagnostics.tail_abs_fraction is not None:
        arrays["tail_fraction"] = np.asarray(
            diagnostics.tail_abs_fraction,
            dtype=np.float64,
        )
    if diagnostics.cancellation_ratio is not None:
        arrays["cancellation_ratio"] = np.asarray(
            diagnostics.cancellation_ratio,
            dtype=np.float64,
        )
    if getattr(diagnostics, "warning_flags", None) is not None:
        arrays["warning_flags"] = np.asarray(diagnostics.warning_flags, dtype=np.uint32)

    return arrays


def probability_slice_with_diagnostics(
    *,
    probability_diagnostics: ProbabilityDiagnosticsInput | None = None,
    probability_table: pd.DataFrame | None = None,
    probability_columns: Mapping[str, Any] | None = None,
    strike: Any = None,
    meta: Mapping[str, Any] | None = None,
    arrays: Mapping[str, Any] | None = None,
) -> HestonProbabilitySliceDiagnostics:
    """Package one probability slice into a notebook-facing diagnostics artifact.

    Pass either the raw Fourier diagnostics object, an already-built
    ``probability_table``, or plain ``probability_columns``. When the raw
    diagnostics object is supplied, the helper copies through the packaged
    whole-integral and panel-local arrays so downstream tables and plots can use
    the real diagnostics flow without recomputing it in notebooks.

    This helper normalizes and packages data only; it does not validate pricing
    correctness or invent additional diagnostics.
    """

    sources = [
        probability_diagnostics is not None,
        probability_table is not None,
        probability_columns is not None,
    ]
    if sum(sources) == 0:
        raise ValueError(
            "Provide probability_diagnostics, probability_table, or "
            "probability_columns."
        )
    if probability_table is not None and probability_columns is not None:
        raise ValueError(
            "Pass either probability_table or probability_columns, not both."
        )

    if probability_table is not None:
        table = _augment_probability_table(probability_table.copy())
    elif probability_columns is not None:
        table = _augment_probability_table(
            _coerce_columns(
                probability_columns,
                label="probability_columns",
            )
        )
    else:
        assert probability_diagnostics is not None
        table = _probability_diag_to_table(
            probability_diagnostics,
            strike=strike,
        )

    packaged_arrays = {} if arrays is None else dict(arrays)
    packaged_meta = {} if meta is None else dict(meta)

    if probability_diagnostics is not None:
        packaged_arrays = {
            **_probability_diag_to_arrays(probability_diagnostics),
            **packaged_arrays,
        }
        packaged_meta.setdefault("backend", probability_diagnostics.backend)
        packaged_meta.setdefault("probability_index", int(probability_diagnostics.j))
        packaged_meta.setdefault("tau", float(probability_diagnostics.tau))
        packaged_meta.setdefault("source", "option_pricing.models.heston.fourier")

    return HestonProbabilitySliceDiagnostics(
        meta=packaged_meta,
        table=table,
        arrays=packaged_arrays,
    )


def _point_table(
    probability: HestonProbabilitySliceDiagnostics,
) -> pd.DataFrame:
    table = probability.table.copy().reset_index(drop=True)
    table.insert(0, "point_index", np.arange(len(table), dtype=np.int64))
    if "severity" not in table.columns:
        table["severity"] = "ok"
    if "severity_rank" not in table.columns:
        table["severity_rank"] = np.asarray(
            [_severity_rank(str(value)) for value in table["severity"]],
            dtype=np.int64,
        )
    return table


def _reshape_panel_array(
    value: Any,
    *,
    point_count: int,
    dtype: Any,
) -> np.ndarray | None:
    if value is None:
        return None

    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    else:
        arr = arr.reshape(point_count, arr.shape[-1])
    return arr


def _build_panel_table(
    probability: HestonProbabilitySliceDiagnostics,
) -> pd.DataFrame:
    points = _point_table(probability)
    arrays = probability.arrays
    point_count = len(points)

    panel_contribs = _reshape_panel_array(
        arrays.get("panel_contribs"),
        point_count=point_count,
        dtype=np.float64,
    )
    panel_invalid = _reshape_panel_array(
        arrays.get("panel_invalid"),
        point_count=point_count,
        dtype=np.bool_,
    )
    panel_reason = _reshape_panel_array(
        arrays.get("panel_reason"),
        point_count=point_count,
        dtype=np.uint32,
    )
    panel_edges = arrays.get("panel_edges")
    panel_edges_arr = (
        None if panel_edges is None else np.asarray(panel_edges, dtype=np.float64)
    )

    n_panels = None
    for candidate in (panel_contribs, panel_invalid, panel_reason):
        if candidate is not None:
            n_panels = int(candidate.shape[-1])
            break
    if n_panels is None and panel_edges_arr is not None and panel_edges_arr.size >= 2:
        n_panels = int(panel_edges_arr.size - 1)

    has_panel_detail = n_panels is not None and n_panels > 0
    rows: list[dict[str, Any]] = []

    if not has_panel_detail:
        note = (
            "Panel-local diagnostics are unavailable for this backend or were not "
            "returned by the engine."
        )
        for point in points.to_dict(orient="records"):
            rows.append(
                {
                    "point_index": int(point["point_index"]),
                    "probability_index": int(point.get("probability_index", -1)),
                    "backend": point.get("backend"),
                    "log_moneyness": float(point["log_moneyness"]),
                    "strike": point.get("strike"),
                    "point_severity": str(point["severity"]),
                    "point_warning_count": int(point.get("warning_count", 0)),
                    "panel_index": pd.NA,
                    "panel_start": pd.NA,
                    "panel_end": pd.NA,
                    "panel_width": pd.NA,
                    "panel_contribution": pd.NA,
                    "abs_panel_contribution": pd.NA,
                    "panel_invalid": pd.NA,
                    "reason_code": pd.NA,
                    "reason_count": pd.NA,
                    "reason_names": "panel detail unavailable",
                    "reason_labels": "panel detail unavailable",
                    "severity": str(point["severity"]),
                    "severity_rank": int(point["severity_rank"]),
                    "detail_available": False,
                    "notes": note,
                }
            )
        return pd.DataFrame(rows)

    strike_values = (
        None
        if "strike" not in points.columns
        else points["strike"].to_numpy(copy=False)
    )

    assert n_panels is not None
    for point_idx in range(point_count):
        for panel_idx in range(n_panels):
            reason_code = (
                None
                if panel_reason is None
                else int(panel_reason[point_idx, panel_idx])
            )
            invalid = (
                None
                if panel_invalid is None
                else bool(panel_invalid[point_idx, panel_idx])
            )
            reason_labels = (
                [] if reason_code is None else _panel_reason_labels(reason_code)
            )
            reason_names = (
                [] if reason_code is None else _panel_reason_names(reason_code)
            )
            severity = (
                "warning"
                if reason_code is None and invalid
                else (
                    "ok" if reason_code is None else _panel_reason_severity(reason_code)
                )
            )
            notes = ""
            if reason_code is None and invalid:
                notes = "panel_invalid is set but panel_reason is unavailable."
            elif reason_code is None and panel_reason is None:
                notes = "panel_reason is unavailable."

            panel_start = (
                pd.NA if panel_edges_arr is None else float(panel_edges_arr[panel_idx])
            )
            panel_end = (
                pd.NA
                if panel_edges_arr is None
                else float(panel_edges_arr[panel_idx + 1])
            )

            rows.append(
                {
                    "point_index": int(points.at[point_idx, "point_index"]),
                    "probability_index": int(points.at[point_idx, "probability_index"]),
                    "backend": points.at[point_idx, "backend"],
                    "log_moneyness": float(points.at[point_idx, "log_moneyness"]),
                    "strike": (
                        pd.NA
                        if strike_values is None
                        else float(strike_values[point_idx])
                    ),
                    "point_severity": str(points.at[point_idx, "severity"]),
                    "point_warning_count": int(points.at[point_idx, "warning_count"]),
                    "panel_index": int(panel_idx),
                    "panel_start": panel_start,
                    "panel_end": panel_end,
                    "panel_width": (
                        pd.NA
                        if panel_edges_arr is None
                        else float(
                            panel_edges_arr[panel_idx + 1] - panel_edges_arr[panel_idx]
                        )
                    ),
                    "panel_contribution": (
                        pd.NA
                        if panel_contribs is None
                        else float(panel_contribs[point_idx, panel_idx])
                    ),
                    "abs_panel_contribution": (
                        pd.NA
                        if panel_contribs is None
                        else abs(float(panel_contribs[point_idx, panel_idx]))
                    ),
                    "panel_invalid": invalid if invalid is not None else pd.NA,
                    "reason_code": pd.NA if reason_code is None else int(reason_code),
                    "reason_count": (
                        pd.NA
                        if reason_code is None
                        else int(int(reason_code).bit_count())
                    ),
                    "reason_names": _join_labels(reason_names),
                    "reason_labels": _join_labels(reason_labels),
                    "severity": severity,
                    "severity_rank": int(_severity_rank(severity)),
                    "detail_available": True,
                    "notes": notes,
                }
            )

    return pd.DataFrame(rows)


def _build_warning_summary(points: pd.DataFrame) -> pd.DataFrame:
    if "warning_flags" not in points.columns:
        if "warning_count" in points.columns:
            warning_count_series = pd.to_numeric(
                points["warning_count"],
                errors="coerce",
            ).fillna(0)
        else:
            warning_count_series = pd.Series(
                np.zeros(len(points), dtype=np.int64),
                index=points.index,
            )
        warned = int((warning_count_series > 0).sum())
        return pd.DataFrame(
            [
                {
                    "warning_name": "warning_flags_unavailable",
                    "warning_label": "warning flags unavailable",
                    "severity": "info",
                    "count": warned,
                    "point_fraction": (
                        0.0 if len(points) == 0 else float(warned) / float(len(points))
                    ),
                    "point_indices": "",
                    "notes": (
                        "warning_count is present but warning_flags are unavailable, so "
                        "reason-level decoding could not be recovered."
                    ),
                }
            ]
        )

    flags = points["warning_flags"].fillna(0).to_numpy(dtype=np.uint32)
    rows: list[dict[str, Any]] = []
    for warning in HestonIntegralWarning:
        if int(warning) == 0:
            continue
        mask = np.asarray(
            [(int(flag) & int(warning)) != 0 for flag in flags],
            dtype=np.bool_,
        )
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        point_indices = points.loc[mask, "point_index"].to_list()
        rows.append(
            {
                "warning_name": _member_name(warning),
                "warning_label": _WARNING_LABELS.get(warning, _member_name(warning)),
                "severity": _WARNING_SEVERITY.get(warning, "warning"),
                "count": count,
                "point_fraction": (
                    0.0 if len(points) == 0 else float(count) / float(len(points))
                ),
                "point_indices": _format_index_list(point_indices),
                "notes": "",
            }
        )

    if not rows:
        rows.append(
            {
                "warning_name": "none",
                "warning_label": "no warnings",
                "severity": "ok",
                "count": 0,
                "point_fraction": 0.0,
                "point_indices": "",
                "notes": "No warning flags were raised.",
            }
        )

    table = pd.DataFrame(rows)
    return table.sort_values(
        ["severity", "warning_name"],
        key=lambda col: col.map(_severity_rank) if col.name == "severity" else col,
        ascending=[False, True],
        ignore_index=True,
    )


def _build_reason_counts(panel_table: pd.DataFrame) -> pd.DataFrame:
    if panel_table.empty or not bool(panel_table["detail_available"].any()):
        return pd.DataFrame(
            [
                {
                    "reason_name": "panel_detail_unavailable",
                    "reason_label": "panel detail unavailable",
                    "severity": "info",
                    "count": 0,
                    "point_count": 0,
                    "panel_fraction": 0.0,
                    "notes": (
                        "Panel-local diagnostics were unavailable, so reason counts "
                        "could not be decoded."
                    ),
                }
            ]
        )

    available = panel_table.loc[panel_table["detail_available"].fillna(False)].copy()
    if "reason_code" not in available.columns:
        return pd.DataFrame(
            [
                {
                    "reason_name": "panel_reason_unavailable",
                    "reason_label": "panel reasons unavailable",
                    "severity": "info",
                    "count": 0,
                    "point_count": 0,
                    "panel_fraction": 0.0,
                    "notes": "panel_reason was not returned by the engine.",
                }
            ]
        )

    valid_reason_codes = available["reason_code"].dropna().astype(np.uint32)
    rows: list[dict[str, Any]] = []
    total_panels = int(len(available))

    for reason in HestonPanelReason:
        if int(reason) == 0:
            continue
        reason_mask = np.asarray(
            [
                (int(value) & int(reason)) != 0
                for value in valid_reason_codes.to_numpy(dtype=np.uint32, copy=False)
            ],
            dtype=np.bool_,
        )
        count = int(np.count_nonzero(reason_mask))
        if count == 0:
            continue
        affected_rows = available.loc[reason_mask]
        rows.append(
            {
                "reason_name": _member_name(reason),
                "reason_label": _PANEL_REASON_LABELS.get(reason, _member_name(reason)),
                "severity": _PANEL_REASON_SEVERITY.get(reason, "warning"),
                "count": count,
                "point_count": int(affected_rows["point_index"].nunique()),
                "panel_fraction": (
                    0.0 if total_panels == 0 else float(count) / float(total_panels)
                ),
                "notes": "",
            }
        )

    if not rows:
        rows.append(
            {
                "reason_name": "none",
                "reason_label": "no bad panels",
                "severity": "ok",
                "count": 0,
                "point_count": 0,
                "panel_fraction": 0.0,
                "notes": "No panel reasons were raised.",
            }
        )

    table = pd.DataFrame(rows)
    return table.sort_values(
        ["severity", "count", "reason_name"],
        key=lambda col: col.map(_severity_rank) if col.name == "severity" else col,
        ascending=[False, False, True],
        ignore_index=True,
    )


def _build_worst_panels(panel_table: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    if panel_table.empty:
        return panel_table.copy()

    ranked = panel_table.copy()
    if bool(ranked["detail_available"].any()):
        ranked["panel_invalid_sort"] = (
            ranked["panel_invalid"].fillna(False).astype(np.int64)
        )
        ranked["reason_count_sort"] = ranked["reason_count"].fillna(0).astype(np.int64)
        ranked["abs_panel_contribution_sort"] = (
            ranked["abs_panel_contribution"].fillna(-np.inf).astype(np.float64)
        )
        ranked = ranked.sort_values(
            [
                "severity_rank",
                "panel_invalid_sort",
                "reason_count_sort",
                "abs_panel_contribution_sort",
                "point_warning_count",
                "point_index",
                "panel_index",
            ],
            ascending=[False, False, False, False, False, True, True],
            ignore_index=True,
        )
        ranked = ranked.drop(
            columns=[
                "panel_invalid_sort",
                "reason_count_sort",
                "abs_panel_contribution_sort",
            ]
        )
    else:
        ranked = ranked.sort_values(
            ["severity_rank", "point_warning_count", "point_index"],
            ascending=[False, False, True],
            ignore_index=True,
        )

    ranked = ranked.head(int(top_n)).copy()
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=np.int64))
    return ranked


def _normalize_diagnostics_input(
    diagnostics: ProbabilityDiagnosticsInput,
) -> ProbabilityDiagnosticsInput:
    return diagnostics


def integration_diagnostics(
    *,
    x: float | np.ndarray,
    tau: float,
    params: HestonParams,
    j: Literal[0, 1],
    backend: HestonDiagnosticsBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
    strike: Any = None,
    top_n_panels: int = 10,
    meta: Mapping[str, Any] | None = None,
    arrays: Mapping[str, Any] | None = None,
) -> HestonIntegrationDiagnosticsBundle:
    """Wrap the existing Fourier diagnostics in readable Step 2 artifacts."""

    x_arr = np.asarray(x, dtype=np.float64)
    raw: ProbabilityDiagnosticsInput
    if x_arr.ndim == 0:
        raw = P_j_with_diagnostics(
            x=float(x_arr),
            tau=tau,
            params=params,
            j=j,
            backend=backend,
            quad_cfg=quad_cfg,
            rule=rule,
        )
    else:
        raw = P_j_batch_with_diagnostics(
            x=x_arr,
            tau=tau,
            params=params,
            j=j,
            backend=backend,
            quad_cfg=quad_cfg,
            rule=rule,
        )

    diagnostics = _normalize_diagnostics_input(raw)
    probability = probability_slice_with_diagnostics(
        probability_diagnostics=diagnostics,
        strike=strike,
        meta=meta,
        arrays=arrays,
    )
    panels = _build_panel_table(probability)
    points = _point_table(probability)
    warning_summary = _build_warning_summary(points)
    reason_counts = _build_reason_counts(panels)
    worst_panels = _build_worst_panels(panels, top_n=top_n_panels)

    bundle_meta = {
        "backend": diagnostics.backend,
        "probability_index": int(diagnostics.j),
        "tau": float(diagnostics.tau),
        "point_count": int(len(points)),
        "panel_detail_available": (
            bool(panels["detail_available"].any()) if not panels.empty else False
        ),
        "source": "option_pricing.models.heston.fourier",
        **({} if meta is None else dict(meta)),
    }

    return HestonIntegrationDiagnosticsBundle(
        meta=bundle_meta,
        probability=probability,
        tables={
            "panels": panels,
            "warning_summary": warning_summary,
            "worst_panels": worst_panels,
            "reason_counts": reason_counts,
        },
        arrays={
            "probability": dict(probability.arrays),
            "severity_rank": points["severity_rank"].to_numpy(dtype=np.int64),
            "warning_count": points["warning_count"].to_numpy(dtype=np.int64),
        },
    )


def integration_config_sweep(
    *,
    x: float | np.ndarray,
    tau: float,
    params: HestonParams,
    j: Literal[0, 1],
    cases: Sequence[IntegrationSweepCase],
    strike: Any = None,
    top_n_panels: int = 10,
) -> tuple[pd.DataFrame, dict[str, HestonIntegrationDiagnosticsBundle]]:
    """Run a readable config sweep over the existing integration wrappers."""

    if not cases:
        raise ValueError("cases must contain at least one sweep configuration.")

    artifacts: dict[str, HestonIntegrationDiagnosticsBundle] = {}
    rows: list[dict[str, Any]] = []
    baseline_probability: np.ndarray | None = None

    for idx, case in enumerate(cases):
        label = str(case.get("label", f"case_{idx}"))
        if label in artifacts:
            raise ValueError(f"Duplicate integration sweep label: {label!r}.")

        artifact = integration_diagnostics(
            x=x,
            tau=tau,
            params=params,
            j=j,
            backend=case.get("backend", "gauss_legendre"),
            quad_cfg=case.get("quad_cfg"),
            rule=case.get("rule"),
            strike=strike,
            top_n_panels=top_n_panels,
            meta={"config_label": label},
        )
        artifacts[label] = artifact

        probability_values = artifact.probability.table["probability"].to_numpy(
            dtype=np.float64
        )
        if baseline_probability is None:
            baseline_probability = probability_values.copy()

        diff = np.asarray(probability_values - baseline_probability, dtype=np.float64)
        table = artifact.probability.table
        rows.append(
            {
                "config_label": label,
                "backend": artifact.meta["backend"],
                "probability_index": int(j),
                "point_count": int(len(table)),
                "warning_point_count": int((table["warning_count"] > 0).sum()),
                "max_warning_count": (
                    int(table["warning_count"].max()) if len(table) else 0
                ),
                "max_severity": (
                    _worst_severity(table["severity"].astype(str).tolist())
                    if "severity" in table.columns and len(table)
                    else "ok"
                ),
                "panel_detail_available": (
                    bool(artifact.tables["panels"]["detail_available"].any())
                    if not artifact.tables["panels"].empty
                    else False
                ),
                "max_tail_fraction": (
                    None
                    if "tail_fraction" not in table.columns
                    else float(
                        pd.to_numeric(table["tail_fraction"], errors="coerce").max()
                    )
                ),
                "max_cancellation_ratio": (
                    None
                    if "cancellation_ratio" not in table.columns
                    else float(
                        pd.to_numeric(
                            table["cancellation_ratio"],
                            errors="coerce",
                        ).max()
                    )
                ),
                "max_quad_error_estimate": (
                    None
                    if "quad_error_estimate" not in table.columns
                    else float(
                        pd.to_numeric(
                            table["quad_error_estimate"],
                            errors="coerce",
                        ).max()
                    )
                ),
                "max_abs_probability_diff_vs_baseline": float(np.nanmax(np.abs(diff))),
                "notes": "",
            }
        )

    return pd.DataFrame(rows), artifacts


__all__ = [
    "integration_config_sweep",
    "integration_diagnostics",
    "probability_slice_with_diagnostics",
]
