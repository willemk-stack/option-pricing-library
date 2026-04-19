"""One-call Step 1 report assembly for notebook-facing Heston diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .contracts import (
    HESTON_REQUIRED_REPORT_TABLES,
    HESTON_SLICE_TABLE_COLUMNS,
    HESTON_SUMMARY_METRICS,
    HESTON_SUMMARY_TABLE_COLUMNS,
)
from .models import (
    HestonBackendComparisonDiagnostics,
    HestonDiagnosticsReport,
    HestonPriceSliceDiagnostics,
    HestonProbabilitySliceDiagnostics,
)


def _default_summary_table() -> pd.DataFrame:
    rows = []
    for metric in HESTON_SUMMARY_METRICS:
        notes = "Populate from downstream orchestration."
        if metric == "suspicious_strike_count":
            notes = "Approval required before finalizing suspicious-strike thresholds."
        rows.append(
            {
                "metric": metric,
                "value": None,
                "notes": notes,
                "severity": None,
            }
        )

    return pd.DataFrame(rows, columns=HESTON_SUMMARY_TABLE_COLUMNS)


def _empty_table(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _default_report_tables() -> dict[str, pd.DataFrame]:
    return {
        "summary": _default_summary_table(),
        "slice": _empty_table(HESTON_SLICE_TABLE_COLUMNS),
        "worst_strikes": _empty_table(("strike", "notes")),
        "backend_compare": _empty_table(("backend_a", "backend_b", "price_diff")),
        "config_sweep": _empty_table(("config_label", "metric", "value")),
        "worst_panels_p0": _empty_table(("panel_index", "notes")),
        "worst_panels_p1": _empty_table(("panel_index", "notes")),
    }


def _merge_mapping(
    base: dict[str, Any],
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if extra is None:
        return base

    for key, value in extra.items():
        base[str(key)] = value
    return base


def run_heston_slice_diagnostics(
    *,
    meta: Mapping[str, Any] | None = None,
    tables: Mapping[str, pd.DataFrame] | None = None,
    arrays: Mapping[str, Any] | None = None,
    probability: HestonProbabilitySliceDiagnostics | None = None,
    price: HestonPriceSliceDiagnostics | None = None,
    backend_comparison: HestonBackendComparisonDiagnostics | None = None,
) -> HestonDiagnosticsReport:
    """Assemble the full Step 1 Heston diagnostics report.

    This runner intentionally freezes report topology without adding hidden
    recomputation. Callers provide already-packaged artifacts and/or explicit
    tables/arrays, and this helper only assembles them into the full
    ``HestonDiagnosticsReport`` contract.

    Notes
    -----
    - The report always exposes ``meta``, ``tables``, and ``arrays``.
    - Required v1 tables are always present.
    - Plotting code must consume returned tables and arrays only.
    """

    default_tables = _default_report_tables()
    report_tables: dict[str, pd.DataFrame] = {
        name: table.copy() for name, table in default_tables.items()
    }

    if tables is not None:
        for name, table in tables.items():
            report_tables[str(name)] = table.copy()

    if price is not None:
        if tables is not None and "slice" in tables:
            raise ValueError(
                "Pass either a price artifact or tables['slice'], not both."
            )
        report_tables["slice"] = price.table.copy()

    if backend_comparison is not None:
        if tables is not None and "backend_compare" in tables:
            raise ValueError(
                "Pass either a backend_comparison artifact or "
                "tables['backend_compare'], not both."
            )
        report_tables["backend_compare"] = backend_comparison.table.copy()

    if probability is not None:
        if tables is not None and "probability" in tables:
            raise ValueError(
                "Pass either a probability artifact or tables['probability'], "
                "not both."
            )
        report_tables["probability"] = probability.table.copy()

    missing_tables = [
        name for name in HESTON_REQUIRED_REPORT_TABLES if name not in report_tables
    ]
    if missing_tables:
        raise ValueError(
            "run_heston_slice_diagnostics failed to populate required table(s): "
            f"{', '.join(missing_tables)}."
        )

    report_meta: dict[str, Any] = {}
    report_meta = _merge_mapping(report_meta, meta)

    if probability is not None:
        report_meta["probability"] = dict(probability.meta)
    if price is not None:
        report_meta["price"] = dict(price.meta)
    if backend_comparison is not None:
        report_meta["backend_comparison"] = dict(backend_comparison.meta)

    report_meta.setdefault(
        "plotting_contract",
        (
            "plot functions may consume existing tables and arrays only; "
            "they must not recompute diagnostics"
        ),
    )

    report_arrays: dict[str, Any] = {}
    report_arrays = _merge_mapping(report_arrays, arrays)

    if probability is not None:
        report_arrays.setdefault("probability", dict(probability.arrays))
    if price is not None:
        report_arrays.setdefault("slice", dict(price.arrays))
    if backend_comparison is not None:
        report_arrays.setdefault(
            "backend_compare",
            dict(backend_comparison.arrays),
        )

    return HestonDiagnosticsReport(
        meta=report_meta,
        tables=report_tables,
        arrays=report_arrays,
    )
