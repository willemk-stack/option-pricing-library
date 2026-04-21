"""One-call Step 5 orchestration for notebook-facing Heston diagnostics.

This module preserves the frozen Step 1 report topology:

- ``meta``
- ``tables``
- ``arrays``

It only packages already-computed artifacts. Plot objects are rejected so the
report stays lightweight and serialization-friendly.
"""

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

_DEFAULT_WORST_STRIKE_COLUMNS: tuple[str, ...] = (
    "rank",
    *HESTON_SLICE_TABLE_COLUMNS,
    "warning_count_p0",
    "warning_count_p1",
    "severity_p0",
    "severity_p1",
    "combined_warning_labels",
    "config_price_span",
    "smoothness_signal",
    "discontinuity_signal",
    "perturbation_max_absolute_price_change",
    "perturbation_max_relative_price_change",
    "parameter_sensitivity_flag",
    "parameter_sensitivity_reasons",
    "suspicious_flag",
    "suspicious_reasons",
)

_DEFAULT_BACKEND_COMPARE_COLUMNS: tuple[str, ...] = (
    "strike",
    "log_moneyness",
    "backend_a",
    "backend_b",
    "price_a",
    "price_b",
    "price_diff",
    "abs_price_diff",
    "implied_vol_a",
    "implied_vol_b",
    "implied_vol_diff",
    "warning_count_a",
    "warning_count_b",
    "severity_a",
    "severity_b",
)

_DEFAULT_CONFIG_SWEEP_COLUMNS: tuple[str, ...] = (
    "config_label",
    "backend",
    "config_resolution",
    "resolved_u_max",
    "resolved_n_panels",
    "resolved_nodes_per_panel",
    "resolved_panel_spacing",
    "resolved_cluster_strength",
    "p0_max_severity",
    "p1_max_severity",
    "p0_warning_point_count",
    "p1_warning_point_count",
    "max_abs_price_diff_vs_baseline",
    "mean_abs_price_diff_vs_baseline",
    "max_abs_probability_diff_p0",
    "max_abs_probability_diff_p1",
    "notes",
)

_DEFAULT_WORST_PANEL_COLUMNS: tuple[str, ...] = (
    "rank",
    "point_index",
    "probability_index",
    "backend",
    "log_moneyness",
    "strike",
    "point_severity",
    "point_warning_count",
    "panel_index",
    "panel_start",
    "panel_end",
    "panel_width",
    "panel_contribution",
    "abs_panel_contribution",
    "panel_invalid",
    "reason_code",
    "reason_count",
    "reason_names",
    "reason_labels",
    "severity",
    "severity_rank",
    "detail_available",
    "notes",
)

_OPTIONAL_ARRAY_GROUPS: tuple[str, ...] = (
    "probability",
    "slice",
    "backend_compare",
    "probability_p0",
    "probability_p1",
    "comparison_probability_p0",
    "comparison_probability_p1",
    "config_sweep_prices",
    "parameter_perturbation_prices",
    "parameter_perturbation_table",
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
        "worst_strikes": _empty_table(_DEFAULT_WORST_STRIKE_COLUMNS),
        "backend_compare": _empty_table(_DEFAULT_BACKEND_COMPARE_COLUMNS),
        "config_sweep": _empty_table(_DEFAULT_CONFIG_SWEEP_COLUMNS),
        "worst_panels_p0": _empty_table(_DEFAULT_WORST_PANEL_COLUMNS),
        "worst_panels_p1": _empty_table(_DEFAULT_WORST_PANEL_COLUMNS),
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


def _copy_tables(
    tables: Mapping[str, pd.DataFrame] | None,
) -> dict[str, pd.DataFrame]:
    if tables is None:
        return {}
    return {str(name): table.copy() for name, table in tables.items()}


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


def run_heston_slice_diagnostics(
    *,
    meta: Mapping[str, Any] | None = None,
    tables: Mapping[str, pd.DataFrame] | None = None,
    arrays: Mapping[str, Any] | None = None,
    probability: HestonProbabilitySliceDiagnostics | None = None,
    price: HestonPriceSliceDiagnostics | None = None,
    backend_comparison: HestonBackendComparisonDiagnostics | None = None,
) -> HestonDiagnosticsReport:
    """Assemble the full notebook-facing Heston diagnostics report.

    This function is downstream-only orchestration. It merges already-built
    artifacts and explicit tables/arrays into the frozen Step 1 report shape
    without re-running pricing, recomputing diagnostics, or embedding plotting
    objects.

    Required tables are always present:

    - ``summary``
    - ``slice``
    - ``worst_strikes``
    - ``backend_compare``
    - ``config_sweep``
    - ``worst_panels_p0``
    - ``worst_panels_p1``

    There are no required arrays. Optional packaged array groups include:

    - ``probability``
    - ``slice``
    - ``backend_compare``
    - ``probability_p0``
    - ``probability_p1``
    - ``comparison_probability_p0``
    - ``comparison_probability_p1``
    - ``config_sweep_prices``
    - ``parameter_perturbation_prices``
    - ``parameter_perturbation_table``

    The resulting report is meant for review notebooks and serialization. It
    does not prove correctness and it does not finalize policy-sensitive
    semantics such as suspiciousness thresholds.
    """

    default_tables = _default_report_tables()
    report_tables = {name: table.copy() for name, table in default_tables.items()}
    report_tables.update(_copy_tables(tables))

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
    report_meta.setdefault("optional_array_groups", list(_OPTIONAL_ARRAY_GROUPS))

    report_arrays: dict[str, Any] = {}
    report_arrays = _merge_mapping(report_arrays, arrays)

    if probability is not None:
        report_arrays.setdefault("probability", dict(probability.arrays))
    if price is not None:
        report_arrays.setdefault("slice", dict(price.arrays))
    if backend_comparison is not None:
        report_arrays.setdefault("backend_compare", dict(backend_comparison.arrays))

    _reject_plot_objects(report_meta, label="meta")
    _reject_plot_objects(report_arrays, label="arrays")

    return HestonDiagnosticsReport(
        meta=report_meta,
        tables=report_tables,
        arrays=report_arrays,
    )
