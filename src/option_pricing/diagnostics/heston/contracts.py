"""Public Step 1 contract helpers for notebook-facing Heston diagnostics.

This module freezes the stable vocabulary for the downstream diagnostics layer:

- severity ordering
- required report table names
- canonical verbose slice-table column names

Step 1 intentionally does *not* freeze numerical thresholds, warning
heuristics, or direct orchestration signatures.
"""

from __future__ import annotations

from typing import Literal

type HestonDiagnosticsSeverity = Literal[
    "ok",
    "info",
    "warning",
    "severe",
    "critical",
]

type HestonDiagnosticsBackend = Literal["gauss_legendre", "quad"]

HESTON_SEVERITY_ORDER: tuple[HestonDiagnosticsSeverity, ...] = (
    "ok",
    "info",
    "warning",
    "severe",
    "critical",
)

HESTON_REQUIRED_REPORT_TABLES: tuple[str, ...] = (
    "summary",
    "slice",
    "worst_strikes",
    "backend_compare",
    "config_sweep",
    "worst_panels_p0",
    "worst_panels_p1",
)

HESTON_SLICE_TABLE_COLUMNS: tuple[str, ...] = (
    "strike",
    "log_moneyness",
    "price",
    "implied_vol",
    "warning_count",
    "severity",
    "tail_fraction",
    "cancellation_ratio",
    "backend_diff",
    "smoothness_flag",
    "discontinuity_flag",
)

HESTON_SUMMARY_TABLE_COLUMNS: tuple[str, ...] = (
    "metric",
    "value",
    "notes",
    "severity",
)

HESTON_SUMMARY_METRICS: tuple[str, ...] = (
    "worst_strike",
    "total_warnings",
    "max_tail_fraction",
    "max_cancellation_ratio",
    "max_backend_discrepancy",
    "suspicious_strike_count",
)
