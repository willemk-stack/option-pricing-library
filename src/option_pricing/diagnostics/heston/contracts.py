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

# NOTE: Calibration benchmark table names and columns are a diagnostics
# contract distinct from the pricing report topology.
HESTON_CALIBRATION_BENCHMARK_REQUIRED_TABLES: tuple[str, ...] = (
    "runs",
    "summary",
    "parameter_recovery",
    "residuals",
)

HESTON_CALIBRATION_BENCHMARK_RUN_COLUMNS: tuple[str, ...] = (
    "mode",
    "use_analytic_jac",
    "repeat_index",
    "warmup",
    "runtime_ms",
    "nfev",
    "njev",
    "cost",
    "residual_norm",
    "max_abs_residual",
    "success",
    "status",
    "message",
    "method",
    "loss",
    "backend",
    "n_quotes",
)

HESTON_CALIBRATION_BENCHMARK_SUMMARY_COLUMNS: tuple[str, ...] = (
    "mode",
    "median_runtime_ms",
    "mean_runtime_ms",
    "std_runtime_ms",
    "median_nfev",
    "median_njev",
    "median_residual_norm",
    "median_max_abs_residual",
    "speedup_vs_finite_difference",
    "nfev_ratio_vs_finite_difference",
    "notes",
)

HESTON_CALIBRATION_BENCHMARK_PARAMETER_COLUMNS: tuple[str, ...] = (
    "mode",
    "parameter",
    "true",
    "seed",
    "fitted",
    "fit_minus_true",
    "abs_fit_minus_true",
)

HESTON_CALIBRATION_BENCHMARK_RESIDUAL_COLUMNS: tuple[str, ...] = (
    "mode",
    "quote_index",
    "expiry",
    "strike",
    "log_moneyness",
    "residual",
)

# NOTE: Calibration-fit evidence is a production-facing diagnostics contract,
# distinct from the analytic-Jacobian benchmark contract above.
HESTON_CALIBRATION_FIT_REQUIRED_TABLES: tuple[str, ...] = (
    "residuals",
    "smile_fit",
    "iv_residual_grid",
    "parameter_recovery",
    "multistart_runs",
    "held_out_errors",
    "objective_slices",
)

HESTON_CALIBRATION_FIT_RESIDUAL_COLUMNS: tuple[str, ...] = (
    "quote_index",
    "expiry",
    "strike",
    "log_moneyness",
    "is_call",
    "market_price",
    "model_price",
    "price_residual",
    "market_iv",
    "model_iv",
    "iv_residual",
    "iv_residual_bps",
    "bs_vega",
    "sqrt_weight",
    "is_held_out",
    "sample",
)

HESTON_CALIBRATION_FIT_SMILE_COLUMNS: tuple[str, ...] = (
    "quote_index",
    "expiry",
    "strike",
    "log_moneyness",
    "is_call",
    "market_iv",
    "model_iv",
    "iv_residual_bps",
    "is_held_out",
)

HESTON_CALIBRATION_FIT_IV_GRID_COLUMNS: tuple[str, ...] = (
    "quote_index",
    "expiry",
    "strike",
    "log_moneyness",
    "iv_residual_bps",
    "grid_kind",
)

HESTON_MODEL_COMPARISON_REQUIRED_TABLES: tuple[str, ...] = (
    "fit_errors",
    "error_summary",
    "tradeoff_summary",
    "held_out_comparison",
)

HESTON_MODEL_COMPARISON_FIT_ERROR_COLUMNS: tuple[str, ...] = (
    "model",
    "quote_index",
    "expiry",
    "strike",
    "log_moneyness",
    "moneyness_bucket",
    "is_call",
    "market_iv",
    "model_iv",
    "iv_residual_bps",
    "market_price",
    "model_price",
    "price_residual",
    "is_held_out",
    "sample",
)

HESTON_MODEL_COMPARISON_ERROR_SUMMARY_COLUMNS: tuple[str, ...] = (
    "model",
    "bucket",
    "n_quotes",
    "price_rmse",
    "price_mae",
    "price_max_abs",
    "iv_rmse_bps",
    "iv_mae_bps",
    "iv_max_abs_bps",
)
