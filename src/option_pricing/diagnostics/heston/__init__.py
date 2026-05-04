"""Public notebook-facing API for Heston diagnostics."""

from .calibration import (
    build_synthetic_heston_quote_set,
    run_heston_calibration_benchmark_diagnostics,
)
from .calibration_fit import (
    run_heston_calibration_diagnostics,
    run_heston_calibration_fit_diagnostics,
)
from .fixtures import (
    MARKET_LIKE_SYNTHETIC_FIXTURE_LABEL,
    build_market_like_heston_quote_set,
)
from .model_comparison import run_heston_vs_local_vol_comparison
from .models import (
    HestonBackendComparisonDiagnostics,
    HestonCalibrationBenchmarkDiagnostics,
    HestonCalibrationFitDiagnostics,
    HestonDiagnosticsReport,
    HestonModelComparisonDiagnostics,
    HestonPriceSliceDiagnostics,
    HestonProbabilitySliceDiagnostics,
)
from .monte_carlo import (
    HestonMCComparisonCase,
    HestonMCSweepConfig,
    compare_heston_mc_schemes,
    run_heston_mc_comparison_sweep,
    summarize_bias_vs_timestep,
    summarize_runtime_vs_error,
)
from .plot import (
    plot_heston_calibration_iv_residuals,
    plot_heston_calibration_objective_slice,
    plot_heston_calibration_smile_overlay,
    plot_heston_mc_bias_vs_timestep,
    plot_heston_mc_runtime_vs_error,
    plot_heston_mc_scheme_comparison,
    plot_heston_model_comparison_error_buckets,
    plot_heston_model_comparison_iv_residual_heatmap,
    plot_heston_model_comparison_smile_overlay,
    plot_heston_model_comparison_train_heldout,
    plot_heston_multistart_cost_summary,
)
from .pricing import (
    compare_backend_slice,
    price_slice_with_diagnostics,
    probability_slice_with_diagnostics,
    run_heston_pricing_diagnostics,
)
from .quote_policy import heston_calibration_quote_policy_tables
from .report import run_heston_slice_diagnostics

__all__ = [
    "HestonBackendComparisonDiagnostics",
    "HestonCalibrationBenchmarkDiagnostics",
    "HestonCalibrationFitDiagnostics",
    "HestonDiagnosticsReport",
    "HestonMCComparisonCase",
    "HestonModelComparisonDiagnostics",
    "HestonPriceSliceDiagnostics",
    "HestonProbabilitySliceDiagnostics",
    "HestonMCSweepConfig",
    "MARKET_LIKE_SYNTHETIC_FIXTURE_LABEL",
    "compare_backend_slice",
    "compare_heston_mc_schemes",
    "build_market_like_heston_quote_set",
    "build_synthetic_heston_quote_set",
    "plot_heston_calibration_iv_residuals",
    "plot_heston_calibration_objective_slice",
    "plot_heston_calibration_smile_overlay",
    "plot_heston_model_comparison_error_buckets",
    "plot_heston_model_comparison_iv_residual_heatmap",
    "plot_heston_model_comparison_smile_overlay",
    "plot_heston_model_comparison_train_heldout",
    "plot_heston_multistart_cost_summary",
    "plot_heston_mc_bias_vs_timestep",
    "plot_heston_mc_runtime_vs_error",
    "plot_heston_mc_scheme_comparison",
    "price_slice_with_diagnostics",
    "probability_slice_with_diagnostics",
    "heston_calibration_quote_policy_tables",
    "run_heston_calibration_benchmark_diagnostics",
    "run_heston_calibration_diagnostics",
    "run_heston_calibration_fit_diagnostics",
    "run_heston_mc_comparison_sweep",
    "run_heston_pricing_diagnostics",
    "run_heston_slice_diagnostics",
    "run_heston_vs_local_vol_comparison",
    "summarize_bias_vs_timestep",
    "summarize_runtime_vs_error",
]
