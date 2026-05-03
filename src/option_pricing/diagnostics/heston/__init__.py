"""Public notebook-facing API for Heston diagnostics."""

from .calibration import (
    build_synthetic_heston_quote_set,
    run_heston_calibration_benchmark_diagnostics,
)
from .calibration_fit import (
    run_heston_calibration_diagnostics,
    run_heston_calibration_fit_diagnostics,
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
    plot_heston_multistart_cost_summary,
)
from .pricing import (
    compare_backend_slice,
    price_slice_with_diagnostics,
    probability_slice_with_diagnostics,
    run_heston_pricing_diagnostics,
)
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
    "compare_backend_slice",
    "compare_heston_mc_schemes",
    "build_synthetic_heston_quote_set",
    "plot_heston_calibration_iv_residuals",
    "plot_heston_calibration_objective_slice",
    "plot_heston_calibration_smile_overlay",
    "plot_heston_multistart_cost_summary",
    "plot_heston_mc_bias_vs_timestep",
    "plot_heston_mc_runtime_vs_error",
    "plot_heston_mc_scheme_comparison",
    "price_slice_with_diagnostics",
    "probability_slice_with_diagnostics",
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
