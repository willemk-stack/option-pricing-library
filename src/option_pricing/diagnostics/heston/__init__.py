"""Public notebook-facing API for Heston diagnostics."""

from .models import (
    HestonBackendComparisonDiagnostics,
    HestonDiagnosticsReport,
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
    plot_heston_mc_bias_vs_timestep,
    plot_heston_mc_runtime_vs_error,
    plot_heston_mc_scheme_comparison,
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
    "HestonDiagnosticsReport",
    "HestonMCComparisonCase",
    "HestonPriceSliceDiagnostics",
    "HestonProbabilitySliceDiagnostics",
    "HestonMCSweepConfig",
    "compare_backend_slice",
    "compare_heston_mc_schemes",
    "plot_heston_mc_bias_vs_timestep",
    "plot_heston_mc_runtime_vs_error",
    "plot_heston_mc_scheme_comparison",
    "price_slice_with_diagnostics",
    "probability_slice_with_diagnostics",
    "run_heston_mc_comparison_sweep",
    "run_heston_pricing_diagnostics",
    "run_heston_slice_diagnostics",
    "summarize_bias_vs_timestep",
    "summarize_runtime_vs_error",
]
