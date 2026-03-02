"""Public namespace for PDE diagnostics helpers."""

from .plots import plot_convergence, plot_error_vs_runtime, plot_price_scatter

__all__ = [
    "plot_convergence",
    "plot_error_vs_runtime",
    "plot_price_scatter",
]
