"""Public plotting helpers for PDE diagnostics."""

from option_pricing.diagnostics._pde_compare.plots import (  # noqa: F401
    plot_convergence,
    plot_error_vs_runtime,
    plot_price_scatter,
)

__all__ = [
    "plot_convergence",
    "plot_error_vs_runtime",
    "plot_price_scatter",
]
