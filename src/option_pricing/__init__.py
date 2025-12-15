"""
option_pricing

Core option pricing library.

This package exposes the main user-facing functions at the top level, so you
can write, for example:

    from option_pricing import bs_call, mc_european_call
"""

# Re-export key functions from submodules
from .processes import sim_brownian, sim_gbm, plot_sample_paths

__all__ = [
    "sim_brownian",
    "sim_gbm",
    "plot_sample_paths"
]
