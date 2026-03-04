"""Shared PDE-vs-analytic diagnostics helpers.

This internal package is used by multiple diagnostics modules.

It centralizes:
  - robust metadata extraction from :class:`~option_pricing.types.PricingInputs`
  - timing and error computation
  - common batch/sweep runners
  - shared tables and plots
"""

from .engine import run_cases, run_once, sweep_nt, sweep_nx, sweep_nx_nt
from .meta import meta_from_inputs

__all__ = [
    "meta_from_inputs",
    "run_once",
    "run_cases",
    "sweep_nx",
    "sweep_nt",
    "sweep_nx_nt",
]
