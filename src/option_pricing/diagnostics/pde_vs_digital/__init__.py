"""PDE solver vs analytic Black-Scholes diagnostics for digital (cash-or-nothing) options."""

from .recipes import DigitalPdeBaselineResult, run_demo_baseline_sweep

__all__ = ["DigitalPdeBaselineResult", "run_demo_baseline_sweep"]
