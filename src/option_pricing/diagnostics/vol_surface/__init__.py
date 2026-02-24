"""
Vol surface diagnostics (compute + plotting).

- compute.py: pure functions (no matplotlib)
- plot.py: plotting helpers (matplotlib)

Most notebook workflows should start with :func:`run_surface_diagnostics`, which
returns a :class:`SurfaceDiagnosticsReport` containing tables + grids.
"""

from .compute import NoArbWorstPointsReport, first_failing_convexity, noarb_worst_points
from .report import SurfaceDiagnosticsReport, run_surface_diagnostics

__all__ = [
    "NoArbWorstPointsReport",
    "SurfaceDiagnosticsReport",
    "first_failing_convexity",
    "noarb_worst_points",
    "run_surface_diagnostics",
]
