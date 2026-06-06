"""
Vol surface diagnostics (compute + plotting).

- compute.py: pure functions (no matplotlib)
- plot.py: plotting helpers (matplotlib)

Most notebook workflows should start with :func:`run_surface_diagnostics`, which
returns a :class:`SurfaceDiagnosticsReport` containing tables + grids.
"""

from .localvol import (
    fixed_strikes_from_forward_y,
    localvol_compare_gatheral_vs_dupire,
    localvol_compare_gatheral_vs_dupire_on_forward_y,
    localvol_grid_diagnostics,
    localvol_summary,
)
from .models import NoArbWorstPointsReport
from .noarb import first_failing_convexity, noarb_worst_points
from .report import SurfaceDiagnosticsReport, run_surface_diagnostics

__all__ = [
    "NoArbWorstPointsReport",
    "SurfaceDiagnosticsReport",
    "fixed_strikes_from_forward_y",
    "first_failing_convexity",
    "localvol_compare_gatheral_vs_dupire",
    "localvol_compare_gatheral_vs_dupire_on_forward_y",
    "localvol_grid_diagnostics",
    "localvol_summary",
    "noarb_worst_points",
    "run_surface_diagnostics",
]
