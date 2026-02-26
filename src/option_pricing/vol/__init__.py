"""Volatility tools and surfaces.

This subpackage re-exports the common, user-facing entrypoints so callers can
import from ``option_pricing.vol`` without drilling into module paths.
"""

from __future__ import annotations

# Implied-vol inversion
from .implied_vol_scalar import implied_vol_bs, implied_vol_bs_result
from .implied_vol_types import ImpliedVolResult, ImpliedVolSliceResult

# Local volatility
from .local_vol_dupire import (
    local_vol_from_call_grid,
    local_vol_from_call_grid_diagnostics,
)
from .local_vol_surface import LocalVolSurface
from .local_vol_types import (
    DupireLVReport,
    GatheralLVReport,
    LocalVolResult,
    LVInvalidReason,
)

# Volatility surfaces and smiles
from .smile_grid import Smile
from .surface_core import VolSurface
from .vol_types import DifferentiableSmileSlice, GridSmileSlice, SmileSlice

__all__ = [
    # Implied vol
    "implied_vol_bs",
    "implied_vol_bs_result",
    "ImpliedVolResult",
    "ImpliedVolSliceResult",
    # Smiles and surfaces
    "Smile",
    "VolSurface",
    "SmileSlice",
    "DifferentiableSmileSlice",
    "GridSmileSlice",
    # Local vol objects
    "LocalVolSurface",
    "local_vol_from_call_grid",
    "local_vol_from_call_grid_diagnostics",
    "LocalVolResult",
    "DupireLVReport",
    "GatheralLVReport",
    "LVInvalidReason",
]
