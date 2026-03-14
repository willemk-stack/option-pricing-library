"""Volatility tools and surfaces.

This subpackage re-exports the common, user-facing entrypoints so callers can
import from ``option_pricing.vol`` without drilling into module paths.
"""

from __future__ import annotations

# No-arbitrage checks
from .arbitrage import (
    CalendarVarianceReport,
    ConvexityReport,
    MonotonicityReport,
    SurfaceNoArbReport,
    check_smile_call_convexity,
    check_smile_price_monotonicity,
    check_surface_noarb,
)

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
from .smile_interpolated import (
    LinearWInterpolatedSmileSlice,
    NoArbInterpolatedSmileSlice,
)

# eSSVI surfaces
from .ssvi import (
    ATMThetaDiagnostics,
    ESSVICalibrationConfig,
    ESSVIConstraintReport,
    ESSVIFitDiagnostics,
    ESSVIFitResult,
    ESSVIImpliedSurface,
    ESSVITermStructures,
    ESSVIValidationReport,
    EtaTermStructure,
    PsiTermStructure,
    SampledThetaTermStructure,
    ThetaTermStructure,
    build_theta_term_from_quotes,
    calibrate_essvi,
    evaluate_essvi_constraints,
    validate_essvi_surface,
)
from .surface_core import VolSurface
from .vol_types import DifferentiableSmileSlice, GridSmileSlice, SmileSlice

__all__ = [
    # Implied vol
    "implied_vol_bs",
    "implied_vol_bs_result",
    "ImpliedVolResult",
    "ImpliedVolSliceResult",
    # No-arbitrage checks
    "check_smile_call_convexity",
    "check_smile_price_monotonicity",
    "check_surface_noarb",
    "MonotonicityReport",
    "ConvexityReport",
    "CalendarVarianceReport",
    "SurfaceNoArbReport",
    # Smiles and surfaces
    "Smile",
    "NoArbInterpolatedSmileSlice",
    "LinearWInterpolatedSmileSlice",
    "VolSurface",
    "SmileSlice",
    "DifferentiableSmileSlice",
    "GridSmileSlice",
    "ATMThetaDiagnostics",
    "SampledThetaTermStructure",
    "ESSVICalibrationConfig",
    "ESSVIFitDiagnostics",
    "ESSVIFitResult",
    "ESSVIConstraintReport",
    "ESSVIValidationReport",
    "ThetaTermStructure",
    "PsiTermStructure",
    "EtaTermStructure",
    "ESSVITermStructures",
    "ESSVIImpliedSurface",
    "build_theta_term_from_quotes",
    "calibrate_essvi",
    "evaluate_essvi_constraints",
    "validate_essvi_surface",
    # Local vol objects
    "LocalVolSurface",
    "local_vol_from_call_grid",
    "local_vol_from_call_grid_diagnostics",
    "LocalVolResult",
    "DupireLVReport",
    "GatheralLVReport",
    "LVInvalidReason",
]
