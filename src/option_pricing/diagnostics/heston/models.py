"""
high-level report dataclasses for notebook use

Examples:
HestonPanelDiagnosticsReport
HestonSliceDiagnosticsReport
HestonPricingDiagnosticsReport
HestonDiagnosticsReport
"""

from dataclasses import dataclass

from ...models.heston.fourier import (
    HestonIntegralBatchDiagnostics,
    HestonIntegralDiagnostics,
)
from ...typing import FloatArray


@dataclass(frozen=True, slots=True)
class HestonPriceDiagnostics:
    """Scalar pricing diagnostics assembled from ``P_0`` and ``P_1``."""

    strike: float
    tau: float
    forward: float
    df: float
    p0: HestonIntegralDiagnostics
    p1: HestonIntegralDiagnostics
    price: float


@dataclass(frozen=True, slots=True)
class HestonPriceBatchDiagnostics:
    """Batch pricing diagnostics container for a strike slice.

    Notes
    -----
    This shape is primarily suited to vectorized fixed-rule evaluation. The
    current public pricing helpers do not yet expose a batch diagnostics API.
    """

    strike: FloatArray  # batch_shape
    tau: float
    forward: float
    df: float
    p0: HestonIntegralBatchDiagnostics
    p1: HestonIntegralBatchDiagnostics
    price: FloatArray  # batch_shape
