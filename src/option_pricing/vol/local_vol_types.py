"""Type definitions and enums for local volatility computations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]


@dataclass(frozen=True, slots=True)
class LocalVolResult:
    sigma: FloatArray  # (nT, nK)
    local_var: FloatArray  # (nT, nK)
    invalid: NDArray[np.bool_]  # (nT, nK)
    invalid_count: int


class LVInvalidReason(IntFlag):
    """Reason codes for invalid local-vol values (bitmask)."""

    NONFINITE_INPUT = 1 << 0
    W_TOO_SMALL = 1 << 1

    DENOM_NONFINITE = 1 << 2
    DENOM_TOO_SMALL = 1 << 3
    DENOM_NONPOSITIVE = 1 << 4

    CURVATURE_TOO_SMALL = 1 << 5

    LOCALVAR_NONFINITE = 1 << 6
    LOCALVAR_NEGATIVE = 1 << 7

    TRIM_T = 1 << 8
    TRIM_K = 1 << 9


@dataclass(frozen=True, slots=True)
class GatheralLVReport:
    """Diagnostics for Gatheral local variance computed from total variance."""

    y: FloatArray
    w: FloatArray
    w_y: FloatArray
    w_yy: FloatArray
    w_T: FloatArray

    denom: FloatArray
    local_var: FloatArray
    sigma: FloatArray

    invalid: NDArray[np.bool_]
    reason: NDArray[np.uint32]
    invalid_count: int


@dataclass(frozen=True, slots=True)
class DupireLVReport:
    """Diagnostics for Dupire local variance computed from a (tau, strike) call grid."""

    sigma: FloatArray
    local_var: FloatArray
    num: FloatArray
    denom: FloatArray
    curvature: FloatArray

    invalid: NDArray[np.bool_]
    reason: NDArray[np.uint32]
    invalid_count: int

    trim_t: int
    trim_k: int
    price_convention: str
    strike_coordinate: str
