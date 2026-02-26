from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from option_pricing.vol.local_vol_types import DupireLVReport


@dataclass(frozen=True)
class SurfaceSlice:
    """Surface snapshot at a single expiry on the smile grid."""

    T: float
    y: np.ndarray
    K: np.ndarray
    F: float
    w: np.ndarray
    iv: np.ndarray


@dataclass(frozen=True)
class NoArbWorstPointsReport:
    """Worst arbitrage violations across monotonicity/convexity/calendar checks."""

    monotonicity: pd.DataFrame
    convexity: pd.DataFrame
    calendar: pd.DataFrame
    summary: dict[str, float | int | bool | str]
    suggestions: tuple[str, ...]


@dataclass(frozen=True)
class LocalVolGridReport:
    """Local-vol diagnostics sampled on a (T, y) grid."""

    expiries: np.ndarray  # (nT,)
    y: np.ndarray  # (ny,)
    K: np.ndarray  # (nT, ny)
    local_var: np.ndarray  # (nT, ny)
    sigma: np.ndarray  # (nT, ny)
    denom: np.ndarray  # (nT, ny)
    invalid: np.ndarray  # (nT, ny) bool
    reason: np.ndarray  # (nT, ny) uint32 bitmask
    invalid_count: int
    invalid_frac: float
    reason_counts: dict[str, int]
    worst_points: pd.DataFrame


@dataclass(frozen=True)
class LocalVolCompareReport:
    """Compare Gatheral-from-w vs Dupire-from-call-grid local vol on a shared grid."""

    expiries: np.ndarray  # (nT,)
    strikes: np.ndarray  # (nK,)
    forwards: np.ndarray  # (nT,)

    y: np.ndarray  # (nT, nK)
    g_sigma: np.ndarray  # (nT, nK)
    g_local_var: np.ndarray  # (nT, nK)
    g_denom: np.ndarray  # (nT, nK)
    g_invalid: np.ndarray  # (nT, nK)
    g_reason: np.ndarray  # (nT, nK)

    dupire: DupireLVReport

    diff_sigma: np.ndarray  # (nT, nK)
    diff_local_var: np.ndarray  # (nT, nK)
    invalid_union: np.ndarray  # (nT, nK)

    summary: dict[str, float | int]
    worst_diffs: pd.DataFrame
    gatheral_reason_counts: dict[str, int]
    dupire_reason_counts: dict[str, int]
