"""Quote-cleaning contracts for marketdata normalization workflows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class QuoteRejectionReason(StrEnum):
    """Primary rejection reason for a quote rejected by cleaning policy v1."""

    NEGATIVE_BID = "negative_bid"
    NONPOSITIVE_ASK = "nonpositive_ask"
    CROSSED_MARKET = "crossed_market"
    EXPIRED_CONTRACT = "expired_contract"
    NONPOSITIVE_STRIKE = "nonpositive_strike"
    MISSING_REQUIRED_PRICE = "missing_required_price"
    INVALID_MID = "invalid_mid"
    BELOW_INTRINSIC_TOLERANCE = "below_intrinsic_tolerance"
    SPREAD_TOO_WIDE = "spread_too_wide"
    MISSING_IV_FOR_IV_REQUIRED_WORKFLOW = "missing_iv_for_iv_required_workflow"
    MISSING_VEGA_FOR_WEIGHTED_CALIBRATION = "missing_vega_for_weighted_calibration"


@dataclass(frozen=True, slots=True)
class QuoteCleaningPolicyV1:
    """Policy parameters for the first quote-cleaning contract."""

    max_relative_spread: float = 1.00
    intrinsic_tolerance: float = 1e-8
    require_iv: bool = False
    require_vega: bool = False
    day_count: str = "ACT/365"


@dataclass(frozen=True, slots=True)
class QuoteCleaningResult:
    """Container for future cleaned and rejected quote outputs."""

    cleaned_quotes: pd.DataFrame
    rejected_quotes: pd.DataFrame
    reason_counts: dict[str, int]
    warnings: tuple[str, ...]


__all__ = [
    "QuoteCleaningPolicyV1",
    "QuoteCleaningResult",
    "QuoteRejectionReason",
]
