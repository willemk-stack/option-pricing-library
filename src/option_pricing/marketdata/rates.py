"""Rate selection policies for normalized marketdata frames."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from option_pricing.marketdata.providers.fred import FredRateUnavailableError
from option_pricing.marketdata.schemas import DatasetName
from option_pricing.marketdata.validation import coerce_frame


@dataclass(frozen=True, slots=True)
class FredRateSelection:
    """Pricing-ready selection from a normalized FRED rate series."""

    series_id: str
    observation_date: pd.Timestamp
    value_percent: float
    rate: float
    asof: pd.Timestamp
    source: str = "fred"
    rate_compounding: str = "continuous"

    @property
    def continuous_decimal(self) -> float:
        """Alias for the pricing-ready continuous decimal rate."""

        return self.rate


def fred_percent_to_continuous_decimal(value_percent: float) -> float:
    """Convert a FRED annualized percent quote to a continuous decimal rate."""

    return math.log1p(value_percent / 100.0)


def select_latest_fred_rate_at_or_before_asof(
    frame: pd.DataFrame,
    *,
    series_id: str,
    asof: str | pd.Timestamp,
) -> FredRateSelection:
    """Select the latest non-missing FRED observation at or before ``asof``."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            "fred_series frame must be a pandas DataFrame, "
            f"got {type(frame).__name__}"
        )

    cleaned_series_id = _clean_series_id(series_id)
    asof_timestamp = _asof_timestamp(asof)
    asof_date = pd.Timestamp(asof_timestamp.date())

    coerced = coerce_frame(frame, DatasetName.FRED_SERIES, allow_extra=True)
    same_series = coerced["series_id"].astype("string") == cleaned_series_id
    usable = coerced.loc[
        same_series
        & coerced["value"].notna()
        & (coerced["observation_date"] <= asof_date)
    ]

    if usable.empty:
        raise FredRateUnavailableError(
            "No usable FRED rate observation exists for "
            f"series_id={cleaned_series_id!r} at or before {asof_date.date()}"
        )

    selected = usable.sort_values(
        ["observation_date", "realtime_start", "realtime_end"],
        kind="mergesort",
    ).iloc[-1]
    value_percent = float(selected["value"])
    observation_date = pd.Timestamp(selected["observation_date"])
    source = "fred" if pd.isna(selected["source"]) else str(selected["source"])

    return FredRateSelection(
        series_id=cleaned_series_id,
        observation_date=observation_date,
        value_percent=value_percent,
        rate=fred_percent_to_continuous_decimal(value_percent),
        asof=asof_timestamp,
        source=source,
    )


def _clean_series_id(series_id: str) -> str:
    if not isinstance(series_id, str):
        raise TypeError("series_id must be a string")
    cleaned = series_id.strip()
    if not cleaned:
        raise ValueError("series_id must be a non-empty string")
    return cleaned


def _asof_timestamp(asof: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(asof)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


__all__ = [
    "FredRateSelection",
    "fred_percent_to_continuous_decimal",
    "select_latest_fred_rate_at_or_before_asof",
]
