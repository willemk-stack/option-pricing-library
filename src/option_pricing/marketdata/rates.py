"""Rate selection policies for normalized marketdata frames."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from option_pricing.marketdata.providers.fred import FredRateUnavailableError
from option_pricing.marketdata.schemas import DatasetName
from option_pricing.marketdata.validation import coerce_frame

RATE_POLICY_FRED_TREASURY_ZERO_PROXY_LINEAR_CC = "fred_treasury_zero_proxy_linear_cc"
RATE_CURVE_SOURCE_FRED = "fred"
RATE_INTERPOLATION_LINEAR = "linear"
RATE_COMPOUNDING_CONTINUOUS = "continuous"
RATE_EXTRAPOLATION_CLAMP_WITH_WARNING = "clamp_with_warning"
FRED_TREASURY_TENOR_YEARS = {
    "DGS1MO": 1.0 / 12.0,
    "DGS3MO": 3.0 / 12.0,
    "DGS6MO": 6.0 / 12.0,
    "DGS1": 1.0,
    "DGS2": 2.0,
}


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


@dataclass(frozen=True, slots=True)
class FredTreasuryRateCurveResolution:
    """Resolved rate from a FRED Treasury zero-rate proxy curve."""

    rate: float
    time_to_expiry_years: float | None
    curve_points: pd.DataFrame
    selected_rate_fallback_used: bool
    warnings: tuple[str, ...]

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "rate_policy": RATE_POLICY_FRED_TREASURY_ZERO_PROXY_LINEAR_CC,
            "rate_curve_source": RATE_CURVE_SOURCE_FRED,
            "rate_curve_series_ids": [
                str(series_id) for series_id in self.curve_points["series_id"].tolist()
            ],
            "rate_interpolation": RATE_INTERPOLATION_LINEAR,
            "rate_compounding": RATE_COMPOUNDING_CONTINUOUS,
            "rate_extrapolation": RATE_EXTRAPOLATION_CLAMP_WITH_WARNING,
            "rate_is_bootstrapped": False,
            "selected_rate_fallback_used": bool(self.selected_rate_fallback_used),
            "rate_warnings": list(self.warnings),
        }


def fred_percent_to_continuous_decimal(value_percent: float) -> float:
    """Convert a FRED annualized percent quote to a continuous decimal rate."""

    return math.log1p(value_percent / 100.0)


def build_fred_treasury_zero_proxy_curve(
    frames_by_series_id: Mapping[str, pd.DataFrame],
    *,
    asof: str | pd.Timestamp,
) -> pd.DataFrame:
    """Build the FRED Treasury zero-rate proxy curve used by snapshots."""

    asof_timestamp = _asof_timestamp(asof)
    rows: list[dict[str, object]] = []
    for series_id, tenor_years in FRED_TREASURY_TENOR_YEARS.items():
        frame = frames_by_series_id.get(series_id)
        if frame is None:
            continue
        try:
            selection = select_latest_fred_rate_at_or_before_asof(
                frame,
                series_id=series_id,
                asof=asof_timestamp,
            )
        except FredRateUnavailableError:
            continue
        decimal_rate = float(selection.value_percent) / 100.0
        rows.append(
            {
                "tenor_years": float(tenor_years),
                "series_id": selection.series_id,
                "observation_date": selection.observation_date,
                "raw_percent_rate": float(selection.value_percent),
                "decimal_rate": decimal_rate,
                "continuous_rate": math.log1p(decimal_rate),
                "source": selection.source,
                "asof": selection.asof,
            }
        )
    return _coerce_rate_curve_frame(pd.DataFrame(rows))


def resolve_fred_treasury_zero_proxy_rate(
    curve_points: pd.DataFrame,
    *,
    time_to_expiry_years: float | None,
    selected_rate_fallback: float | None,
) -> FredTreasuryRateCurveResolution:
    """Interpolate a continuous rate from a FRED Treasury proxy curve."""

    curve = _coerce_rate_curve_frame(curve_points)
    warnings: list[str] = []
    valid = curve.dropna(subset=["tenor_years", "continuous_rate"]).sort_values(
        "tenor_years",
        kind="mergesort",
    )

    if len(valid) < 2:
        if selected_rate_fallback is None:
            raise FredRateUnavailableError(
                "FRED Treasury proxy curve requires at least two valid points "
                "and no selected flat-rate fallback was provided"
            )
        warnings.append(
            "rate_curve_fallback: fewer_than_two_valid_curve_points; "
            "selected_flat_rate_used"
        )
        return FredTreasuryRateCurveResolution(
            rate=float(selected_rate_fallback),
            time_to_expiry_years=time_to_expiry_years,
            curve_points=curve,
            selected_rate_fallback_used=True,
            warnings=tuple(warnings),
        )

    if time_to_expiry_years is None or not math.isfinite(float(time_to_expiry_years)):
        if selected_rate_fallback is None:
            raise FredRateUnavailableError(
                "time_to_expiry_years is required when no selected flat-rate "
                "fallback is provided"
            )
        warnings.append(
            "rate_curve_fallback: missing_time_to_expiry_years; "
            "selected_flat_rate_used"
        )
        return FredTreasuryRateCurveResolution(
            rate=float(selected_rate_fallback),
            time_to_expiry_years=time_to_expiry_years,
            curve_points=curve,
            selected_rate_fallback_used=True,
            warnings=tuple(warnings),
        )

    target = float(time_to_expiry_years)
    tenors = valid["tenor_years"].astype(float).to_list()
    rates = valid["continuous_rate"].astype(float).to_list()
    if target <= tenors[0]:
        if target < tenors[0]:
            warnings.append(
                "rate_curve_clamped: expiry_shorter_than_shortest_tenor; "
                f"time_to_expiry_years={target:.12g}, tenor_years={tenors[0]:.12g}"
            )
        rate = rates[0]
    elif target >= tenors[-1]:
        if target > tenors[-1]:
            warnings.append(
                "rate_curve_clamped: expiry_longer_than_longest_tenor; "
                f"time_to_expiry_years={target:.12g}, tenor_years={tenors[-1]:.12g}"
            )
        rate = rates[-1]
    else:
        rate = _linear_interpolate(tenors, rates, target)

    return FredTreasuryRateCurveResolution(
        rate=float(rate),
        time_to_expiry_years=target,
        curve_points=curve,
        selected_rate_fallback_used=False,
        warnings=tuple(warnings),
    )


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


def representative_time_to_expiry_years(
    expiries: Sequence[object],
    *,
    asof: str | pd.Timestamp,
) -> float | None:
    """Return a stable representative option expiry for a flat MarketData rate."""

    asof_timestamp = _asof_timestamp(asof)
    values: list[float] = []
    for expiry in expiries:
        if bool(pd.isna(cast(Any, expiry))):
            continue
        expiry_timestamp = pd.Timestamp(cast(Any, expiry))
        if expiry_timestamp.tzinfo is None:
            expiry_timestamp = expiry_timestamp.tz_localize("UTC")
        else:
            expiry_timestamp = expiry_timestamp.tz_convert("UTC")
        years = (expiry_timestamp - asof_timestamp).total_seconds() / (365 * 24 * 3600)
        if math.isfinite(years) and years > 0.0:
            values.append(float(years))
    if not values:
        return None
    values.sort()
    midpoint = len(values) // 2
    if len(values) % 2:
        return values[midpoint]
    return (values[midpoint - 1] + values[midpoint]) / 2.0


def _coerce_rate_curve_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "tenor_years",
        "series_id",
        "observation_date",
        "raw_percent_rate",
        "decimal_rate",
        "continuous_rate",
        "source",
        "asof",
    )
    if frame.empty:
        frame = pd.DataFrame(columns=list(columns))
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"rate curve frame is missing columns: {missing}")
    out = frame.loc[:, list(columns)].copy()
    out["tenor_years"] = pd.to_numeric(
        out["tenor_years"],
        errors="coerce",
    ).astype("Float64")
    out["series_id"] = out["series_id"].astype("string")
    out["observation_date"] = pd.to_datetime(
        out["observation_date"],
        errors="coerce",
    )
    for column in ("raw_percent_rate", "decimal_rate", "continuous_rate"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Float64")
    out["source"] = out["source"].astype("string")
    out["asof"] = pd.to_datetime(out["asof"], errors="coerce", utc=True)
    return out.sort_values("tenor_years", kind="mergesort").reset_index(drop=True)


def _linear_interpolate(
    x_values: Sequence[float],
    y_values: Sequence[float],
    target: float,
) -> float:
    for index in range(1, len(x_values)):
        left_x = float(x_values[index - 1])
        right_x = float(x_values[index])
        if left_x <= target <= right_x:
            left_y = float(y_values[index - 1])
            right_y = float(y_values[index])
            weight = (target - left_x) / (right_x - left_x)
            return left_y + weight * (right_y - left_y)
    return float(y_values[-1])


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
    "FRED_TREASURY_TENOR_YEARS",
    "FredRateSelection",
    "FredTreasuryRateCurveResolution",
    "RATE_COMPOUNDING_CONTINUOUS",
    "RATE_CURVE_SOURCE_FRED",
    "RATE_EXTRAPOLATION_CLAMP_WITH_WARNING",
    "RATE_INTERPOLATION_LINEAR",
    "RATE_POLICY_FRED_TREASURY_ZERO_PROXY_LINEAR_CC",
    "build_fred_treasury_zero_proxy_curve",
    "fred_percent_to_continuous_decimal",
    "representative_time_to_expiry_years",
    "resolve_fred_treasury_zero_proxy_rate",
    "select_latest_fred_rate_at_or_before_asof",
]
