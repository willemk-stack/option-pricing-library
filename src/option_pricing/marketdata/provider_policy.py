from __future__ import annotations

from collections.abc import Sequence
from typing import Any

DEFAULT_RATE_SERIES_ID = "DGS3MO"
DEFAULT_SNAPSHOT_RATE_LOOKBACK_DAYS = 90
DEFAULT_RATE_CURVE_SERIES_IDS = (
    "DGS1MO",
    "DGS3MO",
    "DGS6MO",
    "DGS1",
    "DGS2",
)
PROVIDER_RATE_CURVE_COLUMNS = (
    "series_id",
    "tenor",
    "observation_date",
    "value_percent",
    "continuous_decimal",
    "source",
    "asof",
    "day_count",
)
RATE_CURVE_TENORS = {
    "DGS1MO": "1M",
    "DGS3MO": "3M",
    "DGS6MO": "6M",
    "DGS1": "1Y",
    "DGS2": "2Y",
}
DEFAULT_BARS_TIMEFRAME = "1Day"
DEFAULT_DAY_COUNT = "ACT/365"
_DIVIDEND_ASSUMPTION_WARNING = (
    "documented_assumption: dividend_yield=0.0, "
    "dividend_yield_source=assumption, dividend_inference=not_enabled"
)
_NO_OPTION_CHAIN_BACKFILL_WARNING = (
    "current_provider_scope: option_chain_backfill=not_enabled"
)
_NO_SCHEDULING_WARNING = "current_provider_scope: scheduling=not_enabled"
_FRED_BACKFILL_WARNING = (
    "current_provider_scope: fred_backfill_storage=single_series_observations, "
    "curve_interpolation=not_enabled"
)
_BARS_BACKFILL_WARNING = (
    "current_provider_scope: bars_backfill=equity_only, "
    "option_chain_backfill=not_enabled, scheduling=not_enabled"
)


def _provider_snapshot_warnings(
    *,
    cleaning_warnings: Sequence[str],
    dropped_before_cleaning_count: int,
    raw_option_contract_count: int,
    normalized_option_contract_count: int,
    provider_rejected_contract_count: int,
    rate_series_id: str,
    dividend_yield: float,
    dividend_yield_source: str,
) -> tuple[str, ...]:
    warnings = [
        *_snapshot_assumption_warnings(
            rate_series_id=rate_series_id,
            dividend_yield=dividend_yield,
            dividend_yield_source=dividend_yield_source,
        ),
        str(_NO_OPTION_CHAIN_BACKFILL_WARNING),
        str(_NO_SCHEDULING_WARNING),
        *(str(warning) for warning in cleaning_warnings),
    ]
    if dropped_before_cleaning_count > 0:
        warnings.append(
            "alpaca_option_contracts_dropped_before_cleaning: "
            f"dropped={dropped_before_cleaning_count}, "
            f"raw={raw_option_contract_count}, "
            f"normalized={normalized_option_contract_count}, "
            f"provider_rejected_contracts={provider_rejected_contract_count}, "
            "stage=provider_normalization"
        )
    return tuple(warnings)


def _snapshot_assumption_warnings(
    *,
    rate_series_id: str,
    dividend_yield: float,
    dividend_yield_source: str,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if dividend_yield == 0.0 and dividend_yield_source.strip().lower() == "assumption":
        warnings.append(_DIVIDEND_ASSUMPTION_WARNING)
    warnings.append(
        "documented_assumption: "
        f"rate_series_id={rate_series_id}, "
        f"default_rate_series_id={DEFAULT_RATE_SERIES_ID}, "
        "curve_interpolation=not_enabled"
    )
    return tuple(warnings)


def _current_provider_scope() -> dict[str, str]:
    return {
        "curve_interpolation": "not_enabled",
        "dividend_inference": "not_enabled",
        "option_chain_backfill": "not_enabled",
        "scheduling": "not_enabled",
    }


def _merge_unique_warnings(warnings: Sequence[str] | Any) -> tuple[str, ...]:
    seen: set[str] = set()
    merged: list[str] = []
    for warning in warnings:
        cleaned_warning = str(warning)
        if cleaned_warning in seen:
            continue
        seen.add(cleaned_warning)
        merged.append(cleaned_warning)
    return tuple(merged)


__all__ = [
    "DEFAULT_BARS_TIMEFRAME",
    "DEFAULT_DAY_COUNT",
    "DEFAULT_RATE_CURVE_SERIES_IDS",
    "DEFAULT_RATE_SERIES_ID",
    "DEFAULT_SNAPSHOT_RATE_LOOKBACK_DAYS",
    "PROVIDER_RATE_CURVE_COLUMNS",
    "RATE_CURVE_TENORS",
]
