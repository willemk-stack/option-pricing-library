from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from option_pricing.marketdata.cleaning import (
    MODEL_VALIDATION_POLICY_MODEL_READY_QUOTES_V1,
    OPTION_CLEANING_POLICY_STAGED_RECOVERABLE_QUOTES_V1,
    QuoteCleaningResult,
    QuoteRejectionReason,
)
from option_pricing.marketdata.config import (
    DIVIDEND_POLICY_IMPLIED_CARRY,
    DIVIDEND_POLICY_MANUAL_STATIC,
    DIVIDEND_POLICY_PROVIDER_TRAILING_YIELD,
    DIVIDEND_POLICY_ZERO_ASSUMPTION,
    MarketDataPolicyConfig,
)
from option_pricing.marketdata.rates import (
    RATE_COMPOUNDING_CONTINUOUS,
    RATE_CURVE_SOURCE_FRED,
    RATE_EXTRAPOLATION_CLAMP_WITH_WARNING,
    RATE_INTERPOLATION_LINEAR,
    RATE_POLICY_FRED_TREASURY_ZERO_PROXY_LINEAR_CC,
)
from option_pricing.marketdata.schemas import (
    CLEANED_QUOTES_COLUMNS,
    REJECTED_QUOTES_COLUMNS,
    DatasetName,
)
from option_pricing.marketdata.validation import (
    coerce_frame,
    order_columns,
    validate_dtypes,
)

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
    "tenor_years",
    "series_id",
    "observation_date",
    "raw_percent_rate",
    "decimal_rate",
    "continuous_rate",
    "source",
    "asof",
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
DATA_POLICY_SCHEMA_VERSION = "provider_snapshot_data_policy.v1"
OPTION_CLEANING_POLICY_QUOTE_CLEANING_V1 = "quote_cleaning_v1"
OPTION_CLEANING_POLICY_ID = "quote_cleaning_policy.v1"
QUOTE_FRESHNESS_MODE_DEMO_LENIENT = "demo_lenient"
QUOTE_FRESHNESS_MODE_END_OF_DAY = "end_of_day"
QUOTE_FRESHNESS_MODE_INTRADAY_STRICT = "intraday_strict"
STALE_QUOTE_ACTION_WARN = "warn"
STALE_QUOTE_ACTION_REJECT = "reject"
STALE_QUOTE_ACTION_FAIL = "fail"
SUPPORTED_QUOTE_FRESHNESS_MODES = frozenset(
    {
        QUOTE_FRESHNESS_MODE_DEMO_LENIENT,
        QUOTE_FRESHNESS_MODE_END_OF_DAY,
        QUOTE_FRESHNESS_MODE_INTRADAY_STRICT,
    }
)
SUPPORTED_STALE_QUOTE_ACTIONS = frozenset(
    {STALE_QUOTE_ACTION_WARN, STALE_QUOTE_ACTION_REJECT, STALE_QUOTE_ACTION_FAIL}
)
_DIVIDEND_ASSUMPTION_WARNING = (
    "documented_assumption: dividend_yield=0.0, "
    "dividend_policy=zero_assumption, dividend_inference=not_enabled"
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
_SPOT_CHAIN_PROXY_MIN_MID_SPOT_RATIO = 0.05
_SPOT_CHAIN_PROXY_MIN_MID_STRIKE_RATIO = 0.02
_SPOT_CHAIN_MATERIAL_DEVIATION_RATIO = 0.25


@dataclass(frozen=True, slots=True)
class ProviderSnapshotQualityPolicy:
    """Freshness and minimum-shape checks for provider-backed snapshots."""

    quote_freshness_mode: str = QUOTE_FRESHNESS_MODE_DEMO_LENIENT
    max_quote_age_seconds: float | None = None
    allow_prior_session: bool = False
    stale_quote_action: str = STALE_QUOTE_ACTION_WARN
    max_equity_quote_age_seconds: float | None = None
    max_option_quote_age_seconds: float | None = None
    warn_on_stale_quotes: bool = True
    reject_stale_option_quotes: bool = False
    reject_option_quotes_after_asof: bool = False
    reject_stale_equity_quote: bool = False
    require_option_quotes_on_or_before_asof: bool = True
    require_equity_quote_on_or_before_asof: bool = True
    min_accepted_contracts: int = 1
    min_accepted_calls: int | None = None
    min_accepted_puts: int | None = None
    min_expiries: int | None = None

    def __post_init__(self) -> None:
        _validate_choice(
            self.quote_freshness_mode,
            SUPPORTED_QUOTE_FRESHNESS_MODES,
            "quote_freshness_mode",
        )
        _validate_optional_nonnegative_number(
            self.max_quote_age_seconds,
            "max_quote_age_seconds",
        )
        _validate_bool(self.allow_prior_session, "allow_prior_session")
        _validate_choice(
            self.stale_quote_action,
            SUPPORTED_STALE_QUOTE_ACTIONS,
            "stale_quote_action",
        )
        _validate_optional_nonnegative_number(
            self.max_equity_quote_age_seconds,
            "max_equity_quote_age_seconds",
        )
        _validate_optional_nonnegative_number(
            self.max_option_quote_age_seconds,
            "max_option_quote_age_seconds",
        )
        _validate_bool(self.warn_on_stale_quotes, "warn_on_stale_quotes")
        _validate_bool(self.reject_stale_option_quotes, "reject_stale_option_quotes")
        _validate_bool(
            self.reject_option_quotes_after_asof,
            "reject_option_quotes_after_asof",
        )
        _validate_bool(self.reject_stale_equity_quote, "reject_stale_equity_quote")
        _validate_bool(
            self.require_option_quotes_on_or_before_asof,
            "require_option_quotes_on_or_before_asof",
        )
        _validate_bool(
            self.require_equity_quote_on_or_before_asof,
            "require_equity_quote_on_or_before_asof",
        )
        _validate_nonnegative_int(
            self.min_accepted_contracts,
            "min_accepted_contracts",
        )
        _validate_optional_nonnegative_int(
            self.min_accepted_calls,
            "min_accepted_calls",
        )
        _validate_optional_nonnegative_int(
            self.min_accepted_puts,
            "min_accepted_puts",
        )
        _validate_optional_nonnegative_int(self.min_expiries, "min_expiries")

    def as_dict(self) -> dict[str, object]:
        return {
            "quote_freshness_mode": self.quote_freshness_mode,
            "max_quote_age_seconds": self.max_quote_age_seconds,
            "allow_prior_session": self.allow_prior_session,
            "stale_quote_action": self.stale_quote_action,
            "max_equity_quote_age_seconds": self.max_equity_quote_age_seconds,
            "max_option_quote_age_seconds": self.max_option_quote_age_seconds,
            "warn_on_stale_quotes": self.warn_on_stale_quotes,
            "reject_stale_option_quotes": self.reject_stale_option_quotes,
            "reject_option_quotes_after_asof": (self.reject_option_quotes_after_asof),
            "reject_stale_equity_quote": self.reject_stale_equity_quote,
            "require_option_quotes_on_or_before_asof": (
                self.require_option_quotes_on_or_before_asof
            ),
            "require_equity_quote_on_or_before_asof": (
                self.require_equity_quote_on_or_before_asof
            ),
            "min_accepted_contracts": self.min_accepted_contracts,
            "min_accepted_calls": self.min_accepted_calls,
            "min_accepted_puts": self.min_accepted_puts,
            "min_expiries": self.min_expiries,
        }

    def effective_max_equity_quote_age_seconds(self) -> float | None:
        return (
            self.max_equity_quote_age_seconds
            if self.max_equity_quote_age_seconds is not None
            else self.max_quote_age_seconds
        )

    def effective_max_option_quote_age_seconds(self) -> float | None:
        return (
            self.max_option_quote_age_seconds
            if self.max_option_quote_age_seconds is not None
            else self.max_quote_age_seconds
        )

    def rejects_stale_option_quotes(self) -> bool:
        if self.reject_stale_option_quotes:
            return True
        return (
            self.quote_freshness_mode == QUOTE_FRESHNESS_MODE_INTRADAY_STRICT
            and self.stale_quote_action != STALE_QUOTE_ACTION_FAIL
        )

    def fails_stale_option_quotes(self) -> bool:
        return (
            self.quote_freshness_mode == QUOTE_FRESHNESS_MODE_INTRADAY_STRICT
            and self.stale_quote_action == STALE_QUOTE_ACTION_FAIL
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
    dividend_source = dividend_yield_source.strip().lower()
    if dividend_yield == 0.0 and dividend_source in {"assumption", "zero_assumption"}:
        warnings.append(_DIVIDEND_ASSUMPTION_WARNING)
    warnings.append(
        "documented_assumption: "
        f"rate_series_id={rate_series_id}, "
        f"default_rate_series_id={DEFAULT_RATE_SERIES_ID}, "
        "rate_curve=FRED Treasury zero-rate proxy, "
        "interpolation=linear, compounding=continuous, "
        "rate_is_bootstrapped=false"
    )
    return tuple(warnings)


def _current_provider_scope() -> dict[str, str]:
    return {
        "curve_interpolation": RATE_INTERPOLATION_LINEAR,
        "dividend_inference": "manual_static_or_zero_assumption",
        "option_chain_backfill": "not_enabled",
        "scheduling": "not_enabled",
    }


def _provider_snapshot_rate_policy(
    *,
    rate_series_id: str,
    rate_source: str,
    rate_observation_date: pd.Timestamp,
    selected_rate: float,
    lookback_days: int,
    curve_series_ids: Sequence[str],
    flat_rate: float | None = None,
    rate_warnings: Sequence[str] = (),
    selected_rate_fallback_used: bool = False,
) -> dict[str, object]:
    observation_date = pd.Timestamp(rate_observation_date).date().isoformat()
    return {
        "policy": RATE_POLICY_FRED_TREASURY_ZERO_PROXY_LINEAR_CC,
        "provider": "fred",
        "rate_curve_source": RATE_CURVE_SOURCE_FRED,
        "series_id": str(rate_series_id),
        "rate_source": str(rate_source),
        "rate_observation_date": observation_date,
        "selected_rate": float(selected_rate),
        "flat_rate": float(selected_rate if flat_rate is None else flat_rate),
        "lookback_days": int(lookback_days),
        "curve_series_ids": [str(series_id) for series_id in curve_series_ids],
        "rate_curve_series_ids": [str(series_id) for series_id in curve_series_ids],
        "rate_interpolation": RATE_INTERPOLATION_LINEAR,
        "curve_interpolation": RATE_INTERPOLATION_LINEAR,
        "rate_compounding": RATE_COMPOUNDING_CONTINUOUS,
        "rate_extrapolation": RATE_EXTRAPOLATION_CLAMP_WITH_WARNING,
        "rate_is_bootstrapped": False,
        "selected_rate_fallback_used": bool(selected_rate_fallback_used),
        "rate_warnings": [str(warning) for warning in rate_warnings],
    }


def _provider_snapshot_dividend_policy(
    *,
    dividend_yield: float,
    dividend_yield_source: str,
    dividend_note: str | None = None,
    dividend_is_explicit: bool | None = None,
    dividend_fallback_used: bool | None = None,
) -> dict[str, object]:
    policy = (
        DIVIDEND_POLICY_ZERO_ASSUMPTION
        if float(dividend_yield) == 0.0
        and str(dividend_yield_source).strip().lower()
        in {"assumption", "zero_assumption"}
        else DIVIDEND_POLICY_MANUAL_STATIC
    )
    return {
        "policy": policy,
        "dividend_yield": float(dividend_yield),
        "source": str(dividend_yield_source),
        "dividend_source": str(dividend_yield_source),
        "dividend_is_explicit": (
            bool(dividend_is_explicit)
            if dividend_is_explicit is not None
            else policy == DIVIDEND_POLICY_MANUAL_STATIC
        ),
        "dividend_fallback_used": (
            bool(dividend_fallback_used)
            if dividend_fallback_used is not None
            else policy == DIVIDEND_POLICY_ZERO_ASSUMPTION
        ),
        "dividend_inference": "not_enabled",
        **({} if dividend_note is None else {"dividend_note": str(dividend_note)}),
    }


def _resolve_provider_snapshot_dividend_policy(
    *,
    underlying: str,
    policy_config: MarketDataPolicyConfig,
    dividend_yield: float,
    dividend_yield_source: str,
) -> dict[str, object]:
    override_source = str(dividend_yield_source).strip()
    if float(dividend_yield) != 0.0 or override_source.lower() not in {
        "assumption",
        "zero_assumption",
    }:
        return _provider_snapshot_dividend_policy(
            dividend_yield=float(dividend_yield),
            dividend_yield_source=override_source,
            dividend_is_explicit=True,
            dividend_fallback_used=False,
        )

    symbol = str(underlying).strip().upper()
    static = policy_config.dividends.static_yields.get(symbol)
    if static is not None:
        if static.source == DIVIDEND_POLICY_PROVIDER_TRAILING_YIELD:
            raise NotImplementedError(
                "provider_trailing_yield is recognized as a future dividend "
                "policy but is not implemented"
            )
        if static.source == DIVIDEND_POLICY_IMPLIED_CARRY:
            raise NotImplementedError(
                "implied_carry is recognized as a future research/diagnostic "
                "dividend policy but is not implemented"
            )
        return _provider_snapshot_dividend_policy(
            dividend_yield=float(static.dividend_yield),
            dividend_yield_source=static.source,
            dividend_note=static.note,
            dividend_is_explicit=True,
            dividend_fallback_used=False,
        )

    default_policy = policy_config.dividends.default_policy
    if default_policy == DIVIDEND_POLICY_ZERO_ASSUMPTION:
        return _provider_snapshot_dividend_policy(
            dividend_yield=0.0,
            dividend_yield_source=DIVIDEND_POLICY_ZERO_ASSUMPTION,
            dividend_is_explicit=False,
            dividend_fallback_used=True,
        )
    if default_policy == DIVIDEND_POLICY_PROVIDER_TRAILING_YIELD:
        raise NotImplementedError(
            "provider_trailing_yield is recognized as a future dividend policy "
            "but is not implemented"
        )
    if default_policy == DIVIDEND_POLICY_IMPLIED_CARRY:
        raise NotImplementedError(
            "implied_carry is recognized as a future research/diagnostic artifact "
            "and is not implemented"
        )
    raise ValueError(f"Unsupported dividend policy {default_policy!r}")


def _provider_snapshot_option_cleaning_policy() -> dict[str, object]:
    return {
        "policy": OPTION_CLEANING_POLICY_STAGED_RECOVERABLE_QUOTES_V1,
        "legacy_policy": OPTION_CLEANING_POLICY_QUOTE_CLEANING_V1,
        "policy_id": OPTION_CLEANING_POLICY_ID,
        "raw_option_quotes_layer": "raw_option_quotes",
        "clean_option_quotes_layer": "clean_option_quotes",
        "model_validation_quotes_layer": "model_validation_quotes",
        "rejected_quotes_preserved": True,
        "reason_codes": [reason.value for reason in QuoteRejectionReason],
        "recoverable_missing_fields": [
            "mid",
            "time_to_expiry",
            "moneyness",
            "log_moneyness",
            "forward_moneyness",
            "iv",
            "delta",
            "gamma",
            "theta",
            "vega",
            "rho",
        ],
    }


def _provider_snapshot_data_policy(
    *,
    equity_provider: str,
    equity_feed: str,
    option_provider: str,
    option_feed: str,
    rate_policy: Mapping[str, object],
    dividend_policy: Mapping[str, object],
    option_cleaning_policy: Mapping[str, object],
    quote_freshness_mode: str = QUOTE_FRESHNESS_MODE_DEMO_LENIENT,
    model_validation_policy: str = MODEL_VALIDATION_POLICY_MODEL_READY_QUOTES_V1,
) -> dict[str, object]:
    return {
        "schema_version": DATA_POLICY_SCHEMA_VERSION,
        "equity_provider": str(equity_provider),
        "equity_feed": str(equity_feed),
        "option_provider": str(option_provider),
        "option_feed": str(option_feed),
        "rate_policy": dict(rate_policy),
        "dividend_policy": dict(dividend_policy),
        "quote_freshness_mode": str(quote_freshness_mode),
        "option_cleaning_policy": dict(option_cleaning_policy),
        "model_validation_policy": str(model_validation_policy),
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


def _coerce_provider_snapshot_quality_policy(
    policy: ProviderSnapshotQualityPolicy | Mapping[str, object] | None,
) -> ProviderSnapshotQualityPolicy:
    if policy is None:
        return ProviderSnapshotQualityPolicy()
    if isinstance(policy, ProviderSnapshotQualityPolicy):
        return policy
    if not isinstance(policy, Mapping):
        raise TypeError(
            "quality_policy must be a ProviderSnapshotQualityPolicy, mapping, or None"
        )

    allowed = set(ProviderSnapshotQualityPolicy.__dataclass_fields__)
    unknown = sorted(str(key) for key in policy if str(key) not in allowed)
    if unknown:
        raise ValueError(f"Unknown provider snapshot quality policy fields: {unknown}")
    kwargs = {str(key): value for key, value in policy.items()}
    return ProviderSnapshotQualityPolicy(**cast(Any, kwargs))


def _provider_snapshot_freshness_stats(
    *,
    equity_quotes: pd.DataFrame,
    option_chain: pd.DataFrame,
    cleaned_quotes: pd.DataFrame,
    asof: pd.Timestamp,
    policy: ProviderSnapshotQualityPolicy,
) -> dict[str, object]:
    asof_utc = _utc_timestamp(asof)
    equity_ages = _quote_age_seconds(equity_quotes, asof_utc)
    option_ages = _quote_age_seconds(option_chain, asof_utc)
    accepted_ages = _quote_age_seconds(cleaned_quotes, asof_utc)
    max_equity_age = policy.effective_max_equity_quote_age_seconds()
    max_option_age = policy.effective_max_option_quote_age_seconds()

    equity_age = _last_float(equity_ages)
    option_summary = _age_summary(option_ages)
    accepted_calls = _right_count(cleaned_quotes, "call")
    accepted_puts = _right_count(cleaned_quotes, "put")
    accepted_expiries = (
        int(cleaned_quotes["expiry"].nunique()) if "expiry" in cleaned_quotes else 0
    )
    accepted_expiry_years = _numeric_summary(cleaned_quotes, "expiry_years")
    accepted_strikes = _numeric_summary(cleaned_quotes, "strike")

    stale_option_count = _count_stale(option_ages, max_option_age)
    stale_accepted_count = _count_stale(accepted_ages, max_option_age)
    warnings = _quote_freshness_warning_codes(
        policy=policy,
        equity_age=equity_age,
        stale_equity_quote=bool(
            equity_age is not None
            and max_equity_age is not None
            and equity_age > max_equity_age
        ),
        option_quotes_after_asof_count=_count_after_asof(option_ages),
        stale_option_quote_count=stale_option_count,
        stale_accepted_quote_count=stale_accepted_count,
    )

    return {
        "quote_freshness_mode": policy.quote_freshness_mode,
        "max_quote_age_seconds": max_option_age,
        "allow_prior_session": policy.allow_prior_session,
        "stale_quote_action": policy.stale_quote_action,
        "equity_quote_age_seconds": equity_age,
        "equity_quote_after_asof": bool(
            equity_age is not None
            and policy.require_equity_quote_on_or_before_asof
            and equity_age < 0.0
        ),
        "stale_equity_quote": bool(
            equity_age is not None
            and max_equity_age is not None
            and equity_age > max_equity_age
        ),
        "option_quote_age_seconds_min": option_summary["min"],
        "option_quote_age_seconds_median": option_summary["median"],
        "option_quote_age_seconds_max": option_summary["max"],
        "quote_age_summary": dict(option_summary),
        "option_quote_count": int(len(option_chain)),
        "option_quotes_after_asof_count": _count_after_asof(option_ages),
        "stale_quote_count": stale_option_count,
        "stale_option_quote_count": stale_option_count,
        "accepted_quote_count": int(len(cleaned_quotes)),
        "accepted_call_count": accepted_calls,
        "accepted_put_count": accepted_puts,
        "accepted_expiry_count": accepted_expiries,
        "accepted_expiry_years_min": accepted_expiry_years["min"],
        "accepted_expiry_years_max": accepted_expiry_years["max"],
        "accepted_expiry_days_min": _years_to_days(accepted_expiry_years["min"]),
        "accepted_expiry_days_max": _years_to_days(accepted_expiry_years["max"]),
        "accepted_strike_min": accepted_strikes["min"],
        "accepted_strike_max": accepted_strikes["max"],
        "stale_accepted_quote_count": stale_accepted_count,
        "accepted_quotes_after_asof_count": _count_after_asof(accepted_ages),
        "quote_freshness_warnings": list(warnings),
    }


def _provider_snapshot_spot_option_chain_diagnostic(
    *,
    market_inputs: pd.DataFrame,
    cleaned_quotes: pd.DataFrame,
) -> dict[str, object]:
    spot = _market_inputs_spot(market_inputs)
    call_proxies: list[float] = []
    put_proxies: list[float] = []
    if spot is not None and not cleaned_quotes.empty:
        for _, row in cleaned_quotes.iterrows():
            right = _optional_row_text(row, "right").lower()
            strike = _optional_row_float(row, "strike")
            mid = _optional_row_float(row, "mid")
            if (
                right not in {"call", "put"}
                or strike is None
                or mid is None
                or strike <= 0.0
                or mid <= 0.0
            ):
                continue
            if not _is_deep_itm_proxy_candidate(spot=spot, strike=strike, mid=mid):
                continue
            if right == "call":
                call_proxies.append(strike + mid)
            elif strike > mid:
                put_proxies.append(strike - mid)

    proxies = [*call_proxies, *put_proxies]
    summary = _proxy_summary(proxies)
    ratio = (
        None if spot is None or summary["median"] is None else summary["median"] / spot
    )
    relative_deviation = None if ratio is None else abs(ratio - 1.0)
    threshold = _SPOT_CHAIN_MATERIAL_DEVIATION_RATIO
    failed = bool(relative_deviation is not None and relative_deviation > threshold)
    status = (
        "failed"
        if failed
        else "passed" if proxies and spot is not None else "insufficient_data"
    )

    return {
        "check": QuoteRejectionReason.SPOT_OPTION_CHAIN_MISMATCH.value,
        "status": status,
        "spot": spot,
        "proxy_count": int(len(proxies)),
        "call_proxy_count": int(len(call_proxies)),
        "put_proxy_count": int(len(put_proxies)),
        "proxy_min": summary["min"],
        "proxy_median": summary["median"],
        "proxy_max": summary["max"],
        "median_proxy_to_spot_ratio": ratio,
        "relative_deviation": relative_deviation,
        "material_deviation_threshold": threshold,
        "deep_itm_proxy_min_mid_spot_ratio": _SPOT_CHAIN_PROXY_MIN_MID_SPOT_RATIO,
        "deep_itm_proxy_min_mid_strike_ratio": (_SPOT_CHAIN_PROXY_MIN_MID_STRIKE_RATIO),
    }


def _provider_snapshot_quality_warnings(
    *,
    stats: Mapping[str, object],
    policy: ProviderSnapshotQualityPolicy,
) -> tuple[str, ...]:
    if not policy.warn_on_stale_quotes:
        return ()

    warnings: list[str] = []
    max_equity_age = policy.effective_max_equity_quote_age_seconds()
    max_option_age = policy.effective_max_option_quote_age_seconds()
    if bool(stats.get("equity_quote_after_asof")):
        warnings.append(
            "provider_quality: equity_quote_after_asof "
            f"age_seconds={stats.get('equity_quote_age_seconds')}, "
            "require_equity_quote_on_or_before_asof=True"
        )
    if bool(stats.get("stale_equity_quote")):
        warnings.append(
            "provider_quality: stale_equity_quote "
            f"age_seconds={stats.get('equity_quote_age_seconds')}, "
            f"max_age_seconds={max_equity_age}"
        )

    option_after_asof_count = _int_stat(stats, "option_quotes_after_asof_count")
    if policy.require_option_quotes_on_or_before_asof and option_after_asof_count > 0:
        warnings.append(
            "provider_quality: option_quotes_after_asof "
            f"count={option_after_asof_count}, "
            "require_option_quotes_on_or_before_asof=True"
        )

    stale_option_count = _int_stat(stats, "stale_option_quote_count")
    if stale_option_count > 0:
        warnings.append(
            "provider_quality: stale_option_quotes "
            f"count={stale_option_count}, "
            f"max_age_seconds={max_option_age}, "
            f"quote_freshness_mode={policy.quote_freshness_mode}, "
            f"stale_quote_action={policy.stale_quote_action}"
        )

    stale_accepted_count = _int_stat(stats, "stale_accepted_quote_count")
    if stale_accepted_count > 0:
        warnings.append(
            "provider_quality: stale_accepted_option_quotes "
            f"count={stale_accepted_count}, "
            f"max_age_seconds={max_option_age}, "
            f"quote_freshness_mode={policy.quote_freshness_mode}, "
            f"stale_quote_action={policy.stale_quote_action}"
        )
    spot_chain = _spot_chain_diagnostic(stats)
    if spot_chain.get("status") == "failed":
        warnings.append(
            "provider_quality: spot_option_chain_mismatch "
            f"median_proxy_to_spot_ratio="
            f"{spot_chain.get('median_proxy_to_spot_ratio')}, "
            f"relative_deviation={spot_chain.get('relative_deviation')}, "
            f"threshold={spot_chain.get('material_deviation_threshold')}"
        )
    return tuple(warnings)


def _provider_snapshot_quality_failures(
    *,
    stats: Mapping[str, object],
    policy: ProviderSnapshotQualityPolicy,
) -> tuple[str, ...]:
    failures: list[str] = []
    if policy.reject_stale_equity_quote and bool(stats.get("equity_quote_after_asof")):
        failures.append(
            "Equity quote timestamp is after the snapshot asof and "
            "reject_stale_equity_quote=True"
        )
    if policy.reject_stale_equity_quote and bool(stats.get("stale_equity_quote")):
        failures.append(
            "Equity quote is stale under the provider snapshot quality policy "
            f"(age_seconds={stats.get('equity_quote_age_seconds')}, "
            f"max_age_seconds={policy.effective_max_equity_quote_age_seconds()})"
        )
    if (
        policy.fails_stale_option_quotes()
        and _int_stat(stats, "stale_option_quote_count") > 0
    ):
        failures.append(
            "Option quote is stale under intraday_strict freshness policy "
            f"(stale_quote_count={_int_stat(stats, 'stale_option_quote_count')}, "
            f"max_age_seconds={policy.effective_max_option_quote_age_seconds()})"
        )

    spot_chain = _spot_chain_diagnostic(stats)
    if spot_chain.get("status") == "failed":
        failures.append(
            "Provider snapshot failed spot_option_chain_mismatch diagnostic "
            f"(spot={spot_chain.get('spot')}, "
            f"proxy_count={spot_chain.get('proxy_count')}, "
            f"median_proxy={spot_chain.get('proxy_median')}, "
            f"median_proxy_to_spot_ratio="
            f"{spot_chain.get('median_proxy_to_spot_ratio')}, "
            f"relative_deviation={spot_chain.get('relative_deviation')}, "
            f"threshold={spot_chain.get('material_deviation_threshold')})"
        )

    accepted = _int_stat(stats, "accepted_quote_count")
    if accepted < policy.min_accepted_contracts:
        failures.append(
            "No option contracts accepted after cleaning/quality policy: "
            f"accepted={accepted}, min_accepted_contracts="
            f"{policy.min_accepted_contracts}"
        )
    _append_minimum_failure(
        failures,
        name="accepted calls",
        actual=_int_stat(stats, "accepted_call_count"),
        minimum=policy.min_accepted_calls,
    )
    _append_minimum_failure(
        failures,
        name="accepted puts",
        actual=_int_stat(stats, "accepted_put_count"),
        minimum=policy.min_accepted_puts,
    )
    _append_minimum_failure(
        failures,
        name="accepted expiries",
        actual=_int_stat(stats, "accepted_expiry_count"),
        minimum=policy.min_expiries,
    )
    return tuple(failures)


def _apply_provider_snapshot_quality_policy(
    result: QuoteCleaningResult,
    *,
    asof: pd.Timestamp,
    policy: ProviderSnapshotQualityPolicy,
) -> QuoteCleaningResult:
    if result.cleaned_quotes.empty:
        return result

    ages = _quote_age_seconds(result.cleaned_quotes, _utc_timestamp(asof))
    reject_mask = pd.Series(False, index=result.cleaned_quotes.index)
    if policy.reject_option_quotes_after_asof:
        reject_mask = reject_mask | (ages < 0.0)
    max_option_age = policy.effective_max_option_quote_age_seconds()
    if policy.rejects_stale_option_quotes() and max_option_age is not None:
        reject_mask = reject_mask | (ages > float(max_option_age))
    if not bool(reject_mask.any()):
        return result

    cleaned = result.cleaned_quotes.loc[~reject_mask].reset_index(drop=True)
    rejected_quotes = result.cleaned_quotes.loc[reject_mask].reset_index(drop=True)
    quality_rejections = _quality_rejected_quotes(
        rejected_quotes,
        asof=asof,
        policy=policy,
    )
    existing_rejections = _coerce_output(
        result.rejected_quotes,
        DatasetName.REJECTED_QUOTES,
    )
    new_rejections = _coerce_output(quality_rejections, DatasetName.REJECTED_QUOTES)
    rejected = (
        new_rejections
        if existing_rejections.empty
        else pd.concat([existing_rejections, new_rejections], ignore_index=True)
    )
    reason_counts = dict(result.reason_counts)
    for reason, count in (
        quality_rejections["rejection_reason"].astype(str).value_counts().items()
    ):
        reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + int(count)

    return QuoteCleaningResult(
        cleaned_quotes=_coerce_output(cleaned, DatasetName.CLEANED_QUOTES),
        rejected_quotes=_coerce_output(rejected, DatasetName.REJECTED_QUOTES),
        reason_counts=reason_counts,
        warnings=result.warnings,
    )


def _quality_rejected_quotes(
    quotes: pd.DataFrame,
    *,
    asof: pd.Timestamp,
    policy: ProviderSnapshotQualityPolicy,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    max_age_seconds = policy.max_option_quote_age_seconds
    if max_age_seconds is None:
        max_age_seconds = policy.max_quote_age_seconds
    for _, row in quotes.iterrows():
        age_seconds = (
            _utc_timestamp(asof) - _utc_timestamp(row["quote_ts"])
        ).total_seconds()
        if policy.reject_option_quotes_after_asof and age_seconds < 0.0:
            rejection_reason = QuoteRejectionReason.QUOTE_AFTER_ASOF.value
            rejection_detail = "quote_ts is after snapshot asof"
        elif (
            policy.rejects_stale_option_quotes()
            and max_age_seconds is not None
            and age_seconds > float(max_age_seconds)
        ):
            rejection_reason = QuoteRejectionReason.STALE_QUOTE.value
            rejection_detail = (
                f"quote_age_seconds={age_seconds:.6g} exceeds "
                f"max_option_quote_age_seconds={float(max_age_seconds):.6g}"
            )
        else:
            continue
        rows.append(
            {
                "underlying": row["underlying"],
                "contract_symbol": row["contract_symbol"],
                "quote_id": row["quote_id"],
                "quote_ts": row["quote_ts"],
                "asof": row["asof"],
                "expiry": row["expiry"],
                "strike": row["strike"],
                "right": row["right"],
                "bid": row["bid"],
                "ask": row["ask"],
                "mid": row["mid"],
                "iv": row["iv"],
                "vega": row["vega"],
                "source": row["source"],
                "rejection_reason": rejection_reason,
                "rejection_detail": rejection_detail,
                "cleaning_policy": row["cleaning_policy"],
            }
        )
    return pd.DataFrame(rows, columns=list(REJECTED_QUOTES_COLUMNS))


def _coerce_output(frame: pd.DataFrame, dataset_name: DatasetName) -> pd.DataFrame:
    columns = (
        CLEANED_QUOTES_COLUMNS
        if dataset_name == DatasetName.CLEANED_QUOTES
        else REJECTED_QUOTES_COLUMNS
    )
    if frame.empty:
        frame = pd.DataFrame(columns=list(columns))
    coerced = coerce_frame(frame, dataset_name, allow_extra=False)
    ordered = order_columns(coerced, dataset_name).reset_index(drop=True)
    validate_dtypes(ordered, dataset_name, allow_extra=False)
    return ordered


def _quote_age_seconds(frame: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    if frame.empty or "quote_ts" not in frame:
        return pd.Series([], dtype="float64")
    quote_ts = pd.to_datetime(frame["quote_ts"], errors="coerce", utc=True)
    ages = cast(pd.Series, (cast(Any, asof) - cast(Any, quote_ts)).dt.total_seconds())
    return ages


def _age_summary(ages: pd.Series) -> dict[str, float | None]:
    valid = ages.dropna()
    if valid.empty:
        return {"min": None, "median": None, "max": None}
    return {
        "min": float(valid.min()),
        "median": float(valid.median()),
        "max": float(valid.max()),
    }


def _numeric_summary(frame: pd.DataFrame, column: str) -> dict[str, float | None]:
    if frame.empty or column not in frame:
        return {"min": None, "max": None}
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return {"min": None, "max": None}
    return {"min": float(values.min()), "max": float(values.max())}


def _years_to_days(value: float | None) -> float | None:
    return None if value is None else float(value) * 365.0


def _quote_freshness_warning_codes(
    *,
    policy: ProviderSnapshotQualityPolicy,
    equity_age: float | None,
    stale_equity_quote: bool,
    option_quotes_after_asof_count: int,
    stale_option_quote_count: int,
    stale_accepted_quote_count: int,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if stale_equity_quote:
        warnings.append(
            "stale_equity_quote:"
            f"age_seconds={equity_age},"
            f"max_age_seconds={policy.effective_max_equity_quote_age_seconds()}"
        )
    if option_quotes_after_asof_count > 0:
        warnings.append(
            "option_quotes_after_asof:" f"count={option_quotes_after_asof_count}"
        )
    if stale_option_quote_count > 0:
        warnings.append(
            "stale_option_quotes:"
            f"count={stale_option_quote_count},"
            f"mode={policy.quote_freshness_mode},"
            f"action={policy.stale_quote_action},"
            f"max_age_seconds={policy.effective_max_option_quote_age_seconds()}"
        )
    if stale_accepted_quote_count > 0:
        warnings.append(
            "stale_accepted_option_quotes:"
            f"count={stale_accepted_quote_count},"
            f"mode={policy.quote_freshness_mode},"
            f"action={policy.stale_quote_action}"
        )
    if (
        policy.quote_freshness_mode == QUOTE_FRESHNESS_MODE_END_OF_DAY
        and policy.allow_prior_session
    ):
        warnings.append("end_of_day_prior_session_allowed")
    return tuple(warnings)


def _count_after_asof(ages: pd.Series) -> int:
    valid = ages.dropna()
    if valid.empty:
        return 0
    return int((valid < 0.0).sum())


def _count_stale(ages: pd.Series, max_age_seconds: float | None) -> int:
    if max_age_seconds is None:
        return 0
    valid = ages.dropna()
    if valid.empty:
        return 0
    return int((valid > float(max_age_seconds)).sum())


def _last_float(values: pd.Series) -> float | None:
    valid = values.dropna()
    if valid.empty:
        return None
    return float(valid.iloc[-1])


def _right_count(frame: pd.DataFrame, right: str) -> int:
    if frame.empty or "right" not in frame:
        return 0
    return int((frame["right"].astype("string").str.lower() == right).sum())


def _market_inputs_spot(market_inputs: pd.DataFrame) -> float | None:
    if market_inputs.empty or "spot" not in market_inputs:
        return None
    try:
        spot = float(cast(Any, market_inputs.iloc[0]["spot"]))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(spot) or spot <= 0.0:
        return None
    return spot


def _optional_row_text(row: pd.Series, column: str) -> str:
    if column not in row.index or pd.isna(row[column]):
        return ""
    return str(row[column]).strip()


def _optional_row_float(row: pd.Series, column: str) -> float | None:
    if column not in row.index or pd.isna(row[column]):
        return None
    try:
        value = float(cast(Any, row[column]))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _is_deep_itm_proxy_candidate(
    *,
    spot: float,
    strike: float,
    mid: float,
) -> bool:
    return mid >= max(
        _SPOT_CHAIN_PROXY_MIN_MID_SPOT_RATIO * spot,
        _SPOT_CHAIN_PROXY_MIN_MID_STRIKE_RATIO * strike,
    )


def _proxy_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    series = pd.Series(list(values), dtype="float64")
    return {
        "min": float(series.min()),
        "median": float(series.median()),
        "max": float(series.max()),
    }


def _spot_chain_diagnostic(stats: Mapping[str, object]) -> Mapping[str, object]:
    value = stats.get("spot_option_chain_diagnostic")
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return {}


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(cast(Any, value))
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _int_stat(stats: Mapping[str, object], key: str) -> int:
    return int(cast(Any, stats.get(key, 0) or 0))


def _append_minimum_failure(
    failures: list[str],
    *,
    name: str,
    actual: int,
    minimum: int | None,
) -> None:
    if minimum is not None and actual < minimum:
        failures.append(f"Provider snapshot has {actual} {name}; minimum is {minimum}")


def _validate_optional_nonnegative_number(
    value: float | None,
    field_name: str,
) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be numeric when provided")
    if float(value) < 0.0:
        raise ValueError(f"{field_name} must be >= 0")


def _validate_bool(value: bool, field_name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean")


def _validate_choice(value: str, choices: frozenset[str], field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"{field_name} must be one of: {expected}")


def _validate_nonnegative_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")


def _validate_optional_nonnegative_int(value: int | None, field_name: str) -> None:
    if value is None:
        return
    _validate_nonnegative_int(value, field_name)


__all__ = [
    "DEFAULT_BARS_TIMEFRAME",
    "DATA_POLICY_SCHEMA_VERSION",
    "DEFAULT_DAY_COUNT",
    "DEFAULT_RATE_CURVE_SERIES_IDS",
    "DEFAULT_RATE_SERIES_ID",
    "DEFAULT_SNAPSHOT_RATE_LOOKBACK_DAYS",
    "DIVIDEND_POLICY_MANUAL_STATIC",
    "DIVIDEND_POLICY_ZERO_ASSUMPTION",
    "MODEL_VALIDATION_POLICY_MODEL_READY_QUOTES_V1",
    "OPTION_CLEANING_POLICY_ID",
    "OPTION_CLEANING_POLICY_STAGED_RECOVERABLE_QUOTES_V1",
    "OPTION_CLEANING_POLICY_QUOTE_CLEANING_V1",
    "PROVIDER_RATE_CURVE_COLUMNS",
    "QUOTE_FRESHNESS_MODE_DEMO_LENIENT",
    "QUOTE_FRESHNESS_MODE_END_OF_DAY",
    "QUOTE_FRESHNESS_MODE_INTRADAY_STRICT",
    "ProviderSnapshotQualityPolicy",
    "RATE_CURVE_TENORS",
    "STALE_QUOTE_ACTION_FAIL",
    "STALE_QUOTE_ACTION_REJECT",
    "STALE_QUOTE_ACTION_WARN",
    "RATE_POLICY_FRED_TREASURY_ZERO_PROXY_LINEAR_CC",
    "_resolve_provider_snapshot_dividend_policy",
]
