"""Quote-cleaning contracts for marketdata normalization workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NamedTuple

import pandas as pd

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

_CLEANING_POLICY_ID = "quote_cleaning_policy.v1"
OPTION_CLEANING_POLICY_STAGED_RECOVERABLE_QUOTES_V1 = "staged_recoverable_quotes_v1"
MODEL_VALIDATION_POLICY_MODEL_READY_QUOTES_V1 = "model_ready_quotes_v1"
_ACT_365_SECONDS = 365 * 24 * 3600
_VANILLA_NO_ARB_ABS_TOL = 1e-7
_VANILLA_NO_ARB_REL_TOL = 1e-6


class QuoteRejectionReason(StrEnum):
    """Primary rejection reason for a quote rejected by staged cleaning."""

    UNPARSEABLE_CONTRACT = "unparseable_contract"
    BAD_EXPIRY = "bad_expiry"
    EXPIRED_CONTRACT = "expired_contract"
    NONPOSITIVE_MID = "nonpositive_mid"
    NONPOSITIVE_STRIKE = "nonpositive_strike"
    NEGATIVE_BID = "negative_bid"
    NEGATIVE_ASK = "negative_ask"
    CROSSED_BID_ASK = "crossed_bid_ask"
    QUOTE_AFTER_ASOF = "quote_after_asof"
    STALE_QUOTE = "stale_quote"
    MISSING_PRICE_SOURCE = "missing_price_source"
    MISSING_SPOT_FOR_MONEYNESS = "missing_spot_for_moneyness"
    MISSING_RATE_FOR_MODEL = "missing_rate_for_model"
    MISSING_DIVIDEND_FOR_MODEL = "missing_dividend_for_model"
    MISSING_TIME_TO_EXPIRY_FOR_MODEL = "missing_time_to_expiry_for_model"
    MISSING_IV_FOR_IV_VALIDATION = "missing_iv_for_iv_validation"
    UNSUPPORTED_OPTION_RIGHT = "unsupported_option_right"
    NONFINITE_NUMERIC_FIELD = "nonfinite_numeric_field"
    VANILLA_NO_ARBITRAGE_VIOLATION = "vanilla_no_arbitrage_violation"
    NONSTANDARD_OR_ADJUSTED_CONTRACT = "nonstandard_or_adjusted_contract"
    SPOT_OPTION_CHAIN_MISMATCH = "spot_option_chain_mismatch"
    MISSING_BID_OR_ASK = "missing_price_source"
    INVALID_BID_ASK_CROSS = "crossed_bid_ask"
    EXPIRED_OR_BAD_EXPIRY = "bad_expiry"
    MISSING_IV = "missing_iv_for_iv_validation"
    MISSING_GREEK = "nonfinite_numeric_field"
    SPREAD_TOO_WIDE = "nonfinite_numeric_field"
    BELOW_INTRINSIC_TOLERANCE = "nonfinite_numeric_field"


@dataclass(frozen=True, slots=True)
class QuoteCleaningPolicyV1:
    """Policy parameters for the first quote-cleaning contract."""

    max_relative_spread: float | None = None
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


class _Rejection(NamedTuple):
    reason: QuoteRejectionReason
    detail: str


class _DerivedQuoteValues(NamedTuple):
    bid: float | None
    ask: float | None
    mid: float
    spread: float | None
    relative_spread: float | None
    option_price_for_model: float
    mid_computed: bool


class _MarketConvention(NamedTuple):
    spot: float
    rate: float
    dividend_yield: float


def clean_option_quotes(
    option_chain: pd.DataFrame,
    market_inputs: pd.DataFrame,
    *,
    policy: QuoteCleaningPolicyV1 = QuoteCleaningPolicyV1(),  # noqa: B008
) -> QuoteCleaningResult:
    """Clean normalized option quotes into accepted and rejected quote frames."""

    _validate_policy(policy)
    market_frame = _coerce_input_frame(market_inputs, DatasetName.MARKET_INPUTS)
    option_frame = _coerce_input_frame(option_chain, DatasetName.OPTION_CHAIN)

    market = _single_market_convention(market_frame)

    cleaned_records: list[dict[str, object]] = []
    rejected_records: list[dict[str, object]] = []
    reason_counts: dict[str, int] = {}

    for _, row in option_frame.iterrows():
        quote_id = _quote_id(row)
        expiry_years = _expiry_years(row["expiry"], row["asof"])
        derived, rejection = _classify_rejection(row, policy, market, expiry_years)

        if rejection is None:
            if derived is None:
                raise RuntimeError("quote accepted without derived quote values")
            strike = _required_float(row["strike"])
            cleaned_records.append(
                _cleaned_quote_record(
                    row,
                    quote_id=quote_id,
                    expiry_years=expiry_years,
                    moneyness=_moneyness(strike, market.spot),
                    log_moneyness=_log_moneyness(strike, market.spot),
                    derived=derived,
                    model_validation_ready=_model_validation_ready(
                        row,
                        expiry_years=expiry_years,
                        derived=derived,
                        market_inputs=market_frame,
                    ),
                )
            )
            continue

        rejected_records.append(
            _rejected_quote_record(
                row,
                quote_id=quote_id,
                rejection_reason=rejection.reason.value,
                rejection_detail=rejection.detail,
            )
        )
        reason_counts[rejection.reason.value] = (
            reason_counts.get(rejection.reason.value, 0) + 1
        )

    cleaned_quotes = _output_frame(
        cleaned_records,
        DatasetName.CLEANED_QUOTES,
        CLEANED_QUOTES_COLUMNS,
    )
    rejected_quotes = _output_frame(
        rejected_records,
        DatasetName.REJECTED_QUOTES,
        REJECTED_QUOTES_COLUMNS,
    )
    warnings = _warnings(option_frame, cleaned_quotes)

    return QuoteCleaningResult(
        cleaned_quotes=cleaned_quotes,
        rejected_quotes=rejected_quotes,
        reason_counts=reason_counts,
        warnings=warnings,
    )


def _validate_policy(policy: QuoteCleaningPolicyV1) -> None:
    if policy.max_relative_spread is not None:
        if (
            isinstance(policy.max_relative_spread, bool)
            or not isinstance(policy.max_relative_spread, int | float)
            or not math.isfinite(float(policy.max_relative_spread))
            or float(policy.max_relative_spread) < 0.0
        ):
            raise ValueError(
                "QuoteCleaningPolicyV1.max_relative_spread must be finite and "
                ">= 0 when provided"
            )
    if policy.day_count != "ACT/365":
        raise ValueError(
            "QuoteCleaningPolicyV1.day_count only supports 'ACT/365' in S3; "
            f"got {policy.day_count!r}"
        )


def _coerce_input_frame(
    frame: pd.DataFrame,
    dataset_name: DatasetName,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"{dataset_name.value} input must be a pandas DataFrame, "
            f"got {type(frame).__name__}"
        )

    coerced = coerce_frame(frame, dataset_name, allow_extra=False)
    ordered = order_columns(coerced, dataset_name).reset_index(drop=True)
    validate_dtypes(ordered, dataset_name, allow_extra=False)
    return ordered


def _single_market_convention(market_inputs: pd.DataFrame) -> _MarketConvention:
    if len(market_inputs) != 1:
        raise ValueError(
            "market_inputs must contain exactly one row for quote cleaning; "
            f"found {len(market_inputs)}"
        )

    spot = _optional_float(market_inputs.loc[0, "spot"])
    if spot is None or not math.isfinite(spot) or spot <= 0:
        raise ValueError("market_inputs spot must be finite and > 0")

    rate = _optional_float(market_inputs.loc[0, "rate"])
    if rate is None or not math.isfinite(rate):
        raise ValueError("market_inputs rate must be finite")

    dividend_yield = _optional_float(market_inputs.loc[0, "dividend_yield"])
    if dividend_yield is None or not math.isfinite(dividend_yield):
        raise ValueError("market_inputs dividend_yield must be finite")

    return _MarketConvention(
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
    )


def _quote_id(row: pd.Series) -> str:
    return "|".join(
        (
            _text_value(row["underlying"]),
            _text_value(row["contract_symbol"]),
            _timestamp_iso(row["quote_ts"]),
            _timestamp_iso(row["asof"]),
        )
    )


def _timestamp_iso(value: Any) -> str:
    if pd.isna(value):
        return "NaT"

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp.isoformat()


def _expiry_years(expiry: Any, asof: Any) -> float:
    if pd.isna(expiry) or pd.isna(asof):
        return math.nan

    expiry_utc = _as_utc_timestamp(expiry)
    asof_utc = _as_utc_timestamp(asof)
    return (expiry_utc - asof_utc).total_seconds() / _ACT_365_SECONDS


def _as_utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")

    return timestamp.tz_convert("UTC")


def _moneyness(strike: float, spot: float) -> float:
    return strike / spot


def _log_moneyness(strike: float, spot: float) -> float:
    return math.log(strike / spot)


def _classify_rejection(
    row: pd.Series,
    policy: QuoteCleaningPolicyV1,
    market: _MarketConvention,
    expiry_years: float,
) -> tuple[_DerivedQuoteValues | None, _Rejection | None]:
    right = _text_value(row["right"]).strip().lower()
    if right not in {"call", "put"}:
        return None, _Rejection(
            QuoteRejectionReason.UNSUPPORTED_OPTION_RIGHT,
            f"right must be call or put; got {right!r}",
        )

    strike = _optional_float(row["strike"])
    if strike is None or not math.isfinite(strike) or strike <= 0:
        return None, _Rejection(
            QuoteRejectionReason.NONPOSITIVE_STRIKE,
            f"strike must be > 0; got {_format_optional_float(strike)}",
        )

    if not math.isfinite(expiry_years):
        return None, _Rejection(
            QuoteRejectionReason.BAD_EXPIRY,
            "expiry must be parseable and not missing",
        )
    if expiry_years <= 0:
        return None, _Rejection(
            QuoteRejectionReason.EXPIRED_CONTRACT,
            "expiry must be after asof; "
            f"got expiry_years={_format_float(expiry_years)}",
        )

    if pd.isna(row["quote_ts"]):
        return None, _Rejection(
            QuoteRejectionReason.NONFINITE_NUMERIC_FIELD,
            "quote_ts must not be missing",
        )
    if pd.isna(row["asof"]):
        return None, _Rejection(
            QuoteRejectionReason.NONFINITE_NUMERIC_FIELD,
            "asof must not be missing",
        )
    quote_age_seconds = (
        _as_utc_timestamp(row["asof"]) - _as_utc_timestamp(row["quote_ts"])
    ).total_seconds()
    if quote_age_seconds < 0.0:
        return None, _Rejection(
            QuoteRejectionReason.QUOTE_AFTER_ASOF,
            "quote_ts is after asof; "
            f"quote_age_seconds={_format_float(quote_age_seconds)}",
        )

    derived, price_rejection = _derive_quote_values(row)
    if price_rejection is not None:
        return None, price_rejection
    assert derived is not None

    if (
        policy.max_relative_spread is not None
        and derived.relative_spread is not None
        and derived.relative_spread > float(policy.max_relative_spread)
    ):
        return None, _Rejection(
            QuoteRejectionReason.NONFINITE_NUMERIC_FIELD,
            "relative_spread exceeds max_relative_spread; "
            f"relative_spread={_format_float(derived.relative_spread)}, "
            f"max_relative_spread={_format_float(float(policy.max_relative_spread))}",
        )

    iv = _optional_float(row["iv"])
    if policy.require_iv and (iv is None or not math.isfinite(iv) or iv <= 0):
        return None, _Rejection(
            QuoteRejectionReason.MISSING_IV_FOR_IV_VALIDATION,
            f"iv must be finite and > 0; got {_format_optional_float(iv)}",
        )

    vega = _optional_float(row["vega"])
    if policy.require_vega and (vega is None or not math.isfinite(vega) or vega <= 0):
        return None, _Rejection(
            QuoteRejectionReason.NONFINITE_NUMERIC_FIELD,
            f"vega must be finite and > 0; got {_format_optional_float(vega)}",
        )

    no_arb_rejection = _vanilla_no_arbitrage_rejection(
        right=right,
        strike=strike,
        expiry_years=expiry_years,
        mid=derived.mid,
        market=market,
    )
    if no_arb_rejection is not None:
        return None, no_arb_rejection

    return derived, None


def _derive_quote_values(
    row: pd.Series,
) -> tuple[_DerivedQuoteValues | None, _Rejection | None]:
    bid = _optional_float(row["bid"])
    ask = _optional_float(row["ask"])
    supplied_mid = _optional_float(row["mid"])
    last = _optional_float(row["last"]) if "last" in row.index else None

    if bid is not None and not math.isfinite(bid):
        return None, _Rejection(
            QuoteRejectionReason.NONFINITE_NUMERIC_FIELD,
            f"bid must be finite when present; got {_format_float(bid)}",
        )
    if ask is not None and not math.isfinite(ask):
        return None, _Rejection(
            QuoteRejectionReason.NONFINITE_NUMERIC_FIELD,
            f"ask must be finite when present; got {_format_float(ask)}",
        )
    if supplied_mid is not None and not math.isfinite(supplied_mid):
        supplied_mid = None
    if last is not None and not math.isfinite(last):
        last = None

    if bid is not None and bid < 0:
        return None, _Rejection(
            QuoteRejectionReason.NEGATIVE_BID,
            f"bid must be >= 0; got {_format_float(bid)}",
        )
    if ask is not None and ask < 0:
        return None, _Rejection(
            QuoteRejectionReason.NEGATIVE_ASK,
            f"ask must be >= 0; got {_format_float(ask)}",
        )
    if bid is not None and ask is not None and bid > ask:
        return None, _Rejection(
            QuoteRejectionReason.CROSSED_BID_ASK,
            "bid must be <= ask; "
            f"got bid={_format_float(bid)}, ask={_format_float(ask)}",
        )

    spread: float | None = None
    relative_spread: float | None = None
    mid: float | None = None
    mid_computed = False
    if bid is not None and ask is not None:
        spread = ask - bid
        mid = (bid + ask) / 2.0
        mid_computed = supplied_mid is None or not math.isclose(
            supplied_mid,
            mid,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    elif supplied_mid is not None:
        mid = supplied_mid
    elif last is not None:
        mid = last

    if mid is None:
        return None, _Rejection(
            QuoteRejectionReason.MISSING_PRICE_SOURCE,
            "quote requires bid/ask, mid, or last price source",
        )
    if not math.isfinite(mid) or mid <= 0:
        return None, _Rejection(
            QuoteRejectionReason.NONPOSITIVE_MID,
            f"mid must be finite and > 0; got {_format_optional_float(mid)}",
        )

    if spread is not None:
        relative_spread = spread / mid if mid > 0 else None

    return (
        _DerivedQuoteValues(
            bid=bid,
            ask=ask,
            mid=mid,
            spread=spread,
            relative_spread=relative_spread,
            option_price_for_model=mid,
            mid_computed=mid_computed,
        ),
        None,
    )


def _intrinsic_value(right: str, spot: float, strike: float) -> float:
    if right == "call":
        return max(spot - strike, 0.0)
    if right == "put":
        return max(strike - spot, 0.0)

    raise ValueError(f"option_chain right must be 'call' or 'put'; got {right!r}")


def _vanilla_no_arbitrage_rejection(
    *,
    right: str,
    strike: float,
    expiry_years: float,
    mid: float,
    market: _MarketConvention,
) -> _Rejection | None:
    df = math.exp(-market.rate * expiry_years)
    forward = market.spot * math.exp(
        (market.rate - market.dividend_yield) * expiry_years
    )
    if right == "call":
        lower = max(df * (forward - strike), 0.0)
        upper = df * forward
    elif right == "put":
        lower = max(df * (strike - forward), 0.0)
        upper = df * strike
    else:
        raise ValueError(f"option_chain right must be 'call' or 'put'; got {right!r}")

    lower_tolerance = _bound_tolerance(lower)
    upper_tolerance = _bound_tolerance(upper)
    if mid >= lower - lower_tolerance and mid <= upper + upper_tolerance:
        return None

    side = "below lower bound" if mid < lower - lower_tolerance else "above upper bound"
    return _Rejection(
        QuoteRejectionReason.VANILLA_NO_ARBITRAGE_VIOLATION,
        "mid violates broad vanilla no-arbitrage bounds; "
        f"side={side}, right={right}, mid={_format_float(mid)}, "
        f"lower={_format_float(lower)}, upper={_format_float(upper)}, "
        f"abs_tol={_format_float(_VANILLA_NO_ARB_ABS_TOL)}, "
        f"rel_tol={_format_float(_VANILLA_NO_ARB_REL_TOL)}, "
        f"spot={_format_float(market.spot)}, strike={_format_float(strike)}, "
        f"expiry_years={_format_float(expiry_years)}, "
        f"rate={_format_float(market.rate)}, "
        f"dividend_yield={_format_float(market.dividend_yield)}, "
        f"df={_format_float(df)}, forward={_format_float(forward)}",
    )


def _bound_tolerance(bound: float) -> float:
    return _VANILLA_NO_ARB_ABS_TOL + _VANILLA_NO_ARB_REL_TOL * max(abs(bound), 1.0)


def _cleaned_quote_record(
    row: pd.Series,
    *,
    quote_id: str,
    expiry_years: float,
    moneyness: float,
    log_moneyness: float,
    derived: _DerivedQuoteValues,
    model_validation_ready: bool,
) -> dict[str, object]:
    provider_iv_available = _is_finite_positive(row["iv"])
    provider_greeks_available = all(
        _is_finite_number(row[column])
        for column in ("delta", "gamma", "theta", "vega", "rho")
    )
    return {
        "underlying": row["underlying"],
        "contract_symbol": row["contract_symbol"],
        "quote_id": quote_id,
        "quote_ts": row["quote_ts"],
        "asof": row["asof"],
        "expiry": row["expiry"],
        "expiry_years": expiry_years,
        "strike": row["strike"],
        "right": row["right"],
        "bid": derived.bid,
        "ask": derived.ask,
        "mid": derived.mid,
        "spread": derived.spread,
        "relative_spread": derived.relative_spread,
        "iv": row["iv"],
        "vega": row["vega"],
        "delta": row["delta"],
        "gamma": row["gamma"],
        "theta": row["theta"],
        "rho": row["rho"],
        "open_interest": row["open_interest"],
        "moneyness": moneyness,
        "log_moneyness": log_moneyness,
        "time_to_expiry_years": expiry_years,
        "option_price_for_model": derived.option_price_for_model,
        "mid_computed": derived.mid_computed,
        "time_to_expiry_computed": True,
        "moneyness_computed": True,
        "provider_iv_available": provider_iv_available,
        "provider_greeks_available": provider_greeks_available,
        "model_price_available": True,
        "model_validation_ready": model_validation_ready,
        "iv_validation_ready": provider_iv_available,
        "greek_validation_ready": provider_greeks_available,
        "source": row["source"],
        "cleaning_policy": _CLEANING_POLICY_ID,
    }


def _rejected_quote_record(
    row: pd.Series,
    *,
    quote_id: str,
    rejection_reason: str,
    rejection_detail: str,
) -> dict[str, object]:
    return {
        "underlying": row["underlying"],
        "contract_symbol": row["contract_symbol"],
        "quote_id": quote_id,
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
        "cleaning_policy": _CLEANING_POLICY_ID,
    }


def _model_validation_ready(
    row: pd.Series,
    *,
    expiry_years: float,
    derived: _DerivedQuoteValues,
    market_inputs: pd.DataFrame,
) -> bool:
    market_row = market_inputs.iloc[0]
    return all(
        (
            _has_text(row["underlying"]),
            _has_text(row["contract_symbol"]),
            not pd.isna(row["expiry"]),
            _is_finite_positive(row["strike"]),
            _text_value(row["right"]).strip().lower() in {"call", "put"},
            _is_finite_positive(market_row["spot"]),
            math.isfinite(derived.option_price_for_model)
            and derived.option_price_for_model > 0.0,
            _is_finite_number(market_row["rate"]),
            _is_finite_number(market_row["dividend_yield"]),
            math.isfinite(expiry_years) and expiry_years > 0.0,
        )
    )


def _output_frame(
    records: list[dict[str, object]],
    dataset_name: DatasetName,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records, columns=columns)
    coerced = coerce_frame(frame, dataset_name, allow_extra=False)
    out = (
        order_columns(coerced, dataset_name)
        .loc[:, list(columns)]
        .reset_index(drop=True)
    )
    validate_dtypes(out, dataset_name, allow_extra=False)
    return out


def _warnings(
    option_frame: pd.DataFrame, cleaned_quotes: pd.DataFrame
) -> tuple[str, ...]:
    if len(option_frame) > 0 and cleaned_quotes.empty:
        return ("all_quotes_rejected",)

    return ()


def _optional_float(value: Any) -> float | None:
    if pd.isna(value):
        return None

    return float(value)


def _required_float(value: Any) -> float:
    optional = _optional_float(value)
    if optional is None:
        raise ValueError("required numeric value is missing")

    return optional


def _text_value(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"

    return str(value)


def _has_text(value: Any) -> bool:
    if pd.isna(value):
        return False
    return bool(str(value).strip())


def _is_finite_number(value: Any) -> bool:
    if pd.isna(value):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _is_finite_positive(value: Any) -> bool:
    if not _is_finite_number(value):
        return False
    return float(value) > 0.0


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "missing"

    return _format_float(value)


def _format_float(value: float) -> str:
    return str(float(value))


__all__ = [
    "MODEL_VALIDATION_POLICY_MODEL_READY_QUOTES_V1",
    "OPTION_CLEANING_POLICY_STAGED_RECOVERABLE_QUOTES_V1",
    "clean_option_quotes",
    "QuoteCleaningPolicyV1",
    "QuoteCleaningResult",
    "QuoteRejectionReason",
]
