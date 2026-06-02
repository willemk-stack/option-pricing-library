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
_ACT_365_SECONDS = 365 * 24 * 3600


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


class _Rejection(NamedTuple):
    reason: QuoteRejectionReason
    detail: str


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

    spot = _single_spot(market_frame)

    cleaned_records: list[dict[str, object]] = []
    rejected_records: list[dict[str, object]] = []
    reason_counts: dict[str, int] = {}

    for _, row in option_frame.iterrows():
        quote_id = _quote_id(row)
        expiry_years = _expiry_years(row["expiry"], row["asof"])
        rejection = _classify_rejection(row, policy, spot, expiry_years)

        if rejection is None:
            strike = _required_float(row["strike"])
            cleaned_records.append(
                _cleaned_quote_record(
                    row,
                    quote_id=quote_id,
                    expiry_years=expiry_years,
                    moneyness=_moneyness(strike, spot),
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


def _single_spot(market_inputs: pd.DataFrame) -> float:
    if len(market_inputs) != 1:
        raise ValueError(
            "market_inputs must contain exactly one row for quote cleaning; "
            f"found {len(market_inputs)}"
        )

    spot = _optional_float(market_inputs.loc[0, "spot"])
    if spot is None or not math.isfinite(spot) or spot <= 0:
        raise ValueError("market_inputs spot must be finite and > 0")

    return spot


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

    expiry_utc = _expiry_midnight_utc(expiry)
    asof_utc = _as_utc_timestamp(asof)
    return (expiry_utc - asof_utc).total_seconds() / _ACT_365_SECONDS


def _expiry_midnight_utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.normalize().tz_localize("UTC")

    return timestamp.tz_convert("UTC").normalize()


def _as_utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")

    return timestamp.tz_convert("UTC")


def _moneyness(strike: float, spot: float) -> float:
    return strike / spot


def _classify_rejection(
    row: pd.Series,
    policy: QuoteCleaningPolicyV1,
    spot: float,
    expiry_years: float,
) -> _Rejection | None:
    strike = _optional_float(row["strike"])
    if strike is None or not math.isfinite(strike) or strike <= 0:
        return _Rejection(
            QuoteRejectionReason.NONPOSITIVE_STRIKE,
            f"strike must be > 0; got {_format_optional_float(strike)}",
        )

    if not math.isfinite(expiry_years) or expiry_years <= 0:
        return _Rejection(
            QuoteRejectionReason.EXPIRED_CONTRACT,
            "expiry must be after asof; "
            f"got expiry_years={_format_float(expiry_years)}",
        )

    bid = _optional_float(row["bid"])
    ask = _optional_float(row["ask"])
    mid = _optional_float(row["mid"])

    if bid is None and ask is None and mid is None:
        return _Rejection(
            QuoteRejectionReason.MISSING_REQUIRED_PRICE,
            "bid, ask, and mid are required; all are missing",
        )

    if bid is None or ask is None:
        return _Rejection(
            QuoteRejectionReason.MISSING_REQUIRED_PRICE,
            "bid and ask are required; "
            f"got bid={_format_optional_float(bid)}, "
            f"ask={_format_optional_float(ask)}",
        )

    if not math.isfinite(bid) or not math.isfinite(ask):
        return _Rejection(
            QuoteRejectionReason.MISSING_REQUIRED_PRICE,
            "bid and ask must be finite; "
            f"got bid={_format_float(bid)}, ask={_format_float(ask)}",
        )

    if bid < 0:
        return _Rejection(
            QuoteRejectionReason.NEGATIVE_BID,
            f"bid must be >= 0; got {_format_float(bid)}",
        )

    if ask <= 0:
        return _Rejection(
            QuoteRejectionReason.NONPOSITIVE_ASK,
            f"ask must be > 0; got {_format_float(ask)}",
        )

    if ask < bid:
        return _Rejection(
            QuoteRejectionReason.CROSSED_MARKET,
            "ask must be >= bid; "
            f"got bid={_format_float(bid)}, ask={_format_float(ask)}",
        )

    if mid is None or not math.isfinite(mid) or mid <= 0:
        return _Rejection(
            QuoteRejectionReason.INVALID_MID,
            f"mid must be finite and > 0; got {_format_optional_float(mid)}",
        )

    intrinsic = _intrinsic_value(_text_value(row["right"]), spot, strike)
    if mid + policy.intrinsic_tolerance < intrinsic:
        return _Rejection(
            QuoteRejectionReason.BELOW_INTRINSIC_TOLERANCE,
            "mid is below intrinsic value; "
            f"mid={_format_float(mid)}, intrinsic={_format_float(intrinsic)}",
        )

    relative_spread = (ask - bid) / mid
    if (
        not math.isfinite(relative_spread)
        or relative_spread > policy.max_relative_spread
    ):
        return _Rejection(
            QuoteRejectionReason.SPREAD_TOO_WIDE,
            "relative spread exceeds max_relative_spread; "
            f"relative_spread={_format_float(relative_spread)}, "
            f"max_relative_spread={_format_float(policy.max_relative_spread)}",
        )

    iv = _optional_float(row["iv"])
    if policy.require_iv and (iv is None or not math.isfinite(iv) or iv <= 0):
        return _Rejection(
            QuoteRejectionReason.MISSING_IV_FOR_IV_REQUIRED_WORKFLOW,
            f"iv must be finite and > 0; got {_format_optional_float(iv)}",
        )

    vega = _optional_float(row["vega"])
    if policy.require_vega and (vega is None or not math.isfinite(vega) or vega <= 0):
        return _Rejection(
            QuoteRejectionReason.MISSING_VEGA_FOR_WEIGHTED_CALIBRATION,
            f"vega must be finite and > 0; got {_format_optional_float(vega)}",
        )

    return None


def _intrinsic_value(right: str, spot: float, strike: float) -> float:
    if right == "call":
        return max(spot - strike, 0.0)
    if right == "put":
        return max(strike - spot, 0.0)

    raise ValueError(f"option_chain right must be 'call' or 'put'; got {right!r}")


def _cleaned_quote_record(
    row: pd.Series,
    *,
    quote_id: str,
    expiry_years: float,
    moneyness: float,
) -> dict[str, object]:
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
        "bid": row["bid"],
        "ask": row["ask"],
        "mid": row["mid"],
        "iv": row["iv"],
        "vega": row["vega"],
        "delta": row["delta"],
        "gamma": row["gamma"],
        "theta": row["theta"],
        "rho": row["rho"],
        "open_interest": row["open_interest"],
        "moneyness": moneyness,
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


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "missing"

    return _format_float(value)


def _format_float(value: float) -> str:
    return str(float(value))


__all__ = [
    "clean_option_quotes",
    "QuoteCleaningPolicyV1",
    "QuoteCleaningResult",
    "QuoteRejectionReason",
]
