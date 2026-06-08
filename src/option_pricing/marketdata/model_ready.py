"""Model-ready preparation helpers for loaded marketdata bundles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from option_pricing.marketdata.gold import heston_quote_set_from_frame
from option_pricing.marketdata.schemas import HESTON_QUOTES_COLUMNS, DatasetName
from option_pricing.marketdata.validation import coerce_frame, validate_dtypes
from option_pricing.models.heston.calibration.heston_types import (
    HESTON_OBJECTIVE_TYPES,
    HestonObjectiveType,
    HestonQuoteSet,
)
from option_pricing.models.heston.calibration.preflight import (
    HestonQuotePreflight,
    preflight_heston_quotes,
)
from option_pricing.types import MarketData

if TYPE_CHECKING:
    from option_pricing.marketdata.bundles import LoadedModelValidationBundle

_ACT_365_DAYS = 365.0
_SECONDS_PER_DAY = 24 * 60 * 60
_VALID_OPTION_TYPES = frozenset({"call", "put"})
_PRICE_BOUND_TOL = 1.0e-10
_BID_ASK_TOL = 1.0e-12
_COMPUTABLE_EXPIRY_COLUMN = "expiry_years"
_REQUIRED_HESTON_COLUMNS_EXCEPT_COMPUTED = tuple(
    column for column in HESTON_QUOTES_COLUMNS if column != _COMPUTABLE_EXPIRY_COLUMN
)
_REJECTION_REASON_ORDER = (
    "missing_required_column",
    "missing_mid",
    "nonfinite_mid",
    "nonpositive_mid",
    "missing_bid",
    "missing_ask",
    "crossed_market",
    "missing_expiry",
    "nonpositive_time_to_expiry",
    "short_expiry",
    "long_expiry",
    "missing_strike",
    "nonpositive_strike",
    "missing_option_type",
    "invalid_option_type",
    "missing_iv_for_seed",
    "nonfinite_iv",
    "nonpositive_iv",
    "missing_vega_for_objective",
    "nonfinite_vega",
    "nonpositive_vega",
    "nonpositive_spread",
    "price_bound_violation",
    "mid_outside_bid_ask",
    "extreme_moneyness",
)
_PREPARE_HESTON_MARKET_FIT_GUIDANCE = (
    "Use prepare_heston_market_fit(bundle) before calibrating from saved "
    "bundle artifacts."
)


@dataclass(frozen=True, slots=True)
class HestonReadyStats:
    input_quote_count: int
    selected_quote_count: int
    rejected_quote_count: int
    rejection_counts: dict[str, int]
    expiry_count: int
    min_expiry_days: float | None
    max_expiry_days: float | None
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreparedHestonMarketFit:
    model_name: str
    objective_type: HestonObjectiveType
    market_data: MarketData
    selected_quotes: pd.DataFrame
    rejected_quotes: pd.DataFrame
    quote_set: HestonQuoteSet | None
    preflight: HestonQuotePreflight | None
    stats: HestonReadyStats
    status: str
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _NumericValue:
    value: float | None
    missing: bool
    finite: bool

    @property
    def usable(self) -> bool:
        return self.value is not None and self.finite


def prepare_heston_market_fit(
    bundle: LoadedModelValidationBundle,
    *,
    objective_type: HestonObjectiveType | str = "price_rmse",
    min_expiry_days: float = 7.0,
    max_expiry_days: float | None = None,
    min_moneyness: float | None = 0.5,
    max_moneyness: float | None = 2.0,
    require_iv_for_seed: bool = True,
    require_vega_for_objective: bool | None = None,
    raise_on_block: bool = False,
) -> PreparedHestonMarketFit:
    """Prepare a loaded model-validation bundle for Heston calibration.

    This helper starts from ``bundle.heston_quotes`` and only prepares the quote
    universe. It does not fit Heston parameters or call calibration routines.
    """

    objective = _normalize_objective_type(objective_type)
    min_expiry_days = _validate_nonnegative_float(
        "min_expiry_days",
        min_expiry_days,
    )
    max_expiry_days = _validate_optional_nonnegative_float(
        "max_expiry_days",
        max_expiry_days,
    )
    if max_expiry_days is not None and max_expiry_days < min_expiry_days:
        raise ValueError(
            "max_expiry_days must be greater than or equal to min_expiry_days"
        )

    min_moneyness = _validate_optional_positive_float(
        "min_moneyness",
        min_moneyness,
    )
    max_moneyness = _validate_optional_positive_float(
        "max_moneyness",
        max_moneyness,
    )
    if (
        min_moneyness is not None
        and max_moneyness is not None
        and max_moneyness < min_moneyness
    ):
        raise ValueError("max_moneyness must be greater than or equal to min_moneyness")

    require_iv_for_seed = _validate_bool("require_iv_for_seed", require_iv_for_seed)
    raise_on_block = _validate_bool("raise_on_block", raise_on_block)
    require_vega = _resolve_require_vega_for_objective(
        objective,
        require_vega_for_objective,
    )

    market_data = _bundle_market_data(bundle)
    spot = _positive_finite_market_value(market_data, "spot")
    source = _bundle_heston_quotes(bundle)
    _validate_required_heston_columns(source)
    working = _with_expiry_years(source)

    reject_reasons = _heston_rejection_reasons(
        working,
        market_data=market_data,
        spot=spot,
        objective_type=objective,
        min_expiry_days=min_expiry_days,
        max_expiry_days=max_expiry_days,
        min_moneyness=min_moneyness,
        max_moneyness=max_moneyness,
        require_iv_for_seed=require_iv_for_seed,
        require_vega_for_objective=require_vega,
    )
    selected_quotes = _selected_quotes_frame(working, reject_reasons)
    rejected_quotes = _rejected_quotes_frame(working, reject_reasons)
    rejection_counts = _rejection_counts(reject_reasons)

    quote_set: HestonQuoteSet | None = None
    preflight: HestonQuotePreflight | None = None
    warnings: tuple[str, ...] = ()
    status = "ready"

    if selected_quotes.empty:
        status = "empty"
    else:
        quote_set = _quote_set_from_selected_quotes(selected_quotes, market_data)
        warnings = _quote_set_warnings(quote_set)
        preflight = preflight_heston_quotes(quote_set, raise_on_block=False)
        if preflight.recommendation == "block":
            status = "blocked"
            warnings = _dedupe_strings((*warnings, *preflight.messages))
            if raise_on_block:
                raise ValueError(
                    "Heston quote preflight blocked calibration: "
                    + " ".join(preflight.messages)
                )

    stats = _heston_ready_stats(
        input_quote_count=len(working),
        selected_quotes=selected_quotes,
        rejected_quotes=rejected_quotes,
        rejection_counts=rejection_counts,
        warnings=warnings,
    )
    return PreparedHestonMarketFit(
        model_name="heston",
        objective_type=objective,
        market_data=market_data,
        selected_quotes=selected_quotes,
        rejected_quotes=rejected_quotes,
        quote_set=quote_set,
        preflight=preflight,
        stats=stats,
        status=status,
        warnings=warnings,
    )


def _normalize_objective_type(
    objective_type: HestonObjectiveType | str,
) -> HestonObjectiveType:
    if not isinstance(objective_type, str):
        raise TypeError("objective_type must be a HestonObjectiveType or string")
    normalized = objective_type.strip().lower()
    if normalized not in HESTON_OBJECTIVE_TYPES:
        supported = ", ".join(repr(value) for value in HESTON_OBJECTIVE_TYPES)
        raise ValueError(
            f"objective_type must be one of {supported}; got {objective_type!r}"
        )
    return cast(HestonObjectiveType, normalized)


def _validate_required_heston_columns(frame: pd.DataFrame) -> None:
    missing = [
        column
        for column in _REQUIRED_HESTON_COLUMNS_EXCEPT_COMPUTED
        if column not in frame.columns
    ]
    if missing:
        joined = ", ".join(repr(column) for column in missing)
        raise ValueError(
            "bundle.heston_quotes is missing required columns: "
            f"{joined}. heston_quotes is a candidate artifact; "
            f"{_PREPARE_HESTON_MARKET_FIT_GUIDANCE}"
        )


def _bundle_heston_quotes(bundle: Any) -> pd.DataFrame:
    heston_quotes = getattr(bundle, "heston_quotes", None)
    if not isinstance(heston_quotes, pd.DataFrame):
        raise TypeError(
            "bundle.heston_quotes must be a pandas DataFrame. "
            f"{_PREPARE_HESTON_MARKET_FIT_GUIDANCE}"
        )
    return heston_quotes.copy(deep=True).reset_index(drop=True)


def _bundle_market_data(bundle: Any) -> MarketData:
    market_data = getattr(bundle, "market_data", None)
    if not isinstance(market_data, MarketData):
        raise TypeError(
            "bundle.market_data must be a MarketData instance. "
            "Use load_model_validation_bundle(path) to load bundle market data "
            "before preparation."
        )
    for field_name in ("rate", "dividend_yield"):
        _finite_market_value(market_data, field_name)
    return market_data


def _finite_market_value(market_data: MarketData, field_name: str) -> float:
    try:
        value = float(getattr(market_data, field_name))
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"bundle.market_data.{field_name} must be numeric. "
            "Saved model-validation bundles must provide numeric spot, rate, "
            "and dividend_yield assumptions."
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"bundle.market_data.{field_name} must be finite. "
            "Saved model-validation bundles must provide finite market data "
            "assumptions before Heston preparation."
        )
    return value


def _positive_finite_market_value(market_data: MarketData, field_name: str) -> float:
    value = _finite_market_value(market_data, field_name)
    if value <= 0.0:
        raise ValueError(
            f"bundle.market_data.{field_name} must be positive. "
            "Heston preparation requires a positive spot in bundle.market_data."
        )
    return value


def _with_expiry_years(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy(deep=True)
    if _COMPUTABLE_EXPIRY_COLUMN not in working.columns:
        working[_COMPUTABLE_EXPIRY_COLUMN] = np.nan

    values = pd.to_numeric(working[_COMPUTABLE_EXPIRY_COLUMN], errors="coerce")
    value_array = values.to_numpy(dtype=np.float64, na_value=np.nan)
    for position, value in enumerate(value_array):
        if math.isfinite(float(value)):
            continue
        computed = _compute_expiry_years(
            working.at[position, "expiry"],
            working.at[position, "asof"],
        )
        if math.isfinite(computed):
            value_array[position] = computed

    working[_COMPUTABLE_EXPIRY_COLUMN] = value_array
    return working


def _compute_expiry_years(expiry: object, asof: object) -> float:
    if _is_missing(expiry) or _is_missing(asof):
        return math.nan
    try:
        expiry_timestamp = pd.Timestamp(cast(Any, expiry))
        asof_timestamp = pd.Timestamp(cast(Any, asof))
    except (TypeError, ValueError):
        return math.nan
    if pd.isna(expiry_timestamp) or pd.isna(asof_timestamp):
        return math.nan

    if expiry_timestamp.tzinfo is None:
        expiry_timestamp = expiry_timestamp.normalize().tz_localize("UTC")
    else:
        expiry_timestamp = expiry_timestamp.tz_convert("UTC").normalize()
    if asof_timestamp.tzinfo is None:
        asof_timestamp = asof_timestamp.tz_localize("UTC")
    else:
        asof_timestamp = asof_timestamp.tz_convert("UTC")

    return (
        (expiry_timestamp - asof_timestamp).total_seconds()
        / _SECONDS_PER_DAY
        / _ACT_365_DAYS
    )


def _heston_rejection_reasons(
    frame: pd.DataFrame,
    *,
    market_data: MarketData,
    spot: float,
    objective_type: HestonObjectiveType,
    min_expiry_days: float,
    max_expiry_days: float | None,
    min_moneyness: float | None,
    max_moneyness: float | None,
    require_iv_for_seed: bool,
    require_vega_for_objective: bool,
) -> list[tuple[str, ...]]:
    all_reasons: list[tuple[str, ...]] = []
    for _, row in frame.iterrows():
        reasons: list[str] = []
        mid = _numeric_value(row["mid"])
        bid = _numeric_value(row["bid"])
        ask = _numeric_value(row["ask"])
        strike = _numeric_value(row["strike"])
        expiry_years = _numeric_value(row[_COMPUTABLE_EXPIRY_COLUMN])
        iv = _numeric_value(row["iv"])
        vega = _numeric_value(row["vega"])
        right = _text_value(row["right"])
        option_type = _text_value(row["option_type"])

        _add_mid_reasons(reasons, mid)
        _add_bid_ask_reasons(reasons, bid, ask, mid, objective_type)
        _add_expiry_reasons(
            reasons,
            row=row,
            expiry_years=expiry_years,
            min_expiry_days=min_expiry_days,
            max_expiry_days=max_expiry_days,
        )
        _add_strike_reasons(reasons, strike)
        _add_option_type_reasons(reasons, right, option_type)
        if require_iv_for_seed:
            _add_iv_reasons(reasons, iv)
        if require_vega_for_objective:
            _add_vega_reasons(reasons, vega)
        _add_moneyness_reasons(
            reasons,
            strike=strike,
            spot=spot,
            min_moneyness=min_moneyness,
            max_moneyness=max_moneyness,
        )
        _add_price_bound_reason(
            reasons,
            market_data=market_data,
            mid=mid,
            strike=strike,
            expiry_years=expiry_years,
            option_type=option_type,
        )

        all_reasons.append(tuple(_dedupe_strings(reasons)))
    return all_reasons


def _add_mid_reasons(reasons: list[str], mid: _NumericValue) -> None:
    if mid.missing:
        reasons.append("missing_mid")
    elif not mid.finite:
        reasons.append("nonfinite_mid")
    elif cast(float, mid.value) <= 0.0:
        reasons.append("nonpositive_mid")


def _add_bid_ask_reasons(
    reasons: list[str],
    bid: _NumericValue,
    ask: _NumericValue,
    mid: _NumericValue,
    objective_type: HestonObjectiveType,
) -> None:
    if bid.missing or not bid.finite or cast(float, bid.value or 0.0) < 0.0:
        reasons.append("missing_bid")
    if ask.missing or not ask.finite or cast(float, ask.value or 0.0) < 0.0:
        reasons.append("missing_ask")
    if not bid.usable or not ask.usable:
        return

    bid_value = cast(float, bid.value)
    ask_value = cast(float, ask.value)
    if ask_value < bid_value:
        reasons.append("crossed_market")
        return
    if objective_type == "bid_ask_normalized" and ask_value <= bid_value:
        reasons.append("nonpositive_spread")
    if mid.usable:
        mid_value = cast(float, mid.value)
        if mid_value < bid_value - _BID_ASK_TOL or mid_value > ask_value + _BID_ASK_TOL:
            reasons.append("mid_outside_bid_ask")


def _add_expiry_reasons(
    reasons: list[str],
    *,
    row: pd.Series,
    expiry_years: _NumericValue,
    min_expiry_days: float,
    max_expiry_days: float | None,
) -> None:
    if _is_missing(row["expiry"]):
        reasons.append("missing_expiry")
    if not expiry_years.usable or cast(float, expiry_years.value) <= 0.0:
        reasons.append("nonpositive_time_to_expiry")
        return

    expiry_days = cast(float, expiry_years.value) * _ACT_365_DAYS
    if expiry_days < min_expiry_days:
        reasons.append("short_expiry")
    if max_expiry_days is not None and expiry_days > max_expiry_days:
        reasons.append("long_expiry")


def _add_strike_reasons(reasons: list[str], strike: _NumericValue) -> None:
    if strike.missing or not strike.finite:
        reasons.append("missing_strike")
    elif cast(float, strike.value) <= 0.0:
        reasons.append("nonpositive_strike")


def _add_option_type_reasons(
    reasons: list[str],
    right: str | None,
    option_type: str | None,
) -> None:
    if right is None or option_type is None:
        reasons.append("missing_option_type")
        return
    if (
        right not in _VALID_OPTION_TYPES
        or option_type not in _VALID_OPTION_TYPES
        or right != option_type
    ):
        reasons.append("invalid_option_type")


def _add_iv_reasons(reasons: list[str], iv: _NumericValue) -> None:
    if iv.missing:
        reasons.append("missing_iv_for_seed")
    elif not iv.finite:
        reasons.append("nonfinite_iv")
    elif cast(float, iv.value) <= 0.0:
        reasons.append("nonpositive_iv")


def _add_vega_reasons(reasons: list[str], vega: _NumericValue) -> None:
    if vega.missing:
        reasons.append("missing_vega_for_objective")
    elif not vega.finite:
        reasons.append("nonfinite_vega")
    elif cast(float, vega.value) <= 0.0:
        reasons.append("nonpositive_vega")


def _add_moneyness_reasons(
    reasons: list[str],
    *,
    strike: _NumericValue,
    spot: float,
    min_moneyness: float | None,
    max_moneyness: float | None,
) -> None:
    if not strike.usable or cast(float, strike.value) <= 0.0:
        return
    moneyness = cast(float, strike.value) / spot
    if not math.isfinite(moneyness):
        reasons.append("extreme_moneyness")
        return
    if min_moneyness is not None and moneyness < min_moneyness:
        reasons.append("extreme_moneyness")
    if max_moneyness is not None and moneyness > max_moneyness:
        reasons.append("extreme_moneyness")


def _add_price_bound_reason(
    reasons: list[str],
    *,
    market_data: MarketData,
    mid: _NumericValue,
    strike: _NumericValue,
    expiry_years: _NumericValue,
    option_type: str | None,
) -> None:
    if (
        not mid.usable
        or not strike.usable
        or not expiry_years.usable
        or cast(float, mid.value) <= 0.0
        or cast(float, strike.value) <= 0.0
        or cast(float, expiry_years.value) <= 0.0
        or option_type not in _VALID_OPTION_TYPES
    ):
        return

    tau = cast(float, expiry_years.value)
    discount = market_data.df(tau)
    forward = market_data.forward(tau)
    strike_value = cast(float, strike.value)
    if option_type == "call":
        lower = discount * max(forward - strike_value, 0.0)
        upper = discount * forward
    else:
        lower = discount * max(strike_value - forward, 0.0)
        upper = discount * strike_value

    mid_value = cast(float, mid.value)
    if mid_value < lower - _PRICE_BOUND_TOL or mid_value > upper + _PRICE_BOUND_TOL:
        reasons.append("price_bound_violation")


def _selected_quotes_frame(
    frame: pd.DataFrame,
    reject_reasons: list[tuple[str, ...]],
) -> pd.DataFrame:
    selected_mask = [not reasons for reasons in reject_reasons]
    selected = frame.loc[selected_mask, list(HESTON_QUOTES_COLUMNS)].reset_index(
        drop=True,
    )
    coerced = coerce_frame(selected, DatasetName.HESTON_QUOTES, allow_extra=False)
    validate_dtypes(coerced, DatasetName.HESTON_QUOTES, allow_extra=False)
    return coerced.loc[:, list(HESTON_QUOTES_COLUMNS)].reset_index(drop=True)


def _rejected_quotes_frame(
    frame: pd.DataFrame,
    reject_reasons: list[tuple[str, ...]],
) -> pd.DataFrame:
    rejected_mask = [bool(reasons) for reasons in reject_reasons]
    rejected = frame.loc[rejected_mask].reset_index(drop=True).copy(deep=True)
    rejected["reject_reasons"] = [reasons for reasons in reject_reasons if reasons]
    if "reject_reasons" not in rejected:
        rejected["reject_reasons"] = pd.Series(dtype=object)
    return rejected


def _quote_set_from_selected_quotes(
    selected_quotes: pd.DataFrame,
    market_data: MarketData,
) -> HestonQuoteSet:
    try:
        return heston_quote_set_from_frame(selected_quotes, market_data)
    except Exception as exc:
        raise ValueError(
            "Failed to build HestonQuoteSet from selected Heston quotes: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _quote_set_warnings(quote_set: HestonQuoteSet) -> tuple[str, ...]:
    metadata = quote_set.metadata
    if not metadata:
        return ()
    warnings = metadata.get("optional_data_warnings")
    if not isinstance(warnings, tuple):
        return ()
    return tuple(str(warning) for warning in warnings)


def _heston_ready_stats(
    *,
    input_quote_count: int,
    selected_quotes: pd.DataFrame,
    rejected_quotes: pd.DataFrame,
    rejection_counts: dict[str, int],
    warnings: tuple[str, ...],
) -> HestonReadyStats:
    expiry_days = _selected_expiry_days(selected_quotes)
    return HestonReadyStats(
        input_quote_count=int(input_quote_count),
        selected_quote_count=int(len(selected_quotes)),
        rejected_quote_count=int(len(rejected_quotes)),
        rejection_counts=dict(rejection_counts),
        expiry_count=int(np.unique(expiry_days).size),
        min_expiry_days=(None if expiry_days.size == 0 else float(np.min(expiry_days))),
        max_expiry_days=(None if expiry_days.size == 0 else float(np.max(expiry_days))),
        warnings=warnings,
    )


def _selected_expiry_days(selected_quotes: pd.DataFrame) -> np.ndarray:
    if selected_quotes.empty:
        return np.asarray([], dtype=np.float64)
    expiry_years = selected_quotes["expiry_years"].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    expiry_days = expiry_years[np.isfinite(expiry_years)] * _ACT_365_DAYS
    return np.asarray(expiry_days, dtype=np.float64)


def _rejection_counts(
    reject_reasons: list[tuple[str, ...]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for reasons in reject_reasons:
        for reason in reasons:
            counts[reason] = counts.get(reason, 0) + 1

    ordered: dict[str, int] = {}
    for reason in _REJECTION_REASON_ORDER:
        if reason in counts:
            ordered[reason] = counts[reason]
    for reason in sorted(set(counts) - set(ordered)):
        ordered[reason] = counts[reason]
    return ordered


def _numeric_value(value: object) -> _NumericValue:
    if _is_missing(value):
        return _NumericValue(value=None, missing=True, finite=False)
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return _NumericValue(value=None, missing=False, finite=False)
    return _NumericValue(
        value=number,
        missing=False,
        finite=math.isfinite(number),
    )


def _text_value(value: object) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip().lower()
    return text or None


def _is_missing(value: object) -> bool:
    try:
        result = pd.isna(cast(Any, value))
    except (TypeError, ValueError):
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


def _resolve_require_vega_for_objective(
    objective_type: HestonObjectiveType,
    require_vega_for_objective: bool | None,
) -> bool:
    objective_requires_vega = objective_type == "vega_scaled_price"
    if require_vega_for_objective is None:
        return objective_requires_vega
    require_vega_for_objective = _validate_bool(
        "require_vega_for_objective",
        require_vega_for_objective,
    )
    if objective_requires_vega and not require_vega_for_objective:
        raise ValueError(
            "objective_type='vega_scaled_price' requires finite positive vega; "
            "require_vega_for_objective=False would make the objective impossible "
            "to evaluate"
        )
    return require_vega_for_objective


def _validate_bool(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _validate_nonnegative_float(name: str, value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _validate_optional_nonnegative_float(
    name: str,
    value: float | None,
) -> float | None:
    if value is None:
        return None
    return _validate_nonnegative_float(name, value)


def _validate_optional_positive_float(
    name: str,
    value: float | None,
) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric when provided") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive when provided")
    return number


def _dedupe_strings(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return tuple(seen)


__all__ = [
    "HestonReadyStats",
    "PreparedHestonMarketFit",
    "prepare_heston_market_fit",
]
