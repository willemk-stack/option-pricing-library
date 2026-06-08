"""Surface-ready preparation helpers for model-validation bundles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from option_pricing.marketdata.schemas import SURFACE_INPUTS_COLUMNS
from option_pricing.types import MarketData

if TYPE_CHECKING:
    from option_pricing.marketdata.bundles import LoadedModelValidationBundle

_ACT_365_DAYS = 365.0
_SECONDS_PER_DAY = 24 * 60 * 60
_VALID_RIGHTS = frozenset({"call", "put"})
_COMPUTABLE_EXPIRY_COLUMN = "expiry_years"
_SVI_COMPUTED_COLUMNS = (
    "expiry_days",
    "forward",
    "discount",
    "log_moneyness",
    "total_variance",
    "sqrt_weight",
    "option_type",
    "is_call",
)
_REQUIRED_SURFACE_COLUMNS_EXCEPT_COMPUTED = tuple(
    column for column in SURFACE_INPUTS_COLUMNS if column != _COMPUTABLE_EXPIRY_COLUMN
)
_REJECTION_REASON_ORDER = (
    "missing_required_column",
    "missing_expiry",
    "nonpositive_time_to_expiry",
    "short_expiry",
    "long_expiry",
    "missing_strike",
    "nonpositive_strike",
    "missing_iv",
    "nonfinite_iv",
    "nonpositive_iv",
    "missing_mid",
    "nonfinite_mid",
    "nonpositive_mid",
    "missing_right",
    "invalid_right",
    "extreme_moneyness",
    "sparse_expiry",
)
_PREPARE_SVI_MARKET_FIT_GUIDANCE = (
    "Use prepare_svi_market_fit(bundle) before SVI fitting from saved "
    "bundle surface artifacts."
)
_PREPARE_ESSVI_MARKET_FIT_GUIDANCE = (
    "Use prepare_essvi_market_fit(bundle) before eSSVI fitting from saved "
    "bundle surface artifacts."
)


@dataclass(frozen=True, slots=True)
class SurfaceReadyStats:
    input_point_count: int
    selected_point_count: int
    rejected_point_count: int
    rejection_counts: dict[str, int]
    expiry_count: int
    min_expiry_days: float | None
    max_expiry_days: float | None
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreparedSVIMarketFit:
    model_name: str
    market_data: MarketData
    selected_points: pd.DataFrame
    rejected_points: pd.DataFrame
    stats: SurfaceReadyStats
    status: str
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreparedESSVIMarketFit:
    model_name: str
    market_data: MarketData
    selected_points: pd.DataFrame
    rejected_points: pd.DataFrame
    stats: SurfaceReadyStats
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


def prepare_svi_market_fit(
    bundle: LoadedModelValidationBundle,
    *,
    min_expiry_days: float = 7.0,
    max_expiry_days: float | None = None,
    min_points_per_expiry: int = 5,
    min_moneyness: float | None = 0.5,
    max_moneyness: float | None = 2.0,
    require_iv: bool = True,
    require_mid: bool = False,
    raise_on_block: bool = False,
) -> PreparedSVIMarketFit:
    """Prepare a loaded model-validation bundle for future SVI fitting.

    This helper starts from ``bundle.surface_inputs`` and computes SVI-specific
    point fields in memory. It does not fit SVI parameters or mutate the
    persisted ``surface_inputs.v1`` artifact.

    ``raise_on_block`` is reserved for future structural SVI preparation
    blocks. Today this helper returns ``ready`` when points are selected and
    ``empty`` when selection filters leave no usable points.
    """

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
    min_points_per_expiry = _validate_positive_int(
        "min_points_per_expiry",
        min_points_per_expiry,
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
    require_iv = _validate_bool("require_iv", require_iv)
    require_mid = _validate_bool("require_mid", require_mid)
    raise_on_block = _validate_bool("raise_on_block", raise_on_block)

    market_data = _bundle_market_data(bundle)
    spot = _positive_finite_market_value(market_data, "spot")
    source = _bundle_surface_inputs(bundle)
    _validate_required_surface_columns(source)
    working = _with_expiry_years(source)

    reject_reason_lists = _svi_rejection_reason_lists(
        working,
        spot=spot,
        min_expiry_days=min_expiry_days,
        max_expiry_days=max_expiry_days,
        min_moneyness=min_moneyness,
        max_moneyness=max_moneyness,
        require_iv=require_iv,
        require_mid=require_mid,
    )
    _add_sparse_expiry_reasons(
        working,
        reject_reason_lists,
        min_points_per_expiry=min_points_per_expiry,
    )
    enriched = _with_svi_fields(working, market_data)
    reject_reasons = [
        tuple(_dedupe_strings(reasons)) for reasons in reject_reason_lists
    ]
    selected_points = _selected_points_frame(enriched, reject_reasons)
    rejected_points = _rejected_points_frame(enriched, reject_reasons)
    rejection_counts = _rejection_counts(reject_reasons)

    warnings: tuple[str, ...] = ()
    status = "empty" if selected_points.empty else "ready"
    if status == "blocked" and raise_on_block:
        raise ValueError("SVI preparation blocked calibration.")

    stats = _surface_ready_stats(
        input_point_count=len(working),
        selected_points=selected_points,
        rejected_points=rejected_points,
        rejection_counts=rejection_counts,
        warnings=warnings,
    )
    return PreparedSVIMarketFit(
        model_name="svi",
        market_data=market_data,
        selected_points=selected_points,
        rejected_points=rejected_points,
        stats=stats,
        status=status,
        warnings=warnings,
    )


def prepare_essvi_market_fit(
    bundle: LoadedModelValidationBundle,
    *,
    min_expiry_days: float = 7.0,
    max_expiry_days: float | None = None,
    min_points_per_expiry: int = 5,
    min_expiry_count: int = 3,
    min_moneyness: float | None = 0.5,
    max_moneyness: float | None = 2.0,
    require_mid: bool = True,
    require_iv: bool = False,
    raise_on_block: bool = False,
) -> PreparedESSVIMarketFit:
    """Prepare a loaded model-validation bundle for future eSSVI fitting.

    This helper starts from ``bundle.surface_inputs`` and computes the
    cross-maturity point fields needed by ``calibrate_essvi_global``. It does
    not fit eSSVI parameters or mutate the persisted ``surface_inputs.v1``
    artifact.
    """

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
    min_points_per_expiry = _validate_positive_int(
        "min_points_per_expiry",
        min_points_per_expiry,
    )
    min_expiry_count = _validate_positive_int(
        "min_expiry_count",
        min_expiry_count,
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
    require_mid = _validate_bool("require_mid", require_mid)
    require_iv = _validate_bool("require_iv", require_iv)
    raise_on_block = _validate_bool("raise_on_block", raise_on_block)

    market_data = _bundle_market_data(bundle)
    spot = _positive_finite_market_value(market_data, "spot")
    source = _bundle_surface_inputs(
        bundle,
        guidance=_PREPARE_ESSVI_MARKET_FIT_GUIDANCE,
    )
    _validate_required_surface_columns(
        source,
        guidance=_PREPARE_ESSVI_MARKET_FIT_GUIDANCE,
    )
    working = _with_expiry_years(source)

    reject_reason_lists = _svi_rejection_reason_lists(
        working,
        spot=spot,
        min_expiry_days=min_expiry_days,
        max_expiry_days=max_expiry_days,
        min_moneyness=min_moneyness,
        max_moneyness=max_moneyness,
        require_iv=require_iv,
        require_mid=require_mid,
    )
    _add_sparse_expiry_reasons(
        working,
        reject_reason_lists,
        min_points_per_expiry=min_points_per_expiry,
    )
    enriched = _with_essvi_fields(working, market_data)
    reject_reasons = [
        tuple(_dedupe_strings(reasons)) for reasons in reject_reason_lists
    ]
    selected_points = _selected_points_frame(enriched, reject_reasons)
    rejected_points = _rejected_points_frame(enriched, reject_reasons)
    rejection_counts = _rejection_counts(reject_reasons)

    expiry_count = _selected_expiry_count(selected_points)
    warnings = _essvi_surface_warnings(
        selected_point_count=len(selected_points),
        expiry_count=expiry_count,
        min_expiry_count=min_expiry_count,
    )
    if selected_points.empty:
        status = "empty"
    elif expiry_count < min_expiry_count:
        status = "blocked"
    else:
        status = "ready"

    if status == "blocked" and raise_on_block:
        raise ValueError(
            "eSSVI preparation blocked global calibration: " + " ".join(warnings)
        )

    stats = _surface_ready_stats(
        input_point_count=len(working),
        selected_points=selected_points,
        rejected_points=rejected_points,
        rejection_counts=rejection_counts,
        warnings=warnings,
    )
    return PreparedESSVIMarketFit(
        model_name="essvi",
        market_data=market_data,
        selected_points=selected_points,
        rejected_points=rejected_points,
        stats=stats,
        status=status,
        warnings=warnings,
    )


def _bundle_surface_inputs(
    bundle: Any,
    *,
    guidance: str = _PREPARE_SVI_MARKET_FIT_GUIDANCE,
) -> pd.DataFrame:
    surface_inputs = getattr(bundle, "surface_inputs", None)
    if not isinstance(surface_inputs, pd.DataFrame):
        raise TypeError(
            "bundle.surface_inputs must be a pandas DataFrame. " f"{guidance}"
        )
    return surface_inputs.copy(deep=True).reset_index(drop=True)


def _bundle_market_data(bundle: Any) -> MarketData:
    market_data = getattr(bundle, "market_data", None)
    if not isinstance(market_data, MarketData):
        raise TypeError(
            "bundle.market_data must be a MarketData instance. "
            "Use load_model_validation_bundle(path) to load bundle market data "
            "before SVI preparation."
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
            "assumptions before SVI preparation."
        )
    return value


def _positive_finite_market_value(market_data: MarketData, field_name: str) -> float:
    value = _finite_market_value(market_data, field_name)
    if value <= 0.0:
        raise ValueError(
            f"bundle.market_data.{field_name} must be positive. "
            "SVI preparation requires a positive spot in bundle.market_data."
        )
    return value


def _validate_required_surface_columns(
    frame: pd.DataFrame,
    *,
    guidance: str = _PREPARE_SVI_MARKET_FIT_GUIDANCE,
) -> None:
    missing = [
        column
        for column in _REQUIRED_SURFACE_COLUMNS_EXCEPT_COMPUTED
        if column not in frame.columns
    ]
    if missing:
        joined = ", ".join(repr(column) for column in missing)
        raise ValueError(
            "bundle.surface_inputs is missing required columns: "
            f"{joined}. surface_inputs is a candidate artifact; "
            f"{guidance}"
        )


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


def _svi_rejection_reason_lists(
    frame: pd.DataFrame,
    *,
    spot: float,
    min_expiry_days: float,
    max_expiry_days: float | None,
    min_moneyness: float | None,
    max_moneyness: float | None,
    require_iv: bool,
    require_mid: bool,
) -> list[list[str]]:
    all_reasons: list[list[str]] = []
    for _, row in frame.iterrows():
        reasons: list[str] = []
        expiry_years = _numeric_value(row[_COMPUTABLE_EXPIRY_COLUMN])
        strike = _numeric_value(row["strike"])
        iv = _numeric_value(row["iv"])
        mid = _numeric_value(row["mid"])
        right = _text_value(row["right"])

        _add_expiry_reasons(
            reasons,
            row=row,
            expiry_years=expiry_years,
            min_expiry_days=min_expiry_days,
            max_expiry_days=max_expiry_days,
        )
        _add_strike_reasons(reasons, strike)
        if require_iv:
            _add_iv_reasons(reasons, iv)
        if require_mid:
            _add_mid_reasons(reasons, mid)
        _add_right_reasons(reasons, right)
        _add_moneyness_reasons(
            reasons,
            strike=strike,
            spot=spot,
            min_moneyness=min_moneyness,
            max_moneyness=max_moneyness,
        )
        all_reasons.append(reasons)
    return all_reasons


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


def _add_iv_reasons(reasons: list[str], iv: _NumericValue) -> None:
    if iv.missing:
        reasons.append("missing_iv")
    elif not iv.finite:
        reasons.append("nonfinite_iv")
    elif cast(float, iv.value) <= 0.0:
        reasons.append("nonpositive_iv")


def _add_mid_reasons(reasons: list[str], mid: _NumericValue) -> None:
    if mid.missing:
        reasons.append("missing_mid")
    elif not mid.finite:
        reasons.append("nonfinite_mid")
    elif cast(float, mid.value) <= 0.0:
        reasons.append("nonpositive_mid")


def _add_right_reasons(reasons: list[str], right: str | None) -> None:
    if right is None:
        reasons.append("missing_right")
        return
    if right not in _VALID_RIGHTS:
        reasons.append("invalid_right")


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


def _add_sparse_expiry_reasons(
    frame: pd.DataFrame,
    reject_reason_lists: list[list[str]],
    *,
    min_points_per_expiry: int,
) -> None:
    expiry_counts: dict[str, int] = {}
    expiry_keys: list[str | None] = []
    for position, (_, row) in enumerate(frame.iterrows()):
        key = _expiry_bucket_key(row)
        expiry_keys.append(key)
        if reject_reason_lists[position] or key is None:
            continue
        expiry_counts[key] = expiry_counts.get(key, 0) + 1

    for position, key in enumerate(expiry_keys):
        if reject_reason_lists[position] or key is None:
            continue
        if expiry_counts.get(key, 0) < min_points_per_expiry:
            reject_reason_lists[position].append("sparse_expiry")


def _expiry_bucket_key(row: pd.Series) -> str | None:
    expiry = row["expiry"]
    if not _is_missing(expiry):
        try:
            timestamp = cast(Any, pd.Timestamp(cast(Any, expiry)))
        except (TypeError, ValueError):
            timestamp = None
        if timestamp is not None and not pd.isna(timestamp):
            timestamp = cast(pd.Timestamp, timestamp)
            if timestamp.tzinfo is None:
                timestamp = timestamp.normalize().tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC").normalize()
            return timestamp.isoformat()

    expiry_years = _numeric_value(row[_COMPUTABLE_EXPIRY_COLUMN])
    if expiry_years.usable and cast(float, expiry_years.value) > 0.0:
        return f"{cast(float, expiry_years.value):.12g}"
    return None


def _with_svi_fields(frame: pd.DataFrame, market_data: MarketData) -> pd.DataFrame:
    enriched = frame.copy(deep=True)
    expiry_years = pd.to_numeric(
        enriched[_COMPUTABLE_EXPIRY_COLUMN],
        errors="coerce",
    ).to_numpy(dtype=np.float64, na_value=np.nan)
    strikes = pd.to_numeric(enriched["strike"], errors="coerce").to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    ivs = pd.to_numeric(enriched["iv"], errors="coerce").to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    forwards: list[float] = []
    discounts: list[float] = []
    log_moneyness: list[float] = []
    total_variance: list[float] = []
    option_types: list[object] = []
    is_call_values: list[object] = []

    for tau, strike, iv, right_value in zip(
        expiry_years,
        strikes,
        ivs,
        enriched["right"],
        strict=True,
    ):
        forward = _market_forward(market_data, tau)
        discount = _market_discount(market_data, tau)
        forwards.append(forward)
        discounts.append(discount)
        log_moneyness.append(_log_forward_moneyness(strike, forward))
        total_variance.append(_total_variance(tau, iv))

        option_type = _text_value(right_value)
        if option_type in _VALID_RIGHTS:
            option_types.append(option_type)
            is_call_values.append(option_type == "call")
        else:
            option_types.append(pd.NA)
            is_call_values.append(pd.NA)

    enriched["expiry_days"] = expiry_years * _ACT_365_DAYS
    enriched["forward"] = forwards
    enriched["discount"] = discounts
    enriched["log_moneyness"] = log_moneyness
    enriched["total_variance"] = total_variance
    enriched["sqrt_weight"] = 1.0
    enriched["option_type"] = option_types
    enriched["is_call"] = is_call_values
    return enriched


def _with_essvi_fields(frame: pd.DataFrame, market_data: MarketData) -> pd.DataFrame:
    enriched = _with_svi_fields(frame, market_data)
    enriched["y"] = enriched["log_moneyness"]
    enriched["T"] = enriched[_COMPUTABLE_EXPIRY_COLUMN]
    enriched["price_mkt"] = pd.to_numeric(
        enriched["mid"],
        errors="coerce",
    ).to_numpy(dtype=np.float64, na_value=np.nan)
    enriched["implied_vol"] = pd.to_numeric(
        enriched["iv"],
        errors="coerce",
    ).to_numpy(dtype=np.float64, na_value=np.nan)
    return enriched


def _market_forward(market_data: MarketData, tau: float) -> float:
    if not math.isfinite(float(tau)) or tau <= 0.0:
        return math.nan
    try:
        value = float(market_data.forward(float(tau)))
    except (TypeError, ValueError, OverflowError):
        return math.nan
    if not math.isfinite(value) or value <= 0.0:
        return math.nan
    return value


def _market_discount(market_data: MarketData, tau: float) -> float:
    if not math.isfinite(float(tau)) or tau <= 0.0:
        return math.nan
    try:
        value = float(market_data.df(float(tau)))
    except (TypeError, ValueError, OverflowError):
        return math.nan
    if not math.isfinite(value) or value <= 0.0:
        return math.nan
    return value


def _log_forward_moneyness(strike: float, forward: float) -> float:
    if (
        not math.isfinite(float(strike))
        or not math.isfinite(float(forward))
        or strike <= 0.0
        or forward <= 0.0
    ):
        return math.nan
    return math.log(float(strike) / float(forward))


def _total_variance(expiry_years: float, iv: float) -> float:
    if (
        not math.isfinite(float(expiry_years))
        or not math.isfinite(float(iv))
        or expiry_years <= 0.0
        or iv <= 0.0
    ):
        return math.nan
    return float(expiry_years) * float(iv) ** 2


def _selected_points_frame(
    frame: pd.DataFrame,
    reject_reasons: list[tuple[str, ...]],
) -> pd.DataFrame:
    selected_mask = [not reasons for reasons in reject_reasons]
    columns = _surface_point_columns(frame)
    return frame.loc[selected_mask, columns].reset_index(drop=True).copy(deep=True)


def _rejected_points_frame(
    frame: pd.DataFrame,
    reject_reasons: list[tuple[str, ...]],
) -> pd.DataFrame:
    rejected_mask = [bool(reasons) for reasons in reject_reasons]
    rejected = frame.loc[rejected_mask, _surface_point_columns(frame)].reset_index(
        drop=True,
    )
    rejected = rejected.copy(deep=True)
    rejected["reject_reasons"] = [reasons for reasons in reject_reasons if reasons]
    if "reject_reasons" not in rejected:
        rejected["reject_reasons"] = pd.Series(dtype=object)
    return rejected


def _surface_point_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in SURFACE_INPUTS_COLUMNS:
        if column in frame.columns:
            columns.append(column)
    for column in frame.columns:
        if column in columns or column in _SVI_COMPUTED_COLUMNS:
            continue
        columns.append(column)
    columns.extend(
        column for column in _SVI_COMPUTED_COLUMNS if column in frame.columns
    )
    return columns


def _surface_ready_stats(
    *,
    input_point_count: int,
    selected_points: pd.DataFrame,
    rejected_points: pd.DataFrame,
    rejection_counts: dict[str, int],
    warnings: tuple[str, ...],
) -> SurfaceReadyStats:
    expiry_days = _selected_expiry_days(selected_points)
    return SurfaceReadyStats(
        input_point_count=int(input_point_count),
        selected_point_count=int(len(selected_points)),
        rejected_point_count=int(len(rejected_points)),
        rejection_counts=dict(rejection_counts),
        expiry_count=int(np.unique(expiry_days).size),
        min_expiry_days=(None if expiry_days.size == 0 else float(np.min(expiry_days))),
        max_expiry_days=(None if expiry_days.size == 0 else float(np.max(expiry_days))),
        warnings=warnings,
    )


def _selected_expiry_count(selected_points: pd.DataFrame) -> int:
    return int(np.unique(_selected_expiry_days(selected_points)).size)


def _essvi_surface_warnings(
    *,
    selected_point_count: int,
    expiry_count: int,
    min_expiry_count: int,
) -> tuple[str, ...]:
    if selected_point_count == 0:
        return (
            "eSSVI preparation selected no surface points; inspect rejected_points "
            "and stats.rejection_counts before fitting.",
        )
    if expiry_count < min_expiry_count:
        return (
            "eSSVI global calibration requires at least "
            f"{min_expiry_count} expiries; selected {expiry_count}.",
        )
    return ()


def _selected_expiry_days(selected_points: pd.DataFrame) -> np.ndarray:
    if selected_points.empty or "expiry_days" not in selected_points.columns:
        return np.asarray([], dtype=np.float64)
    expiry_days = selected_points["expiry_days"].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    return np.asarray(expiry_days[np.isfinite(expiry_days)], dtype=np.float64)


def _rejection_counts(reject_reasons: list[tuple[str, ...]]) -> dict[str, int]:
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


def _validate_bool(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _validate_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an int") from exc
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


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
    "PreparedESSVIMarketFit",
    "PreparedSVIMarketFit",
    "SurfaceReadyStats",
    "prepare_essvi_market_fit",
    "prepare_svi_market_fit",
]
