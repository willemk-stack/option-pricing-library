"""Internal Alpaca request contracts and validation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from option_pricing.marketdata.providers._alpaca_errors import (
    AlpacaMissingCredentialsError,
)


@dataclass(frozen=True, slots=True)
class _LatestQuoteRequest:
    symbol_or_symbols: tuple[str, ...]
    feed: str


@dataclass(frozen=True, slots=True)
class _EquityBarsRequest:
    symbol_or_symbols: tuple[str, ...]
    start: datetime
    end: datetime
    timeframe: str
    limit: int | None
    adjustment: str | None
    sort: str | None
    feed: str
    asof: str | None


@dataclass(frozen=True, slots=True)
class _OptionChainRequest:
    underlying: str
    feed: str
    expiry_gte: date
    expiry_lte: date
    strike_gte: float | None
    strike_lte: float | None
    option_type: str | None
    root_symbol: str | None
    updated_since: datetime | None

    @property
    def underlying_symbol(self) -> str:
        return self.underlying

    @property
    def expiration_date_gte(self) -> date:
        return self.expiry_gte

    @property
    def expiration_date_lte(self) -> date:
        return self.expiry_lte

    @property
    def strike_price_gte(self) -> float | None:
        return self.strike_gte

    @property
    def strike_price_lte(self) -> float | None:
        return self.strike_lte

    @property
    def type(self) -> str | None:
        return self.option_type


def _clean_symbols(symbols: str | Sequence[str]) -> tuple[str, ...]:
    raw_symbols: tuple[str, ...]
    if isinstance(symbols, str):
        raw_symbols = (symbols,)
    else:
        raw_symbols = tuple(symbols)

    cleaned = tuple(_clean_symbol(symbol) for symbol in raw_symbols)
    if not cleaned:
        raise ValueError("symbols must contain at least one symbol")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("symbols must not contain duplicates")
    return cleaned


def _option_chain_expiry_bounds(
    expiry_gte: date | str | None,
    expiry_lte: date | str | None,
) -> tuple[date, date]:
    current_date = datetime.now(UTC).date()
    lower = (
        current_date
        if expiry_gte is None
        else _normalize_date(expiry_gte, "expiry_gte")
    )
    upper = (
        current_date + timedelta(days=45)
        if expiry_lte is None
        else _normalize_date(expiry_lte, "expiry_lte")
    )
    if lower > upper:
        raise ValueError("expiry_gte must be less than or equal to expiry_lte")
    return lower, upper


def _normalize_date(value: date | str, field_name: str) -> date:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.date()
        return value.astimezone(UTC).date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        cleaned = _clean_required_text(value, field_name)
        try:
            return date.fromisoformat(cleaned)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO date string") from exc
    raise TypeError(f"{field_name} must be a date or ISO date string")


def _normalize_datetime(value: datetime | str, field_name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        cleaned = _clean_required_text(value, field_name)
        try:
            parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO datetime string") from exc
    else:
        raise TypeError(f"{field_name} must be a datetime or ISO datetime string")

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _normalize_optional_datetime(
    value: datetime | str | None,
    field_name: str,
) -> datetime | None:
    if value is None:
        return None
    return _normalize_datetime(value, field_name)


def _clean_required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    return cleaned


def _clean_optional_text(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _clean_required_text(value, field_name)


def _clean_optional_symbol(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _clean_required_text(value, field_name).upper()


def _clean_underlying_symbol(value: str) -> str:
    return _clean_required_text(value, "underlying").upper()


def _clean_optional_float(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric")
    try:
        cleaned = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be numeric") from exc
    if not cleaned == cleaned or cleaned in (float("inf"), float("-inf")):
        raise ValueError(f"{field_name} must be finite")
    return cleaned


def _clean_option_type(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = _clean_required_text(value, "option_type").lower()
    if cleaned not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'")
    return cleaned


def _option_chain_request_metadata(
    underlying: str,
    *,
    feed: str,
    expiry_gte: date,
    expiry_lte: date,
    strike_gte: float | None,
    strike_lte: float | None,
    option_type: str | None,
    root_symbol: str | None,
    updated_since: datetime | None,
) -> dict[str, object]:
    return {
        "underlying": underlying,
        "feed": feed,
        "expiry_gte": expiry_gte,
        "expiry_lte": expiry_lte,
        "strike_gte": strike_gte,
        "strike_lte": strike_lte,
        "option_type": option_type,
        "root_symbol": root_symbol,
        "updated_since": updated_since,
    }


def _clean_limit(value: int | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("limit must be an integer")
    if value <= 0:
        raise ValueError("limit must be a positive integer")
    return value


def _clean_symbol(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("symbols must contain strings")
    cleaned = value.strip().upper()
    if not cleaned:
        raise ValueError("symbols must contain non-empty strings")
    return cleaned


def _clean_credential(value: str, *, credential_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Alpaca {credential_name} must be a string")
    if not value.strip():
        raise AlpacaMissingCredentialsError(credential_name=credential_name)
    return value
