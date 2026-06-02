"""Alpaca market data provider shell for latest equity quotes and equity bars."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from typing import Protocol, cast

from option_pricing.marketdata.config import AlpacaConfig
from option_pricing.marketdata.errors import (
    MarketDataProviderError,
    MissingProviderCredentialError,
    ProviderDataUnavailableError,
    ProviderRequestError,
)


class AlpacaProviderError(MarketDataProviderError):
    """Base error for Alpaca provider failures."""


class AlpacaMissingCredentialsError(
    MissingProviderCredentialError,
    AlpacaProviderError,
):
    """Raised when an Alpaca API key or secret key is missing."""

    def __init__(
        self,
        env_var_name: str | None = None,
        *,
        credential_name: str = "credential",
    ) -> None:
        super().__init__(
            env_var_name,
            provider_name="Alpaca",
            credential_name=credential_name,
        )


class AlpacaRequestError(ProviderRequestError, AlpacaProviderError):
    """Raised when an Alpaca request fails."""

    def __init__(
        self,
        message: str,
        *,
        symbols: Sequence[str] | None = None,
    ) -> None:
        self.symbols = tuple(symbols) if symbols is not None else None
        suffix = f" (symbols={self.symbols!r})" if self.symbols is not None else ""
        super().__init__(f"{message}{suffix}")


class AlpacaDataUnavailableError(ProviderDataUnavailableError, AlpacaProviderError):
    """Raised when Alpaca returns no usable data for the requested quote."""


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


class _StockDataClient(Protocol):
    def get_stock_latest_quote(self, request_params: object) -> object:
        """Return latest quote data for the request."""

    def get_stock_bars(self, request_params: object) -> object:
        """Return historical stock bars for the request."""


class AlpacaClient:
    """Small injectable client for Alpaca equity market data."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        config: AlpacaConfig | None = None,
        stock_data_client: _StockDataClient | None = None,
    ) -> None:
        self.config = config or AlpacaConfig()
        self._api_key = _clean_credential(
            api_key,
            credential_name="API key",
        )
        self._secret_key = _clean_credential(
            secret_key,
            credential_name="secret key",
        )
        self._stock_data_client = stock_data_client

    @classmethod
    def from_env(
        cls,
        config: AlpacaConfig | None = None,
        stock_data_client: _StockDataClient | None = None,
    ) -> AlpacaClient:
        """Create a client from the environment variables named by ``AlpacaConfig``."""

        resolved_config = config or AlpacaConfig()
        api_key = os.environ.get(resolved_config.api_key_env)
        if api_key is None or not api_key.strip():
            raise AlpacaMissingCredentialsError(
                resolved_config.api_key_env,
                credential_name="API key",
            )

        secret_key = os.environ.get(resolved_config.secret_key_env)
        if secret_key is None or not secret_key.strip():
            raise AlpacaMissingCredentialsError(
                resolved_config.secret_key_env,
                credential_name="secret key",
            )

        return cls(
            api_key,
            secret_key,
            config=resolved_config,
            stock_data_client=stock_data_client,
        )

    def get_latest_equity_quotes(
        self,
        symbols: str | Sequence[str],
        *,
        asof: object | None = None,
    ) -> dict[str, object]:
        """Fetch latest quote data for one or more equity symbols."""

        cleaned_symbols = _clean_symbols(symbols)
        stock_client = self._resolve_stock_data_client()
        request = self._latest_quote_request(cleaned_symbols)

        getter = getattr(stock_client, "get_stock_latest_quote", None)
        if not callable(getter):
            raise TypeError(
                "Alpaca stock_data_client must provide a callable "
                "get_stock_latest_quote method"
            )

        try:
            response = getter(request)
        except Exception:
            raise AlpacaRequestError(
                "Alpaca latest equity quote request failed",
                symbols=cleaned_symbols,
            ) from None

        quotes = _coerce_quote_mapping(response, cleaned_symbols)
        return {
            "symbols": cleaned_symbols,
            "quotes": quotes,
            "source": "alpaca",
            "feed": self.config.feed,
            "asof": asof,
        }

    def get_equity_bars(
        self,
        symbols: str | Sequence[str],
        *,
        start: datetime | str,
        end: datetime | str,
        timeframe: str,
        limit: int | None = None,
        adjustment: str | None = None,
        sort: str | None = "asc",
        feed: str | None = None,
        asof: str | None = None,
    ) -> dict[str, object]:
        """Fetch historical equity bars for one or more symbols."""

        cleaned_symbols = _clean_symbols(symbols)
        normalized_start = _normalize_datetime(start, "start")
        normalized_end = _normalize_datetime(end, "end")
        if normalized_start >= normalized_end:
            raise ValueError("start must be before end for Alpaca equity bars")

        cleaned_timeframe = _clean_required_text(timeframe, "timeframe")
        cleaned_limit = _clean_limit(limit)
        cleaned_adjustment = _clean_optional_text(adjustment, "adjustment")
        cleaned_sort = _clean_optional_text(sort, "sort")
        effective_feed = _clean_optional_text(feed, "feed") or self.config.feed
        cleaned_asof = _clean_optional_text(asof, "asof")

        stock_client = self._resolve_stock_data_client()
        request = self._equity_bars_request(
            cleaned_symbols,
            start=normalized_start,
            end=normalized_end,
            timeframe=cleaned_timeframe,
            limit=cleaned_limit,
            adjustment=cleaned_adjustment,
            sort=cleaned_sort,
            feed=effective_feed,
            asof=cleaned_asof,
        )

        getter = getattr(stock_client, "get_stock_bars", None)
        if not callable(getter):
            raise TypeError(
                "Alpaca stock_data_client must provide a callable "
                "get_stock_bars method"
            )

        try:
            response = getter(request)
        except Exception:
            raise AlpacaRequestError(
                "Alpaca equity bars request failed",
                symbols=cleaned_symbols,
            ) from None

        return {
            "symbols": cleaned_symbols,
            "bars": getattr(response, "data", response),
            "source": "alpaca",
            "feed": effective_feed,
            "timeframe": cleaned_timeframe,
            "start": normalized_start,
            "end": normalized_end,
            "asof": cleaned_asof,
        }

    def _resolve_stock_data_client(self) -> object:
        if self._stock_data_client is not None:
            return self._stock_data_client

        try:
            client_cls, _, _ = _alpaca_sdk_objects()
            return client_cls(
                api_key=self._api_key,
                secret_key=self._secret_key,
                sandbox=self.config.sandbox,
            )
        except AlpacaProviderError:
            raise
        except Exception:
            raise AlpacaRequestError(
                "Could not construct Alpaca stock data client"
            ) from None

    def _latest_quote_request(self, symbols: tuple[str, ...]) -> object:
        if self._stock_data_client is not None:
            return _LatestQuoteRequest(
                symbol_or_symbols=symbols,
                feed=self.config.feed,
            )

        try:
            _, request_cls, data_feed_factory = _alpaca_sdk_objects()
            feed = data_feed_factory(self.config.feed)
            return request_cls(symbol_or_symbols=list(symbols), feed=feed)
        except AlpacaProviderError:
            raise
        except Exception:
            raise AlpacaRequestError(
                "Could not construct Alpaca latest quote request",
                symbols=symbols,
            ) from None

    def _equity_bars_request(
        self,
        symbols: tuple[str, ...],
        *,
        start: datetime,
        end: datetime,
        timeframe: str,
        limit: int | None,
        adjustment: str | None,
        sort: str | None,
        feed: str,
        asof: str | None,
    ) -> object:
        if self._stock_data_client is not None:
            return _EquityBarsRequest(
                symbol_or_symbols=symbols,
                start=start,
                end=end,
                timeframe=timeframe,
                limit=limit,
                adjustment=adjustment,
                sort=sort,
                feed=feed,
                asof=asof,
            )

        try:
            (
                request_cls,
                timeframe_factory,
                data_feed_factory,
                adjustment_factory,
                sort_factory,
            ) = _alpaca_bars_sdk_objects()
            request_kwargs: dict[str, object] = {
                "symbol_or_symbols": list(symbols),
                "start": start,
                "end": end,
                "timeframe": _alpaca_timeframe(timeframe, timeframe_factory),
                "feed": data_feed_factory(feed),
            }
            if limit is not None:
                request_kwargs["limit"] = limit
            if adjustment is not None:
                request_kwargs["adjustment"] = _optional_sdk_enum(
                    adjustment,
                    adjustment_factory,
                )
            if sort is not None:
                request_kwargs["sort"] = _optional_sdk_enum(sort, sort_factory)
            if asof is not None:
                request_kwargs["asof"] = asof
            return request_cls(**request_kwargs)
        except AlpacaProviderError:
            raise
        except Exception:
            raise AlpacaRequestError(
                "Could not construct Alpaca equity bars request",
                symbols=symbols,
            ) from None

    def __repr__(self) -> str:
        injected = self._stock_data_client is not None
        return (
            "AlpacaClient("
            f"config={self.config!r}, "
            "api_key=<redacted>, "
            "secret_key=<redacted>, "
            f"stock_data_client_injected={injected!r}"
            ")"
        )


def _alpaca_sdk_objects() -> (
    tuple[Callable[..., object], Callable[..., object], Callable[[str], object]]
):
    try:
        historical_module = import_module("alpaca.data.historical")
        requests_module = import_module("alpaca.data.requests")
        enums_module = import_module("alpaca.data.enums")
    except ModuleNotFoundError:
        raise AlpacaRequestError(
            "alpaca-py is required to fetch Alpaca latest equity quotes"
        ) from None

    client_cls = getattr(historical_module, "StockHistoricalDataClient", None)
    request_cls = getattr(requests_module, "StockLatestQuoteRequest", None)
    data_feed_factory = getattr(enums_module, "DataFeed", None)

    if not callable(client_cls):
        raise AlpacaRequestError("alpaca-py StockHistoricalDataClient is unavailable")
    if not callable(request_cls):
        raise AlpacaRequestError("alpaca-py StockLatestQuoteRequest is unavailable")
    if not callable(data_feed_factory):
        raise AlpacaRequestError("alpaca-py DataFeed enum is unavailable")

    return (
        cast(Callable[..., object], client_cls),
        cast(Callable[..., object], request_cls),
        cast(Callable[[str], object], data_feed_factory),
    )


def _alpaca_bars_sdk_objects() -> tuple[
    Callable[..., object],
    Callable[..., object],
    Callable[[str], object],
    Callable[[str], object] | None,
    Callable[[str], object] | None,
]:
    try:
        requests_module = import_module("alpaca.data.requests")
        enums_module = import_module("alpaca.data.enums")
        timeframe_module = import_module("alpaca.data.timeframe")
    except ModuleNotFoundError:
        raise AlpacaRequestError(
            "alpaca-py is required to fetch Alpaca equity bars"
        ) from None

    request_cls = getattr(requests_module, "StockBarsRequest", None)
    timeframe_factory = getattr(timeframe_module, "TimeFrame", None)
    data_feed_factory = getattr(enums_module, "DataFeed", None)
    adjustment_factory = getattr(enums_module, "Adjustment", None)
    sort_factory = getattr(enums_module, "Sort", None)

    if not callable(request_cls):
        raise AlpacaRequestError("alpaca-py StockBarsRequest is unavailable")
    if timeframe_factory is None:
        raise AlpacaRequestError("alpaca-py TimeFrame is unavailable")
    if not callable(data_feed_factory):
        raise AlpacaRequestError("alpaca-py DataFeed enum is unavailable")

    return (
        cast(Callable[..., object], request_cls),
        cast(Callable[..., object], timeframe_factory),
        cast(Callable[[str], object], data_feed_factory),
        cast(Callable[[str], object] | None, adjustment_factory),
        cast(Callable[[str], object] | None, sort_factory),
    )


def _alpaca_timeframe(value: str, timeframe_factory: Callable[..., object]) -> object:
    normalized = value.strip().lower().replace("_", "").replace(" ", "")
    timeframe_attrs = {
        "1min": "Minute",
        "1minute": "Minute",
        "minute": "Minute",
        "min": "Minute",
        "1hour": "Hour",
        "hour": "Hour",
        "1h": "Hour",
        "1day": "Day",
        "day": "Day",
        "daily": "Day",
        "1d": "Day",
        "1week": "Week",
        "week": "Week",
        "weekly": "Week",
        "1month": "Month",
        "month": "Month",
        "monthly": "Month",
    }
    attr_name = timeframe_attrs.get(normalized)
    if attr_name is not None and hasattr(timeframe_factory, attr_name):
        return getattr(timeframe_factory, attr_name)

    try:
        return timeframe_factory(value)
    except Exception:
        return value


def _optional_sdk_enum(
    value: str,
    factory: Callable[[str], object] | None,
) -> object:
    if factory is None:
        return value
    return factory(value)


def _coerce_quote_mapping(
    response: object,
    symbols: tuple[str, ...],
) -> dict[str, object]:
    data = getattr(response, "data", response)
    if not isinstance(data, Mapping):
        raise AlpacaDataUnavailableError(
            "Alpaca latest equity quote response must be a mapping"
        )

    quotes = {str(symbol): quote for symbol, quote in data.items()}
    missing = [symbol for symbol in symbols if symbol not in quotes]
    if missing:
        raise AlpacaDataUnavailableError(
            f"Alpaca latest equity quote response is missing symbols: {missing}"
        )
    return quotes


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


__all__ = [
    "AlpacaClient",
    "AlpacaDataUnavailableError",
    "AlpacaMissingCredentialsError",
    "AlpacaProviderError",
    "AlpacaRequestError",
]
