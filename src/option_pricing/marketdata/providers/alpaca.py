"""Alpaca market data provider shell for latest equity quotes."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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


class _StockDataClient(Protocol):
    def get_stock_latest_quote(self, request_params: object) -> object:
        """Return latest quote data for the request."""


class AlpacaClient:
    """Small injectable client for Alpaca latest equity quotes."""

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
