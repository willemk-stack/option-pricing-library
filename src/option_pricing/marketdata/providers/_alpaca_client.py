"""High-level Alpaca market data client."""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import date, datetime

from option_pricing.marketdata.config import AlpacaConfig
from option_pricing.marketdata.providers._alpaca_errors import (
    AlpacaMissingCredentialsError,
    AlpacaProviderError,
    AlpacaRequestError,
)
from option_pricing.marketdata.providers._alpaca_payloads import _coerce_quote_mapping
from option_pricing.marketdata.providers._alpaca_protocols import (
    _OptionDataClient,
    _StockDataClient,
)
from option_pricing.marketdata.providers._alpaca_requests import (
    _clean_credential,
    _clean_limit,
    _clean_option_type,
    _clean_optional_float,
    _clean_optional_symbol,
    _clean_optional_text,
    _clean_required_text,
    _clean_symbols,
    _clean_underlying_symbol,
    _EquityBarsRequest,
    _LatestQuoteRequest,
    _normalize_datetime,
    _normalize_optional_datetime,
    _option_chain_expiry_bounds,
    _option_chain_request_metadata,
    _OptionChainRequest,
)
from option_pricing.marketdata.providers._alpaca_sdk import (
    _alpaca_bars_sdk_objects,
    _alpaca_option_sdk_objects,
    _alpaca_sdk_objects,
    _alpaca_timeframe,
    _optional_sdk_enum,
)


class AlpacaClient:
    """Small injectable client for Alpaca market data."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        config: AlpacaConfig | None = None,
        stock_data_client: _StockDataClient | None = None,
        option_data_client: _OptionDataClient | None = None,
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
        self._option_data_client = option_data_client

    @classmethod
    def from_env(
        cls,
        config: AlpacaConfig | None = None,
        stock_data_client: _StockDataClient | None = None,
        option_data_client: _OptionDataClient | None = None,
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
            option_data_client=option_data_client,
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

    def get_option_chain(
        self,
        underlying: str,
        *,
        expiry_gte: date | str | None = None,
        expiry_lte: date | str | None = None,
        strike_gte: float | None = None,
        strike_lte: float | None = None,
        option_type: str | None = None,
        root_symbol: str | None = None,
        updated_since: datetime | str | None = None,
        feed: str | None = None,
        asof: object | None = None,
    ) -> dict[str, object]:
        """Fetch option chain snapshots for one underlying symbol."""

        cleaned_underlying = _clean_underlying_symbol(underlying)
        effective_feed = _clean_optional_text(feed, "feed") or self.config.feed
        normalized_expiry_gte, normalized_expiry_lte = _option_chain_expiry_bounds(
            expiry_gte,
            expiry_lte,
        )
        normalized_strike_gte = _clean_optional_float(strike_gte, "strike_gte")
        normalized_strike_lte = _clean_optional_float(strike_lte, "strike_lte")
        if (
            normalized_strike_gte is not None
            and normalized_strike_lte is not None
            and normalized_strike_gte > normalized_strike_lte
        ):
            raise ValueError("strike_gte must be less than or equal to strike_lte")

        cleaned_option_type = _clean_option_type(option_type)
        cleaned_root_symbol = _clean_optional_symbol(root_symbol, "root_symbol")
        normalized_updated_since = _normalize_optional_datetime(
            updated_since,
            "updated_since",
        )

        option_client = self._resolve_option_data_client()
        request = self._option_chain_request(
            cleaned_underlying,
            feed=effective_feed,
            expiry_gte=normalized_expiry_gte,
            expiry_lte=normalized_expiry_lte,
            strike_gte=normalized_strike_gte,
            strike_lte=normalized_strike_lte,
            option_type=cleaned_option_type,
            root_symbol=cleaned_root_symbol,
            updated_since=normalized_updated_since,
        )

        getter = getattr(option_client, "get_option_chain", None)
        if not callable(getter):
            raise TypeError(
                "Alpaca option_data_client must provide a callable "
                "get_option_chain method"
            )

        try:
            response = getter(request)
        except Exception:
            raise AlpacaRequestError(
                "Alpaca option chain request failed",
                symbols=(cleaned_underlying,),
            ) from None

        request_metadata = _option_chain_request_metadata(
            cleaned_underlying,
            feed=effective_feed,
            expiry_gte=normalized_expiry_gte,
            expiry_lte=normalized_expiry_lte,
            strike_gte=normalized_strike_gte,
            strike_lte=normalized_strike_lte,
            option_type=cleaned_option_type,
            root_symbol=cleaned_root_symbol,
            updated_since=normalized_updated_since,
        )
        return {
            "underlying": cleaned_underlying,
            "contracts": getattr(response, "data", response),
            "source": "alpaca",
            "feed": effective_feed,
            "request": request_metadata,
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

    def _resolve_option_data_client(self) -> object:
        if self._option_data_client is not None:
            return self._option_data_client

        try:
            client_cls, _, _, _ = _alpaca_option_sdk_objects()
            return client_cls(
                api_key=self._api_key,
                secret_key=self._secret_key,
                sandbox=self.config.sandbox,
            )
        except AlpacaProviderError:
            raise
        except Exception:
            raise AlpacaRequestError(
                "Could not construct Alpaca option data client"
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

    def _option_chain_request(
        self,
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
    ) -> object:
        if self._option_data_client is not None:
            return _OptionChainRequest(
                underlying=underlying,
                feed=feed,
                expiry_gte=expiry_gte,
                expiry_lte=expiry_lte,
                strike_gte=strike_gte,
                strike_lte=strike_lte,
                option_type=option_type,
                root_symbol=root_symbol,
                updated_since=updated_since,
            )

        try:
            (
                _,
                request_cls,
                options_feed_factory,
                contract_type_factory,
            ) = _alpaca_option_sdk_objects()
            request_kwargs: dict[str, object] = {
                "underlying_symbol": underlying,
                "feed": _optional_sdk_enum(feed, options_feed_factory),
                "expiration_date_gte": expiry_gte,
                "expiration_date_lte": expiry_lte,
            }
            if strike_gte is not None:
                request_kwargs["strike_price_gte"] = strike_gte
            if strike_lte is not None:
                request_kwargs["strike_price_lte"] = strike_lte
            if option_type is not None:
                request_kwargs["type"] = _optional_sdk_enum(
                    option_type,
                    contract_type_factory,
                )
            if root_symbol is not None:
                request_kwargs["root_symbol"] = root_symbol
            if updated_since is not None:
                request_kwargs["updated_since"] = updated_since
            return request_cls(**request_kwargs)
        except AlpacaProviderError:
            raise
        except Exception:
            raise AlpacaRequestError(
                "Could not construct Alpaca option chain request",
                symbols=(underlying,),
            ) from None

    def __repr__(self) -> str:
        stock_injected = self._stock_data_client is not None
        option_injected = self._option_data_client is not None
        return (
            "AlpacaClient("
            f"config={self.config!r}, "
            "api_key=<redacted>, "
            "secret_key=<redacted>, "
            f"stock_data_client_injected={stock_injected!r}, "
            f"option_data_client_injected={option_injected!r}"
            ")"
        )
