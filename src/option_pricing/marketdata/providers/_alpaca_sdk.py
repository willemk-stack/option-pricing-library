"""Lazy Alpaca SDK adapters."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import cast

from option_pricing.marketdata.providers._alpaca_errors import AlpacaRequestError


def _alpaca_sdk_objects() -> tuple[
    Callable[..., object],
    Callable[..., object],
    Callable[[str], object],
]:
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


def _alpaca_option_sdk_objects() -> tuple[
    Callable[..., object],
    Callable[..., object],
    Callable[[str], object] | None,
    Callable[[str], object] | None,
]:
    try:
        historical_module = import_module("alpaca.data.historical.option")
        requests_module = import_module("alpaca.data.requests")
        enums_module = import_module("alpaca.data.enums")
    except ModuleNotFoundError:
        raise AlpacaRequestError(
            "alpaca-py is required to fetch Alpaca option chains"
        ) from None

    client_cls = getattr(historical_module, "OptionHistoricalDataClient", None)
    request_cls = getattr(requests_module, "OptionChainRequest", None)
    options_feed_factory = getattr(enums_module, "OptionsFeed", None) or getattr(
        requests_module,
        "OptionsFeed",
        None,
    )
    contract_type_factory = getattr(enums_module, "ContractType", None) or getattr(
        requests_module,
        "ContractType",
        None,
    )
    if contract_type_factory is None:
        try:
            trading_enums_module = import_module("alpaca.trading.enums")
        except ModuleNotFoundError:
            trading_enums_module = None
        if trading_enums_module is not None:
            contract_type_factory = getattr(
                trading_enums_module,
                "ContractType",
                None,
            )
    if not callable(options_feed_factory):
        options_feed_factory = None
    if not callable(contract_type_factory):
        contract_type_factory = None

    if not callable(client_cls):
        raise AlpacaRequestError("alpaca-py OptionHistoricalDataClient is unavailable")
    if not callable(request_cls):
        raise AlpacaRequestError("alpaca-py OptionChainRequest is unavailable")

    return (
        cast(Callable[..., object], client_cls),
        cast(Callable[..., object], request_cls),
        cast(Callable[[str], object] | None, options_feed_factory),
        cast(Callable[[str], object] | None, contract_type_factory),
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
