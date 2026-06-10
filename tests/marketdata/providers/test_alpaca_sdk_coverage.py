from __future__ import annotations

from types import SimpleNamespace

import pytest

import option_pricing.marketdata.providers._alpaca_sdk as sdk
from option_pricing.marketdata.providers._alpaca_errors import AlpacaRequestError


def test_alpaca_sdk_objects_raise_when_alpaca_py_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_import(_name: str) -> object:
        raise ModuleNotFoundError("alpaca")

    monkeypatch.setattr(sdk, "import_module", missing_import)

    with pytest.raises(AlpacaRequestError, match="alpaca-py is required"):
        sdk._alpaca_sdk_objects()


def test_alpaca_sdk_objects_validate_required_callables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules = {
        "alpaca.data.historical": SimpleNamespace(
            StockHistoricalDataClient=lambda **_kwargs: object()
        ),
        "alpaca.data.requests": SimpleNamespace(StockLatestQuoteRequest=None),
        "alpaca.data.enums": SimpleNamespace(DataFeed=lambda value: f"feed:{value}"),
    }

    monkeypatch.setattr(sdk, "import_module", lambda name: modules[name])

    with pytest.raises(AlpacaRequestError, match="StockLatestQuoteRequest"):
        sdk._alpaca_sdk_objects()


def test_alpaca_bars_sdk_objects_allow_optional_adjustment_and_sort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Request:
        pass

    class TimeFrame:
        Minute = object()

    modules = {
        "alpaca.data.requests": SimpleNamespace(StockBarsRequest=Request),
        "alpaca.data.enums": SimpleNamespace(DataFeed=lambda value: value),
        "alpaca.data.timeframe": SimpleNamespace(TimeFrame=TimeFrame),
    }

    monkeypatch.setattr(sdk, "import_module", lambda name: modules[name])

    request_cls, timeframe_cls, data_feed, adjustment, sort = (
        sdk._alpaca_bars_sdk_objects()
    )

    assert request_cls is Request
    assert timeframe_cls is TimeFrame
    assert data_feed("iex") == "iex"
    assert adjustment is None
    assert sort is None


def test_alpaca_option_sdk_objects_falls_back_to_trading_contract_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OptionClient:
        pass

    class OptionRequest:
        pass

    class ContractType:
        def __init__(self, value: str) -> None:
            self.value = value

    modules = {
        "alpaca.data.historical.option": SimpleNamespace(
            OptionHistoricalDataClient=OptionClient
        ),
        "alpaca.data.requests": SimpleNamespace(OptionChainRequest=OptionRequest),
        "alpaca.data.enums": SimpleNamespace(OptionsFeed=lambda value: f"feed:{value}"),
        "alpaca.trading.enums": SimpleNamespace(ContractType=ContractType),
    }

    monkeypatch.setattr(sdk, "import_module", lambda name: modules[name])

    client_cls, request_cls, feed_factory, contract_factory = (
        sdk._alpaca_option_sdk_objects()
    )

    assert client_cls is OptionClient
    assert request_cls is OptionRequest
    assert feed_factory is not None
    assert feed_factory("indicative") == "feed:indicative"
    assert contract_factory is ContractType


@pytest.mark.parametrize(
    ("raw", "expected_attr"),
    [
        ("1 min", "Minute"),
        ("1_hour", "Hour"),
        ("daily", "Day"),
        ("weekly", "Week"),
        ("monthly", "Month"),
    ],
)
def test_alpaca_timeframe_uses_known_aliases(raw: str, expected_attr: str) -> None:
    class TimeFrame:
        Minute = object()
        Hour = object()
        Day = object()
        Week = object()
        Month = object()

    assert sdk._alpaca_timeframe(raw, TimeFrame) is getattr(TimeFrame, expected_attr)


def test_alpaca_timeframe_falls_back_to_factory_then_raw_value() -> None:
    assert sdk._alpaca_timeframe("custom", lambda value: f"tf:{value}") == "tf:custom"

    def broken_factory(_value: str) -> object:
        raise RuntimeError("unsupported")

    assert sdk._alpaca_timeframe("custom", broken_factory) == "custom"


def test_optional_sdk_enum_returns_raw_value_when_factory_absent() -> None:
    assert sdk._optional_sdk_enum("asc", None) == "asc"
    assert sdk._optional_sdk_enum("asc", lambda value: f"enum:{value}") == "enum:asc"
