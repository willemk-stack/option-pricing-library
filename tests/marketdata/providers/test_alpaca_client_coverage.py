from __future__ import annotations

from datetime import UTC, datetime

import pytest

from option_pricing.marketdata.config import AlpacaConfig
from option_pricing.marketdata.providers._alpaca_client import AlpacaClient
from option_pricing.marketdata.providers._alpaca_errors import (
    AlpacaMissingCredentialsError,
    AlpacaRequestError,
)


class _Response:
    def __init__(self, data: object) -> None:
        self.data = data


class _StockClient:
    def __init__(self) -> None:
        self.latest_requests: list[object] = []
        self.bar_requests: list[object] = []

    def get_stock_latest_quote(self, request: object) -> _Response:
        self.latest_requests.append(request)
        return _Response({"AAPL": {"bid": 10.0}, "MSFT": {"bid": 20.0}})

    def get_stock_bars(self, request: object) -> _Response:
        self.bar_requests.append(request)
        return _Response({"AAPL": [{"close": 101.0}]})


class _OptionClient:
    def __init__(self) -> None:
        self.requests: list[object] = []

    def get_option_chain(self, request: object) -> _Response:
        self.requests.append(request)
        return _Response({"AAPL260116C00100000": {"strike": 100.0}})


def _client(
    *,
    stock_data_client: object | None = None,
    option_data_client: object | None = None,
) -> AlpacaClient:
    return AlpacaClient(
        "key",
        "secret",
        config=AlpacaConfig(equity_feed="iex", option_feed="indicative"),
        stock_data_client=stock_data_client,
        option_data_client=option_data_client,
    )


def test_from_env_requires_both_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    with pytest.raises(AlpacaMissingCredentialsError, match="ALPACA_API_KEY"):
        AlpacaClient.from_env()

    monkeypatch.setenv("ALPACA_API_KEY", "key")
    with pytest.raises(AlpacaMissingCredentialsError, match="ALPACA_SECRET_KEY"):
        AlpacaClient.from_env()


def test_get_latest_equity_quotes_uses_injected_client_and_request_contract() -> None:
    stock = _StockClient()
    result = _client(stock_data_client=stock).get_latest_equity_quotes(
        ["aapl", " msft "]
    )

    assert result["symbols"] == ("AAPL", "MSFT")
    assert result["feed"] == "iex"
    assert result["quotes"]["AAPL"]["bid"] == 10.0
    assert stock.latest_requests[0].symbol_or_symbols == ("AAPL", "MSFT")
    assert stock.latest_requests[0].feed == "iex"


def test_get_latest_equity_quotes_wraps_provider_errors() -> None:
    class BrokenStockClient:
        def get_stock_latest_quote(self, _request: object) -> object:
            raise RuntimeError("boom")

    with pytest.raises(AlpacaRequestError, match="latest equity quote request failed"):
        _client(stock_data_client=BrokenStockClient()).get_latest_equity_quotes("AAPL")


def test_get_latest_equity_quotes_requires_callable_getter() -> None:
    class BadStockClient:
        get_stock_latest_quote = "not-callable"

    with pytest.raises(TypeError, match="get_stock_latest_quote"):
        _client(stock_data_client=BadStockClient()).get_latest_equity_quotes("AAPL")


def test_get_equity_bars_validates_dates_and_builds_request() -> None:
    stock = _StockClient()
    client = _client(stock_data_client=stock)

    with pytest.raises(ValueError, match="start must be before end"):
        client.get_equity_bars(
            "AAPL",
            start="2026-01-02T00:00:00Z",
            end="2026-01-01T00:00:00Z",
            timeframe="1Day",
        )

    result = client.get_equity_bars(
        "AAPL",
        start="2026-01-01T00:00:00Z",
        end="2026-01-02T00:00:00Z",
        timeframe="1Day",
        limit=10,
        adjustment="raw",
        sort=None,
        asof="2026-01-03",
    )

    request = stock.bar_requests[0]
    assert result["bars"] == {"AAPL": [{"close": 101.0}]}
    assert request.symbol_or_symbols == ("AAPL",)
    assert request.limit == 10
    assert request.adjustment == "raw"
    assert request.sort is None
    assert request.asof == "2026-01-03"


def test_get_option_chain_validates_strikes_and_records_request_metadata() -> None:
    option = _OptionClient()
    client = _client(option_data_client=option)

    with pytest.raises(ValueError, match="strike_gte"):
        client.get_option_chain(
            "aapl",
            expiry_gte="2026-01-01",
            expiry_lte="2026-02-01",
            strike_gte=105.0,
            strike_lte=100.0,
        )

    result = client.get_option_chain(
        "aapl",
        expiry_gte="2026-01-01",
        expiry_lte="2026-02-01",
        strike_gte=90.0,
        strike_lte=110.0,
        option_type="CALL",
        root_symbol=" aapl ",
        updated_since=datetime(2025, 12, 31, tzinfo=UTC),
        asof="snapshot",
    )

    request = option.requests[0]
    assert result["underlying"] == "AAPL"
    assert result["contracts"] == {"AAPL260116C00100000": {"strike": 100.0}}
    assert result["request"]["option_type"] == "call"
    assert result["asof"] == "snapshot"
    assert request.underlying == "AAPL"
    assert request.root_symbol == "AAPL"


def test_get_option_chain_requires_callable_getter() -> None:
    class BadOptionClient:
        get_option_chain = None

    with pytest.raises(TypeError, match="get_option_chain"):
        _client(option_data_client=BadOptionClient()).get_option_chain(
            "AAPL",
            expiry_gte="2026-01-01",
            expiry_lte="2026-02-01",
        )


def test_repr_redacts_credentials_and_shows_injection_state() -> None:
    text = repr(_client(stock_data_client=object(), option_data_client=object()))

    assert "api_key=<redacted>" in text
    assert "secret_key=<redacted>" in text
    assert "stock_data_client_injected=True" in text
    assert "option_data_client_injected=True" in text
    assert "abc" not in text
    assert "def" not in text
