from __future__ import annotations

import builtins
import importlib.util
import os
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from importlib import import_module
from typing import Any

import pytest

from option_pricing.marketdata.config import AlpacaConfig
from option_pricing.marketdata.normalize import normalize_alpaca_option_chain
from option_pricing.marketdata.providers.alpaca import (
    AlpacaClient,
    AlpacaMissingCredentialsError,
    AlpacaRequestError,
)


@dataclass(frozen=True, slots=True)
class _FakeQuote:
    symbol: str = "SPY"
    timestamp: str = "2026-05-22T15:30:00Z"
    bid_price: float = 499.0
    ask_price: float = 499.2
    bid_size: int = 3
    ask_size: int = 4


@dataclass(frozen=True, slots=True)
class _FakeBar:
    symbol: str = "SPY"
    timestamp: str = "2026-05-21T13:30:00Z"
    open: float = 499.0
    high: float = 501.0
    low: float = 498.5
    close: float = 500.5
    volume: int = 1000
    trade_count: int = 75
    vwap: float = 500.1


class _FakeStockClient:
    def __init__(
        self,
        response: object | None = None,
        bars_response: object | None = None,
        error: Exception | None = None,
        bars_error: Exception | None = None,
    ) -> None:
        self.response = {"SPY": _FakeQuote()} if response is None else response
        self.bars_response = (
            {"SPY": [_FakeBar()]} if bars_response is None else bars_response
        )
        self.error = error
        self.bars_error = bars_error
        self.calls: list[object] = []
        self.bars_calls: list[object] = []

    def get_stock_latest_quote(self, request_params: object) -> object:
        self.calls.append(request_params)
        if self.error is not None:
            raise self.error
        return self.response

    def get_stock_bars(self, request_params: object) -> object:
        self.bars_calls.append(request_params)
        if self.bars_error is not None:
            raise self.bars_error
        return self.bars_response


class _FakeOptionClient:
    def __init__(
        self,
        response: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = (
            {"SPY260619C00500000": {"latest_quote": {}}}
            if response is None
            else response
        )
        self.error = error
        self.calls: list[object] = []

    def get_option_chain(self, request_params: object) -> object:
        self.calls.append(request_params)
        if self.error is not None:
            raise self.error
        return self.response


def test_alpaca_client_from_env_missing_api_key_raises_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.setenv("ALPACA_SECRET_KEY", "present-secret")

    with pytest.raises(
        AlpacaMissingCredentialsError, match="ALPACA_API_KEY"
    ) as excinfo:
        AlpacaClient.from_env(stock_data_client=_FakeStockClient())

    assert excinfo.value.env_var_name == "ALPACA_API_KEY"


def test_alpaca_client_from_env_missing_secret_key_raises_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "present-key")
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    with pytest.raises(
        AlpacaMissingCredentialsError,
        match="ALPACA_SECRET_KEY",
    ) as excinfo:
        AlpacaClient.from_env(stock_data_client=_FakeStockClient())

    assert excinfo.value.env_var_name == "ALPACA_SECRET_KEY"


def test_alpaca_client_from_env_uses_configured_env_var_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUSTOM_ALPACA_KEY", "custom-alpaca-key")
    monkeypatch.setenv("CUSTOM_ALPACA_SECRET", "custom-alpaca-secret")

    client = AlpacaClient.from_env(
        AlpacaConfig(
            api_key_env="CUSTOM_ALPACA_KEY",
            secret_key_env="CUSTOM_ALPACA_SECRET",
            equity_feed="iex",
            option_feed="indicative",
        ),
        stock_data_client=_FakeStockClient(),
    )

    representation = repr(client)
    assert "custom-alpaca-key" not in representation
    assert "custom-alpaca-secret" not in representation
    assert "api_key=<redacted>" in representation
    assert "secret_key=<redacted>" in representation


def test_alpaca_client_repr_does_not_leak_credentials() -> None:
    client = AlpacaClient(
        "alpaca-key-value",
        "alpaca-secret-value",
        stock_data_client=_FakeStockClient(),
    )

    representation = repr(client)

    assert "alpaca-key-value" not in representation
    assert "alpaca-secret-value" not in representation
    assert "stock_data_client_injected=True" in representation


def test_alpaca_client_uses_injected_fake_stock_client() -> None:
    fake_client = _FakeStockClient(
        response={
            "SPY": _FakeQuote(symbol="SPY"),
            "AAPL": _FakeQuote(symbol="AAPL"),
        }
    )
    client = AlpacaClient(
        "alpaca-key",
        "alpaca-secret",
        config=AlpacaConfig(equity_feed="iex", option_feed="indicative"),
        stock_data_client=fake_client,
    )

    result = client.get_latest_equity_quotes(
        ["spy", "AAPL"],
        asof="2026-05-22T15:31:00Z",
    )

    assert result["symbols"] == ("SPY", "AAPL")
    assert result["source"] == "alpaca"
    assert result["feed"] == "iex"
    assert result["asof"] == "2026-05-22T15:31:00Z"
    assert result["quotes"] == fake_client.response
    assert len(fake_client.calls) == 1
    request = fake_client.calls[0]
    assert request.symbol_or_symbols == ("SPY", "AAPL")
    assert request.feed == "iex"


def test_alpaca_client_get_equity_bars_uses_injected_fake_stock_client() -> None:
    fake_client = _FakeStockClient(
        bars_response={
            "SPY": [_FakeBar(symbol="SPY")],
            "AAPL": [_FakeBar(symbol="AAPL")],
        }
    )
    client = AlpacaClient(
        "alpaca-key",
        "alpaca-secret",
        config=AlpacaConfig(equity_feed="iex", option_feed="indicative"),
        stock_data_client=fake_client,
    )

    result = client.get_equity_bars(
        ["spy", "AAPL"],
        start="2026-05-21T13:30:00-04:00",
        end=datetime(2026, 5, 22, 13, 30, tzinfo=UTC),
        timeframe="1Day",
        feed="sip",
        adjustment="raw",
        limit=100,
        sort="desc",
        asof="2026-05-22",
    )

    assert result["symbols"] == ("SPY", "AAPL")
    assert result["source"] == "alpaca"
    assert result["feed"] == "sip"
    assert result["timeframe"] == "1Day"
    assert result["bars"] == fake_client.bars_response
    assert result["asof"] == "2026-05-22"
    assert len(fake_client.bars_calls) == 1
    request = fake_client.bars_calls[0]
    assert request.symbol_or_symbols == ("SPY", "AAPL")
    assert request.start == datetime(2026, 5, 21, 17, 30, tzinfo=UTC)
    assert request.end == datetime(2026, 5, 22, 13, 30, tzinfo=UTC)
    assert request.timeframe == "1Day"
    assert request.feed == "sip"
    assert request.adjustment == "raw"
    assert request.limit == 100
    assert request.sort == "desc"
    assert request.asof == "2026-05-22"


def test_alpaca_client_get_equity_bars_defaults_to_config_equity_feed() -> None:
    fake_client = _FakeStockClient()
    client = AlpacaClient(
        "alpaca-key",
        "alpaca-secret",
        config=AlpacaConfig(equity_feed="sip", option_feed="indicative"),
        stock_data_client=fake_client,
    )

    result = client.get_equity_bars(
        "spy",
        start="2026-05-21T00:00:00Z",
        end="2026-05-22T00:00:00Z",
        timeframe="1Day",
    )

    assert result["feed"] == "sip"
    assert fake_client.bars_calls[0].feed == "sip"


def test_alpaca_client_legacy_feed_does_not_override_equity_feed() -> None:
    fake_client = _FakeStockClient()
    client = AlpacaClient(
        "alpaca-key",
        "alpaca-secret",
        config=AlpacaConfig(feed="opra"),
        stock_data_client=fake_client,
    )

    result = client.get_latest_equity_quotes("spy")

    assert result["feed"] == "iex"
    assert fake_client.calls[0].feed == "iex"


def test_alpaca_client_get_equity_bars_start_must_precede_end() -> None:
    client = AlpacaClient(
        "alpaca-key",
        "alpaca-secret",
        stock_data_client=_FakeStockClient(),
    )

    with pytest.raises(ValueError, match="start must be before end"):
        client.get_equity_bars(
            "SPY",
            start="2026-05-22T00:00:00Z",
            end="2026-05-22T00:00:00Z",
            timeframe="1Day",
        )


def test_alpaca_client_get_equity_bars_timeframe_must_not_be_blank() -> None:
    client = AlpacaClient(
        "alpaca-key",
        "alpaca-secret",
        stock_data_client=_FakeStockClient(),
    )

    with pytest.raises(ValueError, match="timeframe must be a non-empty string"):
        client.get_equity_bars(
            "SPY",
            start="2026-05-21T00:00:00Z",
            end="2026-05-22T00:00:00Z",
            timeframe=" ",
        )


def test_alpaca_client_get_equity_bars_provider_exception_is_wrapped() -> None:
    secret = "alpaca-secret-value"
    fake_client = _FakeStockClient(bars_error=RuntimeError(f"boom near {secret}"))
    client = AlpacaClient("alpaca-key-value", secret, stock_data_client=fake_client)

    with pytest.raises(AlpacaRequestError) as excinfo:
        client.get_equity_bars(
            "SPY",
            start="2026-05-21T00:00:00Z",
            end="2026-05-22T00:00:00Z",
            timeframe="1Day",
        )

    details = f"{excinfo.value!s} {excinfo.value!r} {client!r}"
    assert "equity bars request failed" in str(excinfo.value)
    assert secret not in details
    assert "alpaca-key-value" not in details


def test_alpaca_client_get_option_chain_uses_injected_fake_option_client() -> None:
    fake_client = _FakeOptionClient(response={"SPY260619C00500000": object()})
    client = AlpacaClient(
        "alpaca-key",
        "alpaca-secret",
        config=AlpacaConfig(equity_feed="iex", option_feed="indicative"),
        option_data_client=fake_client,
    )

    result = client.get_option_chain(
        " spy ",
        expiry_gte=date(2026, 6, 1),
        expiry_lte="2026-06-30",
        strike_gte=400,
        strike_lte=550.0,
        option_type="Call",
        root_symbol="spy",
        updated_since="2026-05-22T11:30:00-04:00",
        feed="opra",
        asof="2026-05-22T15:31:00Z",
    )

    assert result["underlying"] == "SPY"
    assert result["source"] == "alpaca"
    assert result["feed"] == "opra"
    assert result["contracts"] == fake_client.response
    assert result["asof"] == "2026-05-22T15:31:00Z"
    assert len(fake_client.calls) == 1
    request = fake_client.calls[0]
    assert request.underlying == "SPY"
    assert request.underlying_symbol == "SPY"
    assert request.feed == "opra"
    assert request.expiry_gte == date(2026, 6, 1)
    assert request.expiry_lte == date(2026, 6, 30)
    assert request.expiration_date_gte == date(2026, 6, 1)
    assert request.expiration_date_lte == date(2026, 6, 30)
    assert request.strike_gte == 400.0
    assert request.strike_lte == 550.0
    assert request.strike_price_gte == 400.0
    assert request.strike_price_lte == 550.0
    assert request.option_type == "call"
    assert request.type == "call"
    assert request.root_symbol == "SPY"
    assert request.updated_since == datetime(2026, 5, 22, 15, 30, tzinfo=UTC)

    metadata = result["request"]
    assert metadata["underlying"] == "SPY"
    assert metadata["feed"] == "opra"
    assert metadata["expiry_gte"] == date(2026, 6, 1)
    assert metadata["expiry_lte"] == date(2026, 6, 30)
    assert "alpaca-key" not in repr(metadata)
    assert "alpaca-secret" not in repr(metadata)


def test_alpaca_client_get_option_chain_defaults_to_config_option_feed() -> None:
    fake_client = _FakeOptionClient()
    client = AlpacaClient(
        "alpaca-key",
        "alpaca-secret",
        config=AlpacaConfig(equity_feed="iex", option_feed="opra"),
        option_data_client=fake_client,
    )

    result = client.get_option_chain(
        "SPY",
        expiry_gte="2026-06-01",
        expiry_lte="2026-06-30",
    )

    assert result["feed"] == "opra"
    assert fake_client.calls[0].feed == "opra"


def test_alpaca_client_get_option_chain_applies_default_expiry_bounds() -> None:
    fake_client = _FakeOptionClient()
    client = AlpacaClient(
        "alpaca-key",
        "alpaca-secret",
        option_data_client=fake_client,
    )

    before = datetime.now(UTC).date()
    result = client.get_option_chain("SPY")
    after = datetime.now(UTC).date()

    request = fake_client.calls[0]
    assert before <= request.expiry_gte <= after
    assert request.expiry_lte == request.expiry_gte + timedelta(days=45)
    metadata = result["request"]
    assert metadata["expiry_gte"] == request.expiry_gte
    assert metadata["expiry_lte"] == request.expiry_lte


def test_alpaca_client_get_option_chain_default_lte_uses_current_date_window() -> None:
    fake_client = _FakeOptionClient()
    client = AlpacaClient(
        "alpaca-key",
        "alpaca-secret",
        option_data_client=fake_client,
    )

    current_date = datetime.now(UTC).date()
    client.get_option_chain("SPY", expiry_gte=current_date + timedelta(days=10))

    request = fake_client.calls[0]
    assert request.expiry_gte == current_date + timedelta(days=10)
    assert request.expiry_lte == current_date + timedelta(days=45)


def test_alpaca_client_get_option_chain_explicit_expiry_bounds_override_defaults() -> (
    None
):
    fake_client = _FakeOptionClient()
    client = AlpacaClient(
        "alpaca-key",
        "alpaca-secret",
        option_data_client=fake_client,
    )

    client.get_option_chain(
        "SPY",
        expiry_gte="2026-06-19",
        expiry_lte="2026-07-17",
    )

    request = fake_client.calls[0]
    assert request.expiry_gte == date(2026, 6, 19)
    assert request.expiry_lte == date(2026, 7, 17)


def test_alpaca_client_get_option_chain_expiry_bounds_must_be_ordered() -> None:
    client = AlpacaClient(
        "alpaca-key",
        "alpaca-secret",
        option_data_client=_FakeOptionClient(),
    )

    with pytest.raises(ValueError, match="expiry_gte"):
        client.get_option_chain(
            "SPY",
            expiry_gte="2026-07-17",
            expiry_lte="2026-06-19",
        )


def test_alpaca_client_get_option_chain_strike_bounds_must_be_ordered() -> None:
    client = AlpacaClient(
        "alpaca-key",
        "alpaca-secret",
        option_data_client=_FakeOptionClient(),
    )

    with pytest.raises(ValueError, match="strike_gte"):
        client.get_option_chain("SPY", strike_gte=550.0, strike_lte=500.0)


def test_alpaca_client_get_option_chain_option_type_must_be_call_or_put() -> None:
    client = AlpacaClient(
        "alpaca-key",
        "alpaca-secret",
        option_data_client=_FakeOptionClient(),
    )

    with pytest.raises(ValueError, match="option_type"):
        client.get_option_chain("SPY", option_type="straddle")


def test_alpaca_client_get_option_chain_provider_exception_is_wrapped() -> None:
    secret = "alpaca-secret-value"
    fake_client = _FakeOptionClient(error=RuntimeError(f"boom near {secret}"))
    client = AlpacaClient(
        "alpaca-key-value",
        secret,
        option_data_client=fake_client,
    )

    with pytest.raises(AlpacaRequestError) as excinfo:
        client.get_option_chain("SPY")

    details = f"{excinfo.value!s} {excinfo.value!r} {client!r}"
    assert "option chain request failed" in str(excinfo.value)
    assert secret not in details
    assert "alpaca-key-value" not in details


def test_alpaca_request_failures_raise_without_leaking_secrets() -> None:
    secret = "alpaca-secret-value"
    fake_client = _FakeStockClient(error=ValueError(f"boom near {secret}"))
    client = AlpacaClient("alpaca-key-value", secret, stock_data_client=fake_client)

    with pytest.raises(AlpacaRequestError) as excinfo:
        client.get_latest_equity_quotes("SPY")

    details = f"{excinfo.value!s} {excinfo.value!r} {client!r}"
    assert "latest equity quote request failed" in str(excinfo.value)
    assert secret not in details
    assert "alpaca-key-value" not in details


def test_alpaca_provider_import_does_not_require_alpaca_py(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if level == 0 and name.split(".", maxsplit=1)[0] == "alpaca":
            raise ModuleNotFoundError(f"No module named {name!r}")
        return original_import(name, globals, locals, fromlist, level)

    sys.modules.pop("option_pricing.marketdata.providers.alpaca", None)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = import_module("option_pricing.marketdata.providers.alpaca")

    assert module.AlpacaClient is not None


@pytest.mark.skipif(
    not (
        os.environ.get("ALPACA_API_KEY")
        and os.environ.get("ALPACA_SECRET_KEY")
        and importlib.util.find_spec("alpaca") is not None
    ),
    reason="ALPACA_API_KEY/ALPACA_SECRET_KEY or alpaca-py are not available",
)
def test_alpaca_live_fetch_latest_spy_quote_smoke() -> None:
    payload = AlpacaClient.from_env().get_latest_equity_quotes("SPY")

    assert "SPY" in payload["quotes"]


@pytest.mark.skipif(
    not (
        os.environ.get("ALPACA_API_KEY")
        and os.environ.get("ALPACA_SECRET_KEY")
        and importlib.util.find_spec("alpaca") is not None
    ),
    reason="ALPACA_API_KEY/ALPACA_SECRET_KEY or alpaca-py are not available",
)
def test_alpaca_live_fetch_spy_equity_bar_smoke() -> None:
    payload = AlpacaClient.from_env().get_equity_bars(
        "SPY",
        start="2026-05-18T00:00:00Z",
        end="2026-05-19T00:00:00Z",
        timeframe="1Day",
        limit=1,
    )

    assert "SPY" in payload["bars"]


@pytest.mark.skipif(
    not (
        os.environ.get("ALPACA_API_KEY")
        and os.environ.get("ALPACA_SECRET_KEY")
        and importlib.util.find_spec("alpaca") is not None
    ),
    reason="ALPACA_API_KEY/ALPACA_SECRET_KEY or alpaca-py are not available",
)
def test_alpaca_live_fetch_spy_option_chain_smoke() -> None:
    payload = AlpacaClient.from_env().get_option_chain(
        "SPY",
        asof=datetime.now(UTC).isoformat(),
    )

    if not payload["contracts"]:
        pytest.skip("Alpaca returned no SPY option contracts")

    try:
        normalized = normalize_alpaca_option_chain(payload)
    except ValueError as exc:
        if "usable latest quote bid/ask" in str(exc):
            pytest.skip("Alpaca returned no SPY option contracts with bid/ask")
        raise

    assert normalized["underlying"].astype(str).eq("SPY").all()
