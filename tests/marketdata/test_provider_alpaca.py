from __future__ import annotations

import builtins
import importlib.util
import os
import sys
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import pytest

from option_pricing.marketdata.config import AlpacaConfig
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


class _FakeStockClient:
    def __init__(
        self,
        response: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = {"SPY": _FakeQuote()} if response is None else response
        self.error = error
        self.calls: list[object] = []

    def get_stock_latest_quote(self, request_params: object) -> object:
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
            feed="iex",
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
        config=AlpacaConfig(feed="iex"),
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
