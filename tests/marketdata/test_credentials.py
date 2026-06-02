from __future__ import annotations

import pytest

from option_pricing.marketdata.config import AlpacaConfig, FredConfig
from option_pricing.marketdata.credentials import (
    MissingMarketDataCredentialError,
    load_alpaca_credentials,
    load_env_credential,
    load_fred_credentials,
    redact_mapping,
    redact_secret,
)


def test_missing_fred_key_raises_readable_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    with pytest.raises(MissingMarketDataCredentialError) as excinfo:
        load_fred_credentials(FredConfig())

    message = str(excinfo.value)
    assert excinfo.value.env_var_name == "FRED_API_KEY"
    assert "FRED_API_KEY" in message
    assert "unset or blank" in message


def test_blank_fred_key_raises_readable_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FRED_API_KEY", " \t ")

    with pytest.raises(MissingMarketDataCredentialError) as excinfo:
        load_fred_credentials(FredConfig())

    assert excinfo.value.env_var_name == "FRED_API_KEY"
    assert "FRED_API_KEY" in str(excinfo.value)


def test_present_fred_key_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "fred-test-key")

    credentials = load_fred_credentials(FredConfig())

    assert credentials.api_key == "fred-test-key"
    assert load_env_credential("FRED_API_KEY") == "fred-test-key"


@pytest.mark.parametrize(
    ("missing_env_var", "present_env_var"),
    [
        ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"),
        ("ALPACA_SECRET_KEY", "ALPACA_API_KEY"),
    ],
)
def test_missing_alpaca_key_or_secret_raises_typed_error_identifying_env_var(
    monkeypatch: pytest.MonkeyPatch,
    missing_env_var: str,
    present_env_var: str,
) -> None:
    monkeypatch.delenv(missing_env_var, raising=False)
    monkeypatch.setenv(present_env_var, "present-alpaca-value")

    with pytest.raises(MissingMarketDataCredentialError) as excinfo:
        load_alpaca_credentials(AlpacaConfig())

    message = str(excinfo.value)
    assert excinfo.value.env_var_name == missing_env_var
    assert missing_env_var in message
    assert "present-alpaca-value" not in message


def test_present_alpaca_key_and_secret_returns_both_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "alpaca-test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "alpaca-test-secret")

    credentials = load_alpaca_credentials(AlpacaConfig())

    assert credentials.api_key == "alpaca-test-key"
    assert credentials.secret_key == "alpaca-test-secret"


def test_secret_redaction_does_not_expose_original_values() -> None:
    secret = "alpaca-test-secret"

    redacted = redact_secret(secret, label="ALPACA_SECRET_KEY")

    assert redacted == "<redacted:ALPACA_SECRET_KEY>"
    assert secret not in redacted
    assert redact_secret(secret) == "<redacted>"
    assert redact_secret(secret, label=secret) == "<redacted>"
    assert "<redacted>" not in redact_secret("<redacted>")
    assert "e" not in redact_secret("e")


def test_redaction_handles_none_and_empty_strings_safely() -> None:
    assert redact_secret(None) == "<redacted>"
    assert redact_secret(None, label="FRED_API_KEY") == "<redacted:FRED_API_KEY>"
    assert redact_secret("") == "<redacted>"
    assert redact_secret("", label="FRED_API_KEY") == "<redacted:FRED_API_KEY>"


def test_redact_mapping_replaces_selected_secret_keys() -> None:
    mapping = {
        "api_key": "secret-api-key",
        "base_url": "https://example.test",
        "secret_key": "secret-key",
    }

    redacted = redact_mapping(mapping, {"api_key", "secret_key"})

    assert redacted == {
        "api_key": "<redacted:api_key>",
        "base_url": "https://example.test",
        "secret_key": "<redacted:secret_key>",
    }
    assert "secret-api-key" not in str(redacted)
    assert "secret-key" not in str(redacted)


def test_config_construction_does_not_read_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for env_var in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "FRED_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)

    alpaca_config = AlpacaConfig()
    fred_config = FredConfig()

    assert alpaca_config.api_key_env == "ALPACA_API_KEY"
    assert alpaca_config.secret_key_env == "ALPACA_SECRET_KEY"
    assert fred_config.api_key_env == "FRED_API_KEY"
