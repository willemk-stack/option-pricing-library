from __future__ import annotations

import os

import pytest

from option_pricing.marketdata.config import AlpacaConfig, FredConfig


def test_default_alpaca_config_constructs_successfully() -> None:
    assert AlpacaConfig() is not None


def test_default_fred_config_constructs_successfully() -> None:
    assert FredConfig() is not None


def test_default_env_var_names_remain_unchanged() -> None:
    alpaca_config = AlpacaConfig()
    fred_config = FredConfig()

    assert alpaca_config.api_key_env == "ALPACA_API_KEY"
    assert alpaca_config.secret_key_env == "ALPACA_SECRET_KEY"
    assert fred_config.api_key_env == "FRED_API_KEY"


@pytest.mark.parametrize("bad_value", ["", " \t "])
@pytest.mark.parametrize("field_name", ["api_key_env", "secret_key_env"])
def test_alpaca_empty_or_whitespace_env_var_names_raise_value_error(
    field_name: str,
    bad_value: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"alpaca\.{field_name} must be a non-empty string",
    ):
        AlpacaConfig(**{field_name: bad_value})


@pytest.mark.parametrize("bad_value", ["", " \t "])
def test_fred_empty_or_whitespace_env_var_name_raises_value_error(
    bad_value: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"fred\.api_key_env must be a non-empty string",
    ):
        FredConfig(api_key_env=bad_value)


@pytest.mark.parametrize("bad_value", ["", " \t "])
def test_alpaca_empty_or_whitespace_feed_raises_value_error(
    bad_value: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"alpaca\.feed must be a non-empty string",
    ):
        AlpacaConfig(feed=bad_value)


@pytest.mark.parametrize("bad_value", ["", " \t "])
def test_fred_empty_or_whitespace_base_url_raises_value_error(
    bad_value: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"fred\.base_url must be a non-empty string",
    ):
        FredConfig(base_url=bad_value)


def test_config_construction_does_not_inspect_os_environ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RaisingEnviron(dict[str, str]):
        def __getitem__(self, key: str) -> str:
            raise AssertionError(f"os.environ was read for {key!r}")

        def get(self, key: str, default: str | None = None) -> str | None:
            raise AssertionError(f"os.environ was read for {key!r}")

        def __contains__(self, key: object) -> bool:
            raise AssertionError(f"os.environ was inspected for {key!r}")

    monkeypatch.setattr(os, "environ", RaisingEnviron())

    assert AlpacaConfig() is not None
    assert FredConfig() is not None
