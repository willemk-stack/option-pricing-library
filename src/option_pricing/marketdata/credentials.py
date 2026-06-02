"""Credential loading and redaction helpers for market data providers."""

from __future__ import annotations

import os
from collections.abc import Collection, Mapping
from dataclasses import dataclass

from option_pricing.marketdata.config import AlpacaConfig, FredConfig


class MissingMarketDataCredentialError(RuntimeError):
    """Raised when a required market data credential is unset or blank."""

    def __init__(self, env_var_name: str) -> None:
        self.env_var_name = env_var_name
        super().__init__(
            "Missing market data credential: environment variable "
            f"{env_var_name!r} is unset or blank."
        )


@dataclass(frozen=True, slots=True)
class FredCredentials:
    """Credentials required by a FRED market data provider."""

    api_key: str


@dataclass(frozen=True, slots=True)
class AlpacaCredentials:
    """Credentials required by an Alpaca market data provider."""

    api_key: str
    secret_key: str


def load_env_credential(env_var_name: str) -> str:
    """Read one required credential from the process environment."""

    env_var = _clean_env_var_name(env_var_name)
    value = os.environ.get(env_var)
    if value is None or not value.strip():
        raise MissingMarketDataCredentialError(env_var)
    return value


def load_fred_credentials(config: FredConfig) -> FredCredentials:
    """Load FRED credentials from the environment named by ``config``."""

    return FredCredentials(api_key=load_env_credential(config.api_key_env))


def load_alpaca_credentials(config: AlpacaConfig) -> AlpacaCredentials:
    """Load Alpaca credentials from the environment named by ``config``."""

    return AlpacaCredentials(
        api_key=load_env_credential(config.api_key_env),
        secret_key=load_env_credential(config.secret_key_env),
    )


def redact_secret(value: str | None, *, label: str | None = None) -> str:
    """Return a stable placeholder that does not expose ``value``."""

    cleaned_label = _clean_optional_label(label)
    candidates = (
        (f"<redacted:{cleaned_label}>",) if cleaned_label is not None else ()
    ) + ("<redacted>", "<masked>", "<withheld>", "[redacted]", "[***]")
    if not value:
        return candidates[0]

    for candidate in candidates:
        if value not in candidate:
            return candidate
    return "<secret>"


def redact_mapping(
    mapping: Mapping[str, object],
    secret_keys: Collection[str],
) -> dict[str, object]:
    """Copy ``mapping`` with selected keys replaced by redaction placeholders."""

    secret_key_names = set(secret_keys)
    return {
        key: (
            redact_secret(value if isinstance(value, str) else None, label=key)
            if key in secret_key_names
            else value
        )
        for key, value in mapping.items()
    }


def _clean_env_var_name(env_var_name: str) -> str:
    cleaned = env_var_name.strip()
    if not cleaned:
        raise ValueError("env_var_name must be a non-empty string")
    return cleaned


def _clean_optional_label(label: str | None) -> str | None:
    if label is None:
        return None
    cleaned = label.strip()
    return cleaned or None


__all__ = [
    "AlpacaCredentials",
    "FredCredentials",
    "MissingMarketDataCredentialError",
    "load_alpaca_credentials",
    "load_env_credential",
    "load_fred_credentials",
    "redact_mapping",
    "redact_secret",
]
