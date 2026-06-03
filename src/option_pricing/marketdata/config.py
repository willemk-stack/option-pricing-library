"""Configuration contracts for marketdata providers, pipeline, and storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True, init=False)
class AlpacaConfig:
    api_key_env: str = "ALPACA_API_KEY"
    secret_key_env: str = "ALPACA_SECRET_KEY"
    equity_feed: str = "iex"
    option_feed: str = "indicative"
    sandbox: bool = False

    def __init__(
        self,
        api_key_env: str = "ALPACA_API_KEY",
        secret_key_env: str = "ALPACA_SECRET_KEY",
        feed: str | None = None,
        sandbox: bool = False,
        *,
        equity_feed: str | None = None,
        option_feed: str | None = None,
    ) -> None:
        resolved_equity_feed = "iex" if equity_feed is None else equity_feed
        resolved_option_feed = (
            feed
            if option_feed is None and feed is not None
            else ("indicative" if option_feed is None else option_feed)
        )

        object.__setattr__(self, "api_key_env", api_key_env)
        object.__setattr__(self, "secret_key_env", secret_key_env)
        object.__setattr__(self, "equity_feed", resolved_equity_feed)
        object.__setattr__(self, "option_feed", resolved_option_feed)
        object.__setattr__(self, "sandbox", sandbox)
        self._validate(legacy_feed_provided=feed is not None)

    def __post_init__(self) -> None:
        self._validate(legacy_feed_provided=False)

    def _validate(self, *, legacy_feed_provided: bool) -> None:
        _validate_non_empty_string(self.api_key_env, "alpaca.api_key_env")
        _validate_non_empty_string(self.secret_key_env, "alpaca.secret_key_env")
        _validate_non_empty_string(self.equity_feed, "alpaca.equity_feed")
        _validate_non_empty_string(
            self.option_feed,
            "alpaca.feed" if legacy_feed_provided else "alpaca.option_feed",
        )

    @property
    def feed(self) -> str:
        """Backward-compatible alias for the option snapshot feed."""

        return self.option_feed


@dataclass(frozen=True, slots=True)
class FredConfig:
    api_key_env: str = "FRED_API_KEY"
    base_url: str = "https://api.stlouisfed.org/fred"

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.api_key_env, "fred.api_key_env")
        _validate_non_empty_string(self.base_url, "fred.base_url")


@dataclass(frozen=True, slots=True)
class StorageConfig:
    root: Path
    compression: str = "zstd"

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise TypeError("storage.root must be a pathlib.Path")


@dataclass(frozen=True, slots=True)
class ProviderRetryConfig:
    retry_enabled: bool = True
    max_attempts: int = 3
    wait_initial_seconds: float = 0.05
    wait_max_seconds: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.retry_enabled, bool):
            raise TypeError("retry.retry_enabled must be a boolean")
        _validate_positive_int(self.max_attempts, "retry.max_attempts")
        _validate_nonnegative_number(
            self.wait_initial_seconds,
            "retry.wait_initial_seconds",
        )
        _validate_nonnegative_number(
            self.wait_max_seconds,
            "retry.wait_max_seconds",
        )


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    alpaca: AlpacaConfig
    fred: FredConfig
    storage: StorageConfig
    retry: ProviderRetryConfig = field(default_factory=ProviderRetryConfig)


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1")


def _validate_nonnegative_number(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be numeric")
    if float(value) < 0.0:
        raise ValueError(f"{field_name} must be >= 0")


__all__ = [
    "AlpacaConfig",
    "FredConfig",
    "PipelineConfig",
    "ProviderRetryConfig",
    "StorageConfig",
]
