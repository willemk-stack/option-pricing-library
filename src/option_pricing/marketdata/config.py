"""Configuration contracts for marketdata providers, pipeline, and storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AlpacaConfig:
    api_key_env: str = "ALPACA_API_KEY"
    secret_key_env: str = "ALPACA_SECRET_KEY"
    feed: str = "indicative"
    sandbox: bool = False

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.api_key_env, "alpaca.api_key_env")
        _validate_non_empty_string(self.secret_key_env, "alpaca.secret_key_env")
        _validate_non_empty_string(self.feed, "alpaca.feed")


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
