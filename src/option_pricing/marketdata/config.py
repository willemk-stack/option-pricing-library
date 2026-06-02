"""Configuration contracts for marketdata providers, pipeline, and storage."""

from __future__ import annotations

from dataclasses import dataclass
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
class PipelineConfig:
    alpaca: AlpacaConfig
    fred: FredConfig
    storage: StorageConfig


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


__all__ = [
    "AlpacaConfig",
    "FredConfig",
    "PipelineConfig",
    "StorageConfig",
]
