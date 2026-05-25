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


@dataclass(frozen=True, slots=True)
class FredConfig:
    api_key_env: str = "FRED_API_KEY"
    base_url: str = "https://api.stlouisfed.org/fred"


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


__all__ = [
    "AlpacaConfig",
    "FredConfig",
    "PipelineConfig",
    "StorageConfig",
]
