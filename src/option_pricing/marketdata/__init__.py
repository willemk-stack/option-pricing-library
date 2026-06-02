"""Public marketdata exports with lazy loading for optional runtime dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "AlpacaConfig",
    "FredConfig",
    "LocalStorage",
    "MarketDataPipeline",
    "PipelineConfig",
    "ProviderSnapshotDataUnavailableError",
    "ProviderSnapshotResult",
    "StorageConfig",
]

if TYPE_CHECKING:
    from .config import AlpacaConfig, FredConfig, PipelineConfig, StorageConfig
    from .pipeline import (
        MarketDataPipeline,
        ProviderSnapshotDataUnavailableError,
        ProviderSnapshotResult,
    )
    from .storage import LocalStorage

_PUBLIC_EXPORTS = {
    "AlpacaConfig": "option_pricing.marketdata.config",
    "FredConfig": "option_pricing.marketdata.config",
    "PipelineConfig": "option_pricing.marketdata.config",
    "StorageConfig": "option_pricing.marketdata.config",
    "LocalStorage": "option_pricing.marketdata.storage",
    "MarketDataPipeline": "option_pricing.marketdata.pipeline",
    "ProviderSnapshotDataUnavailableError": "option_pricing.marketdata.pipeline",
    "ProviderSnapshotResult": "option_pricing.marketdata.pipeline",
}


def __getattr__(name: str) -> Any:
    module_name = _PUBLIC_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
