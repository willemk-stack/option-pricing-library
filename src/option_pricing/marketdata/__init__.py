"""Public marketdata exports with lazy loading for optional runtime dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "AlpacaConfig",
    "FredConfig",
    "HestonReadyStats",
    "LoadedModelValidationBundle",
    "LocalStorage",
    "MarketDataPipeline",
    "PipelineConfig",
    "PreparedHestonMarketFit",
    "ProviderCallDiagnostic",
    "ProviderRetryConfig",
    "ProviderSnapshotBundleValidationResult",
    "ProviderSnapshotDataUnavailableError",
    "ProviderSnapshotQualityPolicy",
    "ProviderSnapshotResult",
    "PreparedSVIMarketFit",
    "SurfaceReadyStats",
    "StorageConfig",
    "load_model_validation_bundle",
    "prepare_heston_market_fit",
    "prepare_svi_market_fit",
    "provider_snapshot_public_summary",
    "validate_provider_snapshot_bundle",
]

if TYPE_CHECKING:
    from .bundles import LoadedModelValidationBundle, load_model_validation_bundle
    from .config import (
        AlpacaConfig,
        FredConfig,
        PipelineConfig,
        ProviderRetryConfig,
        StorageConfig,
    )
    from .model_ready import (
        HestonReadyStats,
        PreparedHestonMarketFit,
        prepare_heston_market_fit,
    )
    from .pipeline import (
        MarketDataPipeline,
        ProviderSnapshotDataUnavailableError,
        ProviderSnapshotResult,
        provider_snapshot_public_summary,
    )
    from .provider_confidence import (
        ProviderSnapshotBundleValidationResult,
        validate_provider_snapshot_bundle,
    )
    from .provider_policy import ProviderSnapshotQualityPolicy
    from .provider_results import ProviderCallDiagnostic
    from .storage import LocalStorage
    from .surface_ready import (
        PreparedSVIMarketFit,
        SurfaceReadyStats,
        prepare_svi_market_fit,
    )

_PUBLIC_EXPORTS = {
    "AlpacaConfig": "option_pricing.marketdata.config",
    "FredConfig": "option_pricing.marketdata.config",
    "HestonReadyStats": "option_pricing.marketdata.model_ready",
    "PipelineConfig": "option_pricing.marketdata.config",
    "PreparedHestonMarketFit": "option_pricing.marketdata.model_ready",
    "ProviderRetryConfig": "option_pricing.marketdata.config",
    "StorageConfig": "option_pricing.marketdata.config",
    "PreparedSVIMarketFit": "option_pricing.marketdata.surface_ready",
    "SurfaceReadyStats": "option_pricing.marketdata.surface_ready",
    "LoadedModelValidationBundle": "option_pricing.marketdata.bundles",
    "LocalStorage": "option_pricing.marketdata.storage",
    "MarketDataPipeline": "option_pricing.marketdata.pipeline",
    "ProviderCallDiagnostic": "option_pricing.marketdata.provider_results",
    "ProviderSnapshotBundleValidationResult": (
        "option_pricing.marketdata.provider_confidence"
    ),
    "ProviderSnapshotDataUnavailableError": "option_pricing.marketdata.pipeline",
    "ProviderSnapshotQualityPolicy": "option_pricing.marketdata.provider_policy",
    "ProviderSnapshotResult": "option_pricing.marketdata.pipeline",
    "load_model_validation_bundle": "option_pricing.marketdata.bundles",
    "prepare_heston_market_fit": "option_pricing.marketdata.model_ready",
    "prepare_svi_market_fit": "option_pricing.marketdata.surface_ready",
    "provider_snapshot_public_summary": "option_pricing.marketdata.pipeline",
    "validate_provider_snapshot_bundle": "option_pricing.marketdata.provider_confidence",
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
