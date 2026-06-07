"""Public workflow helpers."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "HestonCalibrationConfig",
    "HestonMarketFitError",
    "HestonMarketFitResult",
    "fit_heston_from_bundle",
    "fit_heston_market",
]

if TYPE_CHECKING:
    from .market_fit import (
        HestonCalibrationConfig,
        HestonMarketFitError,
        HestonMarketFitResult,
        fit_heston_from_bundle,
        fit_heston_market,
    )

_PUBLIC_EXPORTS = {
    "HestonCalibrationConfig": "option_pricing.workflows.market_fit",
    "HestonMarketFitError": "option_pricing.workflows.market_fit",
    "HestonMarketFitResult": "option_pricing.workflows.market_fit",
    "fit_heston_from_bundle": "option_pricing.workflows.market_fit",
    "fit_heston_market": "option_pricing.workflows.market_fit",
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
