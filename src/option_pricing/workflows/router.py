"""Thin convenience router for supported market-fit one-shot workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from option_pricing.marketdata.bundles import LoadedModelValidationBundle

from .market_fit import HestonMarketFitResult, fit_heston_from_bundle
from .surface_fit import (
    ESSVIMarketFitResult,
    SVIMarketFitResult,
    fit_essvi_from_bundle,
    fit_svi_from_bundle,
)

_UNSUPPORTED_MARKET_FIT_MODEL_MESSAGE = (
    "Supported market-fit models are 'heston', 'svi', and 'essvi'. "
    "Black-Scholes, trees, Monte Carlo, PDE, and direct local-vol are not "
    "market-fit workflows; use their explicit pricing, simulation, diagnostics, "
    "or surface-derived APIs instead."
)


def fit_market_model(
    model: Literal["heston", "svi", "essvi"],
    path_or_bundle: str | Path | LoadedModelValidationBundle,
    **kwargs: Any,
) -> HestonMarketFitResult | SVIMarketFitResult | ESSVIMarketFitResult:
    """Delegate to the explicit one-shot helper for a supported market-fit model.

    Prefer ``fit_heston_from_bundle(...)``, ``fit_svi_from_bundle(...)``, or
    ``fit_essvi_from_bundle(...)`` in model-specific code and documentation.
    This helper exists for narrow dispatch cases where the model name is
    already data.
    """

    if model == "heston":
        return fit_heston_from_bundle(path_or_bundle, **kwargs)
    if model == "svi":
        return fit_svi_from_bundle(path_or_bundle, **kwargs)
    if model == "essvi":
        return fit_essvi_from_bundle(path_or_bundle, **kwargs)

    raise ValueError(
        f"Unsupported market-fit model {model!r}. "
        f"{_UNSUPPORTED_MARKET_FIT_MODEL_MESSAGE}"
    )


__all__ = ["fit_market_model"]
