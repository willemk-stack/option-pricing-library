from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from option_pricing.marketdata.contracts import ModelValidationBundleResult
from option_pricing.marketdata.gold import GoldConversionPaths
from option_pricing.marketdata.provider_serialization import _sanitized_request_metadata

ProviderCallStatus = Literal["ok", "failed", "empty", "partial"]


@dataclass(frozen=True, slots=True)
class ProviderCallDiagnostic:
    """Sanitized timing and outcome details for one provider operation."""

    provider: str
    operation: str
    status: ProviderCallStatus
    request_metadata: Mapping[str, object]
    started_at: str
    ended_at: str
    elapsed_ms: float
    exception_type: str | None = None
    message: str | None = None
    retry_count: int = 0
    rows_or_contracts_in: int | None = None
    rows_or_contracts_out: int | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "provider": self.provider,
            "operation": self.operation,
            "status": self.status,
            "request_metadata": _sanitized_request_metadata(self.request_metadata),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed_ms": float(self.elapsed_ms),
            "retry_count": int(self.retry_count),
        }
        if self.exception_type is not None:
            payload["exception_type"] = self.exception_type
        if self.message is not None:
            payload["message"] = self.message
        if self.rows_or_contracts_in is not None:
            payload["rows_or_contracts_in"] = int(self.rows_or_contracts_in)
        if self.rows_or_contracts_out is not None:
            payload["rows_or_contracts_out"] = int(self.rows_or_contracts_out)
        return payload


@dataclass(frozen=True, slots=True)
class ProviderSnapshotBronzePaths:
    """Filesystem paths for one provider-backed Bronze snapshot evidence bundle."""

    root: Path
    manifest: Path
    latest_equity_quotes: Path
    option_chain: Path
    fred_observations: Path


@dataclass(frozen=True, slots=True)
class ProviderSnapshotSilverPaths:
    """Filesystem paths for one provider-backed Silver snapshot output set."""

    market_inputs: Path
    option_chain: Path
    fred_series: Path
    cleaned_quotes: Path
    rejected_quotes: Path
    manifest: Path
    provider_rejected_contracts: Path | None = None


@dataclass(frozen=True, slots=True)
class ProviderSnapshotRateCurvePaths:
    """Filesystem paths for one provider-backed Gold rate-curve artifact."""

    rate_curve: Path
    manifest: Path


@dataclass(frozen=True, slots=True)
class ProviderSnapshotResult:
    """Typed result for one provider-backed market-data snapshot."""

    underlying: str
    asof: pd.Timestamp
    run_id: str
    spot: float
    rate: float
    rate_source: str
    rate_observation_date: pd.Timestamp
    rate_series_id: str
    dividend_yield: float
    dividend_yield_source: str
    feed: str
    raw_option_contract_count: int
    normalized_option_contract_count: int
    accepted_quote_count: int
    rejected_quote_count: int
    dropped_before_cleaning_count: int
    warnings: tuple[str, ...]
    artifact_paths: tuple[Path, ...]
    bronze_paths: ProviderSnapshotBronzePaths
    silver_paths: ProviderSnapshotSilverPaths
    gold_paths: GoldConversionPaths
    model_validation_bundle: ModelValidationBundleResult
    provider_rejected_contract_count: int = 0
    rate_curve_paths: ProviderSnapshotRateCurvePaths | None = None
    diagnostics: tuple[ProviderCallDiagnostic, ...] = ()
    quality_policy: Mapping[str, object] = field(default_factory=dict)
    quote_freshness: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProviderRefreshDailyCounts:
    """Aggregate counts for one provider-backed refresh_daily run."""

    raw_option_contract_count: int
    normalized_option_contract_count: int
    dropped_before_cleaning_count: int
    provider_rejected_contract_count: int
    accepted_quote_count: int
    rejected_quote_count: int


@dataclass(frozen=True, slots=True)
class ProviderRefreshDailyResult:
    """Typed aggregate result for one provider-backed refresh_daily run."""

    aggregate_run_id: str
    child_run_ids: tuple[str, ...]
    underlyings: tuple[str, ...]
    artifact_paths: tuple[Path, ...]
    counts: ProviderRefreshDailyCounts
    warnings: tuple[str, ...]
    results: tuple[ProviderSnapshotResult, ...]


__all__ = [
    "ProviderCallDiagnostic",
    "ProviderCallStatus",
    "ProviderRefreshDailyCounts",
    "ProviderRefreshDailyResult",
    "ProviderSnapshotBronzePaths",
    "ProviderSnapshotRateCurvePaths",
    "ProviderSnapshotResult",
    "ProviderSnapshotSilverPaths",
]
