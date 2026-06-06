from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

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
    failure_kind: str | None = None
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
            payload["sanitized_message"] = self.message
            payload["message"] = self.message
        if self.failure_kind is not None:
            payload["failure_kind"] = self.failure_kind
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
    equity_provider: str = "alpaca"
    equity_feed: str = "iex"
    option_provider: str = "alpaca"
    option_feed: str = "indicative"
    selected_rate: float | None = None
    flat_rate: float | None = None
    rate_policy: Mapping[str, object] = field(default_factory=dict)
    dividend_policy: Mapping[str, object] = field(default_factory=dict)
    option_cleaning_policy: Mapping[str, object] = field(default_factory=dict)
    data_policy: Mapping[str, object] = field(default_factory=dict)

    def public_summary(self) -> dict[str, object]:
        """Return the compact reviewer-facing provider snapshot summary."""

        return provider_snapshot_public_summary(self)


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


def provider_snapshot_public_summary(result: object) -> dict[str, object]:
    """Normalize one provider snapshot result into a stable public summary.

    The summary intentionally uses already-normalized result metadata and local
    artifact references. It does not include provider payload bodies.
    """

    rate_policy = _mapping_attr(result, "rate_policy")
    dividend_policy = _mapping_attr(result, "dividend_policy")
    data_policy = _mapping_attr(result, "data_policy")
    freshness = _mapping_attr(result, "quote_freshness")
    warnings = _warnings_attr(result)

    return {
        "underlying": _json_value(_attr(result, "underlying")),
        "asof": _json_value(_attr(result, "asof")),
        "run_id": _json_value(_attr(result, "run_id")),
        "equity_provider": _json_value(_attr(result, "equity_provider")),
        "equity_feed": _json_value(_attr(result, "equity_feed")),
        "option_provider": _json_value(_attr(result, "option_provider")),
        "option_feed": _json_value(_attr(result, "option_feed", _attr(result, "feed"))),
        "rate_source": _json_value(_attr(result, "rate_source")),
        "selected_rate": _json_value(
            _attr(result, "selected_rate", _attr(result, "rate"))
        ),
        "flat_rate": _json_value(_attr(result, "flat_rate", _attr(result, "rate"))),
        "rate_policy_name": _json_value(rate_policy.get("policy")),
        "dividend_yield": _json_value(_attr(result, "dividend_yield")),
        "dividend_policy_name": _json_value(dividend_policy.get("policy")),
        "dividend_yield_source": _json_value(
            dividend_policy.get("source", _attr(result, "dividend_yield_source"))
        ),
        "raw_option_contract_count": _json_value(
            _attr(result, "raw_option_contract_count")
        ),
        "normalized_option_contract_count": _json_value(
            _attr(result, "normalized_option_contract_count")
        ),
        "provider_rejected_contract_count": _json_value(
            _attr(result, "provider_rejected_contract_count")
        ),
        "accepted_quote_count": _json_value(_attr(result, "accepted_quote_count")),
        "rejected_quote_count": _json_value(_attr(result, "rejected_quote_count")),
        "accepted_expiry_count": _json_value(freshness.get("accepted_expiry_count")),
        "accepted_expiry_days_min": _json_value(
            freshness.get("accepted_expiry_days_min")
        ),
        "accepted_expiry_days_max": _json_value(
            freshness.get("accepted_expiry_days_max")
        ),
        "accepted_expiry_years_min": _json_value(
            freshness.get("accepted_expiry_years_min")
        ),
        "accepted_expiry_years_max": _json_value(
            freshness.get("accepted_expiry_years_max")
        ),
        "accepted_strike_min": _json_value(freshness.get("accepted_strike_min")),
        "accepted_strike_max": _json_value(freshness.get("accepted_strike_max")),
        "accepted_call_count": _json_value(freshness.get("accepted_call_count")),
        "accepted_put_count": _json_value(freshness.get("accepted_put_count")),
        "quote_freshness_mode": _json_value(
            freshness.get(
                "quote_freshness_mode",
                data_policy.get("quote_freshness_mode"),
            )
        ),
        "warning_count": len(warnings),
        "main_artifact_paths": _main_artifact_path_summary(result),
    }


def _attr(result: object, name: str, default: object = None) -> object:
    return getattr(result, name, default)


def _mapping_attr(result: object, name: str) -> Mapping[str, object]:
    value = _attr(result, name, {})
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return {}


def _warnings_attr(result: object) -> tuple[str, ...]:
    warnings = _attr(result, "warnings", ())
    if warnings is None:
        return ()
    if isinstance(warnings, str):
        return (warnings,)
    if isinstance(warnings, Sequence):
        return tuple(str(warning) for warning in warnings)
    return (str(warnings),)


def _main_artifact_path_summary(result: object) -> dict[str, str]:
    bronze_paths = _attr(result, "bronze_paths")
    silver_paths = _attr(result, "silver_paths")
    gold_paths = _attr(result, "gold_paths")
    rate_curve_paths = _attr(result, "rate_curve_paths")
    bundle = _attr(result, "model_validation_bundle")
    candidates = {
        "bronze_manifest": _nested_attr(bronze_paths, "manifest"),
        "silver_manifest": _nested_attr(silver_paths, "manifest"),
        "provider_rejected_contracts": _nested_attr(
            silver_paths,
            "provider_rejected_contracts",
        ),
        "market_data": _nested_attr(gold_paths, "market_data"),
        "market_manifest": _nested_attr(gold_paths, "market_manifest"),
        "rate_curve": _nested_attr(rate_curve_paths, "rate_curve"),
        "rate_curve_manifest": _nested_attr(rate_curve_paths, "manifest"),
        "bundle_manifest": _nested_attr(bundle, "manifest_path"),
    }
    return {
        name: path
        for name, value in candidates.items()
        if (path := _path_reference(value)) is not None
    }


def _nested_attr(value: object, name: str) -> object:
    if value is None:
        return None
    return getattr(value, name, None)


def _path_reference(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value.as_posix()
    return str(value).replace("\\", "/")


def _json_value(value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat) and not isinstance(value, str):
        return isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_json_value(item) for item in value]
    return value


__all__ = [
    "ProviderCallDiagnostic",
    "ProviderCallStatus",
    "ProviderRefreshDailyCounts",
    "ProviderRefreshDailyResult",
    "ProviderSnapshotBronzePaths",
    "ProviderSnapshotRateCurvePaths",
    "ProviderSnapshotResult",
    "ProviderSnapshotSilverPaths",
    "provider_snapshot_public_summary",
]
