"""Model-validation bundle contracts and manifest construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from option_pricing.marketdata.contracts import ModelValidationBundleResult
from option_pricing.marketdata.manifests import validate_model_validation_manifest
from option_pricing.marketdata.schemas import MODEL_VALIDATION_BUNDLE_VERSION

_MODEL_VALIDATION_ARTIFACTS = {
    "market_data": "market_data.json",
    "cleaned_quotes": "cleaned_quotes.parquet",
    "rejected_quotes": "rejected_quotes.parquet",
    "heston_quotes": "heston_quotes.parquet",
    "surface_inputs": "surface_inputs.parquet",
    "heston_fit_summary": "heston_fit_summary.csv",
    "warnings": "warnings.json",
}


@dataclass(frozen=True, slots=True)
class ModelValidationBundlePaths:
    root: Path
    manifest: Path
    market_data: Path
    cleaned_quotes: Path
    rejected_quotes: Path
    heston_quotes: Path
    surface_inputs: Path
    heston_fit_summary: Path
    warnings: Path


@dataclass(frozen=True, slots=True)
class HestonSmokeResult:
    status: Literal["success", "failed", "skipped"]
    message: str
    objective_type: str
    quote_count: int
    success_count: int | None = None
    failure_count: int | None = None
    best_cost: float | None = None
    parameters: Mapping[str, float] | None = None


@dataclass(frozen=True, slots=True)
class ModelValidationBundleConfig:
    run_heston_smoke: bool = True
    heston_objective_type: str = "price_rmse"
    heston_max_seeds: int = 1
    heston_max_nfev: int | None = 1
    fail_on_heston_smoke_failure: bool = False


def build_model_validation_manifest(
    *,
    run_id: str,
    snapshot_id: str,
    underlying: str,
    valuation_timestamp_utc: str,
    market_data_payload: Mapping[str, object],
    rows: Mapping[str, int],
    reason_counts: Mapping[str, int],
    warnings: Sequence[str],
    artifacts: Mapping[str, str],
    heston_smoke: HestonSmokeResult,
    library_commit: str | None = None,
) -> dict[str, object]:
    """Build the self-contained model-validation bundle manifest payload."""

    manifest: dict[str, object] = {
        "artifact_schema_version": MODEL_VALIDATION_BUNDLE_VERSION,
        "run_id": _required_text("run_id", run_id),
        "snapshot_id": _required_text("snapshot_id", snapshot_id),
        "created_at_utc": _utc_now_isoformat(),
        "library_commit": _optional_text("library_commit", library_commit),
        "underlying": _required_text("underlying", underlying),
        "valuation_timestamp_utc": _required_text(
            "valuation_timestamp_utc",
            valuation_timestamp_utc,
        ),
        "spot_source": _market_source_text(market_data_payload, "spot_source"),
        "rate_source": _market_source_text(market_data_payload, "rate_source"),
        "rate_compounding": _required_mapping_text(
            market_data_payload,
            "rate_compounding",
            source_name="market_data_payload",
        ),
        "dividend_yield_source": _market_source_text(
            market_data_payload,
            "dividend_yield_source",
        ),
        "day_count": _required_mapping_text(
            market_data_payload,
            "day_count",
            source_name="market_data_payload",
        ),
        "quote_cleaning_policy": _required_mapping_text(
            market_data_payload,
            "quote_cleaning_policy",
            source_name="market_data_payload",
        ),
        "rows": _int_mapping_payload("rows", rows),
        "reason_counts": _int_mapping_payload(
            "reason_counts",
            reason_counts,
            sort_keys=True,
        ),
        "warnings": _warnings_payload(warnings),
        "artifacts": _artifact_payload(artifacts),
        "heston_smoke": _heston_smoke_payload(heston_smoke),
    }
    validate_model_validation_manifest(manifest)
    return manifest


def _artifact_payload(artifacts: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(artifacts, Mapping):
        raise TypeError("artifacts must be a mapping")

    artifact_map = dict(artifacts)
    if artifact_map != _MODEL_VALIDATION_ARTIFACTS:
        raise ValueError(
            "model-validation bundle artifacts must match the frozen "
            f"A5-S1 filenames: {_MODEL_VALIDATION_ARTIFACTS!r}"
        )
    return dict(_MODEL_VALIDATION_ARTIFACTS)


def _heston_smoke_payload(result: HestonSmokeResult) -> dict[str, object]:
    if not isinstance(result, HestonSmokeResult):
        raise TypeError("heston_smoke must be a HestonSmokeResult")
    if result.status not in {"success", "failed", "skipped"}:
        raise ValueError("heston_smoke status must be success, failed, or skipped")

    parameters: dict[str, float] | None
    if result.parameters is None:
        parameters = None
    else:
        parameters = {
            str(name): float(value)
            for name, value in sorted(
                result.parameters.items(),
                key=lambda item: str(item[0]),
            )
        }

    return {
        "status": result.status,
        "message": result.message,
        "objective_type": result.objective_type,
        "quote_count": int(result.quote_count),
        "success_count": result.success_count,
        "failure_count": result.failure_count,
        "best_cost": result.best_cost,
        "parameters": parameters,
    }


def _market_source_text(payload: Mapping[str, object], key: str) -> str:
    sources = payload.get("sources")
    if isinstance(sources, Mapping) and key in sources:
        return _required_mapping_text(
            cast(Mapping[str, object], sources),
            key,
            source_name="market_data_payload.sources",
        )
    return _required_mapping_text(payload, key, source_name="market_data_payload")


def _int_mapping_payload(
    name: str,
    values: Mapping[str, int],
    *,
    sort_keys: bool = False,
) -> dict[str, int]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")

    items = list(values.items())
    if sort_keys:
        items.sort(key=lambda item: str(item[0]))

    payload: dict[str, int] = {}
    for key, value in items:
        count = int(value)
        if count < 0:
            raise ValueError(f"{name} counts must be non-negative")
        payload[str(key)] = count
    return payload


def _warnings_payload(warnings: Sequence[str]) -> list[str]:
    if isinstance(warnings, str | bytes | bytearray):
        raise TypeError("warnings must be a sequence of strings")

    payload: list[str] = []
    for warning in warnings:
        if not isinstance(warning, str):
            raise TypeError("warnings must be a sequence of strings")
        payload.append(warning)
    return payload


def _required_mapping_text(
    payload: Mapping[str, object],
    key: str,
    *,
    source_name: str,
) -> str:
    value = payload.get(key)
    return _required_text(f"{source_name}.{key}", value)


def _required_text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string")
    return text


def _optional_text(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    return _required_text(name, value)


def _utc_now_isoformat() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "HestonSmokeResult",
    "ModelValidationBundleConfig",
    "ModelValidationBundlePaths",
    "ModelValidationBundleResult",
    "build_model_validation_manifest",
]
