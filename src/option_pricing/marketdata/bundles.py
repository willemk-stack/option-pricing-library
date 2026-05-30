"""Model-validation bundle contracts and manifest construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import pandas as pd

from option_pricing.marketdata.contracts import (
    ModelValidationBundleResult,
    ResultStats,
    RunMetadata,
)
from option_pricing.marketdata.gold import (
    GoldHestonQuotesResult,
    GoldMarketDataSnapshot,
    _cleaning_policy_from_cleaned_quotes,
    build_heston_quotes,
    build_market_data_snapshot,
    heston_quote_set_from_frame,
    market_data_snapshot_to_json,
)
from option_pricing.marketdata.manifests import validate_model_validation_manifest
from option_pricing.marketdata.schemas import (
    MODEL_VALIDATION_BUNDLE_VERSION,
    SURFACE_INPUTS_COLUMNS,
    DatasetName,
)
from option_pricing.marketdata.storage import LocalStorage, PartitionValue
from option_pricing.marketdata.validation import coerce_frame, validate_dtypes
from option_pricing.models.heston.calibration import calibrate_heston_multistart

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


class _HestonParamsLike(Protocol):
    @property
    def kappa(self) -> float: ...

    @property
    def vbar(self) -> float: ...

    @property
    def eta(self) -> float: ...

    @property
    def rho(self) -> float: ...

    @property
    def v(self) -> float: ...


class _HestonCalibrationRunLike(Protocol):
    @property
    def cost(self) -> float: ...


class _HestonMultistartResultLike(Protocol):
    @property
    def best_params(self) -> _HestonParamsLike: ...

    @property
    def best_run(self) -> _HestonCalibrationRunLike: ...

    @property
    def quote_count(self) -> int: ...

    @property
    def success_count(self) -> int: ...

    @property
    def failure_count(self) -> int: ...

    @property
    def jacobian_mode(self) -> str: ...

    @property
    def backend(self) -> str: ...


@dataclass(frozen=True, slots=True)
class _HestonSmokeRun:
    result: HestonSmokeResult
    jacobian_mode: str | None = None
    backend: str | None = None


class _ModelValidationLocalSnapshot(Protocol):
    @property
    def snapshot_id(self) -> str: ...

    @property
    def run_id(self) -> str | None: ...

    @property
    def underlying(self) -> str: ...

    @property
    def asof(self) -> object: ...


def build_surface_inputs(cleaned_quotes: pd.DataFrame) -> pd.DataFrame:
    """Build the schema-stable surface-input frame from cleaned quotes."""

    if not isinstance(cleaned_quotes, pd.DataFrame):
        raise TypeError(
            "cleaned_quotes must be a pandas DataFrame, "
            f"got {type(cleaned_quotes).__name__}"
        )
    validate_dtypes(cleaned_quotes, DatasetName.CLEANED_QUOTES, allow_extra=False)

    source = cleaned_quotes.reset_index(drop=True)
    surface_inputs = pd.DataFrame(
        {column: source[column] for column in SURFACE_INPUTS_COLUMNS},
        columns=SURFACE_INPUTS_COLUMNS,
    )
    coerced = coerce_frame(
        surface_inputs,
        DatasetName.SURFACE_INPUTS,
        allow_extra=False,
    )
    validate_dtypes(coerced, DatasetName.SURFACE_INPUTS, allow_extra=False)
    return coerced.reset_index(drop=True)


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


def _write_model_validation_bundle_artifacts(
    storage: LocalStorage,
    *,
    local_snapshot: _ModelValidationLocalSnapshot,
    market_inputs: pd.DataFrame,
    cleaned_quotes: pd.DataFrame,
    rejected_quotes: pd.DataFrame,
    reason_counts: Mapping[str, int],
    warnings: Sequence[str],
    config: ModelValidationBundleConfig | None = None,
    overwrite: bool = False,
    library_commit: str | None = None,
) -> ModelValidationBundleResult:
    """Write the A5-S2 self-contained model-validation bundle artifacts."""

    _require_local_storage(storage)
    config = _model_validation_config(config)

    run_id = _required_run_id(local_snapshot.run_id)
    snapshot_id = _required_text(
        "local_snapshot.snapshot_id", local_snapshot.snapshot_id
    )
    underlying = _required_text("local_snapshot.underlying", local_snapshot.underlying)
    valuation_timestamp = _utc_timestamp(
        local_snapshot.asof,
        field_name="local_snapshot.asof",
    )
    library_commit = _optional_text("library_commit", library_commit)

    validate_dtypes(rejected_quotes, DatasetName.REJECTED_QUOTES, allow_extra=False)
    _require_matching_underlying(
        rejected_quotes,
        frame_name="rejected_quotes",
        expected=underlying,
    )

    cleaning_policy = _cleaning_policy_from_cleaned_quotes(cleaned_quotes)
    market_snapshot = build_market_data_snapshot(
        market_inputs,
        run_id=run_id,
        snapshot_id=snapshot_id,
        cleaning_policy=cleaning_policy,
        library_commit=library_commit,
    )
    market_data_payload = market_data_snapshot_to_json(market_snapshot)
    _require_payload_matches_snapshot(
        market_data_payload,
        underlying=underlying,
        valuation_timestamp=valuation_timestamp,
    )

    heston_result = build_heston_quotes(cleaned_quotes)
    _require_matching_underlying(
        heston_result.heston_quotes,
        frame_name="heston_quotes",
        expected=underlying,
    )
    surface_inputs = build_surface_inputs(cleaned_quotes)
    _require_matching_underlying(
        surface_inputs,
        frame_name="surface_inputs",
        expected=underlying,
    )

    partitions: dict[str, PartitionValue] = {
        "underlying": underlying,
        "date": valuation_timestamp.date(),
        "run_id": run_id,
    }
    paths = _expected_model_validation_bundle_paths(storage, partitions)
    _ensure_model_validation_bundle_targets_available(paths, overwrite=overwrite)

    heston_smoke_run = _heston_smoke_run(
        config=config,
        market_data_snapshot=market_snapshot,
        heston_result=heston_result,
    )
    heston_smoke = heston_smoke_run.result
    if heston_smoke.status == "failed" and config.fail_on_heston_smoke_failure:
        raise RuntimeError(f"Heston smoke failed: {heston_smoke.message}")

    warning_payload = _warnings_payload(warnings)
    data_quality_warnings = list(heston_result.warnings)
    bundle_warnings = [*warning_payload, *data_quality_warnings]
    manifest = build_model_validation_manifest(
        run_id=run_id,
        snapshot_id=snapshot_id,
        underlying=underlying,
        valuation_timestamp_utc=_utc_isoformat(valuation_timestamp),
        market_data_payload=market_data_payload,
        rows={
            "market_inputs": int(len(market_inputs)),
            "cleaned_quotes": int(len(cleaned_quotes)),
            "rejected_quotes": int(len(rejected_quotes)),
            "heston_quotes": int(heston_result.quote_count),
            "surface_inputs": int(len(surface_inputs)),
        },
        reason_counts=reason_counts,
        warnings=bundle_warnings,
        artifacts=_MODEL_VALIDATION_ARTIFACTS,
        heston_smoke=heston_smoke,
        library_commit=library_commit,
    )

    market_data_path = storage.write_json(
        market_data_payload,
        layer="gold",
        dataset=DatasetName.MODEL_VALIDATION_BUNDLE.value,
        partitions=partitions,
        filename=_MODEL_VALIDATION_ARTIFACTS["market_data"],
        overwrite=overwrite,
    )
    cleaned_quotes_path = storage.write_frame(
        cleaned_quotes,
        layer="gold",
        dataset=DatasetName.MODEL_VALIDATION_BUNDLE.value,
        partitions=partitions,
        filename=_MODEL_VALIDATION_ARTIFACTS["cleaned_quotes"],
        overwrite=overwrite,
    )
    rejected_quotes_path = storage.write_frame(
        rejected_quotes,
        layer="gold",
        dataset=DatasetName.MODEL_VALIDATION_BUNDLE.value,
        partitions=partitions,
        filename=_MODEL_VALIDATION_ARTIFACTS["rejected_quotes"],
        overwrite=overwrite,
    )
    heston_quotes_path = storage.write_frame(
        heston_result.heston_quotes,
        layer="gold",
        dataset=DatasetName.MODEL_VALIDATION_BUNDLE.value,
        partitions=partitions,
        filename=_MODEL_VALIDATION_ARTIFACTS["heston_quotes"],
        overwrite=overwrite,
    )
    surface_inputs_path = storage.write_frame(
        surface_inputs,
        layer="gold",
        dataset=DatasetName.MODEL_VALIDATION_BUNDLE.value,
        partitions=partitions,
        filename=_MODEL_VALIDATION_ARTIFACTS["surface_inputs"],
        overwrite=overwrite,
    )
    _write_heston_fit_summary_csv(
        paths.heston_fit_summary,
        heston_smoke=heston_smoke,
        config=config,
        smoke_run=heston_smoke_run,
    )
    warnings_path = storage.write_json(
        {
            "warnings": warning_payload,
            "data_quality": data_quality_warnings,
            "heston_smoke": [heston_smoke.message],
        },
        layer="gold",
        dataset=DatasetName.MODEL_VALIDATION_BUNDLE.value,
        partitions=partitions,
        filename=_MODEL_VALIDATION_ARTIFACTS["warnings"],
        overwrite=overwrite,
    )
    manifest_path = storage.write_manifest(
        manifest,
        layer="gold",
        dataset=DatasetName.MODEL_VALIDATION_BUNDLE.value,
        partitions=partitions,
        filename="manifest.json",
        overwrite=overwrite,
    )

    artifact_paths = (
        market_data_path,
        cleaned_quotes_path,
        rejected_quotes_path,
        heston_quotes_path,
        surface_inputs_path,
        paths.heston_fit_summary,
        warnings_path,
    )
    return ModelValidationBundleResult(
        manifest=manifest,
        manifest_path=manifest_path,
        metadata=RunMetadata(
            run_id=run_id,
            asof=valuation_timestamp.to_pydatetime(),
            started_at=datetime.now(UTC),
            git_sha=library_commit,
        ),
        artifact_paths=artifact_paths,
        stats=ResultStats(
            rows_in=int(
                len(market_inputs) + len(cleaned_quotes) + len(rejected_quotes)
            ),
            rows_out=int(len(heston_result.heston_quotes) + len(surface_inputs)),
            files_written=(*artifact_paths, manifest_path),
            warnings=tuple(bundle_warnings),
        ),
    )


def _expected_model_validation_bundle_paths(
    storage: LocalStorage,
    partitions: Mapping[str, PartitionValue],
) -> ModelValidationBundlePaths:
    root = _model_validation_bundle_root(storage, partitions)
    return ModelValidationBundlePaths(
        root=root,
        manifest=root / "manifest.json",
        market_data=root / _MODEL_VALIDATION_ARTIFACTS["market_data"],
        cleaned_quotes=root / _MODEL_VALIDATION_ARTIFACTS["cleaned_quotes"],
        rejected_quotes=root / _MODEL_VALIDATION_ARTIFACTS["rejected_quotes"],
        heston_quotes=root / _MODEL_VALIDATION_ARTIFACTS["heston_quotes"],
        surface_inputs=root / _MODEL_VALIDATION_ARTIFACTS["surface_inputs"],
        heston_fit_summary=root / _MODEL_VALIDATION_ARTIFACTS["heston_fit_summary"],
        warnings=root / _MODEL_VALIDATION_ARTIFACTS["warnings"],
    )


def _model_validation_bundle_root(
    storage: LocalStorage,
    partitions: Mapping[str, PartitionValue],
) -> Path:
    dataset = DatasetName.MODEL_VALIDATION_BUNDLE.value
    ordered_partitions = storage._ordered_partitions(
        layer="gold",
        dataset=dataset,
        partitions=partitions,
    )
    return storage._dataset_dir(
        layer="gold",
        dataset=dataset,
        ordered_partitions=ordered_partitions,
    )


def _ensure_model_validation_bundle_targets_available(
    paths: ModelValidationBundlePaths,
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    for path in (
        paths.manifest,
        paths.market_data,
        paths.cleaned_quotes,
        paths.rejected_quotes,
        paths.heston_quotes,
        paths.surface_inputs,
        paths.heston_fit_summary,
        paths.warnings,
    ):
        if path.exists():
            raise FileExistsError(
                f"{path} already exists; pass overwrite=True to replace it"
            )


def _write_heston_fit_summary_csv(
    path: Path,
    *,
    heston_smoke: HestonSmokeResult,
    config: ModelValidationBundleConfig,
    smoke_run: _HestonSmokeRun,
) -> None:
    parameters = heston_smoke.parameters or {}
    attempted_smoke = heston_smoke.status != "skipped"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "status": heston_smoke.status,
                "message": heston_smoke.message,
                "objective_type": heston_smoke.objective_type,
                "quote_count": heston_smoke.quote_count,
                "success_count": heston_smoke.success_count,
                "failure_count": heston_smoke.failure_count,
                "best_cost": heston_smoke.best_cost,
                "kappa": parameters.get("kappa"),
                "vbar": parameters.get("vbar"),
                "eta": parameters.get("eta"),
                "rho": parameters.get("rho"),
                "v": parameters.get("v"),
                "jacobian_mode": smoke_run.jacobian_mode,
                "backend": smoke_run.backend,
                "max_seeds": (
                    int(config.heston_max_seeds) if attempted_smoke else None
                ),
                "max_nfev": (
                    None
                    if config.heston_max_nfev is None or not attempted_smoke
                    else int(config.heston_max_nfev)
                ),
            }
        ],
        columns=(
            "status",
            "message",
            "objective_type",
            "quote_count",
            "success_count",
            "failure_count",
            "best_cost",
            "kappa",
            "vbar",
            "eta",
            "rho",
            "v",
            "jacobian_mode",
            "backend",
            "max_seeds",
            "max_nfev",
        ),
    ).to_csv(path, index=False)


def _model_validation_config(
    config: ModelValidationBundleConfig | None,
) -> ModelValidationBundleConfig:
    if config is None:
        return ModelValidationBundleConfig()
    if not isinstance(config, ModelValidationBundleConfig):
        raise TypeError(
            "config must be a ModelValidationBundleConfig, "
            f"got {type(config).__name__}"
        )
    return config


def _skipped_heston_smoke_result(
    config: ModelValidationBundleConfig,
    *,
    quote_count: int,
) -> HestonSmokeResult:
    return HestonSmokeResult(
        status="skipped",
        message="Heston smoke skipped because config.run_heston_smoke is False.",
        objective_type=_required_text(
            "config.heston_objective_type",
            config.heston_objective_type,
        ),
        quote_count=int(quote_count),
    )


def _heston_smoke_run(
    *,
    config: ModelValidationBundleConfig,
    market_data_snapshot: GoldMarketDataSnapshot,
    heston_result: GoldHestonQuotesResult,
) -> _HestonSmokeRun:
    if not config.run_heston_smoke:
        return _HestonSmokeRun(
            result=_skipped_heston_smoke_result(
                config,
                quote_count=heston_result.quote_count,
            )
        )
    return _run_heston_smoke(
        config=config,
        market_data_snapshot=market_data_snapshot,
        heston_result=heston_result,
    )


def _run_heston_smoke(
    *,
    config: ModelValidationBundleConfig,
    market_data_snapshot: GoldMarketDataSnapshot,
    heston_result: GoldHestonQuotesResult,
) -> _HestonSmokeRun:
    objective_type = _required_text(
        "config.heston_objective_type",
        config.heston_objective_type,
    )
    quote_count = int(heston_result.quote_count)
    try:
        quote_set = heston_quote_set_from_frame(
            heston_result.heston_quotes,
            market_data_snapshot.market_data,
        )
        quote_count = int(quote_set.n_quotes)
        calibration_result = calibrate_heston_multistart(
            quote_set,
            objective_type=cast(Any, objective_type),
            max_seeds=config.heston_max_seeds,
            max_nfev=config.heston_max_nfev,
        )
    except Exception as exc:
        return _HestonSmokeRun(
            result=_failed_heston_smoke_result(
                exc,
                objective_type=objective_type,
                quote_count=quote_count,
            )
        )

    return _HestonSmokeRun(
        result=_successful_heston_smoke_result(
            calibration_result,
            objective_type=objective_type,
        ),
        jacobian_mode=str(calibration_result.jacobian_mode),
        backend=str(calibration_result.backend),
    )


def _failed_heston_smoke_result(
    exc: Exception,
    *,
    objective_type: str,
    quote_count: int = 0,
) -> HestonSmokeResult:
    return HestonSmokeResult(
        status="failed",
        message=f"{type(exc).__name__}: {exc}",
        objective_type=objective_type,
        quote_count=int(quote_count),
    )


def _successful_heston_smoke_result(
    calibration_result: _HestonMultistartResultLike,
    *,
    objective_type: str,
) -> HestonSmokeResult:
    parameters = calibration_result.best_params
    return HestonSmokeResult(
        status="success",
        message=(
            "Heston smoke calibration completed "
            f"with {int(calibration_result.success_count)} successful seed(s), "
            f"{int(calibration_result.failure_count)} failed seed(s), "
            f"best_cost={float(calibration_result.best_run.cost):.6g}."
        ),
        objective_type=objective_type,
        quote_count=int(calibration_result.quote_count),
        success_count=int(calibration_result.success_count),
        failure_count=int(calibration_result.failure_count),
        best_cost=float(calibration_result.best_run.cost),
        parameters={
            "kappa": float(parameters.kappa),
            "vbar": float(parameters.vbar),
            "eta": float(parameters.eta),
            "rho": float(parameters.rho),
            "v": float(parameters.v),
        },
    )


def _require_payload_matches_snapshot(
    payload: Mapping[str, object],
    *,
    underlying: str,
    valuation_timestamp: pd.Timestamp,
) -> None:
    payload_underlying = _required_mapping_text(
        payload,
        "underlying",
        source_name="market_data_payload",
    )
    if payload_underlying != underlying:
        raise ValueError(
            "market_data underlying must match local_snapshot.underlying "
            f"{underlying!r}; found {payload_underlying!r}"
        )

    payload_asof = _required_mapping_text(
        payload,
        "valuation_timestamp_utc",
        source_name="market_data_payload",
    )
    if payload_asof != _utc_isoformat(valuation_timestamp):
        raise ValueError(
            "market_data valuation_timestamp_utc must match local_snapshot.asof "
            f"{_utc_isoformat(valuation_timestamp)!r}; found {payload_asof!r}"
        )


def _require_matching_underlying(
    frame: pd.DataFrame,
    *,
    frame_name: str,
    expected: str,
) -> None:
    if frame.empty:
        return

    values: list[str] = []
    for value in frame["underlying"]:
        if pd.isna(value):
            rendered = ""
        else:
            rendered = str(value).strip()
        if rendered and rendered not in values:
            values.append(rendered)

    if values == [expected]:
        return

    found = ", ".join(repr(value) for value in values) or "<missing>"
    raise ValueError(
        f"{frame_name} underlying must match local_snapshot.underlying "
        f"{expected!r}; found {found}"
    )


def _require_run_id_text(name: str, value: object) -> str:
    return _required_text(name, value)


def _required_run_id(value: str | None) -> str:
    if value is None:
        raise ValueError("local_snapshot.run_id is required")
    return _require_run_id_text("local_snapshot.run_id", value)


def _require_local_storage(storage: LocalStorage) -> None:
    if not isinstance(storage, LocalStorage):
        raise TypeError(
            "storage must be a LocalStorage instance, " f"got {type(storage).__name__}"
        )


def _utc_timestamp(value: object, *, field_name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(cast(Any, value))
    if pd.isna(timestamp):
        raise ValueError(f"{field_name} must not be missing")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(UTC)
    return timestamp.tz_convert(UTC)


def _utc_isoformat(value: object) -> str:
    timestamp = _utc_timestamp(value, field_name="valuation_timestamp")
    return timestamp.to_pydatetime().isoformat().replace("+00:00", "Z")


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
    "build_surface_inputs",
]
