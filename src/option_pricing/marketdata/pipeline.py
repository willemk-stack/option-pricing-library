"""Local-first marketdata pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from pathlib import Path

import pandas as pd

from option_pricing.marketdata.bundles import (
    ModelValidationBundleConfig,
    ModelValidationBundlePaths,
    write_model_validation_bundle_artifacts,
)
from option_pricing.marketdata.cleaning import (
    QuoteCleaningPolicyV1,
    QuoteCleaningResult,
    clean_option_quotes,
)
from option_pricing.marketdata.config import StorageConfig
from option_pricing.marketdata.contracts import ModelValidationBundleResult
from option_pricing.marketdata.gold import GoldConversionPaths, write_gold_artifacts
from option_pricing.marketdata.normalize import (
    normalize_market_inputs,
    normalize_option_chain,
)
from option_pricing.marketdata.providers.local import (
    LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1,
    LocalSnapshotBronzePaths,
    LocalSnapshotConfig,
    LocalSnapshotProvider,
    LocalSnapshotResult,
    write_local_snapshot_bronze,
)
from option_pricing.marketdata.schemas import DatasetName
from option_pricing.marketdata.silver import (
    SilverCleaningPaths,
    write_cleaned_quotes_silver,
)
from option_pricing.marketdata.storage import LocalStorage, PartitionValue


@dataclass(frozen=True, slots=True)
class LocalModelValidationPipelineResult:
    """Typed result for one local model-validation pipeline run."""

    local_snapshot: LocalSnapshotResult
    market_inputs: pd.DataFrame
    option_chain: pd.DataFrame
    quote_cleaning: QuoteCleaningResult
    bronze_paths: LocalSnapshotBronzePaths
    silver_paths: SilverCleaningPaths
    gold_paths: GoldConversionPaths
    model_validation_bundle: ModelValidationBundleResult


@dataclass(frozen=True, slots=True)
class _PipelineTargetPaths:
    bronze_paths: LocalSnapshotBronzePaths
    silver_paths: SilverCleaningPaths
    gold_paths: GoldConversionPaths
    model_validation_bundle_paths: ModelValidationBundlePaths


def run_local_model_validation_pipeline(
    *,
    storage: LocalStorage | StorageConfig | Path,
    run_id: str,
    fixture_name: str = LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1,
    fixture_root: Path | None = None,
    expected_underlying: str | None = None,
    cleaning_policy: QuoteCleaningPolicyV1 | None = None,
    bundle_config: ModelValidationBundleConfig | None = None,
    overwrite: bool = False,
    library_commit: str | None = None,
) -> LocalModelValidationPipelineResult:
    """Run the narrow A5 local fixture-to-model-validation bundle pipeline."""

    required_run_id = _required_run_id(run_id)
    effective_cleaning_policy = _cleaning_policy(cleaning_policy)
    local_storage = _coerce_storage(storage)
    local_snapshot = LocalSnapshotProvider(
        LocalSnapshotConfig(
            fixture_root=fixture_root,
            fixture_name=fixture_name,
            expected_underlying=expected_underlying,
            run_id=required_run_id,
        )
    ).load_snapshot()
    _preflight_pipeline_targets(
        local_storage,
        local_snapshot,
        overwrite=overwrite,
    )

    bronze_paths = write_local_snapshot_bronze(
        local_storage,
        local_snapshot,
        overwrite=overwrite,
        library_commit=library_commit,
    )
    market_inputs = normalize_market_inputs(local_snapshot.market_inputs_raw)
    option_chain = normalize_option_chain(local_snapshot.option_chain_raw)
    quote_cleaning = clean_option_quotes(
        option_chain,
        market_inputs,
        policy=effective_cleaning_policy,
    )
    silver_paths = write_cleaned_quotes_silver(
        local_storage,
        local_snapshot=local_snapshot,
        market_inputs=market_inputs,
        result=quote_cleaning,
        overwrite=overwrite,
        library_commit=library_commit,
    )
    gold_paths = write_gold_artifacts(
        local_storage,
        local_snapshot=local_snapshot,
        market_inputs=market_inputs,
        cleaned_quotes=quote_cleaning.cleaned_quotes,
        rejected_quotes=quote_cleaning.rejected_quotes,
        reason_counts=quote_cleaning.reason_counts,
        warnings=quote_cleaning.warnings,
        overwrite=overwrite,
        library_commit=library_commit,
    )
    model_validation_bundle = write_model_validation_bundle_artifacts(
        local_storage,
        local_snapshot=local_snapshot,
        market_inputs=market_inputs,
        cleaned_quotes=quote_cleaning.cleaned_quotes,
        rejected_quotes=quote_cleaning.rejected_quotes,
        reason_counts=quote_cleaning.reason_counts,
        warnings=quote_cleaning.warnings,
        config=bundle_config,
        overwrite=overwrite,
        library_commit=library_commit,
    )

    return LocalModelValidationPipelineResult(
        local_snapshot=local_snapshot,
        market_inputs=market_inputs,
        option_chain=option_chain,
        quote_cleaning=quote_cleaning,
        bronze_paths=bronze_paths,
        silver_paths=silver_paths,
        gold_paths=gold_paths,
        model_validation_bundle=model_validation_bundle,
    )


def _required_run_id(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("run_id must be a string")
    run_id = value.strip()
    if not run_id:
        raise ValueError("run_id is required")
    return run_id


def _coerce_storage(storage: LocalStorage | StorageConfig | Path) -> LocalStorage:
    if isinstance(storage, LocalStorage):
        return storage
    if isinstance(storage, StorageConfig):
        return LocalStorage(storage)
    if isinstance(storage, Path):
        return LocalStorage(StorageConfig(root=storage))
    raise TypeError(
        "storage must be a LocalStorage, StorageConfig, or pathlib.Path, "
        f"got {type(storage).__name__}"
    )


def _cleaning_policy(
    policy: QuoteCleaningPolicyV1 | None,
) -> QuoteCleaningPolicyV1:
    if policy is None:
        return QuoteCleaningPolicyV1()
    if not isinstance(policy, QuoteCleaningPolicyV1):
        raise TypeError(
            "cleaning_policy must be a QuoteCleaningPolicyV1, "
            f"got {type(policy).__name__}"
        )
    return policy


def _preflight_pipeline_targets(
    storage: LocalStorage,
    local_snapshot: LocalSnapshotResult,
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        return

    paths = _expected_pipeline_target_paths(storage, local_snapshot)
    for path in _iter_pipeline_target_paths(paths):
        if path.exists():
            raise FileExistsError(
                f"{path} already exists; pass overwrite=True to replace it"
            )


def _expected_pipeline_target_paths(
    storage: LocalStorage,
    local_snapshot: LocalSnapshotResult,
) -> _PipelineTargetPaths:
    partitions = _pipeline_partitions(local_snapshot)
    bronze_root = storage.dataset_dir(
        layer="bronze",
        dataset="local_snapshot",
        partitions=partitions,
    )
    bundle_root = storage.dataset_dir(
        layer="gold",
        dataset=DatasetName.MODEL_VALIDATION_BUNDLE.value,
        partitions=partitions,
    )
    return _PipelineTargetPaths(
        bronze_paths=LocalSnapshotBronzePaths(
            root=bronze_root,
            manifest=bronze_root / "manifest.json",
            market_inputs=bronze_root / "market_inputs.parquet",
            option_chain=bronze_root / "option_chain.parquet",
        ),
        silver_paths=SilverCleaningPaths(
            market_inputs=_target_path(
                storage,
                layer="silver",
                dataset=DatasetName.MARKET_INPUTS.value,
                partitions=partitions,
                filename="market_inputs.parquet",
            ),
            cleaned_quotes=_target_path(
                storage,
                layer="silver",
                dataset=DatasetName.CLEANED_QUOTES.value,
                partitions=partitions,
                filename="cleaned_quotes.parquet",
            ),
            rejected_quotes=_target_path(
                storage,
                layer="silver",
                dataset=DatasetName.REJECTED_QUOTES.value,
                partitions=partitions,
                filename="rejected_quotes.parquet",
            ),
            manifest=_target_path(
                storage,
                layer="silver",
                dataset=DatasetName.CLEANED_QUOTES.value,
                partitions=partitions,
                filename="manifest.json",
            ),
        ),
        gold_paths=GoldConversionPaths(
            market_data=_target_path(
                storage,
                layer="gold",
                dataset=DatasetName.MARKET_SNAPSHOT.value,
                partitions=partitions,
                filename="market_data.json",
            ),
            market_manifest=_target_path(
                storage,
                layer="gold",
                dataset=DatasetName.MARKET_SNAPSHOT.value,
                partitions=partitions,
                filename="manifest.json",
            ),
            heston_quotes=_target_path(
                storage,
                layer="gold",
                dataset=DatasetName.HESTON_QUOTES.value,
                partitions=partitions,
                filename="heston_quotes.parquet",
            ),
            heston_manifest=_target_path(
                storage,
                layer="gold",
                dataset=DatasetName.HESTON_QUOTES.value,
                partitions=partitions,
                filename="manifest.json",
            ),
        ),
        model_validation_bundle_paths=ModelValidationBundlePaths(
            root=bundle_root,
            manifest=bundle_root / "manifest.json",
            market_data=bundle_root / "market_data.json",
            cleaned_quotes=bundle_root / "cleaned_quotes.parquet",
            rejected_quotes=bundle_root / "rejected_quotes.parquet",
            heston_quotes=bundle_root / "heston_quotes.parquet",
            surface_inputs=bundle_root / "surface_inputs.parquet",
            heston_fit_summary=bundle_root / "heston_fit_summary.csv",
            warnings=bundle_root / "warnings.json",
        ),
    )


def _pipeline_partitions(
    local_snapshot: LocalSnapshotResult,
) -> dict[str, PartitionValue]:
    valuation_timestamp = _utc_timestamp(local_snapshot.asof)
    return {
        "underlying": local_snapshot.underlying,
        "date": valuation_timestamp.date(),
        "run_id": _required_snapshot_run_id(local_snapshot.run_id),
    }


def _required_snapshot_run_id(value: str | None) -> str:
    if value is None:
        raise ValueError("local_snapshot.run_id is required")
    return _required_run_id(value)


def _utc_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(UTC)
    return timestamp.tz_convert(UTC)


def _target_path(
    storage: LocalStorage,
    *,
    layer: str,
    dataset: str,
    partitions: dict[str, PartitionValue],
    filename: str,
) -> Path:
    return (
        storage.dataset_dir(
            layer=layer,
            dataset=dataset,
            partitions=partitions,
        )
        / filename
    )


def _iter_pipeline_target_paths(paths: _PipelineTargetPaths) -> tuple[Path, ...]:
    return (
        paths.bronze_paths.market_inputs,
        paths.bronze_paths.option_chain,
        paths.bronze_paths.manifest,
        paths.silver_paths.market_inputs,
        paths.silver_paths.cleaned_quotes,
        paths.silver_paths.rejected_quotes,
        paths.silver_paths.manifest,
        paths.gold_paths.market_data,
        paths.gold_paths.market_manifest,
        paths.gold_paths.heston_quotes,
        paths.gold_paths.heston_manifest,
        paths.model_validation_bundle_paths.manifest,
        paths.model_validation_bundle_paths.market_data,
        paths.model_validation_bundle_paths.cleaned_quotes,
        paths.model_validation_bundle_paths.rejected_quotes,
        paths.model_validation_bundle_paths.heston_quotes,
        paths.model_validation_bundle_paths.surface_inputs,
        paths.model_validation_bundle_paths.heston_fit_summary,
        paths.model_validation_bundle_paths.warnings,
    )


__all__ = [
    "LocalModelValidationPipelineResult",
    "run_local_model_validation_pipeline",
]
