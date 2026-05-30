"""Local-first marketdata pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from option_pricing.marketdata.bundles import (
    ModelValidationBundleConfig,
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
from option_pricing.marketdata.silver import (
    SilverCleaningPaths,
    write_cleaned_quotes_silver,
)
from option_pricing.marketdata.storage import LocalStorage


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
    local_storage = _coerce_storage(storage)
    local_snapshot = LocalSnapshotProvider(
        LocalSnapshotConfig(
            fixture_root=fixture_root,
            fixture_name=fixture_name,
            expected_underlying=expected_underlying,
            run_id=required_run_id,
        )
    ).load_snapshot()

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
        policy=_cleaning_policy(cleaning_policy),
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
        local_snapshot=cast(Any, local_snapshot),
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


__all__ = [
    "LocalModelValidationPipelineResult",
    "run_local_model_validation_pipeline",
]
