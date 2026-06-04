from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

import pandas as pd

from option_pricing.marketdata.bundles import ModelValidationBundlePaths
from option_pricing.marketdata.cleaning import QuoteCleaningResult
from option_pricing.marketdata.contracts import ModelValidationBundleResult
from option_pricing.marketdata.gold import GoldConversionPaths
from option_pricing.marketdata.provider_diagnostics import _diagnostics_payload
from option_pricing.marketdata.provider_policy import (
    DEFAULT_DAY_COUNT,
    DEFAULT_RATE_SERIES_ID,
    PROVIDER_RATE_CURVE_COLUMNS,
    _current_provider_scope,
    _provider_snapshot_data_policy,
    _provider_snapshot_dividend_policy,
    _provider_snapshot_option_cleaning_policy,
    _provider_snapshot_rate_policy,
)
from option_pricing.marketdata.provider_results import (
    ProviderSnapshotBronzePaths,
    ProviderSnapshotRateCurvePaths,
    ProviderSnapshotSilverPaths,
)
from option_pricing.marketdata.provider_serialization import (
    _provider_payload_document,
    _relative_artifact_references,
    _sanitized_request_metadata,
    _utc_isoformat,
    _utc_timestamp,
)
from option_pricing.marketdata.rates import build_fred_treasury_zero_proxy_curve
from option_pricing.marketdata.schemas import DatasetName
from option_pricing.marketdata.silver import write_cleaned_quotes_silver
from option_pricing.marketdata.storage import LocalStorage, PartitionValue

PROVIDER_SNAPSHOT_BRONZE_SCHEMA_VERSION = "provider_snapshot_bronze.v1"
PROVIDER_SNAPSHOT_FIXTURE_NAME = "provider_snapshot_v1"
PROVIDER_SNAPSHOT_SOURCE_TYPE = "provider_snapshot"
_PROVIDER_REJECTED_CONTRACTS_DATASET = "provider_rejected_contracts"
_PROVIDER_RATE_CURVE_DATASET = "curves"
_PROVIDER_RATE_CURVE_SCHEMA_VERSION = "provider_rate_curve_gold.v1"


@dataclass(frozen=True, slots=True, eq=False)
class _ProviderSnapshot:
    fixture_name: str
    snapshot_id: str
    run_id: str
    underlying: str
    asof: pd.Timestamp
    manifest: dict[str, Any]
    market_inputs_raw: pd.DataFrame
    option_chain_raw: pd.DataFrame
    metadata: dict[str, Any]
    row_counts: dict[str, int]
    warnings: tuple[str, ...] = ()

    @property
    def source_type(self) -> str:
        return PROVIDER_SNAPSHOT_SOURCE_TYPE


@dataclass(frozen=True, slots=True)
class _ProviderSnapshotTargetPaths:
    bronze_paths: ProviderSnapshotBronzePaths
    silver_paths: ProviderSnapshotSilverPaths
    gold_paths: GoldConversionPaths
    rate_curve_paths: ProviderSnapshotRateCurvePaths | None
    model_validation_bundle_paths: ModelValidationBundlePaths


def _build_provider_snapshot_rate_curve(
    *,
    asof: pd.Timestamp,
    day_count: str,
    primary_series_id: str,
    primary_fred_series: pd.DataFrame,
    requested_series_ids: Sequence[str],
    load_series_frame: Callable[[str], pd.DataFrame | None],
) -> pd.DataFrame:
    frames_by_series_id: dict[str, pd.DataFrame] = {}
    for series_id in requested_series_ids:
        frame = (
            primary_fred_series
            if series_id == primary_series_id
            else load_series_frame(series_id)
        )
        if frame is None:
            continue
        frames_by_series_id[str(series_id)] = frame
    del day_count
    return build_fred_treasury_zero_proxy_curve(frames_by_series_id, asof=asof).loc[
        :,
        list(PROVIDER_RATE_CURVE_COLUMNS),
    ]


def _provider_snapshot_request_metadata(
    *,
    underlying: str,
    asof: pd.Timestamp,
    expiry_gte: date | str | None,
    expiry_lte: date | str | None,
    strike_gte: float | None,
    strike_lte: float | None,
    option_type: str | None,
    equity_feed: str,
    option_feed: str,
    rate_series_id: str,
    rate_lookback_days: int,
    curve_series_ids: Sequence[str],
    fred_observation_end: date | None = None,
    fred_observation_start: date | None = None,
) -> dict[str, object]:
    request_metadata: dict[str, Any] = {
        "underlying": underlying,
        "asof": asof,
        "expiry_gte": expiry_gte,
        "expiry_lte": expiry_lte,
        "strike_gte": strike_gte,
        "strike_lte": strike_lte,
        "option_type": option_type,
        "equity_feed": equity_feed,
        "option_feed": option_feed,
        "rate_series_id": rate_series_id,
        "rate_lookback_days": int(rate_lookback_days),
        "curve_series_ids": tuple(str(series_id) for series_id in curve_series_ids),
    }
    if fred_observation_start is not None:
        request_metadata["fred_observation_start"] = fred_observation_start
    if fred_observation_end is not None:
        request_metadata["fred_observation_end"] = fred_observation_end
    return _sanitized_request_metadata(request_metadata)


def _provider_snapshot_target_stub(
    *,
    underlying: str,
    asof: pd.Timestamp,
    run_id: str,
) -> _ProviderSnapshot:
    return _ProviderSnapshot(
        fixture_name=PROVIDER_SNAPSHOT_FIXTURE_NAME,
        snapshot_id=_provider_snapshot_id(
            underlying=underlying,
            asof=asof,
            run_id=run_id,
        ),
        run_id=run_id,
        underlying=underlying,
        asof=asof,
        manifest={},
        market_inputs_raw=pd.DataFrame(),
        option_chain_raw=pd.DataFrame(),
        metadata={},
        row_counts={},
    )


def _quote_cleaning_result_with_warnings(
    result: QuoteCleaningResult,
    warnings: Sequence[str],
) -> QuoteCleaningResult:
    return QuoteCleaningResult(
        cleaned_quotes=result.cleaned_quotes,
        rejected_quotes=result.rejected_quotes,
        reason_counts=result.reason_counts,
        warnings=tuple(warnings),
    )


def _provider_snapshot_id(
    *,
    underlying: str,
    asof: pd.Timestamp,
    run_id: str,
) -> str:
    return (
        f"{PROVIDER_SNAPSHOT_FIXTURE_NAME}:{underlying}:{_utc_isoformat(asof)}:{run_id}"
    )


def _preflight_provider_snapshot_targets(
    storage: LocalStorage,
    provider_snapshot: _ProviderSnapshot,
    *,
    rate_series_id: str,
    include_rate_curve: bool,
    overwrite: bool,
) -> None:
    if overwrite:
        return

    paths = _expected_provider_snapshot_target_paths(
        storage,
        provider_snapshot,
        rate_series_id=rate_series_id,
        include_rate_curve=include_rate_curve,
    )
    for path in _iter_provider_snapshot_target_paths(paths):
        if path.exists():
            raise FileExistsError(
                f"{path} already exists; pass overwrite=True to replace it"
            )


def _write_provider_snapshot_bronze(
    storage: LocalStorage,
    provider_snapshot: _ProviderSnapshot,
    *,
    equity_quote_payload: Mapping[str, Any],
    option_chain_payload: Mapping[str, Any],
    fred_payload: Mapping[str, Any],
    rate_series_id: str,
    request_metadata: Mapping[str, Any],
    diagnostics: Sequence[object],
    quality_policy: Mapping[str, object],
    quote_freshness: Mapping[str, object],
    overwrite: bool,
    library_commit: str | None,
) -> ProviderSnapshotBronzePaths:
    partitions = _provider_snapshot_partitions(provider_snapshot)
    paths = _expected_provider_bronze_paths(storage, provider_snapshot)

    latest_equity_quotes_path = storage.write_json(
        _provider_payload_document(equity_quote_payload),
        layer="bronze",
        dataset=PROVIDER_SNAPSHOT_SOURCE_TYPE,
        partitions=partitions,
        filename="latest_equity_quotes.json",
        overwrite=overwrite,
    )
    option_chain_path = storage.write_json(
        _provider_payload_document(option_chain_payload),
        layer="bronze",
        dataset=PROVIDER_SNAPSHOT_SOURCE_TYPE,
        partitions=partitions,
        filename="option_chain.json",
        overwrite=overwrite,
    )
    fred_observations_path = storage.write_json(
        _provider_payload_document(fred_payload),
        layer="bronze",
        dataset=PROVIDER_SNAPSHOT_SOURCE_TYPE,
        partitions=partitions,
        filename="fred_observations.json",
        overwrite=overwrite,
    )
    manifest_path = storage.write_manifest(
        _provider_bronze_manifest(
            provider_snapshot,
            rate_series_id=rate_series_id,
            request_metadata=request_metadata,
            diagnostics=diagnostics,
            quality_policy=quality_policy,
            quote_freshness=quote_freshness,
            library_commit=library_commit,
        ),
        layer="bronze",
        dataset=PROVIDER_SNAPSHOT_SOURCE_TYPE,
        partitions=partitions,
        filename="manifest.json",
        overwrite=overwrite,
    )
    return ProviderSnapshotBronzePaths(
        root=paths.root,
        manifest=manifest_path,
        latest_equity_quotes=latest_equity_quotes_path,
        option_chain=option_chain_path,
        fred_observations=fred_observations_path,
    )


def _write_provider_snapshot_silver(
    storage: LocalStorage,
    provider_snapshot: _ProviderSnapshot,
    *,
    option_chain: pd.DataFrame,
    fred_series: pd.DataFrame,
    market_inputs: pd.DataFrame,
    quote_cleaning: QuoteCleaningResult,
    provider_rejected_contracts: pd.DataFrame,
    rate_series_id: str,
    overwrite: bool,
    library_commit: str | None,
) -> ProviderSnapshotSilverPaths:
    cleaning_paths = write_cleaned_quotes_silver(
        storage,
        local_snapshot=cast(Any, provider_snapshot),
        market_inputs=market_inputs,
        result=quote_cleaning,
        overwrite=overwrite,
        library_commit=library_commit,
    )
    valuation_timestamp = _utc_timestamp(provider_snapshot.asof)
    option_chain_path = storage.write_frame(
        option_chain,
        layer="silver",
        dataset=DatasetName.OPTION_CHAIN.value,
        partitions={
            "underlying": provider_snapshot.underlying,
            "asof_date": valuation_timestamp.date(),
            "run_id": provider_snapshot.run_id,
        },
        filename="option_chain.parquet",
        overwrite=overwrite,
    )
    fred_series_path = storage.write_frame(
        fred_series,
        layer="silver",
        dataset=DatasetName.FRED_SERIES.value,
        partitions={
            "series_id": rate_series_id,
            "date": valuation_timestamp.date(),
            "run_id": provider_snapshot.run_id,
        },
        filename="fred_series.parquet",
        overwrite=overwrite,
    )
    provider_rejected_contracts_path = storage.write_frame(
        provider_rejected_contracts,
        layer="silver",
        dataset=_PROVIDER_REJECTED_CONTRACTS_DATASET,
        partitions=_provider_snapshot_partitions(provider_snapshot),
        filename="provider_rejected_contracts.parquet",
        overwrite=overwrite,
    )
    return ProviderSnapshotSilverPaths(
        market_inputs=cleaning_paths.market_inputs,
        option_chain=option_chain_path,
        fred_series=fred_series_path,
        cleaned_quotes=cleaning_paths.cleaned_quotes,
        rejected_quotes=cleaning_paths.rejected_quotes,
        manifest=cleaning_paths.manifest,
        provider_rejected_contracts=provider_rejected_contracts_path,
    )


def _write_provider_snapshot_rate_curve_gold(
    storage: LocalStorage,
    provider_snapshot: _ProviderSnapshot,
    *,
    rate_curve: pd.DataFrame,
    requested_series_ids: Sequence[str],
    lookback_days: int,
    overwrite: bool,
    library_commit: str | None,
) -> ProviderSnapshotRateCurvePaths | None:
    if not requested_series_ids:
        return None

    partitions = _provider_snapshot_partitions(provider_snapshot)
    rate_curve_path = storage.write_frame(
        rate_curve,
        layer="gold",
        dataset=_PROVIDER_RATE_CURVE_DATASET,
        partitions=partitions,
        filename="rate_curve.parquet",
        overwrite=overwrite,
    )
    manifest_path = storage.write_manifest(
        _provider_snapshot_rate_curve_manifest(
            provider_snapshot,
            rate_curve=rate_curve,
            requested_series_ids=requested_series_ids,
            lookback_days=lookback_days,
            library_commit=library_commit,
        ),
        layer="gold",
        dataset=_PROVIDER_RATE_CURVE_DATASET,
        partitions=partitions,
        filename="manifest.json",
        overwrite=overwrite,
    )
    return ProviderSnapshotRateCurvePaths(
        rate_curve=rate_curve_path,
        manifest=manifest_path,
    )


def _provider_snapshot_rate_curve_manifest(
    provider_snapshot: _ProviderSnapshot,
    *,
    rate_curve: pd.DataFrame,
    requested_series_ids: Sequence[str],
    lookback_days: int,
    library_commit: str | None,
) -> dict[str, object]:
    included_series_ids = [
        str(series_id) for series_id in rate_curve["series_id"].tolist()
    ]
    return {
        "provider_snapshot_rate_curve_schema_version": _PROVIDER_RATE_CURVE_SCHEMA_VERSION,
        "artifact": "rate_curve",
        "run_id": provider_snapshot.run_id,
        "snapshot_id": provider_snapshot.snapshot_id,
        "underlying": provider_snapshot.underlying,
        "valuation_timestamp_utc": _utc_isoformat(provider_snapshot.asof),
        "rate_compounding": "continuous",
        "day_count": DEFAULT_DAY_COUNT,
        "lookback_days": int(lookback_days),
        "requested_series_ids": [str(series_id) for series_id in requested_series_ids],
        "included_series_ids": included_series_ids,
        "rows": {"rate_curve": int(len(rate_curve))},
        "artifacts": {"rate_curve": "rate_curve.parquet"},
        "source": {
            "source_type": PROVIDER_SNAPSHOT_SOURCE_TYPE,
            "fixture_name": provider_snapshot.fixture_name,
        },
        "rate_policy": {
            **_metadata_mapping(provider_snapshot, "rate_policy"),
            "provider": "fred",
            "primary_series_id": str(provider_snapshot.metadata["rate_series_id"]),
            "requested_series_ids": [
                str(series_id) for series_id in requested_series_ids
            ],
            "lookback_days": int(lookback_days),
        },
        "current_provider_scope": _current_provider_scope(),
        "library_commit": library_commit,
    }


def _provider_bronze_manifest(
    provider_snapshot: _ProviderSnapshot,
    *,
    rate_series_id: str,
    request_metadata: Mapping[str, Any],
    diagnostics: Sequence[object],
    quality_policy: Mapping[str, object],
    quote_freshness: Mapping[str, object],
    library_commit: str | None,
) -> dict[str, object]:
    sanitized_request = _sanitized_request_metadata(request_metadata)
    return {
        "provider_snapshot_schema_version": PROVIDER_SNAPSHOT_BRONZE_SCHEMA_VERSION,
        "fixture_name": provider_snapshot.fixture_name,
        "snapshot_id": provider_snapshot.snapshot_id,
        "run_id": provider_snapshot.run_id,
        "source_type": PROVIDER_SNAPSHOT_SOURCE_TYPE,
        "underlying": provider_snapshot.underlying,
        "valuation_timestamp_utc": _utc_isoformat(provider_snapshot.asof),
        "providers": provider_snapshot.metadata["providers"],
        "equity_provider": provider_snapshot.metadata["equity_provider"],
        "equity_feed": provider_snapshot.metadata["equity_feed"],
        "option_provider": provider_snapshot.metadata["option_provider"],
        "option_feed": provider_snapshot.metadata["option_feed"],
        "rate_series_id": rate_series_id,
        "rate_assumptions": {
            "provider": "fred",
            "series_id": rate_series_id,
            "default_series_id": DEFAULT_RATE_SERIES_ID,
            "rate_policy": "fred_treasury_zero_proxy_linear_cc",
            "rate_curve_source": "fred",
            "rate_interpolation": "linear",
            "rate_compounding": "continuous",
            "rate_extrapolation": "clamp_with_warning",
            "rate_is_bootstrapped": False,
        },
        "selected_rate": provider_snapshot.metadata["selected_rate"],
        "flat_rate": provider_snapshot.metadata["flat_rate"],
        "rate_policy": _metadata_mapping(provider_snapshot, "rate_policy"),
        "dividend_assumptions": _provider_snapshot_dividend_assumptions(
            provider_snapshot
        ),
        "dividend_policy": _metadata_mapping(provider_snapshot, "dividend_policy"),
        "option_cleaning_policy": _metadata_mapping(
            provider_snapshot,
            "option_cleaning_policy",
        ),
        "data_policy": _metadata_mapping(provider_snapshot, "data_policy"),
        "current_provider_scope": _current_provider_scope(),
        "quality_policy": dict(quality_policy),
        "quote_freshness": dict(quote_freshness),
        "provider_operation_diagnostics": _diagnostics_payload(cast(Any, diagnostics)),
        "request_metadata": sanitized_request,
        "rows": dict(provider_snapshot.row_counts),
        "warnings": list(provider_snapshot.warnings),
        "artifacts": {
            "latest_equity_quotes": "latest_equity_quotes.json",
            "option_chain": "option_chain.json",
            "fred_observations": "fred_observations.json",
        },
        "library_commit": library_commit,
    }


def _provider_snapshot_dividend_assumptions(
    provider_snapshot: _ProviderSnapshot,
) -> dict[str, object]:
    market_row = provider_snapshot.market_inputs_raw.iloc[0]
    dividend_policy = _metadata_mapping(provider_snapshot, "dividend_policy")
    return {
        "dividend_yield": _finite_float(market_row["dividend_yield"], "dividend_yield"),
        "source": _required_text(
            str(market_row["dividend_yield_source"]),
            "dividend_yield_source",
        ),
        "dividend_is_explicit": bool(
            dividend_policy.get("dividend_is_explicit", False)
        ),
        "dividend_fallback_used": bool(
            dividend_policy.get("dividend_fallback_used", False)
        ),
        "dividend_inference": "not_enabled",
    }


def _expected_provider_snapshot_target_paths(
    storage: LocalStorage,
    provider_snapshot: _ProviderSnapshot,
    *,
    rate_series_id: str,
    include_rate_curve: bool,
) -> _ProviderSnapshotTargetPaths:
    partitions = _provider_snapshot_partitions(provider_snapshot)
    valuation_timestamp = _utc_timestamp(provider_snapshot.asof)
    bundle_root = storage.dataset_dir(
        layer="gold",
        dataset=DatasetName.MODEL_VALIDATION_BUNDLE.value,
        partitions=partitions,
    )
    return _ProviderSnapshotTargetPaths(
        bronze_paths=_expected_provider_bronze_paths(storage, provider_snapshot),
        silver_paths=ProviderSnapshotSilverPaths(
            market_inputs=_target_path(
                storage,
                layer="silver",
                dataset=DatasetName.MARKET_INPUTS.value,
                partitions=partitions,
                filename="market_inputs.parquet",
            ),
            option_chain=_target_path(
                storage,
                layer="silver",
                dataset=DatasetName.OPTION_CHAIN.value,
                partitions={
                    "underlying": provider_snapshot.underlying,
                    "asof_date": valuation_timestamp.date(),
                    "run_id": provider_snapshot.run_id,
                },
                filename="option_chain.parquet",
            ),
            fred_series=_target_path(
                storage,
                layer="silver",
                dataset=DatasetName.FRED_SERIES.value,
                partitions={
                    "series_id": rate_series_id,
                    "date": valuation_timestamp.date(),
                    "run_id": provider_snapshot.run_id,
                },
                filename="fred_series.parquet",
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
            provider_rejected_contracts=_target_path(
                storage,
                layer="silver",
                dataset=_PROVIDER_REJECTED_CONTRACTS_DATASET,
                partitions=partitions,
                filename="provider_rejected_contracts.parquet",
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
        rate_curve_paths=(
            ProviderSnapshotRateCurvePaths(
                rate_curve=_target_path(
                    storage,
                    layer="gold",
                    dataset=_PROVIDER_RATE_CURVE_DATASET,
                    partitions=partitions,
                    filename="rate_curve.parquet",
                ),
                manifest=_target_path(
                    storage,
                    layer="gold",
                    dataset=_PROVIDER_RATE_CURVE_DATASET,
                    partitions=partitions,
                    filename="manifest.json",
                ),
            )
            if include_rate_curve
            else None
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


def _expected_provider_bronze_paths(
    storage: LocalStorage,
    provider_snapshot: _ProviderSnapshot,
) -> ProviderSnapshotBronzePaths:
    root = storage.dataset_dir(
        layer="bronze",
        dataset=PROVIDER_SNAPSHOT_SOURCE_TYPE,
        partitions=_provider_snapshot_partitions(provider_snapshot),
    )
    return ProviderSnapshotBronzePaths(
        root=root,
        manifest=root / "manifest.json",
        latest_equity_quotes=root / "latest_equity_quotes.json",
        option_chain=root / "option_chain.json",
        fred_observations=root / "fred_observations.json",
    )


def _provider_snapshot_partitions(
    provider_snapshot: _ProviderSnapshot,
) -> dict[str, PartitionValue]:
    valuation_timestamp = _utc_timestamp(provider_snapshot.asof)
    return {
        "underlying": provider_snapshot.underlying,
        "date": valuation_timestamp.date(),
        "run_id": provider_snapshot.run_id,
    }


def _iter_provider_snapshot_target_paths(
    paths: _ProviderSnapshotTargetPaths,
) -> tuple[Path, ...]:
    curve_paths: tuple[Path, ...] = ()
    if paths.rate_curve_paths is not None:
        curve_paths = (
            paths.rate_curve_paths.rate_curve,
            paths.rate_curve_paths.manifest,
        )

    return (
        paths.bronze_paths.latest_equity_quotes,
        paths.bronze_paths.option_chain,
        paths.bronze_paths.fred_observations,
        paths.bronze_paths.manifest,
        paths.silver_paths.market_inputs,
        paths.silver_paths.option_chain,
        paths.silver_paths.fred_series,
        paths.silver_paths.cleaned_quotes,
        paths.silver_paths.rejected_quotes,
        paths.silver_paths.manifest,
        *(
            ()
            if paths.silver_paths.provider_rejected_contracts is None
            else (paths.silver_paths.provider_rejected_contracts,)
        ),
        paths.gold_paths.market_data,
        paths.gold_paths.market_manifest,
        paths.gold_paths.heston_quotes,
        paths.gold_paths.heston_manifest,
        *curve_paths,
        paths.model_validation_bundle_paths.manifest,
        paths.model_validation_bundle_paths.market_data,
        paths.model_validation_bundle_paths.cleaned_quotes,
        paths.model_validation_bundle_paths.rejected_quotes,
        paths.model_validation_bundle_paths.heston_quotes,
        paths.model_validation_bundle_paths.surface_inputs,
        paths.model_validation_bundle_paths.heston_fit_summary,
        paths.model_validation_bundle_paths.warnings,
    )


def _provider_snapshot_artifact_paths(
    *,
    bronze_paths: ProviderSnapshotBronzePaths,
    silver_paths: ProviderSnapshotSilverPaths,
    gold_paths: GoldConversionPaths,
    rate_curve_paths: ProviderSnapshotRateCurvePaths | None,
    model_validation_bundle: ModelValidationBundleResult,
) -> tuple[Path, ...]:
    curve_artifacts: tuple[Path, ...] = ()
    if rate_curve_paths is not None:
        curve_artifacts = (
            rate_curve_paths.rate_curve,
            rate_curve_paths.manifest,
        )

    provider_rejected_contracts_artifact: tuple[Path, ...] = ()
    if silver_paths.provider_rejected_contracts is not None:
        provider_rejected_contracts_artifact = (
            silver_paths.provider_rejected_contracts,
        )

    return (
        bronze_paths.latest_equity_quotes,
        bronze_paths.option_chain,
        bronze_paths.fred_observations,
        bronze_paths.manifest,
        silver_paths.market_inputs,
        silver_paths.option_chain,
        silver_paths.fred_series,
        silver_paths.cleaned_quotes,
        silver_paths.rejected_quotes,
        silver_paths.manifest,
        *provider_rejected_contracts_artifact,
        gold_paths.market_data,
        gold_paths.market_manifest,
        gold_paths.heston_quotes,
        gold_paths.heston_manifest,
        *curve_artifacts,
        *model_validation_bundle.artifact_paths,
        model_validation_bundle.manifest_path,
    )


def _provider_snapshot_run_details(
    *,
    storage: LocalStorage,
    underlying: str,
    asof: pd.Timestamp,
    run_id: str,
    rate_series_id: str,
    rate_source: str,
    rate_observation_date: pd.Timestamp,
    spot_source: str,
    dividend_yield: float,
    dividend_yield_source: str,
    equity_feed: str,
    option_feed: str,
    selected_rate: float,
    flat_rate: float,
    rate_lookback_days: int,
    raw_option_contract_count: int,
    normalized_option_contract_count: int,
    dropped_before_cleaning_count: int,
    provider_rejected_contract_count: int,
    accepted_quote_count: int,
    rejected_quote_count: int,
    diagnostics: Sequence[object],
    quality_policy: Mapping[str, object],
    quote_freshness: Mapping[str, object],
    curve_series_ids: Sequence[str],
    rate_warnings: Sequence[str],
    selected_rate_fallback_used: bool,
    warnings: Sequence[str],
    artifact_paths: Sequence[Path],
    library_commit: str | None,
) -> dict[str, object]:
    rate_policy = _provider_snapshot_rate_policy(
        rate_series_id=rate_series_id,
        rate_source=rate_source,
        rate_observation_date=rate_observation_date,
        selected_rate=selected_rate,
        flat_rate=flat_rate,
        lookback_days=rate_lookback_days,
        curve_series_ids=curve_series_ids,
        rate_warnings=rate_warnings,
        selected_rate_fallback_used=selected_rate_fallback_used,
    )
    dividend_policy = _provider_snapshot_dividend_policy(
        dividend_yield=dividend_yield,
        dividend_yield_source=dividend_yield_source,
    )
    option_cleaning_policy = _provider_snapshot_option_cleaning_policy()
    data_policy = _provider_snapshot_data_policy(
        equity_provider="alpaca",
        equity_feed=equity_feed,
        option_provider="alpaca",
        option_feed=option_feed,
        rate_policy=rate_policy,
        dividend_policy=dividend_policy,
        option_cleaning_policy=option_cleaning_policy,
        quote_freshness_mode=str(
            quote_freshness.get("quote_freshness_mode", "demo_lenient")
        ),
    )
    return {
        "operation": "snapshot",
        "provider": "alpaca+fred",
        "equity_provider": "alpaca",
        "equity_feed": equity_feed,
        "option_provider": "alpaca",
        "option_feed": option_feed,
        "underlying": underlying,
        "asof": _utc_isoformat(asof),
        "run_id": run_id,
        "rate_series_id": rate_series_id,
        "rate_source": rate_source,
        "rate_observation_date": rate_observation_date.date().isoformat(),
        "selected_rate": float(selected_rate),
        "flat_rate": float(flat_rate),
        "spot_source": spot_source,
        "dividend_yield": dividend_yield,
        "dividend_yield_source": dividend_yield_source,
        "dividend_policy": dividend_policy,
        "rate_lookback_days": rate_lookback_days,
        "rate_policy": rate_policy,
        "option_cleaning_policy": option_cleaning_policy,
        "data_policy": data_policy,
        "raw_option_contract_count": raw_option_contract_count,
        "normalized_option_contract_count": normalized_option_contract_count,
        "dropped_before_cleaning_count": dropped_before_cleaning_count,
        "provider_rejected_contract_count": provider_rejected_contract_count,
        "accepted_quote_count": accepted_quote_count,
        "rejected_quote_count": rejected_quote_count,
        "quality_policy": dict(quality_policy),
        "quote_freshness": dict(quote_freshness),
        "provider_operation_diagnostics": _diagnostics_payload(cast(Any, diagnostics)),
        "current_provider_scope": _current_provider_scope(),
        "warnings": list(warnings),
        "artifact_paths": _relative_artifact_references(storage.root, artifact_paths),
        "library_commit": library_commit,
    }


def _metadata_mapping(
    provider_snapshot: _ProviderSnapshot,
    key: str,
) -> dict[str, object]:
    value = provider_snapshot.metadata.get(key)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    return cleaned


def _finite_float(value: object, field_name: str) -> float:
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be numeric") from exc
    if not pd.notna(number):
        raise ValueError(f"{field_name} must be finite")
    if not number == number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{field_name} must be finite")
    return number


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
