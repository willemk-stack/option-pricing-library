"""Local-first marketdata pipeline orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, date, datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

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
from option_pricing.marketdata.config import (
    AlpacaConfig,
    FredConfig,
    PipelineConfig,
    StorageConfig,
)
from option_pricing.marketdata.contracts import (
    BackfillResult,
    ModelValidationBundleResult,
    ResultStats,
    RunMetadata,
)
from option_pricing.marketdata.errors import ProviderDataUnavailableError
from option_pricing.marketdata.gold import GoldConversionPaths, write_gold_artifacts
from option_pricing.marketdata.normalize import (
    normalize_alpaca_bars,
    normalize_alpaca_latest_quotes,
    normalize_alpaca_option_chain,
    normalize_fred_observations,
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
from option_pricing.marketdata.rates import select_latest_fred_rate_at_or_before_asof
from option_pricing.marketdata.schemas import DatasetName
from option_pricing.marketdata.silver import (
    SilverCleaningPaths,
    write_cleaned_quotes_silver,
)
from option_pricing.marketdata.storage import LocalStorage, PartitionValue

PROVIDER_SNAPSHOT_BRONZE_SCHEMA_VERSION = "provider_snapshot_bronze.v1"
PROVIDER_SNAPSHOT_FIXTURE_NAME = "provider_snapshot_v1"
PROVIDER_SNAPSHOT_SOURCE_TYPE = "provider_snapshot"
_DEFAULT_RATE_SERIES_ID = "DGS3MO"
_DEFAULT_BARS_TIMEFRAME = "1Day"
_DEFAULT_DAY_COUNT = "ACT/365"
_SECRET_KEY_PARTS = frozenset(
    {"api_key", "apikey", "secret", "token", "authorization", "password"}
)
FRED_BACKFILL_BRONZE_SCHEMA_VERSION = "fred_backfill_bronze.v1"
FRED_BACKFILL_SILVER_SCHEMA_VERSION = "fred_backfill_silver.v1"
EQUITY_BARS_BACKFILL_BRONZE_SCHEMA_VERSION = "equity_bars_backfill_bronze.v1"
EQUITY_BARS_BACKFILL_SILVER_SCHEMA_VERSION = "equity_bars_backfill_silver.v1"


class _AlpacaClientLike(Protocol):
    def get_latest_equity_quotes(
        self,
        symbols: str | Sequence[str],
        *,
        asof: object | None = None,
    ) -> Mapping[str, Any]: ...

    def get_equity_bars(
        self,
        symbols: str | Sequence[str],
        *,
        start: datetime | str,
        end: datetime | str,
        timeframe: str,
        limit: int | None = None,
        adjustment: str | None = None,
        sort: str | None = "asc",
        feed: str | None = None,
        asof: str | None = None,
    ) -> Mapping[str, Any]: ...

    def get_option_chain(
        self,
        underlying: str,
        *,
        expiry_gte: date | str | None = None,
        expiry_lte: date | str | None = None,
        strike_gte: float | None = None,
        strike_lte: float | None = None,
        option_type: str | None = None,
        root_symbol: str | None = None,
        updated_since: datetime | str | None = None,
        feed: str | None = None,
        asof: object | None = None,
    ) -> Mapping[str, Any]: ...


class _FredClientLike(Protocol):
    def fetch_observations(
        self,
        series_id: str,
        *,
        observation_start: date | datetime | str | None = None,
        observation_end: date | datetime | str | None = None,
        realtime_start: date | datetime | str | None = None,
        realtime_end: date | datetime | str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        sort_order: str | None = None,
        units: str | None = None,
        frequency: str | None = None,
        aggregation_method: str | None = None,
    ) -> Mapping[str, Any]: ...


class ProviderSnapshotDataUnavailableError(ProviderDataUnavailableError):
    """Raised when a provider-backed snapshot lacks required usable data."""


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
    dividend_yield: float
    dividend_yield_source: str
    accepted_quote_count: int
    rejected_quote_count: int
    dropped_before_cleaning_count: int
    warnings: tuple[str, ...]
    artifact_paths: tuple[Path, ...]
    bronze_paths: ProviderSnapshotBronzePaths
    silver_paths: ProviderSnapshotSilverPaths
    gold_paths: GoldConversionPaths
    model_validation_bundle: ModelValidationBundleResult


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
class _PipelineTargetPaths:
    bronze_paths: LocalSnapshotBronzePaths
    silver_paths: SilverCleaningPaths
    gold_paths: GoldConversionPaths
    model_validation_bundle_paths: ModelValidationBundlePaths


@dataclass(frozen=True, slots=True)
class _ProviderSnapshotTargetPaths:
    bronze_paths: ProviderSnapshotBronzePaths
    silver_paths: ProviderSnapshotSilverPaths
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
    """Run the local fixture-to-model-validation bundle pipeline."""

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


class MarketDataPipeline:
    """Provider-backed market-data orchestration for one live-capable snapshot."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        alpaca_client: _AlpacaClientLike | None = None,
        fred_client: _FredClientLike | None = None,
        storage: LocalStorage | StorageConfig | Path | None = None,
        cleaning_policy: QuoteCleaningPolicyV1 | None = None,
        bundle_config: ModelValidationBundleConfig | None = None,
    ) -> None:
        self.config, self.storage = _coerce_provider_pipeline_inputs(
            config,
            storage=storage,
        )
        self._alpaca_client = alpaca_client
        self._fred_client = fred_client
        self.cleaning_policy = _cleaning_policy(cleaning_policy)
        self.bundle_config = bundle_config or ModelValidationBundleConfig(
            run_heston_smoke=False
        )

    def snapshot(
        self,
        underlying: str,
        *,
        asof: str | pd.Timestamp | None = None,
        run_id: str | None = None,
        rate_series_id: str = _DEFAULT_RATE_SERIES_ID,
        expiry_gte: date | str | None = None,
        expiry_lte: date | str | None = None,
        strike_gte: float | None = None,
        strike_lte: float | None = None,
        option_type: str | None = None,
        feed: str | None = None,
        dividend_yield: float = 0.0,
        dividend_yield_source: str = "assumption",
        overwrite: bool = False,
        library_commit: str | None = None,
    ) -> ProviderSnapshotResult:
        """Fetch, normalize, clean, and persist one provider-backed snapshot."""

        cleaned_underlying = _clean_underlying(underlying)
        asof_timestamp = _coerce_asof(asof)
        effective_run_id = _optional_run_id(run_id) or _new_run_id(asof_timestamp)
        cleaned_rate_series_id = _clean_rate_series_id(rate_series_id)
        cleaned_dividend_yield_source = _required_text(
            dividend_yield_source,
            "dividend_yield_source",
        )
        cleaned_library_commit = _optional_text(library_commit, "library_commit")

        asof_label = _utc_isoformat(asof_timestamp)
        equity_quote_payload = self._fetch_latest_equity_quote(
            cleaned_underlying,
            asof=asof_label,
        )
        option_chain_payload = self._fetch_option_chain(
            cleaned_underlying,
            asof=asof_label,
            expiry_gte=expiry_gte,
            expiry_lte=expiry_lte,
            strike_gte=strike_gte,
            strike_lte=strike_lte,
            option_type=option_type,
            feed=feed,
        )
        fred_payload = self._fetch_fred_observations(
            cleaned_rate_series_id,
            asof=asof_timestamp,
        )

        equity_quotes = _normalize_latest_equity_quotes_for_snapshot(
            equity_quote_payload,
            underlying=cleaned_underlying,
            asof=asof_timestamp,
        )
        spot = _spot_from_equity_quotes(
            equity_quotes,
            underlying=cleaned_underlying,
        )
        raw_option_contract_count = _count_alpaca_option_contracts(option_chain_payload)
        provider_option_chain = _normalize_option_chain_for_snapshot(
            option_chain_payload,
            underlying=cleaned_underlying,
            asof=asof_timestamp,
        )
        option_chain = normalize_option_chain(provider_option_chain)
        dropped_before_cleaning_count = max(
            raw_option_contract_count - len(option_chain),
            0,
        )

        fred_series = _normalize_fred_observations_for_snapshot(
            fred_payload,
            series_id=cleaned_rate_series_id,
            asof=asof_timestamp,
        )
        rate_selection = _select_rate_for_snapshot(
            fred_series,
            series_id=cleaned_rate_series_id,
            asof=asof_timestamp,
        )
        rate_source = f"{rate_selection.source}:{rate_selection.series_id}"
        market_inputs = normalize_market_inputs(
            _market_inputs_frame(
                underlying=cleaned_underlying,
                asof=asof_timestamp,
                spot=spot,
                rate=rate_selection.rate,
                rate_source=rate_source,
                rate_observation_date=rate_selection.observation_date,
                rate_compounding=rate_selection.rate_compounding,
                dividend_yield=dividend_yield,
                dividend_yield_source=cleaned_dividend_yield_source,
            )
        )
        quote_cleaning = clean_option_quotes(
            option_chain,
            market_inputs,
            policy=self.cleaning_policy,
        )
        if quote_cleaning.cleaned_quotes.empty:
            raise ProviderSnapshotDataUnavailableError(
                "No usable option contracts remain after quote cleaning for "
                f"{cleaned_underlying!r}"
            )

        warnings = _provider_snapshot_warnings(
            cleaning_warnings=quote_cleaning.warnings,
            dropped_before_cleaning_count=dropped_before_cleaning_count,
            raw_option_contract_count=raw_option_contract_count,
            normalized_option_contract_count=len(option_chain),
        )
        provider_snapshot = _ProviderSnapshot(
            fixture_name=PROVIDER_SNAPSHOT_FIXTURE_NAME,
            snapshot_id=_provider_snapshot_id(
                underlying=cleaned_underlying,
                asof=asof_timestamp,
                run_id=effective_run_id,
            ),
            run_id=effective_run_id,
            underlying=cleaned_underlying,
            asof=asof_timestamp,
            manifest={},
            market_inputs_raw=market_inputs,
            option_chain_raw=option_chain,
            metadata={
                "source_type": PROVIDER_SNAPSHOT_SOURCE_TYPE,
                "providers": {
                    "spot": "alpaca",
                    "option_chain": "alpaca",
                    "rate": "fred",
                },
                "rate_series_id": cleaned_rate_series_id,
                "feed": feed or self.config.alpaca.feed,
            },
            row_counts={
                "equity_quotes": int(len(equity_quotes)),
                "option_contracts_raw": int(raw_option_contract_count),
                "option_contracts_normalized": int(len(option_chain)),
                "fred_observations": int(len(fred_series)),
                "cleaned_quotes": int(len(quote_cleaning.cleaned_quotes)),
                "rejected_quotes": int(len(quote_cleaning.rejected_quotes)),
            },
            warnings=warnings,
        )

        _preflight_provider_snapshot_targets(
            self.storage,
            provider_snapshot,
            rate_series_id=cleaned_rate_series_id,
            overwrite=overwrite,
        )
        bronze_paths = _write_provider_snapshot_bronze(
            self.storage,
            provider_snapshot,
            equity_quote_payload=equity_quote_payload,
            option_chain_payload=option_chain_payload,
            fred_payload=fred_payload,
            rate_series_id=cleaned_rate_series_id,
            feed=feed or self.config.alpaca.feed,
            overwrite=overwrite,
            library_commit=cleaned_library_commit,
        )
        silver_paths = _write_provider_snapshot_silver(
            self.storage,
            provider_snapshot,
            option_chain=option_chain,
            fred_series=fred_series,
            market_inputs=market_inputs,
            quote_cleaning=quote_cleaning,
            rate_series_id=cleaned_rate_series_id,
            overwrite=overwrite,
            library_commit=cleaned_library_commit,
        )
        gold_paths = write_gold_artifacts(
            self.storage,
            local_snapshot=provider_snapshot,
            market_inputs=market_inputs,
            cleaned_quotes=quote_cleaning.cleaned_quotes,
            rejected_quotes=quote_cleaning.rejected_quotes,
            reason_counts=quote_cleaning.reason_counts,
            warnings=warnings,
            overwrite=overwrite,
            library_commit=cleaned_library_commit,
        )
        model_validation_bundle = write_model_validation_bundle_artifacts(
            self.storage,
            local_snapshot=provider_snapshot,
            market_inputs=market_inputs,
            cleaned_quotes=quote_cleaning.cleaned_quotes,
            rejected_quotes=quote_cleaning.rejected_quotes,
            reason_counts=quote_cleaning.reason_counts,
            warnings=warnings,
            config=self.bundle_config,
            overwrite=overwrite,
            library_commit=cleaned_library_commit,
        )
        artifact_paths = _provider_snapshot_artifact_paths(
            bronze_paths=bronze_paths,
            silver_paths=silver_paths,
            gold_paths=gold_paths,
            model_validation_bundle=model_validation_bundle,
        )

        return ProviderSnapshotResult(
            underlying=cleaned_underlying,
            asof=asof_timestamp,
            run_id=effective_run_id,
            spot=spot,
            rate=rate_selection.rate,
            rate_source=rate_source,
            rate_observation_date=rate_selection.observation_date,
            dividend_yield=float(dividend_yield),
            dividend_yield_source=cleaned_dividend_yield_source,
            accepted_quote_count=int(len(quote_cleaning.cleaned_quotes)),
            rejected_quote_count=int(len(quote_cleaning.rejected_quotes)),
            dropped_before_cleaning_count=int(dropped_before_cleaning_count),
            warnings=warnings,
            artifact_paths=artifact_paths,
            bronze_paths=bronze_paths,
            silver_paths=silver_paths,
            gold_paths=gold_paths,
            model_validation_bundle=model_validation_bundle,
        )

    def backfill_fred(
        self,
        series_ids: str | Sequence[str],
        start: date | datetime | str,
        *,
        end: date | datetime | str | None = None,
        run_id: str | None = None,
        overwrite: bool = False,
        library_commit: str | None = None,
    ) -> BackfillResult:
        """Backfill one or more FRED series into Bronze and Silver storage."""

        started_at = datetime.now(UTC)
        metadata = _backfill_metadata(
            run_id=run_id,
            prefix="b4b-fred",
            started_at=started_at,
            library_commit=library_commit,
        )
        cleaned_library_commit = _optional_text(library_commit, "library_commit")
        cleaned_series_ids = _clean_fred_series_ids(series_ids)
        start_date = _coerce_backfill_date(start, "start")
        end_date = (
            started_at.date() if end is None else _coerce_backfill_date(end, "end")
        )
        if start_date > end_date:
            raise ValueError("start must be on or before end for FRED backfill")

        _preflight_fred_backfill_targets(
            self.storage,
            series_ids=cleaned_series_ids,
            end_date=end_date,
            overwrite=overwrite,
        )

        artifact_paths: list[Path] = []
        requests: list[dict[str, object]] = []
        rows_in = 0
        rows_out = 0
        asof = pd.Timestamp(metadata.asof)

        for series_id in cleaned_series_ids:
            request = {
                "series_id": series_id,
                "observation_start": start_date,
                "observation_end": end_date,
                "sort_order": "asc",
            }
            payload = self._resolve_fred_client().fetch_observations(
                series_id,
                observation_start=start_date,
                observation_end=end_date,
                sort_order="asc",
            )
            fred_series = normalize_fred_observations(
                payload,
                series_id=series_id,
                asof=asof,
            )
            raw_rows = _count_fred_observations(payload)
            normalized_rows = int(len(fred_series))
            rows_in += raw_rows
            rows_out += normalized_rows
            requests.append(request)

            partitions = _fred_backfill_partitions(
                series_id=series_id,
                end_date=end_date,
            )
            bronze_json_path = self.storage.write_json(
                _provider_backfill_payload_document(payload, request=request),
                layer="bronze",
                dataset=DatasetName.FRED_SERIES.value,
                partitions=partitions,
                filename="observations.json",
                overwrite=overwrite,
            )
            bronze_manifest_path = self.storage.write_manifest(
                _fred_backfill_manifest(
                    metadata,
                    series_id=series_id,
                    start_date=start_date,
                    end_date=end_date,
                    raw_rows=raw_rows,
                    normalized_rows=normalized_rows,
                    layer="bronze",
                    artifacts={"observations": "observations.json"},
                    library_commit=cleaned_library_commit,
                ),
                layer="bronze",
                dataset=DatasetName.FRED_SERIES.value,
                partitions=partitions,
                filename="manifest.json",
                overwrite=overwrite,
            )
            silver_frame_path = self.storage.write_frame(
                fred_series,
                layer="silver",
                dataset=DatasetName.FRED_SERIES.value,
                partitions=partitions,
                filename="fred_series.parquet",
                overwrite=overwrite,
            )
            silver_manifest_path = self.storage.write_manifest(
                _fred_backfill_manifest(
                    metadata,
                    series_id=series_id,
                    start_date=start_date,
                    end_date=end_date,
                    raw_rows=raw_rows,
                    normalized_rows=normalized_rows,
                    layer="silver",
                    artifacts={"fred_series": "fred_series.parquet"},
                    library_commit=cleaned_library_commit,
                ),
                layer="silver",
                dataset=DatasetName.FRED_SERIES.value,
                partitions=partitions,
                filename="manifest.json",
                overwrite=overwrite,
            )
            artifact_paths.extend(
                (
                    bronze_json_path,
                    bronze_manifest_path,
                    silver_frame_path,
                    silver_manifest_path,
                )
            )

        runs_path = self.storage.record_run(
            metadata,
            artifacts=artifact_paths,
            details=_backfill_run_details(
                operation="backfill_fred",
                provider="fred",
                targets=list(cleaned_series_ids),
                start=start_date,
                end=end_date,
                rows_in=rows_in,
                rows_out=rows_out,
                requests=requests,
                library_commit=cleaned_library_commit,
            ),
        )
        return BackfillResult(
            metadata=metadata,
            run_ids=(metadata.run_id,),
            artifact_paths=tuple(artifact_paths),
            stats=ResultStats(
                rows_in=rows_in,
                rows_out=rows_out,
                files_written=(*artifact_paths, runs_path),
            ),
        )

    def backfill_bars(
        self,
        symbols: str | Sequence[str],
        start: date | datetime | str,
        end: date | datetime | str,
        *,
        timeframe: str = _DEFAULT_BARS_TIMEFRAME,
        limit: int | None = None,
        adjustment: str | None = None,
        sort: str | None = "asc",
        feed: str | None = None,
        run_id: str | None = None,
        overwrite: bool = False,
        library_commit: str | None = None,
    ) -> BackfillResult:
        """Backfill Alpaca historical equity bars into Bronze and Silver storage."""

        started_at = datetime.now(UTC)
        metadata = _backfill_metadata(
            run_id=run_id,
            prefix="b4b-bars",
            started_at=started_at,
            library_commit=library_commit,
        )
        cleaned_library_commit = _optional_text(library_commit, "library_commit")
        cleaned_symbols = _clean_alpaca_symbols(symbols)
        cleaned_timeframe = _required_text(timeframe, "timeframe")
        cleaned_adjustment = _optional_text(adjustment, "adjustment")
        cleaned_sort = _optional_text(sort, "sort")
        cleaned_feed = _optional_text(feed, "feed")
        start_timestamp = _coerce_backfill_timestamp(start, "start")
        end_timestamp = _coerce_backfill_timestamp(end, "end")
        if start_timestamp >= end_timestamp:
            raise ValueError("start must be before end for Alpaca bars backfill")

        _preflight_bars_backfill_targets(
            self.storage,
            symbols=cleaned_symbols,
            timeframe=cleaned_timeframe,
            end_date=end_timestamp.date(),
            overwrite=overwrite,
        )

        asof = pd.Timestamp(metadata.asof)
        asof_label = _utc_isoformat(asof)
        request = {
            "symbols": list(cleaned_symbols),
            "start": start_timestamp.to_pydatetime(),
            "end": end_timestamp.to_pydatetime(),
            "timeframe": cleaned_timeframe,
            "limit": limit,
            "adjustment": cleaned_adjustment,
            "sort": cleaned_sort,
            "feed": cleaned_feed or self.config.alpaca.feed,
            "asof": asof_label,
        }
        payload = self._resolve_alpaca_client().get_equity_bars(
            cleaned_symbols,
            start=start_timestamp.to_pydatetime(),
            end=end_timestamp.to_pydatetime(),
            timeframe=cleaned_timeframe,
            limit=limit,
            adjustment=cleaned_adjustment,
            sort=cleaned_sort,
            feed=cleaned_feed,
            asof=asof_label,
        )
        bars = normalize_alpaca_bars(payload, asof=asof)

        artifact_paths: list[Path] = []
        rows_in = 0
        rows_out = 0
        for symbol in cleaned_symbols:
            symbol_bars = _bars_frame_for_symbol(bars, symbol)
            raw_rows = _count_alpaca_bars_for_symbol(payload, symbol)
            normalized_rows = int(len(symbol_bars))
            rows_in += raw_rows
            rows_out += normalized_rows

            partitions = _bars_backfill_partitions(
                symbol=symbol,
                timeframe=cleaned_timeframe,
                end_date=end_timestamp.date(),
            )
            bronze_json_path = self.storage.write_json(
                _provider_backfill_payload_document(
                    _alpaca_bars_payload_for_symbol(payload, symbol),
                    request={**request, "symbols": [symbol]},
                ),
                layer="bronze",
                dataset=DatasetName.EQUITY_BARS.value,
                partitions=partitions,
                filename="bars.json",
                overwrite=overwrite,
            )
            bronze_manifest_path = self.storage.write_manifest(
                _bars_backfill_manifest(
                    metadata,
                    symbol=symbol,
                    start=start_timestamp,
                    end=end_timestamp,
                    timeframe=cleaned_timeframe,
                    feed=cleaned_feed or self.config.alpaca.feed,
                    raw_rows=raw_rows,
                    normalized_rows=normalized_rows,
                    layer="bronze",
                    artifacts={"bars": "bars.json"},
                    library_commit=cleaned_library_commit,
                ),
                layer="bronze",
                dataset=DatasetName.EQUITY_BARS.value,
                partitions=partitions,
                filename="manifest.json",
                overwrite=overwrite,
            )
            silver_frame_path = self.storage.write_frame(
                symbol_bars,
                layer="silver",
                dataset=DatasetName.EQUITY_BARS.value,
                partitions=partitions,
                filename="equity_bars.parquet",
                overwrite=overwrite,
            )
            silver_manifest_path = self.storage.write_manifest(
                _bars_backfill_manifest(
                    metadata,
                    symbol=symbol,
                    start=start_timestamp,
                    end=end_timestamp,
                    timeframe=cleaned_timeframe,
                    feed=cleaned_feed or self.config.alpaca.feed,
                    raw_rows=raw_rows,
                    normalized_rows=normalized_rows,
                    layer="silver",
                    artifacts={"equity_bars": "equity_bars.parquet"},
                    library_commit=cleaned_library_commit,
                ),
                layer="silver",
                dataset=DatasetName.EQUITY_BARS.value,
                partitions=partitions,
                filename="manifest.json",
                overwrite=overwrite,
            )
            artifact_paths.extend(
                (
                    bronze_json_path,
                    bronze_manifest_path,
                    silver_frame_path,
                    silver_manifest_path,
                )
            )

        runs_path = self.storage.record_run(
            metadata,
            artifacts=artifact_paths,
            details=_backfill_run_details(
                operation="backfill_bars",
                provider="alpaca",
                targets=list(cleaned_symbols),
                start=start_timestamp,
                end=end_timestamp,
                rows_in=rows_in,
                rows_out=rows_out,
                requests=[request],
                library_commit=cleaned_library_commit,
            ),
        )
        return BackfillResult(
            metadata=metadata,
            run_ids=(metadata.run_id,),
            artifact_paths=tuple(artifact_paths),
            stats=ResultStats(
                rows_in=rows_in,
                rows_out=rows_out,
                files_written=(*artifact_paths, runs_path),
            ),
        )

    def _fetch_latest_equity_quote(
        self,
        underlying: str,
        *,
        asof: str,
    ) -> Mapping[str, Any]:
        try:
            return self._resolve_alpaca_client().get_latest_equity_quotes(
                underlying,
                asof=asof,
            )
        except Exception:
            raise ProviderSnapshotDataUnavailableError(
                f"Alpaca latest equity quote is unavailable for {underlying!r}"
            ) from None

    def _fetch_option_chain(
        self,
        underlying: str,
        *,
        asof: str,
        expiry_gte: date | str | None,
        expiry_lte: date | str | None,
        strike_gte: float | None,
        strike_lte: float | None,
        option_type: str | None,
        feed: str | None,
    ) -> Mapping[str, Any]:
        try:
            return self._resolve_alpaca_client().get_option_chain(
                underlying,
                expiry_gte=expiry_gte,
                expiry_lte=expiry_lte,
                strike_gte=strike_gte,
                strike_lte=strike_lte,
                option_type=option_type,
                feed=feed,
                asof=asof,
            )
        except Exception:
            raise ProviderSnapshotDataUnavailableError(
                f"Alpaca option chain is unavailable for {underlying!r}"
            ) from None

    def _fetch_fred_observations(
        self,
        series_id: str,
        *,
        asof: pd.Timestamp,
    ) -> Mapping[str, Any]:
        try:
            return self._resolve_fred_client().fetch_observations(
                series_id,
                observation_end=asof.date(),
                sort_order="asc",
            )
        except Exception:
            raise ProviderSnapshotDataUnavailableError(
                f"FRED observations are unavailable for series_id={series_id!r}"
            ) from None

    def _resolve_alpaca_client(self) -> _AlpacaClientLike:
        if self._alpaca_client is None:
            module = import_module("option_pricing.marketdata.providers.alpaca")
            client_cls = module.AlpacaClient
            self._alpaca_client = cast(
                _AlpacaClientLike,
                client_cls.from_env(self.config.alpaca),
            )
        return self._alpaca_client

    def _resolve_fred_client(self) -> _FredClientLike:
        if self._fred_client is None:
            module = import_module("option_pricing.marketdata.providers.fred")
            client_cls = module.FredClient
            self._fred_client = cast(
                _FredClientLike,
                client_cls.from_env(self.config.fred),
            )
        return self._fred_client


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


def _coerce_provider_pipeline_inputs(
    config: PipelineConfig | None,
    *,
    storage: LocalStorage | StorageConfig | Path | None,
) -> tuple[PipelineConfig, LocalStorage]:
    if config is not None and not isinstance(config, PipelineConfig):
        raise TypeError(
            "config must be a PipelineConfig when provided, "
            f"got {type(config).__name__}"
        )

    if config is None and storage is None:
        raise ValueError("MarketDataPipeline requires PipelineConfig or storage")

    local_storage = (
        _coerce_storage(storage)
        if storage is not None
        else LocalStorage(cast(PipelineConfig, config).storage)
    )
    if config is None:
        resolved_config = PipelineConfig(
            alpaca=AlpacaConfig(),
            fred=FredConfig(),
            storage=local_storage.config,
        )
    else:
        resolved_config = PipelineConfig(
            alpaca=config.alpaca,
            fred=config.fred,
            storage=local_storage.config,
        )
    return resolved_config, local_storage


def _clean_underlying(value: str) -> str:
    return _required_text(value, "underlying").upper()


def _clean_rate_series_id(value: str) -> str:
    return _required_text(value, "rate_series_id").upper()


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    return cleaned


def _optional_text(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _optional_run_id(value: str | None) -> str | None:
    if value is None:
        return None
    return _required_run_id(value)


def _new_run_id(asof: pd.Timestamp) -> str:
    timestamp = asof.strftime("%Y%m%dT%H%M%SZ")
    return f"b4a-{timestamp}-{uuid4().hex[:8]}"


def _backfill_metadata(
    *,
    run_id: str | None,
    prefix: str,
    started_at: datetime,
    library_commit: str | None,
) -> RunMetadata:
    cleaned_run_id = _optional_run_id(run_id)
    cleaned_library_commit = _optional_text(library_commit, "library_commit")
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    return RunMetadata(
        run_id=cleaned_run_id or f"{prefix}-{timestamp}-{uuid4().hex[:8]}",
        asof=started_at,
        started_at=started_at,
        git_sha=cleaned_library_commit,
    )


def _clean_fred_series_ids(series_ids: str | Sequence[str]) -> tuple[str, ...]:
    return tuple(
        _required_text(value, "series_id").upper()
        for value in _one_or_more_text_values(series_ids, "series_ids")
    )


def _clean_alpaca_symbols(symbols: str | Sequence[str]) -> tuple[str, ...]:
    return tuple(
        _required_text(value, "symbol").upper()
        for value in _one_or_more_text_values(symbols, "symbols")
    )


def _one_or_more_text_values(
    values: str | Sequence[str],
    field_name: str,
) -> tuple[str, ...]:
    raw_values: tuple[str, ...]
    if isinstance(values, str):
        raw_values = (values,)
    elif isinstance(values, Sequence) and not isinstance(
        values,
        (bytes, bytearray),
    ):
        raw_values = tuple(values)
    else:
        raise TypeError(f"{field_name} must be a string or sequence of strings")

    if not raw_values:
        raise ValueError(f"{field_name} must contain at least one value")
    for value in raw_values:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must contain only strings")
    return raw_values


def _coerce_backfill_date(value: date | datetime | str, field_name: str) -> date:
    if isinstance(value, datetime):
        timestamp = pd.Timestamp(value)
    elif isinstance(value, date):
        return value
    else:
        timestamp = pd.Timestamp(value)

    if pd.isna(timestamp):
        raise ValueError(f"{field_name} must not be missing")
    if timestamp.tzinfo is None:
        return timestamp.date()
    return timestamp.tz_convert(UTC).date()


def _coerce_backfill_timestamp(
    value: date | datetime | str,
    field_name: str,
) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"{field_name} must not be missing")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(UTC)
    return timestamp.tz_convert(UTC)


def _preflight_fred_backfill_targets(
    storage: LocalStorage,
    *,
    series_ids: Sequence[str],
    end_date: date,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    for series_id in series_ids:
        for path in _expected_fred_backfill_target_paths(
            storage,
            series_id=series_id,
            end_date=end_date,
        ):
            if path.exists():
                raise FileExistsError(
                    f"{path} already exists; pass overwrite=True to replace it"
                )


def _preflight_bars_backfill_targets(
    storage: LocalStorage,
    *,
    symbols: Sequence[str],
    timeframe: str,
    end_date: date,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    for symbol in symbols:
        for path in _expected_bars_backfill_target_paths(
            storage,
            symbol=symbol,
            timeframe=timeframe,
            end_date=end_date,
        ):
            if path.exists():
                raise FileExistsError(
                    f"{path} already exists; pass overwrite=True to replace it"
                )


def _expected_fred_backfill_target_paths(
    storage: LocalStorage,
    *,
    series_id: str,
    end_date: date,
) -> tuple[Path, ...]:
    partitions = _fred_backfill_partitions(series_id=series_id, end_date=end_date)
    return (
        _target_path(
            storage,
            layer="bronze",
            dataset=DatasetName.FRED_SERIES.value,
            partitions=partitions,
            filename="observations.json",
        ),
        _target_path(
            storage,
            layer="bronze",
            dataset=DatasetName.FRED_SERIES.value,
            partitions=partitions,
            filename="manifest.json",
        ),
        _target_path(
            storage,
            layer="silver",
            dataset=DatasetName.FRED_SERIES.value,
            partitions=partitions,
            filename="fred_series.parquet",
        ),
        _target_path(
            storage,
            layer="silver",
            dataset=DatasetName.FRED_SERIES.value,
            partitions=partitions,
            filename="manifest.json",
        ),
    )


def _expected_bars_backfill_target_paths(
    storage: LocalStorage,
    *,
    symbol: str,
    timeframe: str,
    end_date: date,
) -> tuple[Path, ...]:
    partitions = _bars_backfill_partitions(
        symbol=symbol,
        timeframe=timeframe,
        end_date=end_date,
    )
    return (
        _target_path(
            storage,
            layer="bronze",
            dataset=DatasetName.EQUITY_BARS.value,
            partitions=partitions,
            filename="bars.json",
        ),
        _target_path(
            storage,
            layer="bronze",
            dataset=DatasetName.EQUITY_BARS.value,
            partitions=partitions,
            filename="manifest.json",
        ),
        _target_path(
            storage,
            layer="silver",
            dataset=DatasetName.EQUITY_BARS.value,
            partitions=partitions,
            filename="equity_bars.parquet",
        ),
        _target_path(
            storage,
            layer="silver",
            dataset=DatasetName.EQUITY_BARS.value,
            partitions=partitions,
            filename="manifest.json",
        ),
    )


def _fred_backfill_partitions(
    *,
    series_id: str,
    end_date: date,
) -> dict[str, PartitionValue]:
    return {"series_id": series_id, "date": end_date}


def _bars_backfill_partitions(
    *,
    symbol: str,
    timeframe: str,
    end_date: date,
) -> dict[str, PartitionValue]:
    return {"symbol": symbol, "timeframe": timeframe, "date": end_date}


def _provider_backfill_payload_document(
    payload: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "request": _jsonable_provider_value(request),
        "payload": _jsonable_provider_value(payload),
    }


def _fred_backfill_manifest(
    metadata: RunMetadata,
    *,
    series_id: str,
    start_date: date,
    end_date: date,
    raw_rows: int,
    normalized_rows: int,
    layer: str,
    artifacts: Mapping[str, str],
    library_commit: str | None,
) -> dict[str, object]:
    return {
        "schema_version": (
            FRED_BACKFILL_BRONZE_SCHEMA_VERSION
            if layer == "bronze"
            else FRED_BACKFILL_SILVER_SCHEMA_VERSION
        ),
        "operation": "backfill_fred",
        "run_id": metadata.run_id,
        "source_type": "provider_backfill",
        "provider": "fred",
        "series_id": series_id,
        "start_date": start_date,
        "end_date": end_date,
        "rows": {"raw": raw_rows, "normalized": normalized_rows},
        "artifacts": dict(artifacts),
        "library_commit": library_commit,
    }


def _bars_backfill_manifest(
    metadata: RunMetadata,
    *,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeframe: str,
    feed: str,
    raw_rows: int,
    normalized_rows: int,
    layer: str,
    artifacts: Mapping[str, str],
    library_commit: str | None,
) -> dict[str, object]:
    return {
        "schema_version": (
            EQUITY_BARS_BACKFILL_BRONZE_SCHEMA_VERSION
            if layer == "bronze"
            else EQUITY_BARS_BACKFILL_SILVER_SCHEMA_VERSION
        ),
        "operation": "backfill_bars",
        "run_id": metadata.run_id,
        "source_type": "provider_backfill",
        "provider": "alpaca",
        "symbol": symbol,
        "start": start,
        "end": end,
        "timeframe": timeframe,
        "feed": feed,
        "rows": {"raw": raw_rows, "normalized": normalized_rows},
        "artifacts": dict(artifacts),
        "library_commit": library_commit,
    }


def _backfill_run_details(
    *,
    operation: str,
    provider: str,
    targets: Sequence[str],
    start: date | datetime | pd.Timestamp,
    end: date | datetime | pd.Timestamp,
    rows_in: int,
    rows_out: int,
    requests: Sequence[Mapping[str, object]],
    library_commit: str | None,
) -> dict[str, object]:
    return {
        "operation": operation,
        "provider": provider,
        "targets": list(targets),
        "start": start,
        "end": end,
        "rows": {"raw": rows_in, "normalized": rows_out},
        "requests": [dict(request) for request in requests],
        "library_commit": library_commit,
    }


def _count_fred_observations(payload: Mapping[str, Any]) -> int:
    observations = payload.get("observations")
    return len(observations) if isinstance(observations, list) else 0


def _bars_frame_for_symbol(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    matches = frame["symbol"].astype("string").str.upper() == symbol
    return frame.loc[matches].reset_index(drop=True)


def _count_alpaca_bars_for_symbol(payload: Mapping[str, Any], symbol: str) -> int:
    bars_payload = _alpaca_bars_payload(payload)
    if isinstance(bars_payload, Mapping):
        records = _symbol_mapping_value(bars_payload, symbol)
        return 0 if records is None else _count_bar_records(records)
    if isinstance(bars_payload, Sequence) and not isinstance(
        bars_payload,
        (str, bytes, bytearray),
    ):
        return sum(
            1
            for record in bars_payload
            if _record_symbol_matches(record, symbol)
            or _payload_targets_single_symbol(payload, symbol)
        )
    return 1 if _record_symbol_matches(bars_payload, symbol) else 0


def _alpaca_bars_payload_for_symbol(
    payload: Mapping[str, Any],
    symbol: str,
) -> dict[str, object]:
    out = dict(payload)
    out["symbols"] = [symbol]
    bars_payload = _alpaca_bars_payload(payload)
    if isinstance(bars_payload, Mapping):
        records = _symbol_mapping_value(bars_payload, symbol)
        out["bars"] = {symbol: [] if records is None else records}
    else:
        out["bars"] = bars_payload
    return out


def _alpaca_bars_payload(payload: Mapping[str, Any]) -> Any:
    bars_payload = payload.get("bars", payload.get("bar", payload))
    return getattr(bars_payload, "data", bars_payload)


def _symbol_mapping_value(mapping: Mapping[Any, Any], symbol: str) -> Any | None:
    for key, value in mapping.items():
        if str(key).strip().upper() == symbol:
            return value
    return None


def _count_bar_records(records: Any) -> int:
    records = getattr(records, "data", records)
    if _is_bar_record_like(records):
        return 1
    if isinstance(records, Sequence) and not isinstance(
        records,
        (str, bytes, bytearray),
    ):
        return len(records)
    return 0


def _record_symbol_matches(record: Any, symbol: str) -> bool:
    record_symbol = _provider_record_value(record, ("symbol", "S", "s"))
    if record_symbol is None:
        return False
    return str(record_symbol).strip().upper() == symbol


def _payload_targets_single_symbol(payload: Mapping[str, Any], symbol: str) -> bool:
    payload_symbols = payload.get("symbols", payload.get("symbol"))
    if isinstance(payload_symbols, str):
        return payload_symbols.strip().upper() == symbol
    if isinstance(payload_symbols, Sequence) and not isinstance(
        payload_symbols,
        (bytes, bytearray),
    ):
        cleaned = [str(value).strip().upper() for value in payload_symbols]
        return cleaned == [symbol]
    return False


def _is_bar_record_like(value: Any) -> bool:
    if isinstance(value, Mapping):
        raw_data = value.get("raw_data")
        if isinstance(raw_data, Mapping) and _is_bar_record_like(raw_data):
            return True
        return any(
            key in value
            for key in (
                "timestamp",
                "t",
                "open",
                "o",
                "high",
                "h",
                "low",
                "l",
                "close",
                "c",
                "volume",
                "v",
            )
        )
    return any(
        hasattr(value, field_name)
        for field_name in ("timestamp", "open", "high", "low", "close", "volume")
    )


def _provider_record_value(record: Any, aliases: Sequence[str]) -> Any | None:
    if isinstance(record, Mapping):
        for alias in aliases:
            if alias in record:
                return record[alias]
        raw_data = record.get("raw_data")
        if isinstance(raw_data, Mapping):
            return _provider_record_value(raw_data, aliases)
        return None

    for alias in aliases:
        try:
            return getattr(record, alias)
        except AttributeError:
            continue
    raw_data = getattr(record, "raw_data", None)
    if isinstance(raw_data, Mapping):
        return _provider_record_value(raw_data, aliases)
    return None


def _coerce_asof(value: str | pd.Timestamp | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz=UTC)
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("asof must not be missing")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(UTC)
    return timestamp.tz_convert(UTC)


def _utc_isoformat(value: pd.Timestamp) -> str:
    return value.to_pydatetime().astimezone(UTC).isoformat().replace("+00:00", "Z")


def _normalize_latest_equity_quotes_for_snapshot(
    payload: Mapping[str, Any],
    *,
    underlying: str,
    asof: pd.Timestamp,
) -> pd.DataFrame:
    try:
        return normalize_alpaca_latest_quotes(payload, asof=asof)
    except Exception:
        raise ProviderSnapshotDataUnavailableError(
            f"Alpaca latest equity quote is unavailable for {underlying!r}"
        ) from None


def _spot_from_equity_quotes(
    equity_quotes: pd.DataFrame,
    *,
    underlying: str,
) -> float:
    matches = equity_quotes.loc[
        equity_quotes["symbol"].astype("string").str.upper() == underlying
    ]
    if matches.empty:
        raise ProviderSnapshotDataUnavailableError(
            f"Alpaca latest equity quote is unavailable for {underlying!r}"
        )
    mid = matches.iloc[-1]["mid"]
    if pd.isna(mid):
        raise ProviderSnapshotDataUnavailableError(
            f"Alpaca latest equity quote mid is unavailable for {underlying!r}"
        )
    spot = float(mid)
    if not spot > 0:
        raise ProviderSnapshotDataUnavailableError(
            f"Alpaca latest equity quote mid must be positive for {underlying!r}"
        )
    return spot


def _normalize_option_chain_for_snapshot(
    payload: Mapping[str, Any],
    *,
    underlying: str,
    asof: pd.Timestamp,
) -> pd.DataFrame:
    try:
        return normalize_alpaca_option_chain(
            payload,
            underlying=underlying,
            asof=asof,
        )
    except Exception:
        raise ProviderSnapshotDataUnavailableError(
            "No usable Alpaca option contracts remain after provider "
            f"normalization for {underlying!r}"
        ) from None


def _normalize_fred_observations_for_snapshot(
    payload: Mapping[str, Any],
    *,
    series_id: str,
    asof: pd.Timestamp,
) -> pd.DataFrame:
    try:
        return normalize_fred_observations(payload, series_id=series_id, asof=asof)
    except Exception:
        raise ProviderSnapshotDataUnavailableError(
            f"FRED observations are unavailable for series_id={series_id!r}"
        ) from None


def _select_rate_for_snapshot(
    fred_series: pd.DataFrame,
    *,
    series_id: str,
    asof: pd.Timestamp,
) -> Any:
    try:
        return select_latest_fred_rate_at_or_before_asof(
            fred_series,
            series_id=series_id,
            asof=asof,
        )
    except ProviderDataUnavailableError:
        raise ProviderSnapshotDataUnavailableError(
            "No usable FRED rate observation exists for "
            f"series_id={series_id!r} at or before {asof.date()}"
        ) from None


def _market_inputs_frame(
    *,
    underlying: str,
    asof: pd.Timestamp,
    spot: float,
    rate: float,
    rate_source: str,
    rate_observation_date: pd.Timestamp,
    rate_compounding: str,
    dividend_yield: float,
    dividend_yield_source: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "underlying": underlying,
                "asof": asof,
                "spot": spot,
                "spot_source": "alpaca_latest_equity_quote_mid",
                "rate": rate,
                "rate_source": rate_source,
                "rate_observation_date": rate_observation_date,
                "rate_compounding": rate_compounding,
                "dividend_yield": float(dividend_yield),
                "dividend_yield_source": dividend_yield_source,
                "day_count": _DEFAULT_DAY_COUNT,
            }
        ]
    )


def _count_alpaca_option_contracts(payload: Mapping[str, Any]) -> int:
    contracts: object
    if "contracts" in payload:
        contracts = payload["contracts"]
    elif "snapshots" in payload:
        contracts = payload["snapshots"]
    else:
        return 0

    contracts = getattr(contracts, "data", contracts)
    if isinstance(contracts, Mapping):
        return len(contracts)
    if isinstance(contracts, Sequence) and not isinstance(
        contracts,
        (str, bytes, bytearray),
    ):
        return len(contracts)
    return 0


def _provider_snapshot_warnings(
    *,
    cleaning_warnings: Sequence[str],
    dropped_before_cleaning_count: int,
    raw_option_contract_count: int,
    normalized_option_contract_count: int,
) -> tuple[str, ...]:
    warnings = [str(warning) for warning in cleaning_warnings]
    if dropped_before_cleaning_count > 0:
        warnings.append(
            "alpaca_option_contracts_dropped_before_cleaning: "
            f"dropped={dropped_before_cleaning_count}, "
            f"raw={raw_option_contract_count}, "
            f"normalized={normalized_option_contract_count}, "
            "reason=missing_or_unusable_bid_ask"
        )
    return tuple(warnings)


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
    overwrite: bool,
) -> None:
    if overwrite:
        return

    paths = _expected_provider_snapshot_target_paths(
        storage,
        provider_snapshot,
        rate_series_id=rate_series_id,
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
    feed: str,
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
            feed=feed,
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
    return ProviderSnapshotSilverPaths(
        market_inputs=cleaning_paths.market_inputs,
        option_chain=option_chain_path,
        fred_series=fred_series_path,
        cleaned_quotes=cleaning_paths.cleaned_quotes,
        rejected_quotes=cleaning_paths.rejected_quotes,
        manifest=cleaning_paths.manifest,
    )


def _provider_payload_document(payload: Mapping[str, Any]) -> dict[str, object]:
    return {"payload": _jsonable_provider_value(payload)}


def _jsonable_provider_value(value: Any) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable_provider_value(asdict(value))
    if isinstance(value, pd.Timestamp):
        return _utc_isoformat(_coerce_asof(value))
    if isinstance(value, datetime):
        timestamp = pd.Timestamp(value)
        return _utc_isoformat(_coerce_asof(timestamp))
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        out: dict[str, object] = {}
        for key, item in value.items():
            text_key = str(key)
            if _is_secret_payload_key(text_key):
                out[text_key] = "<redacted>"
            else:
                out[text_key] = _jsonable_provider_value(item)
        return out
    raw_data = getattr(value, "raw_data", None)
    if isinstance(raw_data, Mapping):
        return _jsonable_provider_value(raw_data)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable_provider_value(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return {"type": type(value).__name__}


def _is_secret_payload_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(part in lowered for part in _SECRET_KEY_PARTS)


def _provider_bronze_manifest(
    provider_snapshot: _ProviderSnapshot,
    *,
    rate_series_id: str,
    feed: str,
    library_commit: str | None,
) -> dict[str, object]:
    return {
        "provider_snapshot_schema_version": PROVIDER_SNAPSHOT_BRONZE_SCHEMA_VERSION,
        "fixture_name": provider_snapshot.fixture_name,
        "snapshot_id": provider_snapshot.snapshot_id,
        "run_id": provider_snapshot.run_id,
        "source_type": PROVIDER_SNAPSHOT_SOURCE_TYPE,
        "underlying": provider_snapshot.underlying,
        "valuation_timestamp_utc": _utc_isoformat(provider_snapshot.asof),
        "providers": provider_snapshot.metadata["providers"],
        "feed": feed,
        "rate_series_id": rate_series_id,
        "rows": dict(provider_snapshot.row_counts),
        "warnings": list(provider_snapshot.warnings),
        "artifacts": {
            "latest_equity_quotes": "latest_equity_quotes.json",
            "option_chain": "option_chain.json",
            "fred_observations": "fred_observations.json",
        },
        "library_commit": library_commit,
    }


def _expected_provider_snapshot_target_paths(
    storage: LocalStorage,
    provider_snapshot: _ProviderSnapshot,
    *,
    rate_series_id: str,
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


def _provider_snapshot_artifact_paths(
    *,
    bronze_paths: ProviderSnapshotBronzePaths,
    silver_paths: ProviderSnapshotSilverPaths,
    gold_paths: GoldConversionPaths,
    model_validation_bundle: ModelValidationBundleResult,
) -> tuple[Path, ...]:
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
        gold_paths.market_data,
        gold_paths.market_manifest,
        gold_paths.heston_quotes,
        gold_paths.heston_manifest,
        *model_validation_bundle.artifact_paths,
        model_validation_bundle.manifest_path,
    )


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
    "MarketDataPipeline",
    "ProviderSnapshotBronzePaths",
    "ProviderSnapshotDataUnavailableError",
    "ProviderSnapshotResult",
    "ProviderSnapshotSilverPaths",
    "run_local_model_validation_pipeline",
]
