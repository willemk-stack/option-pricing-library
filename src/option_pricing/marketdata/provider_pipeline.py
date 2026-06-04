from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

import pandas as pd

from option_pricing.marketdata.bundles import (
    ModelValidationBundleConfig,
    write_model_validation_bundle_artifacts,
)
from option_pricing.marketdata.cleaning import (
    QuoteCleaningPolicyV1,
    clean_option_quotes,
)
from option_pricing.marketdata.config import (
    AlpacaConfig,
    FredConfig,
    MarketDataPolicyConfig,
    PipelineConfig,
    StorageConfig,
)
from option_pricing.marketdata.contracts import BackfillResult, ResultStats, RunMetadata
from option_pricing.marketdata.errors import ProviderDataUnavailableError
from option_pricing.marketdata.gold import write_gold_artifacts
from option_pricing.marketdata.normalize import (
    AlpacaOptionChainNormalizationAudit,
    normalize_alpaca_bars,
    normalize_alpaca_latest_quotes,
    normalize_alpaca_option_chain_with_audit,
    normalize_fred_observations,
    normalize_market_inputs,
    normalize_option_chain,
)
from option_pricing.marketdata.provider_artifacts import (
    _build_provider_snapshot_rate_curve,
    _preflight_provider_snapshot_targets,
    _provider_snapshot_artifact_paths,
    _provider_snapshot_request_metadata,
    _provider_snapshot_run_details,
    _provider_snapshot_target_stub,
    _ProviderSnapshot,
    _quote_cleaning_result_with_warnings,
    _write_provider_snapshot_bronze,
    _write_provider_snapshot_rate_curve_gold,
    _write_provider_snapshot_silver,
)
from option_pricing.marketdata.provider_backfills import (
    _alpaca_bars_payload_for_symbol,
    _backfill_metadata,
    _backfill_run_details,
    _bars_backfill_manifest,
    _bars_backfill_partitions,
    _bars_frame_for_symbol,
    _clean_alpaca_symbols,
    _clean_fred_series_ids,
    _coerce_backfill_date,
    _coerce_backfill_timestamp,
    _count_alpaca_bars_for_symbol,
    _count_fred_observations,
    _fred_backfill_manifest,
    _fred_backfill_partitions,
    _preflight_bars_backfill_targets,
    _preflight_fred_backfill_targets,
    _provider_backfill_payload_document,
)
from option_pricing.marketdata.provider_diagnostics import (
    ProviderCallFailedError,
    _call_provider_with_diagnostic,
    _normalization_failure_diagnostic,
)
from option_pricing.marketdata.provider_policy import (
    _BARS_BACKFILL_WARNING,
    _FRED_BACKFILL_WARNING,
    DEFAULT_BARS_TIMEFRAME,
    DEFAULT_DAY_COUNT,
    DEFAULT_RATE_CURVE_SERIES_IDS,
    DEFAULT_RATE_SERIES_ID,
    DEFAULT_SNAPSHOT_RATE_LOOKBACK_DAYS,
    ProviderSnapshotQualityPolicy,
    _apply_provider_snapshot_quality_policy,
    _coerce_provider_snapshot_quality_policy,
    _current_provider_scope,
    _merge_unique_warnings,
    _provider_snapshot_data_policy,
    _provider_snapshot_dividend_policy,
    _provider_snapshot_freshness_stats,
    _provider_snapshot_option_cleaning_policy,
    _provider_snapshot_quality_failures,
    _provider_snapshot_quality_warnings,
    _provider_snapshot_rate_policy,
    _provider_snapshot_warnings,
    _resolve_provider_snapshot_dividend_policy,
)
from option_pricing.marketdata.provider_results import (
    ProviderCallDiagnostic,
    ProviderRefreshDailyCounts,
    ProviderRefreshDailyResult,
    ProviderSnapshotResult,
)
from option_pricing.marketdata.provider_serialization import (
    _coerce_asof,
    _relative_artifact_references,
    _utc_isoformat,
    _utc_timestamp,
)
from option_pricing.marketdata.rates import (
    RATE_COMPOUNDING_CONTINUOUS,
    representative_time_to_expiry_years,
    resolve_fred_treasury_zero_proxy_rate,
    select_latest_fred_rate_at_or_before_asof,
)
from option_pricing.marketdata.schemas import DatasetName
from option_pricing.marketdata.storage import LocalStorage


class _AlpacaClientLike(Protocol):
    def get_latest_equity_quotes(
        self,
        symbols: str | Sequence[str],
        *,
        feed: str | None = None,
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

    def __init__(
        self,
        message: str,
        *,
        diagnostic: ProviderCallDiagnostic | None = None,
        failure_kind: str | None = None,
    ) -> None:
        self.diagnostic = diagnostic
        self.failure_kind = failure_kind
        super().__init__(message)


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
        policy_config: (
            MarketDataPolicyConfig | Mapping[str, object] | Path | None
        ) = None,
        quality_policy: (
            ProviderSnapshotQualityPolicy | Mapping[str, object] | None
        ) = None,
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
        self.policy_config = _coerce_marketdata_policy_config(
            policy_config if policy_config is not None else self.config.policy
        )
        self.quality_policy = _coerce_provider_snapshot_quality_policy(quality_policy)

    def snapshot(
        self,
        underlying: str,
        *,
        asof: str | pd.Timestamp | None = None,
        run_id: str | None = None,
        rate_series_id: str = DEFAULT_RATE_SERIES_ID,
        expiry_gte: date | str | None = None,
        expiry_lte: date | str | None = None,
        strike_gte: float | None = None,
        strike_lte: float | None = None,
        option_type: str | None = None,
        feed: str | None = None,
        equity_feed: str | None = None,
        option_feed: str | None = None,
        dividend_yield: float = 0.0,
        dividend_yield_source: str = "assumption",
        rate_lookback_days: int = DEFAULT_SNAPSHOT_RATE_LOOKBACK_DAYS,
        curve_series_ids: Sequence[str] | None = None,
        quality_policy: (
            ProviderSnapshotQualityPolicy | Mapping[str, object] | None
        ) = None,
        overwrite: bool = False,
        library_commit: str | None = None,
    ) -> ProviderSnapshotResult:
        """Fetch, normalize, clean, and persist one provider-backed snapshot."""

        started_at = datetime.now(UTC)
        cleaned_underlying = _clean_underlying(underlying)
        asof_timestamp = _coerce_asof(asof)
        effective_run_id = _optional_run_id(run_id) or _new_run_id(asof_timestamp)
        cleaned_rate_series_id = _clean_rate_series_id(rate_series_id)
        cleaned_dividend_yield_source = _required_text(
            dividend_yield_source,
            "dividend_yield_source",
        )
        cleaned_dividend_yield = _nonnegative_finite_float(
            dividend_yield,
            "dividend_yield",
        )
        cleaned_rate_lookback_days = _nonnegative_int(
            rate_lookback_days,
            "rate_lookback_days",
        )
        cleaned_curve_series_ids = _snapshot_curve_series_ids(
            curve_series_ids,
            primary_series_id=cleaned_rate_series_id,
        )
        effective_quality_policy = _coerce_provider_snapshot_quality_policy(
            quality_policy if quality_policy is not None else self.quality_policy
        )
        quality_policy_payload = effective_quality_policy.as_dict()
        cleaned_library_commit = _optional_text(library_commit, "library_commit")
        cleaned_legacy_feed = _optional_text(feed, "feed")
        cleaned_equity_feed = _optional_text(equity_feed, "equity_feed")
        cleaned_option_feed = _optional_text(option_feed, "option_feed")
        resolved_equity_feed = cleaned_equity_feed or self.config.alpaca.equity_feed
        resolved_option_feed = (
            cleaned_option_feed or cleaned_legacy_feed or self.config.alpaca.option_feed
        )
        fred_observation_start = asof_timestamp.date() - timedelta(
            days=cleaned_rate_lookback_days
        )
        provider_sources = {
            "spot": "alpaca",
            "option_chain": "alpaca",
            "rate": "fred",
        }
        snapshot_request_metadata = _provider_snapshot_request_metadata(
            underlying=cleaned_underlying,
            asof=asof_timestamp,
            expiry_gte=expiry_gte,
            expiry_lte=expiry_lte,
            strike_gte=strike_gte,
            strike_lte=strike_lte,
            option_type=option_type,
            equity_feed=resolved_equity_feed,
            option_feed=resolved_option_feed,
            rate_series_id=cleaned_rate_series_id,
            rate_lookback_days=cleaned_rate_lookback_days,
            curve_series_ids=cleaned_curve_series_ids,
            fred_observation_end=asof_timestamp.date(),
            fred_observation_start=fred_observation_start,
        )

        _preflight_provider_snapshot_targets(
            self.storage,
            _provider_snapshot_target_stub(
                underlying=cleaned_underlying,
                asof=asof_timestamp,
                run_id=effective_run_id,
            ),
            rate_series_id=cleaned_rate_series_id,
            include_rate_curve=bool(cleaned_curve_series_ids),
            overwrite=overwrite,
        )

        asof_label = _utc_isoformat(asof_timestamp)
        diagnostics: list[ProviderCallDiagnostic] = []
        equity_quote_payload = self._fetch_latest_equity_quote(
            cleaned_underlying,
            asof=asof_label,
            feed=resolved_equity_feed,
            diagnostics=diagnostics,
        )
        option_chain_payload = self._fetch_option_chain(
            cleaned_underlying,
            asof=asof_label,
            expiry_gte=expiry_gte,
            expiry_lte=expiry_lte,
            strike_gte=strike_gte,
            strike_lte=strike_lte,
            option_type=option_type,
            feed=resolved_option_feed,
            diagnostics=diagnostics,
        )
        fred_payload = self._fetch_fred_observations(
            cleaned_rate_series_id,
            asof=asof_timestamp,
            lookback_days=cleaned_rate_lookback_days,
            diagnostics=diagnostics,
        )

        equity_quotes = _normalize_latest_equity_quotes_for_snapshot(
            equity_quote_payload,
            underlying=cleaned_underlying,
            asof=asof_timestamp,
            diagnostics=diagnostics,
        )
        spot = _spot_from_equity_quotes(
            equity_quotes,
            underlying=cleaned_underlying,
        )
        raw_option_contract_count = _count_alpaca_option_contracts(option_chain_payload)
        try:
            provider_option_chain = _normalize_option_chain_for_snapshot(
                option_chain_payload,
                underlying=cleaned_underlying,
                asof=asof_timestamp,
                feed=resolved_option_feed,
                diagnostics=diagnostics,
            )
        except ProviderSnapshotDataUnavailableError as exc:
            raise ProviderSnapshotDataUnavailableError(
                f"{exc}; raw_option_contracts={raw_option_contract_count}, "
                "normalized_option_contracts=0, "
                "reason=provider_normalization",
                diagnostic=exc.diagnostic,
                failure_kind=exc.failure_kind or "normalization_failed",
            ) from None
        option_chain = normalize_option_chain(provider_option_chain.option_chain)
        provider_rejected_contracts = provider_option_chain.rejected_contracts
        provider_rejected_contract_count = int(len(provider_rejected_contracts))
        dropped_before_cleaning_count = provider_rejected_contract_count

        fred_series = _normalize_fred_observations_for_snapshot(
            fred_payload,
            series_id=cleaned_rate_series_id,
            asof=asof_timestamp,
            diagnostics=diagnostics,
        )
        rate_selection = _select_rate_for_snapshot(
            fred_series,
            series_id=cleaned_rate_series_id,
            asof=asof_timestamp,
            lookback_days=cleaned_rate_lookback_days,
        )
        rate_curve = _build_provider_snapshot_rate_curve(
            asof=asof_timestamp,
            day_count=DEFAULT_DAY_COUNT,
            primary_series_id=cleaned_rate_series_id,
            primary_fred_series=fred_series,
            requested_series_ids=cleaned_curve_series_ids,
            load_series_frame=lambda series_id: self._load_snapshot_rate_curve_series(
                series_id,
                asof=asof_timestamp,
                lookback_days=cleaned_rate_lookback_days,
                diagnostics=diagnostics,
            ),
        )
        representative_expiry_years = representative_time_to_expiry_years(
            option_chain["expiry"].tolist(),
            asof=asof_timestamp,
        )
        rate_resolution = resolve_fred_treasury_zero_proxy_rate(
            rate_curve,
            time_to_expiry_years=representative_expiry_years,
            selected_rate_fallback=rate_selection.rate,
        )
        selected_rate = float(rate_resolution.rate)
        flat_rate_fallback = float(rate_selection.rate)
        rate_source = (
            f"{rate_selection.source}:{rate_selection.series_id}"
            if rate_resolution.selected_rate_fallback_used
            else "fred:treasury_zero_proxy_curve"
        )
        rate_policy_payload = _provider_snapshot_rate_policy(
            rate_series_id=cleaned_rate_series_id,
            rate_source=rate_source,
            rate_observation_date=rate_selection.observation_date,
            selected_rate=selected_rate,
            lookback_days=cleaned_rate_lookback_days,
            curve_series_ids=cleaned_curve_series_ids,
            flat_rate=flat_rate_fallback,
            rate_warnings=rate_resolution.warnings,
            selected_rate_fallback_used=rate_resolution.selected_rate_fallback_used,
        )
        dividend_policy_payload = _resolve_provider_snapshot_dividend_policy(
            underlying=cleaned_underlying,
            policy_config=self.policy_config,
            dividend_yield=cleaned_dividend_yield,
            dividend_yield_source=cleaned_dividend_yield_source,
        )
        resolved_dividend_yield = float(
            cast(Any, dividend_policy_payload["dividend_yield"])
        )
        resolved_dividend_yield_source = str(dividend_policy_payload["source"])
        option_cleaning_policy_payload = _provider_snapshot_option_cleaning_policy()
        data_policy_payload = _provider_snapshot_data_policy(
            equity_provider="alpaca",
            equity_feed=resolved_equity_feed,
            option_provider="alpaca",
            option_feed=resolved_option_feed,
            rate_policy=rate_policy_payload,
            dividend_policy=dividend_policy_payload,
            option_cleaning_policy=option_cleaning_policy_payload,
            quote_freshness_mode=effective_quality_policy.quote_freshness_mode,
        )
        market_inputs = normalize_market_inputs(
            _market_inputs_frame(
                underlying=cleaned_underlying,
                asof=asof_timestamp,
                spot=spot,
                rate=selected_rate,
                rate_source=rate_source,
                rate_observation_date=rate_selection.observation_date,
                rate_compounding=RATE_COMPOUNDING_CONTINUOUS,
                dividend_yield=resolved_dividend_yield,
                dividend_yield_source=resolved_dividend_yield_source,
            )
        )
        quote_cleaning = clean_option_quotes(
            option_chain,
            market_inputs,
            policy=self.cleaning_policy,
        )
        quote_cleaning = _apply_provider_snapshot_quality_policy(
            quote_cleaning,
            asof=asof_timestamp,
            policy=effective_quality_policy,
        )
        quote_freshness = _provider_snapshot_freshness_stats(
            equity_quotes=equity_quotes,
            option_chain=option_chain,
            cleaned_quotes=quote_cleaning.cleaned_quotes,
            asof=asof_timestamp,
            policy=effective_quality_policy,
        )
        quality_warnings = _provider_snapshot_quality_warnings(
            stats=quote_freshness,
            policy=effective_quality_policy,
        )
        quality_failures = _provider_snapshot_quality_failures(
            stats=quote_freshness,
            policy=effective_quality_policy,
        )
        if quality_failures:
            raise ProviderSnapshotDataUnavailableError(
                "; ".join(quality_failures),
                failure_kind="quality_policy_failed",
            )
        if quote_cleaning.cleaned_quotes.empty:
            raise ProviderSnapshotDataUnavailableError(
                "No usable option contracts remain after quote cleaning for "
                f"{cleaned_underlying!r}; accepted=0, "
                f"rejected={len(quote_cleaning.rejected_quotes)}",
                failure_kind="no_option_contracts_accepted_after_cleaning",
            )

        warnings = _provider_snapshot_warnings(
            cleaning_warnings=(
                *rate_resolution.warnings,
                *quality_warnings,
                *quote_cleaning.warnings,
            ),
            dropped_before_cleaning_count=dropped_before_cleaning_count,
            raw_option_contract_count=raw_option_contract_count,
            normalized_option_contract_count=len(option_chain),
            provider_rejected_contract_count=provider_rejected_contract_count,
            rate_series_id=cleaned_rate_series_id,
            dividend_yield=resolved_dividend_yield,
            dividend_yield_source=resolved_dividend_yield_source,
        )
        quote_cleaning_for_artifacts = _quote_cleaning_result_with_warnings(
            quote_cleaning,
            warnings,
        )
        provider_snapshot = _ProviderSnapshot(
            fixture_name="provider_snapshot_v1",
            snapshot_id=(
                f"provider_snapshot_v1:{cleaned_underlying}:{_utc_isoformat(asof_timestamp)}:{effective_run_id}"
            ),
            run_id=effective_run_id,
            underlying=cleaned_underlying,
            asof=asof_timestamp,
            manifest={},
            market_inputs_raw=market_inputs,
            option_chain_raw=option_chain,
            metadata={
                "source_type": "provider_snapshot",
                "providers": provider_sources,
                "rate_series_id": cleaned_rate_series_id,
                "equity_provider": "alpaca",
                "equity_feed": resolved_equity_feed,
                "option_provider": "alpaca",
                "option_feed": resolved_option_feed,
                "selected_rate": selected_rate,
                "flat_rate": flat_rate_fallback,
                "rate_policy": rate_policy_payload,
                "dividend_policy": dividend_policy_payload,
                "option_cleaning_policy": option_cleaning_policy_payload,
                "quote_freshness_mode": effective_quality_policy.quote_freshness_mode,
                "model_validation_policy": "model_ready_quotes_v1",
                "data_policy": data_policy_payload,
                "current_provider_scope": _current_provider_scope(),
                "quality_policy": quality_policy_payload,
                "quote_freshness": quote_freshness,
            },
            row_counts={
                "equity_quotes": int(len(equity_quotes)),
                "option_contracts_raw": int(raw_option_contract_count),
                "option_contracts_normalized": int(len(option_chain)),
                "provider_rejected_contracts": int(provider_rejected_contract_count),
                "fred_observations": int(len(fred_series)),
                "rate_curve_points": int(len(rate_curve)),
                "cleaned_quotes": int(len(quote_cleaning.cleaned_quotes)),
                "rejected_quotes": int(len(quote_cleaning.rejected_quotes)),
            },
            warnings=warnings,
        )

        _preflight_provider_snapshot_targets(
            self.storage,
            provider_snapshot,
            rate_series_id=cleaned_rate_series_id,
            include_rate_curve=bool(cleaned_curve_series_ids),
            overwrite=overwrite,
        )
        bronze_paths = _write_provider_snapshot_bronze(
            self.storage,
            provider_snapshot,
            equity_quote_payload=equity_quote_payload,
            option_chain_payload=option_chain_payload,
            fred_payload=fred_payload,
            rate_series_id=cleaned_rate_series_id,
            request_metadata=snapshot_request_metadata,
            diagnostics=diagnostics,
            quality_policy=quality_policy_payload,
            quote_freshness=quote_freshness,
            overwrite=overwrite,
            library_commit=cleaned_library_commit,
        )
        silver_paths = _write_provider_snapshot_silver(
            self.storage,
            provider_snapshot,
            option_chain=option_chain,
            fred_series=fred_series,
            market_inputs=market_inputs,
            quote_cleaning=quote_cleaning_for_artifacts,
            provider_rejected_contracts=provider_rejected_contracts,
            rate_series_id=cleaned_rate_series_id,
            overwrite=overwrite,
            library_commit=cleaned_library_commit,
        )
        rate_curve_paths = _write_provider_snapshot_rate_curve_gold(
            self.storage,
            provider_snapshot,
            rate_curve=rate_curve,
            requested_series_ids=cleaned_curve_series_ids,
            lookback_days=cleaned_rate_lookback_days,
            overwrite=overwrite,
            library_commit=cleaned_library_commit,
        )
        gold_paths = write_gold_artifacts(
            self.storage,
            local_snapshot=provider_snapshot,
            market_inputs=market_inputs,
            cleaned_quotes=quote_cleaning_for_artifacts.cleaned_quotes,
            rejected_quotes=quote_cleaning_for_artifacts.rejected_quotes,
            reason_counts=quote_cleaning_for_artifacts.reason_counts,
            warnings=warnings,
            overwrite=overwrite,
            library_commit=cleaned_library_commit,
        )
        model_validation_bundle = write_model_validation_bundle_artifacts(
            self.storage,
            local_snapshot=provider_snapshot,
            market_inputs=market_inputs,
            cleaned_quotes=quote_cleaning_for_artifacts.cleaned_quotes,
            rejected_quotes=quote_cleaning_for_artifacts.rejected_quotes,
            reason_counts=quote_cleaning_for_artifacts.reason_counts,
            warnings=warnings,
            config=self.bundle_config,
            overwrite=overwrite,
            library_commit=cleaned_library_commit,
        )
        artifact_paths = _provider_snapshot_artifact_paths(
            bronze_paths=bronze_paths,
            silver_paths=silver_paths,
            gold_paths=gold_paths,
            rate_curve_paths=rate_curve_paths,
            model_validation_bundle=model_validation_bundle,
        )
        spot_source = _required_text(
            str(equity_quote_payload.get("source", provider_sources["spot"])),
            "spot_source",
        )
        self.storage.record_run(
            RunMetadata(
                run_id=effective_run_id,
                asof=cast(datetime, _utc_timestamp(asof_timestamp).to_pydatetime()),
                started_at=started_at,
                git_sha=cleaned_library_commit,
            ),
            artifacts=artifact_paths,
            details=_provider_snapshot_run_details(
                storage=self.storage,
                underlying=cleaned_underlying,
                asof=asof_timestamp,
                run_id=effective_run_id,
                rate_series_id=cleaned_rate_series_id,
                rate_source=rate_source,
                rate_observation_date=rate_selection.observation_date,
                spot_source=spot_source,
                dividend_yield=resolved_dividend_yield,
                dividend_yield_source=resolved_dividend_yield_source,
                equity_feed=resolved_equity_feed,
                option_feed=resolved_option_feed,
                selected_rate=selected_rate,
                flat_rate=flat_rate_fallback,
                rate_lookback_days=cleaned_rate_lookback_days,
                raw_option_contract_count=raw_option_contract_count,
                normalized_option_contract_count=len(option_chain),
                dropped_before_cleaning_count=dropped_before_cleaning_count,
                provider_rejected_contract_count=provider_rejected_contract_count,
                accepted_quote_count=int(
                    len(quote_cleaning_for_artifacts.cleaned_quotes)
                ),
                rejected_quote_count=int(
                    len(quote_cleaning_for_artifacts.rejected_quotes)
                ),
                diagnostics=diagnostics,
                quality_policy=quality_policy_payload,
                quote_freshness=quote_freshness,
                curve_series_ids=cleaned_curve_series_ids,
                rate_warnings=rate_resolution.warnings,
                selected_rate_fallback_used=(
                    rate_resolution.selected_rate_fallback_used
                ),
                warnings=warnings,
                artifact_paths=artifact_paths,
                library_commit=cleaned_library_commit,
            ),
        )

        return ProviderSnapshotResult(
            underlying=cleaned_underlying,
            asof=asof_timestamp,
            run_id=effective_run_id,
            spot=spot,
            rate=selected_rate,
            rate_source=rate_source,
            rate_observation_date=rate_selection.observation_date,
            rate_series_id=cleaned_rate_series_id,
            dividend_yield=resolved_dividend_yield,
            dividend_yield_source=resolved_dividend_yield_source,
            feed=resolved_option_feed,
            raw_option_contract_count=int(raw_option_contract_count),
            normalized_option_contract_count=int(len(option_chain)),
            accepted_quote_count=int(len(quote_cleaning_for_artifacts.cleaned_quotes)),
            rejected_quote_count=int(len(quote_cleaning_for_artifacts.rejected_quotes)),
            dropped_before_cleaning_count=int(dropped_before_cleaning_count),
            warnings=warnings,
            artifact_paths=artifact_paths,
            bronze_paths=bronze_paths,
            silver_paths=silver_paths,
            gold_paths=gold_paths,
            model_validation_bundle=model_validation_bundle,
            provider_rejected_contract_count=provider_rejected_contract_count,
            rate_curve_paths=rate_curve_paths,
            diagnostics=tuple(diagnostics),
            quality_policy=quality_policy_payload,
            quote_freshness=quote_freshness,
            equity_provider="alpaca",
            equity_feed=resolved_equity_feed,
            option_provider="alpaca",
            option_feed=resolved_option_feed,
            selected_rate=selected_rate,
            flat_rate=flat_rate_fallback,
            rate_policy=rate_policy_payload,
            dividend_policy=dividend_policy_payload,
            option_cleaning_policy=option_cleaning_policy_payload,
            data_policy=data_policy_payload,
        )

    def refresh_daily(
        self,
        underlyings: str | Sequence[str],
        *,
        asof: str | pd.Timestamp | None = None,
        run_id_prefix: str | None = None,
        rate_series_id: str = DEFAULT_RATE_SERIES_ID,
        expiry_gte: date | str | None = None,
        expiry_lte: date | str | None = None,
        strike_gte: float | None = None,
        strike_lte: float | None = None,
        option_type: str | None = None,
        feed: str | None = None,
        equity_feed: str | None = None,
        option_feed: str | None = None,
        dividend_yield: float = 0.0,
        dividend_yield_source: str = "assumption",
        rate_lookback_days: int = DEFAULT_SNAPSHOT_RATE_LOOKBACK_DAYS,
        curve_series_ids: Sequence[str] | None = None,
        quality_policy: (
            ProviderSnapshotQualityPolicy | Mapping[str, object] | None
        ) = None,
        overwrite: bool = False,
        library_commit: str | None = None,
    ) -> ProviderRefreshDailyResult:
        """Run one provider-backed snapshot per underlying and record an aggregate run."""

        started_at = datetime.now(UTC)
        asof_timestamp = _coerce_asof(asof)
        cleaned_underlyings = _clean_alpaca_symbols(underlyings)
        cleaned_library_commit = _optional_text(library_commit, "library_commit")
        cleaned_rate_series_id = _clean_rate_series_id(rate_series_id)
        cleaned_curve_series_ids = _snapshot_curve_series_ids(
            curve_series_ids,
            primary_series_id=cleaned_rate_series_id,
        )
        cleaned_rate_lookback_days = _nonnegative_int(
            rate_lookback_days,
            "rate_lookback_days",
        )
        effective_quality_policy = _coerce_provider_snapshot_quality_policy(
            quality_policy if quality_policy is not None else self.quality_policy
        )
        quality_policy_payload = effective_quality_policy.as_dict()
        cleaned_legacy_feed = _optional_text(feed, "feed")
        cleaned_equity_feed = _optional_text(equity_feed, "equity_feed")
        cleaned_option_feed = _optional_text(option_feed, "option_feed")
        resolved_equity_feed = cleaned_equity_feed or self.config.alpaca.equity_feed
        resolved_option_feed = (
            cleaned_option_feed or cleaned_legacy_feed or self.config.alpaca.option_feed
        )
        aggregate_run_id = _new_refresh_daily_run_id(
            asof_timestamp,
            run_id_prefix=run_id_prefix,
        )

        results: list[ProviderSnapshotResult] = []
        for index, underlying in enumerate(cleaned_underlyings, start=1):
            results.append(
                self.snapshot(
                    underlying,
                    asof=asof_timestamp,
                    run_id=_refresh_daily_child_run_id(
                        aggregate_run_id,
                        underlying,
                        ordinal=index,
                    ),
                    rate_series_id=cleaned_rate_series_id,
                    expiry_gte=expiry_gte,
                    expiry_lte=expiry_lte,
                    strike_gte=strike_gte,
                    strike_lte=strike_lte,
                    option_type=option_type,
                    feed=cleaned_legacy_feed,
                    equity_feed=resolved_equity_feed,
                    option_feed=resolved_option_feed,
                    dividend_yield=dividend_yield,
                    dividend_yield_source=dividend_yield_source,
                    rate_lookback_days=cleaned_rate_lookback_days,
                    curve_series_ids=cleaned_curve_series_ids,
                    quality_policy=effective_quality_policy,
                    overwrite=overwrite,
                    library_commit=cleaned_library_commit,
                )
            )

        artifact_paths = tuple(
            path for result in results for path in result.artifact_paths
        )
        child_run_ids = tuple(result.run_id for result in results)
        warnings = _merge_unique_warnings(
            warning for result in results for warning in result.warnings
        )
        counts = ProviderRefreshDailyCounts(
            raw_option_contract_count=sum(
                int(result.raw_option_contract_count) for result in results
            ),
            normalized_option_contract_count=sum(
                int(result.normalized_option_contract_count) for result in results
            ),
            dropped_before_cleaning_count=sum(
                int(result.dropped_before_cleaning_count) for result in results
            ),
            provider_rejected_contract_count=sum(
                int(result.provider_rejected_contract_count) for result in results
            ),
            accepted_quote_count=sum(
                int(result.accepted_quote_count) for result in results
            ),
            rejected_quote_count=sum(
                int(result.rejected_quote_count) for result in results
            ),
        )
        self.storage.record_run(
            RunMetadata(
                run_id=aggregate_run_id,
                asof=cast(datetime, _utc_timestamp(asof_timestamp).to_pydatetime()),
                started_at=started_at,
                git_sha=cleaned_library_commit,
            ),
            artifacts=artifact_paths,
            details=_provider_refresh_daily_run_details(
                storage=self.storage,
                aggregate_run_id=aggregate_run_id,
                child_run_ids=child_run_ids,
                underlyings=cleaned_underlyings,
                asof=asof_timestamp,
                rate_series_id=cleaned_rate_series_id,
                equity_feed=resolved_equity_feed,
                option_feed=resolved_option_feed,
                selected_rate=float(cast(float, results[0].selected_rate)),
                rate_observation_date=results[0].rate_observation_date,
                rate_lookback_days=cleaned_rate_lookback_days,
                curve_series_ids=cleaned_curve_series_ids,
                quality_policy=quality_policy_payload,
                expiry_gte=expiry_gte,
                expiry_lte=expiry_lte,
                strike_gte=strike_gte,
                strike_lte=strike_lte,
                option_type=option_type,
                dividend_yield=_finite_float(dividend_yield, "dividend_yield"),
                dividend_yield_source=_required_text(
                    dividend_yield_source,
                    "dividend_yield_source",
                ),
                counts=counts,
                warnings=warnings,
                artifact_paths=artifact_paths,
                library_commit=cleaned_library_commit,
            ),
        )
        return ProviderRefreshDailyResult(
            aggregate_run_id=aggregate_run_id,
            child_run_ids=child_run_ids,
            underlyings=cleaned_underlyings,
            artifact_paths=artifact_paths,
            counts=counts,
            warnings=warnings,
            results=tuple(results),
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
            start_date=start_date,
            end_date=end_date,
            run_id=metadata.run_id,
            overwrite=overwrite,
        )

        artifact_paths: list[Path] = []
        requests: list[dict[str, object]] = []
        diagnostics: list[ProviderCallDiagnostic] = []
        warnings = (_FRED_BACKFILL_WARNING,)
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
            payload = self._fetch_fred_observations_window(
                series_id,
                observation_start=start_date,
                observation_end=end_date,
                diagnostics=diagnostics,
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
                start_date=start_date,
                end_date=end_date,
                run_id=metadata.run_id,
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
                    request_metadata=request,
                    raw_rows=raw_rows,
                    normalized_rows=normalized_rows,
                    layer="bronze",
                    artifacts={"observations": "observations.json"},
                    warnings=warnings,
                    diagnostics=diagnostics,
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
                    request_metadata=request,
                    raw_rows=raw_rows,
                    normalized_rows=normalized_rows,
                    layer="silver",
                    artifacts={"fred_series": "fred_series.parquet"},
                    warnings=warnings,
                    diagnostics=diagnostics,
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
                diagnostics=diagnostics,
                warnings=warnings,
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
                warnings=warnings,
            ),
        )

    def backfill_bars(
        self,
        symbols: str | Sequence[str],
        start: date | datetime | str,
        end: date | datetime | str,
        *,
        timeframe: str = DEFAULT_BARS_TIMEFRAME,
        limit: int | None = None,
        adjustment: str | None = None,
        sort: str | None = "asc",
        feed: str | None = None,
        equity_feed: str | None = None,
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
        cleaned_legacy_feed = _optional_text(feed, "feed")
        cleaned_equity_feed = _optional_text(equity_feed, "equity_feed")
        resolved_equity_feed = (
            cleaned_equity_feed or cleaned_legacy_feed or self.config.alpaca.equity_feed
        )
        start_timestamp = _coerce_backfill_timestamp(start, "start")
        end_timestamp = _coerce_backfill_timestamp(end, "end")
        if start_timestamp >= end_timestamp:
            raise ValueError("start must be before end for Alpaca bars backfill")

        _preflight_bars_backfill_targets(
            self.storage,
            symbols=cleaned_symbols,
            timeframe=cleaned_timeframe,
            start_date=start_timestamp.date(),
            end_date=end_timestamp.date(),
            run_id=metadata.run_id,
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
            "feed": resolved_equity_feed,
            "asof": asof_label,
        }
        diagnostics: list[ProviderCallDiagnostic] = []
        payload = self._fetch_equity_bars(
            cleaned_symbols,
            start=start_timestamp.to_pydatetime(),
            end=end_timestamp.to_pydatetime(),
            timeframe=cleaned_timeframe,
            limit=limit,
            adjustment=cleaned_adjustment,
            sort=cleaned_sort,
            feed=resolved_equity_feed,
            asof=None,
            diagnostics=diagnostics,
        )
        bars = normalize_alpaca_bars(payload, asof=asof)

        artifact_paths: list[Path] = []
        warnings = (_BARS_BACKFILL_WARNING,)
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
                start_date=start_timestamp.date(),
                end_date=end_timestamp.date(),
                run_id=metadata.run_id,
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
                    feed=resolved_equity_feed,
                    request_metadata={**request, "symbols": [symbol]},
                    raw_rows=raw_rows,
                    normalized_rows=normalized_rows,
                    layer="bronze",
                    artifacts={"bars": "bars.json"},
                    warnings=warnings,
                    diagnostics=diagnostics,
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
                    feed=resolved_equity_feed,
                    request_metadata={**request, "symbols": [symbol]},
                    raw_rows=raw_rows,
                    normalized_rows=normalized_rows,
                    layer="silver",
                    artifacts={"equity_bars": "equity_bars.parquet"},
                    warnings=warnings,
                    diagnostics=diagnostics,
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
                diagnostics=diagnostics,
                warnings=warnings,
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
                warnings=warnings,
            ),
        )

    def _fetch_latest_equity_quote(
        self,
        underlying: str,
        *,
        asof: str,
        feed: str,
        diagnostics: list[ProviderCallDiagnostic],
    ) -> Mapping[str, Any]:
        try:
            payload, diagnostic = _call_provider_with_diagnostic(
                provider="alpaca",
                operation="latest_equity_quote",
                request_metadata={"symbols": [underlying], "feed": feed, "asof": asof},
                retry_config=self.config.retry,
                call=lambda: self._resolve_alpaca_client().get_latest_equity_quotes(
                    underlying,
                    feed=feed,
                    asof=asof,
                ),
                count_items=_count_alpaca_latest_equity_quotes,
            )
        except ProviderCallFailedError as exc:
            diagnostics.append(exc.diagnostic)
            raise ProviderSnapshotDataUnavailableError(
                "Alpaca latest equity quote is unavailable for "
                f"{underlying!r}; reason={exc.failure_kind}",
                diagnostic=exc.diagnostic,
                failure_kind=exc.failure_kind,
            ) from None
        diagnostics.append(diagnostic)
        return payload

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
        diagnostics: list[ProviderCallDiagnostic],
    ) -> Mapping[str, Any]:
        try:
            payload, diagnostic = _call_provider_with_diagnostic(
                provider="alpaca",
                operation="option_chain",
                request_metadata={
                    "underlying": underlying,
                    "expiry_gte": expiry_gte,
                    "expiry_lte": expiry_lte,
                    "strike_gte": strike_gte,
                    "strike_lte": strike_lte,
                    "option_type": option_type,
                    "feed": feed,
                    "asof": asof,
                },
                retry_config=self.config.retry,
                call=lambda: self._resolve_alpaca_client().get_option_chain(
                    underlying,
                    expiry_gte=expiry_gte,
                    expiry_lte=expiry_lte,
                    strike_gte=strike_gte,
                    strike_lte=strike_lte,
                    option_type=option_type,
                    feed=feed,
                    asof=asof,
                ),
                count_items=_count_alpaca_option_contracts,
            )
        except ProviderCallFailedError as exc:
            diagnostics.append(exc.diagnostic)
            raise ProviderSnapshotDataUnavailableError(
                f"Alpaca option chain is unavailable for {underlying!r}; "
                f"reason={exc.failure_kind}",
                diagnostic=exc.diagnostic,
                failure_kind=exc.failure_kind,
            ) from None
        diagnostics.append(diagnostic)
        return payload

    def _fetch_equity_bars(
        self,
        symbols: Sequence[str],
        *,
        start: datetime,
        end: datetime,
        timeframe: str,
        limit: int | None,
        adjustment: str | None,
        sort: str | None,
        feed: str | None,
        asof: str | None,
        diagnostics: list[ProviderCallDiagnostic],
    ) -> Mapping[str, Any]:
        try:
            payload, diagnostic = _call_provider_with_diagnostic(
                provider="alpaca",
                operation="equity_bars",
                request_metadata={
                    "symbols": list(symbols),
                    "start": start,
                    "end": end,
                    "timeframe": timeframe,
                    "limit": limit,
                    "adjustment": adjustment,
                    "sort": sort,
                    "feed": feed,
                    "asof": asof,
                },
                retry_config=self.config.retry,
                call=lambda: self._resolve_alpaca_client().get_equity_bars(
                    symbols,
                    start=start,
                    end=end,
                    timeframe=timeframe,
                    limit=limit,
                    adjustment=adjustment,
                    sort=sort,
                    feed=feed,
                    asof=asof,
                ),
                count_items=_count_alpaca_equity_bars,
            )
        except ProviderCallFailedError as exc:
            diagnostics.append(exc.diagnostic)
            raise ProviderSnapshotDataUnavailableError(
                f"Alpaca equity bars are unavailable; reason={exc.failure_kind}",
                diagnostic=exc.diagnostic,
                failure_kind=exc.failure_kind,
            ) from None
        diagnostics.append(diagnostic)
        return payload

    def _fetch_fred_observations(
        self,
        series_id: str,
        *,
        asof: pd.Timestamp,
        lookback_days: int,
        diagnostics: list[ProviderCallDiagnostic] | None = None,
    ) -> Mapping[str, Any]:
        observation_start = asof.date() - timedelta(days=lookback_days)
        observation_end = asof.date()
        return self._fetch_fred_observations_window(
            series_id,
            observation_start=observation_start,
            observation_end=observation_end,
            diagnostics=diagnostics,
        )

    def _fetch_fred_observations_window(
        self,
        series_id: str,
        *,
        observation_start: date,
        observation_end: date,
        diagnostics: list[ProviderCallDiagnostic] | None = None,
    ) -> Mapping[str, Any]:
        try:
            payload, diagnostic = _call_provider_with_diagnostic(
                provider="fred",
                operation="fred_observations",
                request_metadata={
                    "series_id": series_id,
                    "observation_start": observation_start,
                    "observation_end": observation_end,
                    "sort_order": "asc",
                },
                retry_config=self.config.retry,
                call=lambda: self._resolve_fred_client().fetch_observations(
                    series_id,
                    observation_start=observation_start,
                    observation_end=observation_end,
                    sort_order="asc",
                ),
                count_items=_count_fred_observations_for_diagnostic,
            )
        except ProviderCallFailedError as exc:
            if diagnostics is not None:
                diagnostics.append(exc.diagnostic)
            raise ProviderSnapshotDataUnavailableError(
                "FRED observations are unavailable for "
                f"series_id={series_id!r}; reason={exc.failure_kind}",
                diagnostic=exc.diagnostic,
                failure_kind=exc.failure_kind,
            ) from None
        if diagnostics is not None:
            diagnostics.append(diagnostic)
        return payload

    def _load_snapshot_rate_curve_series(
        self,
        series_id: str,
        *,
        asof: pd.Timestamp,
        lookback_days: int,
        diagnostics: list[ProviderCallDiagnostic],
    ) -> pd.DataFrame | None:
        try:
            payload = self._fetch_fred_observations(
                series_id,
                asof=asof,
                lookback_days=lookback_days,
                diagnostics=diagnostics,
            )
            return _normalize_fred_observations_for_snapshot(
                payload,
                series_id=series_id,
                asof=asof,
                diagnostics=diagnostics,
            )
        except ProviderSnapshotDataUnavailableError:
            return None

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


def _cleaning_policy(policy: QuoteCleaningPolicyV1 | None) -> QuoteCleaningPolicyV1:
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
            retry=config.retry,
            policy=config.policy,
        )
    return resolved_config, local_storage


def _coerce_marketdata_policy_config(
    policy_config: MarketDataPolicyConfig | Mapping[str, object] | Path,
) -> MarketDataPolicyConfig:
    if isinstance(policy_config, MarketDataPolicyConfig):
        return policy_config
    if isinstance(policy_config, Path):
        return MarketDataPolicyConfig.from_file(policy_config)
    if isinstance(policy_config, Mapping):
        return MarketDataPolicyConfig.from_mapping(policy_config)
    raise TypeError(
        "policy_config must be a MarketDataPolicyConfig, mapping, or pathlib.Path"
    )


def _clean_underlying(value: str) -> str:
    return _required_text(value, "underlying").upper()


def _clean_rate_series_id(value: str) -> str:
    return _required_text(value, "rate_series_id").upper()


def _nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    try:
        integer = int(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be an integer") from exc
    if integer < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return integer


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
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _nonnegative_finite_float(value: object, field_name: str) -> float:
    number = _finite_float(value, field_name)
    if number < 0.0:
        raise ValueError(f"{field_name} must be >= 0")
    return number


def _optional_text(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _required_run_id(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("run_id must be a string")
    run_id = value.strip()
    if not run_id:
        raise ValueError("run_id is required")
    return run_id


def _optional_run_id(value: str | None) -> str | None:
    if value is None:
        return None
    return _required_run_id(value)


def _new_run_id(asof: pd.Timestamp) -> str:
    timestamp = asof.strftime("%Y%m%dT%H%M%SZ")
    return f"b4a-{timestamp}-{uuid4().hex[:8]}"


def _new_refresh_daily_run_id(
    asof: pd.Timestamp,
    *,
    run_id_prefix: str | None,
) -> str:
    cleaned_prefix = _optional_text(run_id_prefix, "run_id_prefix")
    timestamp = asof.strftime("%Y%m%dT%H%M%SZ")
    return f"{cleaned_prefix or 'b4-refresh-daily'}-{timestamp}-{uuid4().hex[:8]}"


def _refresh_daily_child_run_id(
    aggregate_run_id: str,
    underlying: str,
    *,
    ordinal: int,
) -> str:
    return f"{aggregate_run_id}-{ordinal:02d}-{underlying.lower()}"


def _snapshot_curve_series_ids(
    curve_series_ids: Sequence[str] | None,
    *,
    primary_series_id: str,
) -> tuple[str, ...]:
    if curve_series_ids is None:
        cleaned = list(DEFAULT_RATE_CURVE_SERIES_IDS)
    else:
        if isinstance(curve_series_ids, str):
            raw_series_ids: tuple[str, ...] = (curve_series_ids,)
        else:
            raw_series_ids = tuple(curve_series_ids)
        if not raw_series_ids:
            return ()
        cleaned = [_clean_rate_series_id(series_id) for series_id in raw_series_ids]

    seen: set[str] = set()
    deduped: list[str] = []
    for series_id in cleaned:
        if series_id in seen:
            continue
        seen.add(series_id)
        deduped.append(series_id)
    if deduped and primary_series_id not in seen:
        deduped.insert(0, primary_series_id)
    return tuple(deduped)


def _normalize_latest_equity_quotes_for_snapshot(
    payload: Mapping[str, Any],
    *,
    underlying: str,
    asof: pd.Timestamp,
    diagnostics: list[ProviderCallDiagnostic] | None = None,
) -> pd.DataFrame:
    try:
        return normalize_alpaca_latest_quotes(payload, asof=asof)
    except Exception as exc:
        diagnostic = _normalization_failure_diagnostic(
            provider="alpaca",
            operation="latest_equity_quote_normalization",
            request_metadata={"underlying": underlying, "asof": asof},
            exception=exc,
        )
        if diagnostics is not None:
            diagnostics.append(diagnostic)
        raise ProviderSnapshotDataUnavailableError(
            "Alpaca latest equity quote is unavailable for "
            f"{underlying!r}; reason=normalization_failed",
            diagnostic=diagnostic,
            failure_kind="normalization_failed",
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
    feed: str | None,
    diagnostics: list[ProviderCallDiagnostic] | None = None,
) -> AlpacaOptionChainNormalizationAudit:
    try:
        return normalize_alpaca_option_chain_with_audit(
            payload,
            underlying=underlying,
            asof=asof,
            feed=feed,
        )
    except Exception as exc:
        diagnostic = _normalization_failure_diagnostic(
            provider="alpaca",
            operation="option_chain_normalization",
            request_metadata={"underlying": underlying, "asof": asof, "feed": feed},
            exception=exc,
        )
        if diagnostics is not None:
            diagnostics.append(diagnostic)
        raise ProviderSnapshotDataUnavailableError(
            "No usable Alpaca option contracts remain after provider "
            f"normalization for {underlying!r}; reason=normalization_failed",
            diagnostic=diagnostic,
            failure_kind="normalization_failed",
        ) from None


def _normalize_fred_observations_for_snapshot(
    payload: Mapping[str, Any],
    *,
    series_id: str,
    asof: pd.Timestamp,
    diagnostics: list[ProviderCallDiagnostic] | None = None,
) -> pd.DataFrame:
    try:
        return normalize_fred_observations(payload, series_id=series_id, asof=asof)
    except Exception as exc:
        diagnostic = _normalization_failure_diagnostic(
            provider="fred",
            operation="fred_observations_normalization",
            request_metadata={"series_id": series_id, "asof": asof},
            exception=exc,
        )
        if diagnostics is not None:
            diagnostics.append(diagnostic)
        raise ProviderSnapshotDataUnavailableError(
            f"FRED observations are unavailable for series_id={series_id!r}; "
            "reason=normalization_failed",
            diagnostic=diagnostic,
            failure_kind="normalization_failed",
        ) from None


def _select_rate_for_snapshot(
    fred_series: pd.DataFrame,
    *,
    series_id: str,
    asof: pd.Timestamp,
    lookback_days: int,
) -> Any:
    try:
        return select_latest_fred_rate_at_or_before_asof(
            fred_series,
            series_id=series_id,
            asof=asof,
        )
    except ProviderDataUnavailableError:
        observation_start = asof.date() - timedelta(days=lookback_days)
        raise ProviderSnapshotDataUnavailableError(
            "No usable FRED rate observation exists for "
            f"series_id={series_id!r} at or before {asof.date()} "
            f"within the last {lookback_days} days "
            f"(observation_start={observation_start.isoformat()})"
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
                "day_count": DEFAULT_DAY_COUNT,
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


def _count_alpaca_latest_equity_quotes(payload: Mapping[str, Any]) -> int:
    quotes = payload.get("quotes")
    quotes = getattr(quotes, "data", quotes)
    if isinstance(quotes, Mapping):
        return len(quotes)
    if isinstance(quotes, Sequence) and not isinstance(
        quotes,
        (str, bytes, bytearray),
    ):
        return len(quotes)
    return 0


def _count_fred_observations_for_diagnostic(payload: Mapping[str, Any]) -> int:
    observations = payload.get("observations")
    observations = getattr(observations, "data", observations)
    if isinstance(observations, Sequence) and not isinstance(
        observations,
        (str, bytes, bytearray),
    ):
        return len(observations)
    return 0


def _count_alpaca_equity_bars(payload: Mapping[str, Any]) -> int:
    bars = payload.get("bars", payload.get("bar"))
    bars = getattr(bars, "data", bars)
    if isinstance(bars, Mapping):
        return sum(_count_alpaca_bar_records(records) for records in bars.values())
    if isinstance(bars, Sequence) and not isinstance(bars, (str, bytes, bytearray)):
        return len(bars)
    return 0 if bars is None else 1


def _count_alpaca_bar_records(records: Any) -> int:
    records = getattr(records, "data", records)
    if isinstance(records, Sequence) and not isinstance(
        records,
        (str, bytes, bytearray),
    ):
        return len(records)
    return 0 if records is None else 1


def _provider_refresh_daily_run_details(
    *,
    storage: LocalStorage,
    aggregate_run_id: str,
    child_run_ids: Sequence[str],
    underlyings: Sequence[str],
    asof: pd.Timestamp,
    rate_series_id: str,
    equity_feed: str,
    option_feed: str,
    selected_rate: float,
    rate_observation_date: pd.Timestamp,
    rate_lookback_days: int,
    curve_series_ids: Sequence[str],
    quality_policy: Mapping[str, object],
    expiry_gte: date | str | None,
    expiry_lte: date | str | None,
    strike_gte: float | None,
    strike_lte: float | None,
    option_type: str | None,
    dividend_yield: float,
    dividend_yield_source: str,
    counts: ProviderRefreshDailyCounts,
    warnings: Sequence[str],
    artifact_paths: Sequence[Path],
    library_commit: str | None,
) -> dict[str, object]:
    rate_policy = _provider_snapshot_rate_policy(
        rate_series_id=rate_series_id,
        rate_source=f"fred:{rate_series_id}",
        rate_observation_date=rate_observation_date,
        selected_rate=selected_rate,
        lookback_days=rate_lookback_days,
        curve_series_ids=curve_series_ids,
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
    )
    return {
        "operation": "refresh_daily",
        "provider": "alpaca+fred",
        "equity_provider": "alpaca",
        "equity_feed": equity_feed,
        "option_provider": "alpaca",
        "option_feed": option_feed,
        "aggregate_run_id": aggregate_run_id,
        "child_run_ids": list(child_run_ids),
        "underlyings": list(underlyings),
        "asof": _utc_isoformat(asof),
        "rate_series_id": rate_series_id,
        "selected_rate": float(selected_rate),
        "flat_rate": float(selected_rate),
        "rate_policy": rate_policy,
        "dividend_policy": dividend_policy,
        "option_cleaning_policy": option_cleaning_policy,
        "data_policy": data_policy,
        "quality_policy": dict(quality_policy),
        "current_provider_scope": _current_provider_scope(),
        "filters": {
            "expiry_gte": expiry_gte,
            "expiry_lte": expiry_lte,
            "strike_gte": strike_gte,
            "strike_lte": strike_lte,
            "option_type": option_type,
        },
        "dividend_yield": dividend_yield,
        "dividend_yield_source": dividend_yield_source,
        "counts": {
            "raw_option_contract_count": counts.raw_option_contract_count,
            "normalized_option_contract_count": counts.normalized_option_contract_count,
            "dropped_before_cleaning_count": counts.dropped_before_cleaning_count,
            "provider_rejected_contract_count": counts.provider_rejected_contract_count,
            "accepted_quote_count": counts.accepted_quote_count,
            "rejected_quote_count": counts.rejected_quote_count,
        },
        "warnings": list(warnings),
        "artifact_paths": _relative_artifact_references(storage.root, artifact_paths),
        "library_commit": library_commit,
    }


__all__ = ["MarketDataPipeline", "ProviderSnapshotDataUnavailableError"]
