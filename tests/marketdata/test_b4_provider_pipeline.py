from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

import option_pricing.marketdata.cli as cli
from option_pricing.marketdata.config import (
    AlpacaConfig,
    FredConfig,
    PipelineConfig,
    ProviderRetryConfig,
    StorageConfig,
)
from option_pricing.marketdata.errors import (
    MissingProviderCredentialError,
    ProviderRequestError,
)
from option_pricing.marketdata.pipeline import (
    MarketDataPipeline,
    ProviderSnapshotDataUnavailableError,
    ProviderSnapshotResult,
    run_local_model_validation_pipeline,
)


class _FakeAlpacaClient:
    def __init__(
        self,
        *,
        equity_payload: Mapping[str, Any] | None = None,
        option_payload: Mapping[str, Any] | None = None,
        bar_payload: Mapping[str, Any] | None = None,
    ) -> None:
        self.equity_payload = (
            _equity_quote_payload() if equity_payload is None else equity_payload
        )
        self.option_payload = (
            _option_chain_payload() if option_payload is None else option_payload
        )
        self.bar_payload = (
            _bars_payload(("SPY",)) if bar_payload is None else bar_payload
        )
        self.equity_calls: list[dict[str, object]] = []
        self.option_calls: list[dict[str, object]] = []
        self.bar_calls: list[dict[str, object]] = []

    def get_latest_equity_quotes(
        self,
        symbols: str,
        *,
        asof: object | None = None,
    ) -> Mapping[str, Any]:
        self.equity_calls.append({"symbols": symbols, "asof": asof})
        return self.equity_payload

    def get_option_chain(
        self,
        underlying: str,
        *,
        expiry_gte: object | None = None,
        expiry_lte: object | None = None,
        strike_gte: float | None = None,
        strike_lte: float | None = None,
        option_type: str | None = None,
        feed: str | None = None,
        asof: object | None = None,
    ) -> Mapping[str, Any]:
        self.option_calls.append(
            {
                "underlying": underlying,
                "expiry_gte": expiry_gte,
                "expiry_lte": expiry_lte,
                "strike_gte": strike_gte,
                "strike_lte": strike_lte,
                "option_type": option_type,
                "feed": feed,
                "asof": asof,
            }
        )
        return self.option_payload

    def get_equity_bars(
        self,
        symbols: object,
        **kwargs: object,
    ) -> Mapping[str, Any]:
        self.bar_calls.append({"symbols": symbols, **kwargs})
        return self.bar_payload


class _FakeFredClient:
    def __init__(
        self,
        payload: Mapping[str, Any] | None = None,
        *,
        payloads: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.payload = _fred_payload() if payload is None else payload
        self.payloads = payloads or {}
        self.calls: list[dict[str, object]] = []

    def fetch_observations(
        self,
        series_id: str,
        **kwargs: object,
    ) -> Mapping[str, Any]:
        self.calls.append({"series_id": series_id, **kwargs})
        if series_id in self.payloads:
            return self.payloads[series_id]
        return self.payload


@pytest.fixture
def fake_parquet(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_to_parquet(
        self: pd.DataFrame,
        path: str,
        compression: str | None = None,
        index: bool = False,
    ) -> None:
        del compression
        payload = self if index else self.reset_index(drop=True)
        payload.to_pickle(path)

    def _fake_read_parquet(
        path: str,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        frame = cast(pd.DataFrame, pd.read_pickle(path))
        if columns is None:
            return frame
        return cast(pd.DataFrame, frame.loc[:, columns])

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet)
    monkeypatch.setattr(pd, "read_parquet", _fake_read_parquet)


def _pipeline(
    tmp_path: Path,
    *,
    alpaca_client: _FakeAlpacaClient | None = None,
    fred_client: _FakeFredClient | None = None,
) -> MarketDataPipeline:
    return MarketDataPipeline(
        PipelineConfig(
            alpaca=AlpacaConfig(
                api_key_env="B4_TEST_ALPACA_KEY",
                secret_key_env="B4_TEST_ALPACA_SECRET",
            ),
            fred=FredConfig(api_key_env="B4_TEST_FRED_KEY"),
            storage=StorageConfig(root=tmp_path),
        ),
        alpaca_client=alpaca_client or _FakeAlpacaClient(),
        fred_client=fred_client or _FakeFredClient(),
    )


def _equity_quote_payload() -> dict[str, object]:
    return {
        "quotes": {
            "SPY": {
                "timestamp": "2026-05-22T15:30:00Z",
                "bid_price": 499.0,
                "ask_price": 501.0,
                "bid_size": 10,
                "ask_size": 12,
            }
        },
        "source": "alpaca",
        "feed": "indicative",
    }


def _option_chain_payload() -> dict[str, object]:
    return {
        "underlying": "SPY",
        "contracts": {
            "SPY260619C00500000": {
                "latest_quote": {
                    "timestamp": "2026-05-22T15:30:00Z",
                    "bid_price": 4.0,
                    "ask_price": 4.4,
                },
                "implied_volatility": 0.2,
                "greeks": {"vega": 0.18},
            },
            "SPY260619P00500000": {
                "latest_quote": {
                    "timestamp": "2026-05-22T15:30:00Z",
                    "bid_price": 3.8,
                    "ask_price": 4.2,
                },
                "implied_volatility": 0.22,
                "greeks": {"vega": 0.19},
            },
            "SPY260619C00510000": {
                "latest_quote": {
                    "timestamp": "2026-05-22T15:30:00Z",
                    "ask_price": 2.0,
                }
            },
        },
        "source": "alpaca",
        "feed": "indicative",
    }


def _fred_payload() -> dict[str, object]:
    return {
        "observations": [
            {
                "realtime_start": "2026-05-01",
                "realtime_end": "2026-05-31",
                "date": "2026-05-20",
                "value": "4.25",
            },
            {
                "realtime_start": "2026-05-01",
                "realtime_end": "2026-05-31",
                "date": "2026-05-21",
                "value": ".",
            },
        ]
    }


def _bar_record(
    symbol: str,
    *,
    timestamp: str = "2026-05-22T11:30:00-04:00",
    close: float = 500.5,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "timestamp": timestamp,
        "timeframe": "1Day",
        "open": close - 1.5,
        "high": close + 0.5,
        "low": close - 2.0,
        "close": close,
        "volume": 1000,
        "trade_count": 75,
        "vwap": close - 0.4,
    }


def _bars_payload(symbols: tuple[str, ...]) -> dict[str, object]:
    return {
        "symbols": symbols,
        "bars": {
            symbol: [_bar_record(symbol, close=500.5 + index)]
            for index, symbol in enumerate(symbols)
        },
        "source": "alpaca",
        "feed": "iex",
        "timeframe": "1Day",
    }


def _read_json(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        cast(dict[str, object], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _fred_backfill_root(
    tmp_path: Path,
    *,
    layer: str,
    series_id: str,
    start: str,
    end: str,
    run_id: str,
) -> Path:
    return (
        tmp_path
        / layer
        / "fred_series"
        / f"series_id={series_id}"
        / f"start_date={start}"
        / f"end_date={end}"
        / f"run_id={run_id}"
    )


def _bars_backfill_root(
    tmp_path: Path,
    *,
    layer: str,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    run_id: str,
) -> Path:
    return (
        tmp_path
        / layer
        / "equity_bars"
        / f"symbol={symbol}"
        / f"timeframe={timeframe}"
        / f"start_date={start}"
        / f"end_date={end}"
        / f"run_id={run_id}"
    )


def _snapshot(tmp_path: Path, **overrides: object) -> ProviderSnapshotResult:
    snapshot_kwargs: dict[str, object] = {
        "asof": "2026-05-22T15:31:00Z",
        "run_id": "b4-test-run",
        "expiry_gte": "2026-06-01",
        "expiry_lte": "2026-06-30",
        "feed": "indicative",
        "library_commit": "abc123",
    }
    snapshot_kwargs.update(overrides)
    return _pipeline(tmp_path).snapshot("spy", **snapshot_kwargs)


def test_provider_snapshot_works_end_to_end_with_fake_clients(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    result = _snapshot(tmp_path)

    assert result.underlying == "SPY"
    assert result.asof == pd.Timestamp("2026-05-22T15:31:00Z")
    assert result.run_id == "b4-test-run"
    assert result.spot == pytest.approx(500.0)
    assert result.rate == pytest.approx(math.log1p(4.25 / 100.0))
    assert result.rate_source == "fred:DGS3MO"
    assert result.rate_observation_date == pd.Timestamp("2026-05-20")
    assert result.rate_series_id == "DGS3MO"
    assert result.feed == "indicative"
    assert result.dividend_yield == pytest.approx(0.0)
    assert result.dividend_yield_source == "assumption"
    assert result.raw_option_contract_count == 3
    assert result.normalized_option_contract_count == 2
    assert result.accepted_quote_count == 2
    assert result.rejected_quote_count == 0
    assert result.provider_rejected_contract_count == 1
    assert result.dropped_before_cleaning_count == 1
    assert result.warnings[:4] == (
        "documented_assumption: dividend_yield=0.0, "
        "dividend_yield_source=assumption, dividend_inference=not_enabled",
        "documented_assumption: rate_series_id=DGS3MO, "
        "default_rate_series_id=DGS3MO, curve_interpolation=not_enabled",
        "current_provider_scope: option_chain_backfill=not_enabled",
        "current_provider_scope: scheduling=not_enabled",
    )
    assert result.warnings[-1] == (
        "alpaca_option_contracts_dropped_before_cleaning: "
        "dropped=1, raw=3, normalized=2, "
        "provider_rejected_contracts=1, stage=provider_normalization"
    )

    for path in result.artifact_paths:
        assert path.exists()

    assert result.bronze_paths.manifest.exists()
    assert result.silver_paths.option_chain.exists()
    assert result.silver_paths.fred_series.exists()
    assert result.silver_paths.provider_rejected_contracts is not None
    assert result.silver_paths.provider_rejected_contracts.exists()
    assert result.gold_paths.market_data.exists()
    assert result.rate_curve_paths is not None
    assert result.rate_curve_paths.rate_curve.exists()
    assert result.rate_curve_paths.manifest.exists()
    assert result.model_validation_bundle.manifest_path.exists()

    bronze_manifest = _read_json(result.bronze_paths.manifest)
    silver_manifest = _read_json(result.silver_paths.manifest)
    gold_manifest = _read_json(result.gold_paths.market_manifest)
    rate_curve_manifest = _read_json(result.rate_curve_paths.manifest)
    bundle_manifest = _read_json(result.model_validation_bundle.manifest_path)
    warnings_payload = _read_json(
        result.model_validation_bundle.manifest_path.parent / "warnings.json"
    )
    run_entries = _read_jsonl(tmp_path / "_meta" / "runs.jsonl")
    provider_rejected_contracts = pd.read_parquet(
        result.silver_paths.provider_rejected_contracts
    )
    rate_curve = pd.read_parquet(result.rate_curve_paths.rate_curve)

    assert bronze_manifest["providers"] == {
        "spot": "alpaca",
        "option_chain": "alpaca",
        "rate": "fred",
    }
    assert bronze_manifest["feed"] == "indicative"
    assert bronze_manifest["rate_assumptions"] == {
        "provider": "fred",
        "series_id": "DGS3MO",
        "default_series_id": "DGS3MO",
        "curve_interpolation": "not_enabled",
    }
    assert bronze_manifest["dividend_assumptions"] == {
        "dividend_yield": 0.0,
        "source": "assumption",
        "dividend_inference": "not_enabled",
    }
    assert bronze_manifest["dividend_policy"] == bronze_manifest["dividend_assumptions"]
    assert bronze_manifest["rate_policy"] == {
        "provider": "fred",
        "series_id": "DGS3MO",
        "default_series_id": "DGS3MO",
        "lookback_days": 90,
        "curve_series_ids": ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2"],
        "curve_interpolation": "not_enabled",
    }
    assert bronze_manifest["quality_policy"]["min_accepted_contracts"] == 1
    assert bronze_manifest["quote_freshness"] == {
        "equity_quote_age_seconds": 60.0,
        "equity_quote_after_asof": False,
        "stale_equity_quote": False,
        "option_quote_age_seconds_min": 60.0,
        "option_quote_age_seconds_median": 60.0,
        "option_quote_age_seconds_max": 60.0,
        "option_quote_count": 2,
        "option_quotes_after_asof_count": 0,
        "stale_option_quote_count": 0,
        "accepted_quote_count": 2,
        "accepted_call_count": 1,
        "accepted_put_count": 1,
        "accepted_expiry_count": 1,
        "stale_accepted_quote_count": 0,
        "accepted_quotes_after_asof_count": 0,
    }
    diagnostics = cast(
        list[dict[str, object]],
        bronze_manifest["provider_operation_diagnostics"],
    )
    assert [item["operation"] for item in diagnostics[:3]] == [
        "latest_equity_quote",
        "option_chain",
        "fred_observations",
    ]
    assert all(item["status"] == "ok" for item in diagnostics)
    assert diagnostics[0]["rows_or_contracts_in"] == 1
    assert diagnostics[1]["rows_or_contracts_in"] == 3
    assert diagnostics[2]["rows_or_contracts_in"] == 2
    assert bronze_manifest["current_provider_scope"] == {
        "curve_interpolation": "not_enabled",
        "dividend_inference": "not_enabled",
        "option_chain_backfill": "not_enabled",
        "scheduling": "not_enabled",
    }
    assert bronze_manifest["request_metadata"] == {
        "underlying": "SPY",
        "asof": "2026-05-22T15:31:00Z",
        "expiry_gte": "2026-06-01",
        "expiry_lte": "2026-06-30",
        "strike_gte": None,
        "strike_lte": None,
        "option_type": None,
        "feed": "indicative",
        "rate_series_id": "DGS3MO",
        "rate_lookback_days": 90,
        "curve_series_ids": ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2"],
        "fred_observation_start": "2026-02-21",
        "fred_observation_end": "2026-05-22",
    }
    assert bronze_manifest["rows"]["provider_rejected_contracts"] == 1
    assert bronze_manifest["rows"]["rate_curve_points"] == 5
    assert silver_manifest["source_type"] == "provider_snapshot"
    assert silver_manifest["rate_source"] == "fred:DGS3MO"
    assert silver_manifest["dividend_yield_source"] == "assumption"
    assert silver_manifest["warnings"] == list(result.warnings)
    assert gold_manifest["source"]["source_type"] == "provider_snapshot"
    assert gold_manifest["sources"]["rate_source"] == "fred:DGS3MO"
    assert gold_manifest["sources"]["dividend_yield_source"] == "assumption"
    assert rate_curve_manifest == {
        "provider_snapshot_rate_curve_schema_version": "provider_rate_curve_gold.v1",
        "artifact": "rate_curve",
        "run_id": "b4-test-run",
        "snapshot_id": bronze_manifest["snapshot_id"],
        "underlying": "SPY",
        "valuation_timestamp_utc": "2026-05-22T15:31:00Z",
        "rate_compounding": "continuous",
        "day_count": "ACT/365",
        "lookback_days": 90,
        "requested_series_ids": ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2"],
        "included_series_ids": ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2"],
        "rows": {"rate_curve": 5},
        "artifacts": {"rate_curve": "rate_curve.parquet"},
        "source": {
            "source_type": "provider_snapshot",
            "fixture_name": "provider_snapshot_v1",
        },
        "rate_policy": {
            "provider": "fred",
            "primary_series_id": "DGS3MO",
            "requested_series_ids": ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2"],
            "lookback_days": 90,
            "curve_interpolation": "not_enabled",
        },
        "current_provider_scope": {
            "curve_interpolation": "not_enabled",
            "dividend_inference": "not_enabled",
            "option_chain_backfill": "not_enabled",
            "scheduling": "not_enabled",
        },
        "library_commit": "abc123",
        "dataset": "curves",
        "layer": "gold",
        "partitions": {
            "underlying": "SPY",
            "date": "2026-05-22",
            "run_id": "b4-test-run",
        },
        "written_at": rate_curve_manifest["written_at"],
    }
    assert bundle_manifest["rate_source"] == "fred:DGS3MO"
    assert bundle_manifest["dividend_yield_source"] == "assumption"
    assert warnings_payload["warnings"] == list(result.warnings)
    rejected_row = provider_rejected_contracts.iloc[0]
    assert len(provider_rejected_contracts) == 1
    assert rejected_row["underlying"] == "SPY"
    assert rejected_row["contract_symbol"] == "SPY260619C00510000"
    assert rejected_row["payload_contract_key"] == "SPY260619C00510000"
    assert rejected_row["asof"] == pd.Timestamp("2026-05-22T15:31:00Z")
    assert rejected_row["source"] == "alpaca"
    assert rejected_row["feed"] == "indicative"
    assert rejected_row["rejection_stage"] == "provider_normalization"
    assert rejected_row["reason"] == "missing_required_price"
    assert rejected_row["rejection_detail"] == "latest_quote is missing bid"
    assert rejected_row["raw_quote_timestamp"] == "2026-05-22T15:30:00Z"
    assert pd.isna(rejected_row["raw_bid"])
    assert rejected_row["raw_ask"] == "2.0"
    assert pd.isna(rejected_row["raw_expiry"])
    assert pd.isna(rejected_row["raw_strike"])
    assert pd.isna(rejected_row["raw_right"])
    assert rate_curve["series_id"].astype(str).tolist() == [
        "DGS1MO",
        "DGS3MO",
        "DGS6MO",
        "DGS1",
        "DGS2",
    ]
    assert rate_curve["tenor"].astype(str).tolist() == ["1M", "3M", "6M", "1Y", "2Y"]
    assert rate_curve["day_count"].astype(str).tolist() == ["ACT/365"] * 5
    assert len(run_entries) == 1
    assert run_entries[0]["artifacts"] == [
        path.relative_to(tmp_path).as_posix() for path in result.artifact_paths
    ]
    run_details = cast(dict[str, object], run_entries[0]["details"])
    assert run_details["provider_operation_diagnostics"] == diagnostics
    assert run_details == {
        "operation": "snapshot",
        "provider": "alpaca+fred",
        "underlying": "SPY",
        "asof": "2026-05-22T15:31:00Z",
        "run_id": "b4-test-run",
        "rate_series_id": "DGS3MO",
        "rate_source": "fred:DGS3MO",
        "rate_observation_date": "2026-05-20",
        "spot_source": "alpaca",
        "dividend_yield": 0.0,
        "dividend_yield_source": "assumption",
        "dividend_policy": {
            "dividend_yield": 0.0,
            "source": "assumption",
            "dividend_inference": "not_enabled",
        },
        "feed": "indicative",
        "rate_lookback_days": 90,
        "rate_policy": {
            "provider": "fred",
            "series_id": "DGS3MO",
            "rate_source": "fred:DGS3MO",
            "rate_observation_date": "2026-05-20",
            "lookback_days": 90,
            "curve_series_ids": ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2"],
            "curve_interpolation": "not_enabled",
        },
        "raw_option_contract_count": 3,
        "normalized_option_contract_count": 2,
        "dropped_before_cleaning_count": 1,
        "provider_rejected_contract_count": 1,
        "accepted_quote_count": 2,
        "rejected_quote_count": 0,
        "quality_policy": bronze_manifest["quality_policy"],
        "quote_freshness": bronze_manifest["quote_freshness"],
        "provider_operation_diagnostics": diagnostics,
        "current_provider_scope": {
            "curve_interpolation": "not_enabled",
            "dividend_inference": "not_enabled",
            "option_chain_backfill": "not_enabled",
            "scheduling": "not_enabled",
        },
        "warnings": list(result.warnings),
        "artifact_paths": [
            path.relative_to(tmp_path).as_posix() for path in result.artifact_paths
        ],
        "library_commit": "abc123",
    }


def test_provider_snapshot_result_payload_is_json_serializable(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    result = _snapshot(tmp_path)

    payload = cli._result_payload("snapshot", result)

    encoded = json.dumps(payload, sort_keys=True)
    assert json.loads(encoded)["run_id"] == "b4-test-run"
    assert "DGS3MO" in encoded
    assert "dividend_inference=not_enabled" in encoded


def test_refresh_daily_dispatches_snapshots_and_records_aggregate_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = MarketDataPipeline(storage=tmp_path)
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_snapshot(underlying: str, **kwargs: object) -> object:
        calls.append((underlying, kwargs))
        return SimpleNamespace(
            underlying=underlying,
            run_id=cast(str, kwargs["run_id"]),
            artifact_paths=(tmp_path / f"{underlying.lower()}-market_data.json",),
            raw_option_contract_count=3,
            normalized_option_contract_count=2,
            dropped_before_cleaning_count=1,
            provider_rejected_contract_count=1,
            accepted_quote_count=2,
            rejected_quote_count=0,
            warnings=("sample warning",),
        )

    monkeypatch.setattr(pipeline, "snapshot", fake_snapshot)

    result = pipeline.refresh_daily(
        ["spy", "qqq"],
        asof="2026-05-22T15:31:00Z",
        run_id_prefix="daily-close",
        rate_series_id="DGS3MO",
        feed="sip",
        dividend_yield=0.0125,
        dividend_yield_source="manual_override",
        overwrite=True,
    )

    assert result.aggregate_run_id.startswith("daily-close-20260522T153100Z-")
    assert result.underlyings == ("SPY", "QQQ")
    assert [call[0] for call in calls] == ["SPY", "QQQ"]
    assert result.child_run_ids == tuple(
        cast(str, call_kwargs["run_id"]) for _, call_kwargs in calls
    )
    assert all(
        call_kwargs["asof"] == pd.Timestamp("2026-05-22T15:31:00Z")
        for _, call_kwargs in calls
    )
    assert all(call_kwargs["feed"] == "sip" for _, call_kwargs in calls)
    assert result.counts.raw_option_contract_count == 6
    assert result.counts.normalized_option_contract_count == 4
    assert result.counts.dropped_before_cleaning_count == 2
    assert result.counts.provider_rejected_contract_count == 2
    assert result.counts.accepted_quote_count == 4
    assert result.counts.rejected_quote_count == 0
    assert result.warnings == ("sample warning",)
    assert result.artifact_paths == (
        tmp_path / "spy-market_data.json",
        tmp_path / "qqq-market_data.json",
    )

    run_entries = _read_jsonl(tmp_path / "_meta" / "runs.jsonl")
    assert len(run_entries) == 1
    assert run_entries[0]["run_id"] == result.aggregate_run_id
    assert run_entries[0]["artifacts"] == [
        "spy-market_data.json",
        "qqq-market_data.json",
    ]
    run_details = cast(dict[str, object], run_entries[0]["details"])
    assert run_details == {
        "operation": "refresh_daily",
        "provider": "alpaca+fred",
        "aggregate_run_id": result.aggregate_run_id,
        "child_run_ids": list(result.child_run_ids),
        "underlyings": ["SPY", "QQQ"],
        "asof": "2026-05-22T15:31:00Z",
        "rate_series_id": "DGS3MO",
        "feed": "sip",
        "rate_policy": {
            "provider": "fred",
            "series_id": "DGS3MO",
            "lookback_days": 90,
            "curve_series_ids": ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2"],
            "curve_interpolation": "not_enabled",
        },
        "quality_policy": {
            "max_equity_quote_age_seconds": None,
            "max_option_quote_age_seconds": None,
            "warn_on_stale_quotes": True,
            "reject_stale_option_quotes": False,
            "reject_option_quotes_after_asof": False,
            "reject_stale_equity_quote": False,
            "require_option_quotes_on_or_before_asof": True,
            "require_equity_quote_on_or_before_asof": True,
            "min_accepted_contracts": 1,
            "min_accepted_calls": None,
            "min_accepted_puts": None,
            "min_expiries": None,
        },
        "current_provider_scope": {
            "curve_interpolation": "not_enabled",
            "dividend_inference": "not_enabled",
            "option_chain_backfill": "not_enabled",
            "scheduling": "not_enabled",
        },
        "filters": {
            "expiry_gte": None,
            "expiry_lte": None,
            "strike_gte": None,
            "strike_lte": None,
            "option_type": None,
        },
        "dividend_yield": 0.0125,
        "dividend_yield_source": "manual_override",
        "counts": {
            "raw_option_contract_count": 6,
            "normalized_option_contract_count": 4,
            "dropped_before_cleaning_count": 2,
            "provider_rejected_contract_count": 2,
            "accepted_quote_count": 4,
            "rejected_quote_count": 0,
        },
        "warnings": ["sample warning"],
        "artifact_paths": ["spy-market_data.json", "qqq-market_data.json"],
        "library_commit": None,
    }


def test_provider_snapshot_dividend_override_flows_through_metadata(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    result = _snapshot(
        tmp_path,
        run_id="dividend-override",
        dividend_yield=0.015,
        dividend_yield_source="manual_override",
        library_commit="dividend-override",
    )

    bronze_manifest = _read_json(result.bronze_paths.manifest)
    silver_manifest = _read_json(result.silver_paths.manifest)
    market_data = _read_json(result.gold_paths.market_data)
    gold_manifest = _read_json(result.gold_paths.market_manifest)
    bundle_manifest = _read_json(result.model_validation_bundle.manifest_path)

    assert result.dividend_yield == pytest.approx(0.015)
    assert result.dividend_yield_source == "manual_override"
    assert bronze_manifest["dividend_assumptions"] == {
        "dividend_yield": 0.015,
        "source": "manual_override",
        "dividend_inference": "not_enabled",
    }
    assert silver_manifest["dividend_yield"] == pytest.approx(0.015)
    assert silver_manifest["dividend_yield_source"] == "manual_override"
    assert cast(dict[str, object], market_data["market_data"])[
        "dividend_yield"
    ] == pytest.approx(0.015)
    assert (
        cast(dict[str, object], market_data["sources"])["dividend_yield_source"]
        == "manual_override"
    )
    assert (
        cast(dict[str, object], gold_manifest["sources"])["dividend_yield_source"]
        == "manual_override"
    )
    assert bundle_manifest["dividend_yield_source"] == "manual_override"


def test_provider_snapshot_outputs_do_not_leak_secrets(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    secret = "super-secret-value"
    alpaca_client = _FakeAlpacaClient(
        equity_payload={
            **_equity_quote_payload(),
            "authorization": secret,
        },
        option_payload={
            **_option_chain_payload(),
            "nested": {"secret_token": secret},
        },
    )
    fred_client = _FakeFredClient(
        {
            **_fred_payload(),
            "api_key": secret,
            "nested": {"secret_token": secret},
        }
    )

    result = _pipeline(
        tmp_path,
        alpaca_client=alpaca_client,
        fred_client=fred_client,
    ).snapshot(
        "SPY",
        asof="2026-05-22T15:31:00Z",
        run_id="snapshot-secret-check",
        feed="indicative",
    )

    manifest_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in result.artifact_paths
        if path.name == "manifest.json"
    )
    bronze_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            result.bronze_paths.latest_equity_quotes,
            result.bronze_paths.option_chain,
            result.bronze_paths.fred_observations,
        )
    )
    runs_text = (tmp_path / "_meta" / "runs.jsonl").read_text(encoding="utf-8")
    cli_text = json.dumps(cli._result_payload("snapshot", result), sort_keys=True)

    assert secret not in bronze_text
    assert secret not in manifest_text
    assert secret not in runs_text
    assert secret not in repr(result)
    assert secret not in cli_text
    assert "<redacted>" in bronze_text


def test_provider_failure_preserves_sanitized_diagnostic_context(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    secret = "provider-secret-value"

    class _FailingAlpacaClient(_FakeAlpacaClient):
        def get_latest_equity_quotes(
            self,
            symbols: str,
            *,
            asof: object | None = None,
        ) -> Mapping[str, Any]:
            self.equity_calls.append({"symbols": symbols, "asof": asof})
            raise RuntimeError(f"boom near {secret}")

    with pytest.raises(ProviderSnapshotDataUnavailableError) as excinfo:
        _pipeline(
            tmp_path,
            alpaca_client=_FailingAlpacaClient(),
        ).snapshot(
            "SPY",
            asof="2026-05-22T15:31:00Z",
            run_id="failing-diagnostic",
        )

    exc = excinfo.value
    assert exc.failure_kind == "provider_request_failed"
    assert exc.diagnostic is not None
    diagnostic = exc.diagnostic.as_dict()
    assert diagnostic["provider"] == "alpaca"
    assert diagnostic["operation"] == "latest_equity_quote"
    assert diagnostic["status"] == "failed"
    assert diagnostic["exception_type"] == "RuntimeError"
    assert secret not in json.dumps(diagnostic, sort_keys=True)
    assert "RuntimeError raised by provider call" in str(diagnostic["message"])


def test_provider_retry_retries_transient_request_and_records_retry_count(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    class _FlakyAlpacaClient(_FakeAlpacaClient):
        def __init__(self) -> None:
            super().__init__()
            self.remaining_failures = 1

        def get_latest_equity_quotes(
            self,
            symbols: str,
            *,
            asof: object | None = None,
        ) -> Mapping[str, Any]:
            self.equity_calls.append({"symbols": symbols, "asof": asof})
            if self.remaining_failures:
                self.remaining_failures -= 1
                raise ProviderRequestError("transient latest quote failure")
            return self.equity_payload

    alpaca_client = _FlakyAlpacaClient()
    pipeline = MarketDataPipeline(
        PipelineConfig(
            alpaca=AlpacaConfig(),
            fred=FredConfig(),
            storage=StorageConfig(root=tmp_path),
            retry=ProviderRetryConfig(
                max_attempts=2,
                wait_initial_seconds=0.0,
                wait_max_seconds=0.0,
            ),
        ),
        alpaca_client=alpaca_client,
        fred_client=_FakeFredClient(),
    )

    result = pipeline.snapshot(
        "SPY",
        asof="2026-05-22T15:31:00Z",
        run_id="retry-success",
        curve_series_ids=(),
    )

    assert len(alpaca_client.equity_calls) == 2
    assert result.diagnostics[0].operation == "latest_equity_quote"
    assert result.diagnostics[0].retry_count == 1
    assert result.diagnostics[0].status == "ok"


def test_provider_retry_does_not_retry_missing_credentials(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    class _MissingCredentialsAlpacaClient(_FakeAlpacaClient):
        def get_latest_equity_quotes(
            self,
            symbols: str,
            *,
            asof: object | None = None,
        ) -> Mapping[str, Any]:
            self.equity_calls.append({"symbols": symbols, "asof": asof})
            raise MissingProviderCredentialError("ALPACA_API_KEY")

    alpaca_client = _MissingCredentialsAlpacaClient()
    with pytest.raises(ProviderSnapshotDataUnavailableError) as excinfo:
        _pipeline(
            tmp_path,
            alpaca_client=alpaca_client,
        ).snapshot(
            "SPY",
            asof="2026-05-22T15:31:00Z",
            run_id="missing-credentials",
        )

    assert len(alpaca_client.equity_calls) == 1
    assert excinfo.value.failure_kind == "missing_credentials"
    assert excinfo.value.diagnostic is not None
    assert excinfo.value.diagnostic.retry_count == 0


def test_stale_option_quote_warns_by_default_policy(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    result = _snapshot(
        tmp_path,
        run_id="stale-warning",
        quality_policy={"max_option_quote_age_seconds": 30},
    )

    assert result.accepted_quote_count == 2
    assert result.quote_freshness["stale_option_quote_count"] == 2
    assert any("provider_quality: stale_option_quotes" in w for w in result.warnings)
    assert any(
        "provider_quality: stale_accepted_option_quotes" in w for w in result.warnings
    )


def test_stale_option_quote_can_be_rejected_by_policy(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    option_payload = _option_chain_payload()
    contracts = cast(dict[str, Any], option_payload["contracts"])
    cast(dict[str, Any], contracts["SPY260619C00500000"])["latest_quote"] = {
        "timestamp": "2026-05-22T15:31:00Z",
        "bid_price": 4.0,
        "ask_price": 4.4,
    }
    cast(dict[str, Any], contracts["SPY260619P00500000"])["latest_quote"] = {
        "timestamp": "2026-05-22T15:00:00Z",
        "bid_price": 3.8,
        "ask_price": 4.2,
    }
    alpaca_client = _FakeAlpacaClient(option_payload=option_payload)

    result = _pipeline(tmp_path, alpaca_client=alpaca_client).snapshot(
        "SPY",
        asof="2026-05-22T15:31:00Z",
        run_id="stale-rejection",
        curve_series_ids=(),
        quality_policy={
            "max_option_quote_age_seconds": 30,
            "reject_stale_option_quotes": True,
        },
    )

    rejected = pd.read_parquet(result.silver_paths.rejected_quotes)
    assert result.accepted_quote_count == 1
    assert result.rejected_quote_count == 1
    assert rejected["rejection_reason"].astype(str).tolist() == ["stale_quote"]


def test_option_quote_after_asof_warns_by_default_policy(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    option_payload = _option_chain_payload()
    contracts = cast(dict[str, Any], option_payload["contracts"])
    cast(dict[str, Any], contracts["SPY260619C00500000"])["latest_quote"] = {
        "timestamp": "2026-05-22T15:32:00Z",
        "bid_price": 4.0,
        "ask_price": 4.4,
    }
    alpaca_client = _FakeAlpacaClient(option_payload=option_payload)

    result = _pipeline(tmp_path, alpaca_client=alpaca_client).snapshot(
        "SPY",
        asof="2026-05-22T15:31:00Z",
        run_id="after-asof-warning",
        curve_series_ids=(),
    )

    assert result.accepted_quote_count == 2
    assert result.rejected_quote_count == 0
    assert result.quote_freshness["option_quotes_after_asof_count"] == 1
    assert result.quote_freshness["accepted_quotes_after_asof_count"] == 1
    assert any(
        "provider_quality: option_quotes_after_asof" in w for w in result.warnings
    )


def test_option_quote_after_asof_can_be_rejected_by_policy(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    option_payload = _option_chain_payload()
    contracts = cast(dict[str, Any], option_payload["contracts"])
    cast(dict[str, Any], contracts["SPY260619C00500000"])["latest_quote"] = {
        "timestamp": "2026-05-22T15:32:00Z",
        "bid_price": 4.0,
        "ask_price": 4.4,
    }
    alpaca_client = _FakeAlpacaClient(option_payload=option_payload)

    result = _pipeline(tmp_path, alpaca_client=alpaca_client).snapshot(
        "SPY",
        asof="2026-05-22T15:31:00Z",
        run_id="after-asof-rejection",
        curve_series_ids=(),
        quality_policy={"reject_option_quotes_after_asof": True},
    )

    rejected = pd.read_parquet(result.silver_paths.rejected_quotes)
    assert result.accepted_quote_count == 1
    assert result.rejected_quote_count == 1
    assert result.quote_freshness["option_quotes_after_asof_count"] == 1
    assert result.quote_freshness["accepted_quotes_after_asof_count"] == 0
    assert rejected["rejection_reason"].astype(str).tolist() == ["quote_after_asof"]
    assert rejected["rejection_detail"].astype(str).tolist() == [
        "quote_ts is after snapshot asof"
    ]
    assert any(
        "provider_quality: option_quotes_after_asof" in w for w in result.warnings
    )


def test_equity_quote_after_asof_can_fail_quality_policy(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    equity_payload = _equity_quote_payload()
    quote = cast(dict[str, Any], cast(dict[str, Any], equity_payload["quotes"])["SPY"])
    quote["timestamp"] = "2026-05-22T15:32:00Z"

    with pytest.raises(
        ProviderSnapshotDataUnavailableError,
        match="Equity quote timestamp is after",
    ) as excinfo:
        _pipeline(
            tmp_path,
            alpaca_client=_FakeAlpacaClient(equity_payload=equity_payload),
        ).snapshot(
            "SPY",
            asof="2026-05-22T15:31:00Z",
            run_id="equity-after-asof",
            curve_series_ids=(),
            quality_policy={"reject_stale_equity_quote": True},
        )

    assert excinfo.value.failure_kind == "quality_policy_failed"


def test_provider_snapshot_missing_fred_rate_fails_clearly(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    fred_client = _FakeFredClient(
        {
            "observations": [
                {
                    "realtime_start": "2026-05-01",
                    "realtime_end": "2026-05-31",
                    "date": "2026-05-20",
                    "value": ".",
                }
            ]
        }
    )

    with pytest.raises(
        ProviderSnapshotDataUnavailableError,
        match="within the last 90 days",
    ):
        _pipeline(tmp_path, fred_client=fred_client).snapshot(
            "SPY",
            asof="2026-05-22T15:31:00Z",
            run_id="missing-fred",
        )


def test_provider_snapshot_bounds_fred_requests_to_lookback_window(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    fred_client = _FakeFredClient()

    _pipeline(tmp_path, fred_client=fred_client).snapshot(
        "SPY",
        asof="2026-05-22T15:31:00Z",
        run_id="fred-window",
        rate_lookback_days=30,
    )

    assert [call["series_id"] for call in fred_client.calls] == [
        "DGS3MO",
        "DGS1MO",
        "DGS6MO",
        "DGS1",
        "DGS2",
    ]
    assert all(
        call["observation_start"] == date(2026, 4, 22)
        and call["observation_end"] == date(2026, 5, 22)
        and call["sort_order"] == "asc"
        for call in fred_client.calls
    )


def test_provider_snapshot_missing_alpaca_quote_fails_clearly(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    alpaca_client = _FakeAlpacaClient(equity_payload={"quotes": {}})

    with pytest.raises(
        ProviderSnapshotDataUnavailableError,
        match="Alpaca latest equity quote is unavailable",
    ):
        _pipeline(tmp_path, alpaca_client=alpaca_client).snapshot(
            "SPY",
            asof="2026-05-22T15:31:00Z",
            run_id="missing-quote",
        )


def test_provider_snapshot_no_usable_option_chain_fails_clearly(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    alpaca_client = _FakeAlpacaClient(
        option_payload={
            "underlying": "SPY",
            "contracts": {
                "SPY260619C00500000": {
                    "latest_quote": {
                        "timestamp": "2026-05-22T15:30:00Z",
                        "ask_price": 4.4,
                    }
                }
            },
        }
    )

    with pytest.raises(
        ProviderSnapshotDataUnavailableError,
        match="No usable Alpaca option contracts",
    ):
        _pipeline(tmp_path, alpaca_client=alpaca_client).snapshot(
            "SPY",
            asof="2026-05-22T15:31:00Z",
            run_id="missing-chain",
        )


def test_provider_snapshot_fake_clients_do_not_need_provider_credentials(
    tmp_path: Path,
    fake_parquet: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("B4_TEST_ALPACA_KEY", raising=False)
    monkeypatch.delenv("B4_TEST_ALPACA_SECRET", raising=False)
    monkeypatch.delenv("B4_TEST_FRED_KEY", raising=False)

    result = _snapshot(tmp_path)

    assert result.accepted_quote_count == 2
    assert result.rate_source == "fred:DGS3MO"


def test_provider_snapshot_accepts_explicit_dependencies_without_config(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    pipeline = MarketDataPipeline(
        storage=tmp_path,
        alpaca_client=_FakeAlpacaClient(),
        fred_client=_FakeFredClient(),
    )

    result = pipeline.snapshot(
        "SPY",
        asof="2026-05-22T15:31:00Z",
        run_id="explicit-dependencies",
    )

    assert result.run_id == "explicit-dependencies"
    assert result.accepted_quote_count == 2


def test_local_model_validation_pipeline_api_imports_unchanged() -> None:
    assert MarketDataPipeline is not None
    assert callable(run_local_model_validation_pipeline)


def test_backfill_fred_writes_bronze_and_silver_with_fake_client(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    fred_client = _FakeFredClient()
    pipeline = _pipeline(tmp_path, fred_client=fred_client)

    result = pipeline.backfill_fred(
        "dgs3mo",
        start="2026-05-01",
        end="2026-05-22",
        run_id="fred-backfill",
    )

    bronze_root = _fred_backfill_root(
        tmp_path,
        layer="bronze",
        series_id="DGS3MO",
        start="2026-05-01",
        end="2026-05-22",
        run_id="fred-backfill",
    )
    silver_root = _fred_backfill_root(
        tmp_path,
        layer="silver",
        series_id="DGS3MO",
        start="2026-05-01",
        end="2026-05-22",
        run_id="fred-backfill",
    )
    assert (bronze_root / "observations.json").exists()
    assert (bronze_root / "manifest.json").exists()
    assert (silver_root / "fred_series.parquet").exists()
    assert (silver_root / "manifest.json").exists()
    assert (tmp_path / "_meta" / "runs.jsonl").exists()
    assert result.run_ids == ("fred-backfill",)
    assert result.stats.rows_in == 2
    assert result.stats.rows_out == 2
    assert set(result.artifact_paths) == {
        bronze_root / "observations.json",
        bronze_root / "manifest.json",
        silver_root / "fred_series.parquet",
        silver_root / "manifest.json",
    }
    assert fred_client.calls == [
        {
            "series_id": "DGS3MO",
            "observation_start": date(2026, 5, 1),
            "observation_end": date(2026, 5, 22),
            "sort_order": "asc",
        }
    ]

    manifest = _read_json(silver_root / "manifest.json")
    assert manifest["operation"] == "backfill_fred"
    assert manifest["provider"] == "fred"
    assert manifest["request_metadata"] == {
        "series_id": "DGS3MO",
        "observation_start": "2026-05-01",
        "observation_end": "2026-05-22",
        "sort_order": "asc",
    }
    frame = pd.read_parquet(silver_root / "fred_series.parquet")
    assert frame["series_id"].astype(str).tolist() == ["DGS3MO", "DGS3MO"]


def test_backfill_fred_handles_multiple_series(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    fred_client = _FakeFredClient(
        payloads={"DGS3MO": _fred_payload(), "FEDFUNDS": _fred_payload()}
    )

    result = _pipeline(tmp_path, fred_client=fred_client).backfill_fred(
        ["dgs3mo", "fedfunds"],
        start="2026-05-01",
        end="2026-05-22",
        run_id="fred-multi",
    )

    assert [call["series_id"] for call in fred_client.calls] == [
        "DGS3MO",
        "FEDFUNDS",
    ]
    assert result.stats.rows_out == 4
    assert len(result.artifact_paths) == 8
    assert (
        _fred_backfill_root(
            tmp_path,
            layer="silver",
            series_id="FEDFUNDS",
            start="2026-05-01",
            end="2026-05-22",
            run_id="fred-multi",
        )
        / "fred_series.parquet"
    ).exists()


def test_backfill_fred_defaults_end_date_safely(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    fred_client = _FakeFredClient()
    before = datetime.now(UTC).date()

    result = _pipeline(tmp_path, fred_client=fred_client).backfill_fred(
        "DGS3MO",
        start="2026-05-01",
        run_id="fred-default-end",
    )

    after = datetime.now(UTC).date()
    called_end = cast(date, fred_client.calls[0]["observation_end"])
    assert before <= called_end <= after
    assert any(
        f"end_date={called_end.isoformat()}" in path.as_posix()
        for path in result.artifact_paths
    )


def test_backfill_bars_writes_bronze_and_silver_with_fake_client(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    alpaca_client = _FakeAlpacaClient(bar_payload=_bars_payload(("SPY",)))

    result = _pipeline(tmp_path, alpaca_client=alpaca_client).backfill_bars(
        "spy",
        start="2026-05-20",
        end="2026-05-23",
        run_id="bars-backfill",
    )

    bronze_root = _bars_backfill_root(
        tmp_path,
        layer="bronze",
        symbol="SPY",
        timeframe="1Day",
        start="2026-05-20",
        end="2026-05-23",
        run_id="bars-backfill",
    )
    silver_root = _bars_backfill_root(
        tmp_path,
        layer="silver",
        symbol="SPY",
        timeframe="1Day",
        start="2026-05-20",
        end="2026-05-23",
        run_id="bars-backfill",
    )
    assert (bronze_root / "bars.json").exists()
    assert (bronze_root / "manifest.json").exists()
    assert (silver_root / "equity_bars.parquet").exists()
    assert (silver_root / "manifest.json").exists()
    assert result.stats.rows_in == 1
    assert result.stats.rows_out == 1
    assert alpaca_client.bar_calls[0]["symbols"] == ("SPY",)
    assert alpaca_client.bar_calls[0]["timeframe"] == "1Day"

    frame = pd.read_parquet(silver_root / "equity_bars.parquet")
    assert frame["symbol"].astype(str).tolist() == ["SPY"]
    assert float(frame.loc[0, "close"]) == pytest.approx(500.5)
    request_metadata = cast(
        dict[str, object], _read_json(silver_root / "manifest.json")["request_metadata"]
    )
    assert request_metadata["symbols"] == ["SPY"]
    assert request_metadata["start"] == "2026-05-20T00:00:00Z"
    assert request_metadata["end"] == "2026-05-23T00:00:00Z"
    assert request_metadata["timeframe"] == "1Day"
    assert request_metadata["feed"] == "indicative"
    assert isinstance(request_metadata["asof"], str)


def test_backfill_bars_handles_multiple_symbols(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    alpaca_client = _FakeAlpacaClient(bar_payload=_bars_payload(("SPY", "QQQ")))

    result = _pipeline(tmp_path, alpaca_client=alpaca_client).backfill_bars(
        ["spy", "qqq"],
        start="2026-05-20",
        end="2026-05-23",
        run_id="bars-multi",
    )

    assert alpaca_client.bar_calls[0]["symbols"] == ("SPY", "QQQ")
    assert result.stats.rows_in == 2
    assert result.stats.rows_out == 2
    assert len(result.artifact_paths) == 8
    qqq_frame = pd.read_parquet(
        _bars_backfill_root(
            tmp_path,
            layer="silver",
            symbol="QQQ",
            timeframe="1Day",
            start="2026-05-20",
            end="2026-05-23",
            run_id="bars-multi",
        )
        / "equity_bars.parquet"
    )
    assert qqq_frame["symbol"].astype(str).tolist() == ["QQQ"]

    spy_bronze = _read_json(
        _bars_backfill_root(
            tmp_path,
            layer="bronze",
            symbol="SPY",
            timeframe="1Day",
            start="2026-05-20",
            end="2026-05-23",
            run_id="bars-multi",
        )
        / "bars.json"
    )
    assert cast(dict[str, object], spy_bronze["payload"])["symbols"] == ["SPY"]


def test_backfill_fred_same_end_date_different_start_and_run_id_do_not_collide(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    pipeline = _pipeline(tmp_path, fred_client=_FakeFredClient())

    pipeline.backfill_fred(
        "DGS3MO",
        start="2026-05-01",
        end="2026-05-22",
        run_id="fred-window-a",
    )
    pipeline.backfill_fred(
        "DGS3MO",
        start="2026-05-10",
        end="2026-05-22",
        run_id="fred-window-b",
    )

    first_root = _fred_backfill_root(
        tmp_path,
        layer="silver",
        series_id="DGS3MO",
        start="2026-05-01",
        end="2026-05-22",
        run_id="fred-window-a",
    )
    second_root = _fred_backfill_root(
        tmp_path,
        layer="silver",
        series_id="DGS3MO",
        start="2026-05-10",
        end="2026-05-22",
        run_id="fred-window-b",
    )

    assert first_root != second_root
    assert (first_root / "fred_series.parquet").exists()
    assert (second_root / "fred_series.parquet").exists()


def test_backfill_bars_same_window_end_but_different_start_and_run_id_do_not_collide(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    pipeline = _pipeline(tmp_path, alpaca_client=_FakeAlpacaClient())

    pipeline.backfill_bars(
        "SPY",
        start="2026-05-20",
        end="2026-05-23",
        run_id="bars-window-a",
    )
    pipeline.backfill_bars(
        "SPY",
        start="2026-05-21",
        end="2026-05-23",
        run_id="bars-window-b",
    )

    first_root = _bars_backfill_root(
        tmp_path,
        layer="silver",
        symbol="SPY",
        timeframe="1Day",
        start="2026-05-20",
        end="2026-05-23",
        run_id="bars-window-a",
    )
    second_root = _bars_backfill_root(
        tmp_path,
        layer="silver",
        symbol="SPY",
        timeframe="1Day",
        start="2026-05-21",
        end="2026-05-23",
        run_id="bars-window-b",
    )

    assert first_root != second_root
    assert (first_root / "equity_bars.parquet").exists()
    assert (second_root / "equity_bars.parquet").exists()


def test_backfill_overwrite_false_protects_existing_artifacts(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    fred_client = _FakeFredClient()
    pipeline = _pipeline(tmp_path, fred_client=fred_client)
    pipeline.backfill_fred(
        "DGS3MO",
        start="2026-05-01",
        end="2026-05-22",
        run_id="fred-first",
    )

    with pytest.raises(FileExistsError, match="overwrite=True"):
        pipeline.backfill_fred(
            "DGS3MO",
            start="2026-05-01",
            end="2026-05-22",
            run_id="fred-first",
        )

    assert len(fred_client.calls) == 1


def test_provider_snapshot_overwrite_false_protects_before_provider_calls(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    _snapshot(tmp_path)
    alpaca_client = _FakeAlpacaClient()
    fred_client = _FakeFredClient()

    with pytest.raises(FileExistsError, match="overwrite=True"):
        _pipeline(
            tmp_path,
            alpaca_client=alpaca_client,
            fred_client=fred_client,
        ).snapshot(
            "SPY",
            asof="2026-05-22T15:31:00Z",
            run_id="b4-test-run",
        )

    assert alpaca_client.equity_calls == []
    assert alpaca_client.option_calls == []
    assert fred_client.calls == []


def test_provider_snapshot_overwrite_true_replaces_artifacts(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    _snapshot(tmp_path)
    replacement_equity = _equity_quote_payload()
    quote = cast(
        dict[str, object],
        cast(dict[str, object], replacement_equity["quotes"])["SPY"],
    )
    quote["bid_price"] = 509.0
    quote["ask_price"] = 511.0

    result = _pipeline(
        tmp_path,
        alpaca_client=_FakeAlpacaClient(equity_payload=replacement_equity),
    ).snapshot(
        "SPY",
        asof="2026-05-22T15:31:00Z",
        run_id="b4-test-run",
        overwrite=True,
        library_commit="replacement",
    )

    market_data = _read_json(result.gold_paths.market_data)
    manifest = _read_json(result.gold_paths.market_manifest)
    assert cast(dict[str, object], market_data["market_data"])["spot"] == 510.0
    assert manifest["library_commit"] == "replacement"


def test_backfill_overwrite_true_replaces_artifacts(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    pipeline = _pipeline(tmp_path, fred_client=_FakeFredClient())
    pipeline.backfill_fred(
        "DGS3MO",
        start="2026-05-01",
        end="2026-05-22",
        run_id="fred-replace",
        library_commit="first",
    )

    result = pipeline.backfill_fred(
        "DGS3MO",
        start="2026-05-01",
        end="2026-05-22",
        run_id="fred-replace",
        overwrite=True,
        library_commit="replacement",
    )

    manifest = _read_json(
        _fred_backfill_root(
            tmp_path,
            layer="silver",
            series_id="DGS3MO",
            start="2026-05-01",
            end="2026-05-22",
            run_id="fred-replace",
        )
        / "manifest.json"
    )
    assert manifest["library_commit"] == "replacement"
    assert manifest["warnings"] == list(result.stats.warnings)


def test_backfill_outputs_do_not_leak_secrets(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    secret = "super-secret-value"
    fred_client = _FakeFredClient(
        {
            **_fred_payload(),
            "api_key": secret,
            "nested": {"secret_token": secret},
        }
    )

    result = _pipeline(tmp_path, fred_client=fred_client).backfill_fred(
        "DGS3MO",
        start="2026-05-01",
        end="2026-05-22",
        run_id="secret-check",
    )

    manifest_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in result.artifact_paths
        if path.name == "manifest.json"
    )
    runs_text = (tmp_path / "_meta" / "runs.jsonl").read_text(encoding="utf-8")
    bronze_text = (
        _fred_backfill_root(
            tmp_path,
            layer="bronze",
            series_id="DGS3MO",
            start="2026-05-01",
            end="2026-05-22",
            run_id="secret-check",
        )
        / "observations.json"
    ).read_text(encoding="utf-8")
    cli_text = json.dumps(cli._result_payload("backfill-fred", result), sort_keys=True)

    assert secret not in manifest_text
    assert secret not in runs_text
    assert secret not in repr(result)
    assert secret not in bronze_text
    assert secret not in cli_text
    assert "<redacted>" in bronze_text
