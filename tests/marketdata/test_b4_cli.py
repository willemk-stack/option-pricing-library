from __future__ import annotations

import ast
import json
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import option_pricing.marketdata.cli as cli
from option_pricing.marketdata.provider_confidence import (
    ProviderSnapshotBundleValidationResult,
)

FAKE_SECRET = "fake-cli-secret-value"


class _FakePipeline:
    instances: list[_FakePipeline] = []

    def __init__(self, config: object, **kwargs: object) -> None:
        self.config = config
        self.init_kwargs = kwargs
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.instances.append(self)

    def snapshot(self, *args: object, **kwargs: object) -> object:
        self.calls.append(("snapshot", args, kwargs))
        return _snapshot_result()

    def refresh_daily(self, *args: object, **kwargs: object) -> object:
        self.calls.append(("refresh_daily", args, kwargs))
        return _refresh_daily_result()

    def backfill_fred(self, *args: object, **kwargs: object) -> object:
        self.calls.append(("backfill_fred", args, kwargs))
        return _backfill_result("fred-cli-run")

    def backfill_bars(self, *args: object, **kwargs: object) -> object:
        self.calls.append(("backfill_bars", args, kwargs))
        return _backfill_result("bars-cli-run")


@pytest.fixture(autouse=True)
def fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakePipeline.instances.clear()
    monkeypatch.setattr(cli, "MarketDataPipeline", _FakePipeline)


def _snapshot_result(
    *,
    underlying: str = "SPY",
    run_id: str = "snapshot-cli-run",
) -> object:
    return SimpleNamespace(
        underlying=underlying,
        asof=datetime(2026, 5, 22, 15, 31, tzinfo=UTC),
        run_id=run_id,
        spot=500.0,
        rate=0.0416,
        rate_source="fred:treasury_zero_proxy_curve",
        rate_observation_date=date(2026, 5, 20),
        rate_series_id="DGS3MO",
        feed="indicative",
        equity_provider="alpaca",
        equity_feed="iex",
        option_provider="alpaca",
        option_feed="indicative",
        selected_rate=0.0416,
        flat_rate=0.0416,
        rate_policy=_rate_policy(),
        dividend_policy=_dividend_policy(),
        option_cleaning_policy=_option_cleaning_policy(),
        data_policy=_data_policy(),
        dividend_yield=0.0,
        dividend_yield_source="zero_assumption",
        raw_option_contract_count=46,
        normalized_option_contract_count=45,
        dropped_before_cleaning_count=1,
        accepted_quote_count=42,
        rejected_quote_count=3,
        provider_rejected_contract_count=1,
        quality_policy={
            "quote_freshness_mode": "demo_lenient",
            "max_quote_age_seconds": None,
            "allow_prior_session": False,
            "stale_quote_action": "warn",
            "min_accepted_contracts": 1,
            "warn_on_stale_quotes": True,
        },
        quote_freshness={
            "quote_freshness_mode": "demo_lenient",
            "max_quote_age_seconds": None,
            "equity_quote_age_seconds": 60.0,
            "stale_option_quote_count": 0,
            "stale_quote_count": 0,
            "quote_age_summary": {"min": None, "median": None, "max": None},
            "quote_freshness_warnings": [],
        },
        diagnostics=(
            {
                "provider": "alpaca",
                "operation": "latest_equity_quote",
                "status": "ok",
                "request_metadata": {"symbols": ["SPY"]},
                "started_at": "2026-05-22T15:31:00Z",
                "ended_at": "2026-05-22T15:31:00Z",
                "elapsed_ms": 1.0,
                "retry_count": 0,
                "rows_or_contracts_in": 1,
            },
        ),
        warnings=("sample warning",),
        artifact_paths=(
            Path(
                "data/silver/provider_rejected_contracts/provider_rejected_contracts.parquet"
            ),
            Path("data/gold/curves/rate_curve.parquet"),
            Path("data/gold/curves/manifest.json"),
            Path("data/gold/market_snapshot/market_data.json"),
            Path("data/gold/model_validation_bundle/manifest.json"),
        ),
        bronze_paths=SimpleNamespace(
            manifest=Path("data/bronze/provider_snapshot/manifest.json")
        ),
        silver_paths=SimpleNamespace(
            manifest=Path("data/silver/cleaned_quotes/manifest.json"),
            provider_rejected_contracts=Path(
                "data/silver/provider_rejected_contracts/provider_rejected_contracts.parquet"
            ),
        ),
        gold_paths=SimpleNamespace(
            market_data=Path("data/gold/market_snapshot/market_data.json"),
            market_manifest=Path("data/gold/market_snapshot/manifest.json"),
        ),
        rate_curve_paths=SimpleNamespace(
            rate_curve=Path("data/gold/curves/rate_curve.parquet"),
            manifest=Path("data/gold/curves/manifest.json"),
        ),
        model_validation_bundle=SimpleNamespace(
            manifest_path=Path("data/gold/model_validation_bundle/manifest.json")
        ),
    )


def _rate_policy() -> dict[str, object]:
    return {
        "policy": "fred_treasury_zero_proxy_linear_cc",
        "provider": "fred",
        "rate_curve_source": "fred",
        "series_id": "DGS3MO",
        "rate_source": "fred:treasury_zero_proxy_curve",
        "rate_observation_date": "2026-05-20",
        "selected_rate": 0.0416,
        "flat_rate": 0.0416,
        "lookback_days": 90,
        "curve_series_ids": ["DGS1MO", "DGS3MO"],
        "rate_curve_series_ids": ["DGS1MO", "DGS3MO"],
        "rate_interpolation": "linear",
        "curve_interpolation": "linear",
        "rate_compounding": "continuous",
        "rate_extrapolation": "clamp_with_warning",
        "rate_is_bootstrapped": False,
        "selected_rate_fallback_used": False,
        "rate_warnings": [],
    }


def _dividend_policy() -> dict[str, object]:
    return {
        "policy": "zero_assumption",
        "dividend_yield": 0.0,
        "source": "zero_assumption",
        "dividend_source": "zero_assumption",
        "dividend_is_explicit": False,
        "dividend_fallback_used": True,
        "dividend_inference": "not_enabled",
    }


def _option_cleaning_policy() -> dict[str, object]:
    return {
        "policy": "staged_recoverable_quotes_v1",
        "legacy_policy": "quote_cleaning_v1",
        "policy_id": "quote_cleaning_policy.v1",
        "rejected_quotes_preserved": True,
        "raw_option_quotes_layer": "raw_option_quotes",
        "clean_option_quotes_layer": "clean_option_quotes",
        "model_validation_quotes_layer": "model_validation_quotes",
        "reason_codes": ["missing_price_source", "crossed_bid_ask"],
        "recoverable_missing_fields": ["iv", "vega"],
    }


def _data_policy() -> dict[str, object]:
    return {
        "schema_version": "provider_snapshot_data_policy.v1",
        "equity_provider": "alpaca",
        "equity_feed": "iex",
        "option_provider": "alpaca",
        "option_feed": "indicative",
        "rate_policy": _rate_policy(),
        "dividend_policy": _dividend_policy(),
        "option_cleaning_policy": _option_cleaning_policy(),
        "quote_freshness_mode": "demo_lenient",
        "model_validation_policy": "model_ready_quotes_v1",
    }


def _refresh_daily_result() -> object:
    return SimpleNamespace(
        aggregate_run_id="daily-close-20260522T153100Z-abcdef12",
        child_run_ids=(
            "daily-close-20260522T153100Z-abcdef12-01-spy",
            "daily-close-20260522T153100Z-abcdef12-02-qqq",
        ),
        underlyings=("SPY", "QQQ"),
        artifact_paths=(
            Path("data/gold/market_snapshot/spy-market_data.json"),
            Path("data/gold/market_snapshot/qqq-market_data.json"),
        ),
        counts=SimpleNamespace(
            raw_option_contract_count=92,
            normalized_option_contract_count=90,
            dropped_before_cleaning_count=2,
            provider_rejected_contract_count=2,
            accepted_quote_count=84,
            rejected_quote_count=6,
        ),
        warnings=("sample warning", "daily warning"),
        results=(
            _snapshot_result(
                underlying="SPY",
                run_id="daily-close-20260522T153100Z-abcdef12-01-spy",
            ),
            _snapshot_result(
                underlying="QQQ",
                run_id="daily-close-20260522T153100Z-abcdef12-02-qqq",
            ),
        ),
    )


def _p(path: str) -> str:
    return str(Path(path))


def _backfill_result(run_id: str) -> object:
    return SimpleNamespace(
        metadata=SimpleNamespace(run_id=run_id),
        run_ids=(run_id,),
        artifact_paths=(
            Path("data/bronze/manifest.json"),
            Path("data/silver/manifest.json"),
        ),
        stats=SimpleNamespace(rows_in=7, rows_out=5, warnings=("small warning",)),
    )


def _last_call() -> tuple[str, tuple[object, ...], dict[str, object]]:
    return _FakePipeline.instances[-1].calls[-1]


def test_snapshot_cli_parses_arguments_and_calls_pipeline_correctly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(
        [
            "snapshot",
            "--underlying",
            "spy",
            "--asof",
            "2026-05-22T15:31:00Z",
            "--run-id",
            "snapshot-cli-run",
            "--data-root",
            str(tmp_path),
            "--rate-series",
            "fedfunds",
            "--feed",
            "sip",
            "--expiry-gte",
            "2026-06-01",
            "--expiry-lte",
            "2026-06-30",
            "--strike-gte",
            "400",
            "--strike-lte",
            "600",
            "--option-type",
            "call",
            "--dividend-yield",
            "0.0125",
            "--dividend-yield-source",
            "manual_override",
            "--rate-lookback-days",
            "45",
            "--curve-series",
            "DGS1MO",
            "DGS1",
            "--library-commit",
            "abc123",
            "--max-equity-quote-age-seconds",
            "120",
            "--max-option-quote-age-seconds",
            "300",
            "--quote-freshness-mode",
            "intraday_strict",
            "--max-quote-age-seconds",
            "600",
            "--allow-prior-session",
            "--stale-quote-action",
            "reject",
            "--reject-stale-option-quotes",
            "--reject-option-quotes-after-asof",
            "--min-accepted-contracts",
            "10",
            "--min-accepted-calls",
            "4",
            "--min-accepted-puts",
            "4",
            "--min-expiries",
            "2",
            "--run-heston-smoke",
            "--overwrite",
        ]
    )

    assert exit_code == 0
    instance = _FakePipeline.instances[-1]
    assert instance.config.storage.root == tmp_path
    assert instance.config.alpaca.feed == "sip"
    assert instance.config.alpaca.equity_feed == "iex"
    assert instance.config.alpaca.option_feed == "sip"
    assert instance.init_kwargs["bundle_config"].run_heston_smoke is True
    name, args, kwargs = _last_call()
    assert name == "snapshot"
    assert args == ("spy",)
    assert kwargs == {
        "asof": "2026-05-22T15:31:00Z",
        "run_id": "snapshot-cli-run",
        "rate_series_id": "fedfunds",
        "expiry_gte": "2026-06-01",
        "expiry_lte": "2026-06-30",
        "strike_gte": 400.0,
        "strike_lte": 600.0,
        "option_type": "call",
        "feed": "sip",
        "equity_feed": None,
        "option_feed": None,
        "dividend_yield": 0.0125,
        "dividend_yield_source": "manual_override",
        "rate_lookback_days": 45,
        "curve_series_ids": ["DGS1MO", "DGS1"],
        "quality_policy": {
            "quote_freshness_mode": "intraday_strict",
            "max_quote_age_seconds": 600.0,
            "allow_prior_session": True,
            "stale_quote_action": "reject",
            "max_equity_quote_age_seconds": 120.0,
            "max_option_quote_age_seconds": 300.0,
            "reject_stale_option_quotes": True,
            "reject_option_quotes_after_asof": True,
            "min_accepted_contracts": 10,
            "min_accepted_calls": 4,
            "min_accepted_puts": 4,
            "min_expiries": 2,
        },
        "overwrite": True,
        "library_commit": "abc123",
    }
    stdout = capsys.readouterr().out
    assert "Market snapshot completed." in stdout
    assert "rate_series_id: DGS3MO" in stdout
    assert "rate_observation_date: 2026-05-20" in stdout
    assert "equity_feed: iex" in stdout
    assert "option_feed: indicative" in stdout
    assert "dividend_yield_source: zero_assumption" in stdout
    assert "raw_option_contract_count: 46" in stdout
    assert "normalized_option_contract_count: 45" in stdout
    assert "dropped_before_cleaning_count: 1" in stdout
    assert "accepted_quote_count: 42" in stdout
    assert "provider_rejected_contract_count: 1" in stdout
    assert "sample warning" in stdout


def test_snapshot_cli_parses_split_feed_arguments(tmp_path: Path) -> None:
    exit_code = cli.main(
        [
            "snapshot",
            "--underlying",
            "spy",
            "--data-root",
            str(tmp_path),
            "--equity-feed",
            "sip",
            "--option-feed",
            "opra",
        ]
    )

    assert exit_code == 0
    instance = _FakePipeline.instances[-1]
    assert instance.config.alpaca.equity_feed == "sip"
    assert instance.config.alpaca.option_feed == "opra"
    name, args, kwargs = _last_call()
    assert name == "snapshot"
    assert args == ("spy",)
    assert kwargs["feed"] is None
    assert kwargs["equity_feed"] == "sip"
    assert kwargs["option_feed"] == "opra"


def test_snapshot_cli_defaults_to_split_feeds_for_first_real_run(
    tmp_path: Path,
) -> None:
    exit_code = cli.main(
        [
            "snapshot",
            "--underlying",
            "spy",
            "--data-root",
            str(tmp_path),
            "--overwrite",
            "--json",
        ]
    )

    assert exit_code == 0
    instance = _FakePipeline.instances[-1]
    assert instance.config.alpaca.equity_feed == "iex"
    assert instance.config.alpaca.option_feed == "indicative"
    name, args, kwargs = _last_call()
    assert name == "snapshot"
    assert args == ("spy",)
    assert kwargs["feed"] is None
    assert kwargs["equity_feed"] is None
    assert kwargs["option_feed"] is None
    assert kwargs["overwrite"] is True


def test_refresh_daily_cli_parses_arguments_and_calls_pipeline_correctly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(
        [
            "refresh-daily",
            "--underlyings",
            "spy",
            "qqq",
            "--asof",
            "2026-05-22T15:31:00Z",
            "--run-id-prefix",
            "daily-close",
            "--data-root",
            str(tmp_path),
            "--rate-series",
            "DGS2",
            "--feed",
            "sip",
            "--expiry-gte",
            "2026-06-01",
            "--expiry-lte",
            "2026-06-30",
            "--strike-gte",
            "400",
            "--strike-lte",
            "600",
            "--option-type",
            "call",
            "--dividend-yield",
            "0.0125",
            "--dividend-yield-source",
            "manual_override",
            "--overwrite",
        ]
    )

    assert exit_code == 0
    instance = _FakePipeline.instances[-1]
    assert instance.config.storage.root == tmp_path
    assert instance.config.alpaca.feed == "sip"
    assert instance.config.alpaca.equity_feed == "iex"
    assert instance.config.alpaca.option_feed == "sip"
    assert instance.init_kwargs["bundle_config"].run_heston_smoke is False
    name, args, kwargs = _last_call()
    assert name == "refresh_daily"
    assert args == (["spy", "qqq"],)
    assert kwargs == {
        "asof": "2026-05-22T15:31:00Z",
        "run_id_prefix": "daily-close",
        "rate_series_id": "DGS2",
        "expiry_gte": "2026-06-01",
        "expiry_lte": "2026-06-30",
        "strike_gte": 400.0,
        "strike_lte": 600.0,
        "option_type": "call",
        "feed": "sip",
        "equity_feed": None,
        "option_feed": None,
        "dividend_yield": 0.0125,
        "dividend_yield_source": "manual_override",
        "rate_lookback_days": 90,
        "curve_series_ids": None,
        "quality_policy": {
            "quote_freshness_mode": "demo_lenient",
            "allow_prior_session": False,
            "stale_quote_action": "warn",
        },
        "overwrite": True,
        "library_commit": None,
    }
    stdout = capsys.readouterr().out
    assert "Daily refresh completed." in stdout
    assert "aggregate_run_id: daily-close-20260522T153100Z-abcdef12" in stdout
    assert "raw_option_contract_count: 92" in stdout
    assert "accepted_quote_count: 84" in stdout


def test_refresh_daily_cli_passes_quality_minimums(
    tmp_path: Path,
) -> None:
    exit_code = cli.main(
        [
            "refresh-daily",
            "--underlyings",
            "spy",
            "--data-root",
            str(tmp_path),
            "--min-accepted-contracts",
            "8",
            "--min-accepted-calls",
            "3",
            "--min-accepted-puts",
            "3",
            "--min-expiries",
            "2",
            "--reject-option-quotes-after-asof",
        ]
    )

    assert exit_code == 0
    name, args, kwargs = _last_call()
    assert name == "refresh_daily"
    assert args == (["spy"],)
    assert kwargs["quality_policy"] == {
        "quote_freshness_mode": "demo_lenient",
        "allow_prior_session": False,
        "stale_quote_action": "warn",
        "reject_option_quotes_after_asof": True,
        "min_accepted_contracts": 8,
        "min_accepted_calls": 3,
        "min_accepted_puts": 3,
        "min_expiries": 2,
    }


def test_snapshot_json_emits_valid_stable_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(["snapshot", "--underlying", "SPY", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "accepted_quote_count": 42,
        "artifact_paths": [
            _p(
                "data/silver/provider_rejected_contracts/provider_rejected_contracts.parquet"
            ),
            _p("data/gold/curves/rate_curve.parquet"),
            _p("data/gold/curves/manifest.json"),
            _p("data/gold/market_snapshot/market_data.json"),
            _p("data/gold/model_validation_bundle/manifest.json"),
        ],
        "asof": "2026-05-22T15:31:00+00:00",
        "command": "snapshot",
        "data_policy": _data_policy(),
        "dividend_policy": _dividend_policy(),
        "dividend_yield": 0.0,
        "dividend_yield_source": "zero_assumption",
        "dropped_before_cleaning_count": 1,
        "equity_feed": "iex",
        "equity_provider": "alpaca",
        "flat_rate": 0.0416,
        "main_artifact_paths": {
            "bronze_manifest": _p("data/bronze/provider_snapshot/manifest.json"),
            "bundle_manifest": _p("data/gold/model_validation_bundle/manifest.json"),
            "market_data": _p("data/gold/market_snapshot/market_data.json"),
            "market_manifest": _p("data/gold/market_snapshot/manifest.json"),
            "provider_rejected_contracts": _p(
                "data/silver/provider_rejected_contracts/provider_rejected_contracts.parquet"
            ),
            "rate_curve": _p("data/gold/curves/rate_curve.parquet"),
            "rate_curve_manifest": _p("data/gold/curves/manifest.json"),
            "silver_manifest": _p("data/silver/cleaned_quotes/manifest.json"),
        },
        "normalized_option_contract_count": 45,
        "option_cleaning_policy": _option_cleaning_policy(),
        "option_feed": "indicative",
        "option_provider": "alpaca",
        "provider_rejected_contract_count": 1,
        "provider_operation_diagnostics": [
            {
                "elapsed_ms": 1.0,
                "ended_at": "2026-05-22T15:31:00Z",
                "operation": "latest_equity_quote",
                "provider": "alpaca",
                "request_metadata": {"symbols": ["SPY"]},
                "retry_count": 0,
                "rows_or_contracts_in": 1,
                "started_at": "2026-05-22T15:31:00Z",
                "status": "ok",
            }
        ],
        "quality_policy": {
            "quote_freshness_mode": "demo_lenient",
            "max_quote_age_seconds": None,
            "allow_prior_session": False,
            "stale_quote_action": "warn",
            "min_accepted_contracts": 1,
            "warn_on_stale_quotes": True,
        },
        "quote_freshness": {
            "quote_freshness_mode": "demo_lenient",
            "max_quote_age_seconds": None,
            "equity_quote_age_seconds": 60.0,
            "stale_option_quote_count": 0,
            "stale_quote_count": 0,
            "quote_age_summary": {"min": None, "median": None, "max": None},
            "quote_freshness_warnings": [],
        },
        "rate_policy": _rate_policy(),
        "rate_series_id": "DGS3MO",
        "rate": 0.0416,
        "rate_observation_date": "2026-05-20",
        "rate_source": "fred:treasury_zero_proxy_curve",
        "raw_option_contract_count": 46,
        "rejected_quote_count": 3,
        "run_id": "snapshot-cli-run",
        "selected_rate": 0.0416,
        "spot": 500.0,
        "underlying": "SPY",
        "warnings": ["sample warning"],
    }


def test_backfill_fred_parses_series_start_end_and_calls_pipeline_correctly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(
        [
            "backfill-fred",
            "--series",
            "dgs3mo",
            "fedfunds",
            "--start",
            "2026-05-01",
            "--end",
            "2026-05-22",
            "--run-id",
            "fred-cli-run",
        ]
    )

    assert exit_code == 0
    name, args, kwargs = _last_call()
    assert name == "backfill_fred"
    assert args == (["dgs3mo", "fedfunds"],)
    assert kwargs == {
        "start": "2026-05-01",
        "end": "2026-05-22",
        "run_id": "fred-cli-run",
        "overwrite": False,
        "library_commit": None,
    }
    stdout = capsys.readouterr().out
    assert "dataset: fred_series" in stdout
    assert "rows_in: 7" in stdout


def test_backfill_bars_parses_symbols_dates_timeframe_and_calls_pipeline_correctly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(
        [
            "backfill-bars",
            "--symbols",
            "spy",
            "qqq",
            "--start",
            "2026-05-20",
            "--end",
            "2026-05-23",
            "--timeframe",
            "1Hour",
            "--feed",
            "sip",
            "--run-id",
            "bars-cli-run",
            "--overwrite",
        ]
    )

    assert exit_code == 0
    instance = _FakePipeline.instances[-1]
    assert instance.config.alpaca.equity_feed == "sip"
    assert instance.config.alpaca.option_feed == "indicative"
    name, args, kwargs = _last_call()
    assert name == "backfill_bars"
    assert args == (["spy", "qqq"],)
    assert kwargs == {
        "start": "2026-05-20",
        "end": "2026-05-23",
        "timeframe": "1Hour",
        "feed": "sip",
        "equity_feed": None,
        "run_id": "bars-cli-run",
        "overwrite": True,
        "library_commit": None,
    }
    stdout = capsys.readouterr().out
    assert "dataset: equity_bars" in stdout
    assert "small warning" in stdout


def test_validate_bundle_cli_calls_existing_helper(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, Path]] = []

    def fake_validate_provider_snapshot_bundle(
        *,
        market_data_path: Path,
        cleaned_quotes_path: Path,
        heston_quotes_path: Path,
    ) -> ProviderSnapshotBundleValidationResult:
        calls.append(
            {
                "market_data_path": market_data_path,
                "cleaned_quotes_path": cleaned_quotes_path,
                "heston_quotes_path": heston_quotes_path,
            }
        )
        return ProviderSnapshotBundleValidationResult(
            underlying="SPY",
            cleaned_quote_count=42,
            heston_quote_count=42,
            spot=500.0,
            rate=0.0416,
            dividend_yield=0.0,
        )

    monkeypatch.setattr(
        cli,
        "validate_provider_snapshot_bundle",
        fake_validate_provider_snapshot_bundle,
    )

    exit_code = cli.main(
        [
            "validate-bundle",
            "--market-data",
            "market_data.json",
            "--cleaned-quotes",
            "cleaned_quotes.parquet",
            "--heston-quotes",
            "heston_quotes.parquet",
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "market_data_path": Path("market_data.json"),
            "cleaned_quotes_path": Path("cleaned_quotes.parquet"),
            "heston_quotes_path": Path("heston_quotes.parquet"),
        }
    ]
    assert _FakePipeline.instances == []
    stdout = capsys.readouterr().out
    assert "Provider snapshot bundle validation passed." in stdout
    assert "cleaned_quote_count: 42" in stdout


def test_validate_bundle_cli_emits_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "validate_provider_snapshot_bundle",
        lambda **_: ProviderSnapshotBundleValidationResult(
            underlying="SPY",
            cleaned_quote_count=42,
            heston_quote_count=42,
            spot=500.0,
            rate=0.0416,
            dividend_yield=0.0,
        ),
    )

    exit_code = cli.main(
        [
            "validate-bundle",
            "--market-data",
            "market_data.json",
            "--cleaned-quotes",
            "cleaned_quotes.parquet",
            "--heston-quotes",
            "heston_quotes.parquet",
            "--json",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "cleaned_quote_count": 42,
        "command": "validate-bundle",
        "dividend_yield": 0.0,
        "heston_quote_count": 42,
        "rate": 0.0416,
        "spot": 500.0,
        "underlying": "SPY",
    }


def test_missing_required_args_fail_through_argparse(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["snapshot"])

    assert excinfo.value.code == 2
    assert "--underlying" in capsys.readouterr().err


def test_cli_output_does_not_contain_fake_secret_values(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", FAKE_SECRET)
    monkeypatch.setenv("ALPACA_SECRET_KEY", FAKE_SECRET)
    monkeypatch.setenv("FRED_API_KEY", FAKE_SECRET)

    exit_code = cli.main(["snapshot", "--underlying", "SPY"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert FAKE_SECRET not in captured.out
    assert FAKE_SECRET not in captured.err


def test_cli_stays_inside_parse_and_dispatch_boundary() -> None:
    source = Path(cli.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_marketdata_modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module.startswith("option_pricing.marketdata"):
                imported_marketdata_modules.add(node.module)

    assert imported_marketdata_modules == {
        "option_pricing.marketdata.bundles",
        "option_pricing.marketdata.config",
        "option_pricing.marketdata.errors",
        "option_pricing.marketdata.pipeline",
        "option_pricing.marketdata.provider_confidence",
    }
    assert "normalize_" not in source
    assert "clean_option_quotes" not in source
