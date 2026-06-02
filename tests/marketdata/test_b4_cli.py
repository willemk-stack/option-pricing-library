from __future__ import annotations

import ast
import json
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import option_pricing.marketdata.cli as cli

FAKE_SECRET = "fake-cli-secret-value"


class _FakePipeline:
    instances: list[_FakePipeline] = []

    def __init__(self, config: object) -> None:
        self.config = config
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
        rate_source="fred:DGS3MO",
        rate_observation_date=date(2026, 5, 20),
        rate_series_id="DGS3MO",
        feed="indicative",
        dividend_yield=0.0,
        dividend_yield_source="assumption",
        raw_option_contract_count=46,
        normalized_option_contract_count=45,
        dropped_before_cleaning_count=1,
        accepted_quote_count=42,
        rejected_quote_count=3,
        provider_rejected_contract_count=1,
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
            "--overwrite",
        ]
    )

    assert exit_code == 0
    instance = _FakePipeline.instances[-1]
    assert instance.config.storage.root == tmp_path
    assert instance.config.alpaca.feed == "sip"
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
        "dividend_yield": 0.0125,
        "dividend_yield_source": "manual_override",
        "overwrite": True,
    }
    stdout = capsys.readouterr().out
    assert "Market snapshot completed." in stdout
    assert "rate_series_id: DGS3MO" in stdout
    assert "rate_observation_date: 2026-05-20" in stdout
    assert "feed: indicative" in stdout
    assert "dividend_yield_source: assumption" in stdout
    assert "raw_option_contract_count: 46" in stdout
    assert "normalized_option_contract_count: 45" in stdout
    assert "dropped_before_cleaning_count: 1" in stdout
    assert "accepted_quote_count: 42" in stdout
    assert "provider_rejected_contract_count: 1" in stdout
    assert "sample warning" in stdout


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
        "dividend_yield": 0.0125,
        "dividend_yield_source": "manual_override",
        "overwrite": True,
    }
    stdout = capsys.readouterr().out
    assert "Daily refresh completed." in stdout
    assert "aggregate_run_id: daily-close-20260522T153100Z-abcdef12" in stdout
    assert "raw_option_contract_count: 92" in stdout
    assert "accepted_quote_count: 84" in stdout


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
        "dividend_yield": 0.0,
        "dividend_yield_source": "assumption",
        "dropped_before_cleaning_count": 1,
        "feed": "indicative",
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
        "provider_rejected_contract_count": 1,
        "rate_series_id": "DGS3MO",
        "rate": 0.0416,
        "rate_observation_date": "2026-05-20",
        "rate_source": "fred:DGS3MO",
        "raw_option_contract_count": 46,
        "rejected_quote_count": 3,
        "run_id": "snapshot-cli-run",
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
    name, args, kwargs = _last_call()
    assert name == "backfill_bars"
    assert args == (["spy", "qqq"],)
    assert kwargs == {
        "start": "2026-05-20",
        "end": "2026-05-23",
        "timeframe": "1Hour",
        "feed": "sip",
        "run_id": "bars-cli-run",
        "overwrite": True,
    }
    stdout = capsys.readouterr().out
    assert "dataset: equity_bars" in stdout
    assert "small warning" in stdout


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
        "option_pricing.marketdata.config",
        "option_pricing.marketdata.errors",
        "option_pricing.marketdata.pipeline",
    }
    assert "normalize_" not in source
    assert "clean_option_quotes" not in source
