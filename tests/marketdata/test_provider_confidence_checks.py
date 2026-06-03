from __future__ import annotations

import importlib.util
import math
import os
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from option_pricing.marketdata.bundles import ModelValidationBundleConfig
from option_pricing.marketdata.config import (
    AlpacaConfig,
    FredConfig,
    PipelineConfig,
    StorageConfig,
)
from option_pricing.marketdata.pipeline import MarketDataPipeline
from option_pricing.marketdata.schemas import DatasetName
from option_pricing.marketdata.validation import validate_dtypes

_ASOF = "2026-05-22T15:30:00Z"
_TEXT_ARTIFACT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".txt", ".yml", ".yaml"}
_LIVE_CREDENTIAL_ENV_VARS = (
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "FRED_API_KEY",
)


class _FakeAlpacaClient:
    def get_latest_equity_quotes(
        self,
        symbols: str | Sequence[str],
        *,
        asof: object | None = None,
    ) -> Mapping[str, Any]:
        symbol = _single_symbol(symbols)
        quote_ts = str(asof or _ASOF)
        return {
            "source": "fake_alpaca",
            "quotes": {
                symbol: {
                    "symbol": symbol,
                    "timestamp": quote_ts,
                    "bid_price": 499.50,
                    "ask_price": 500.50,
                    "bid_size": 100,
                    "ask_size": 120,
                }
            },
        }

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
        updated_since: object | None = None,
        feed: str | None = None,
        asof: object | None = None,
    ) -> Mapping[str, Any]:
        del expiry_gte, expiry_lte, strike_gte, strike_lte, option_type
        del root_symbol, updated_since
        contract_symbol = f"{underlying.upper()}260619C00500000"
        quote_ts = str(asof or _ASOF)
        return {
            "source": "fake_alpaca",
            "feed": feed or "indicative",
            "underlying": underlying,
            "asof": quote_ts,
            "contracts": {
                contract_symbol: {
                    "contract_symbol": contract_symbol,
                    "expiration_date": "2026-06-19",
                    "strike_price": 500.0,
                    "type": "call",
                    "latest_quote": {
                        "timestamp": quote_ts,
                        "bid_price": 10.00,
                        "ask_price": 10.50,
                    },
                    "latest_trade": {"price": 10.25},
                    "implied_volatility": 0.20,
                    "greeks": {
                        "delta": 0.51,
                        "gamma": 0.02,
                        "theta": -0.03,
                        "vega": 0.11,
                        "rho": 0.04,
                    },
                    "open_interest": 250,
                }
            },
        }

    def get_equity_bars(
        self,
        symbols: str | Sequence[str],
        *,
        start: object,
        end: object,
        timeframe: str,
        limit: int | None = None,
        adjustment: str | None = None,
        sort: str | None = "asc",
        feed: str | None = None,
        asof: str | None = None,
    ) -> Mapping[str, Any]:
        del start, end, limit, adjustment, sort, feed
        symbol = _single_symbol(symbols)
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": {
                symbol: [
                    {
                        "timestamp": "2026-05-22T14:30:00Z",
                        "open": 499.0,
                        "high": 501.0,
                        "low": 498.5,
                        "close": 500.0,
                        "volume": 1_000_000,
                        "trade_count": 10_000,
                        "vwap": 500.1,
                    }
                ]
            },
            "asof": asof or _ASOF,
        }


class _FakeFredClient:
    def fetch_observations(
        self,
        series_id: str,
        *,
        observation_start: date | str | None = None,
        observation_end: date | str | None = None,
        realtime_start: date | str | None = None,
        realtime_end: date | str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        sort_order: str | None = None,
        units: str | None = None,
        frequency: str | None = None,
        aggregation_method: str | None = None,
    ) -> Mapping[str, Any]:
        del observation_start, observation_end, realtime_start, realtime_end
        del limit, offset, sort_order, units, frequency, aggregation_method
        return {
            "observations": [
                {
                    "date": "2026-05-22",
                    "value": "5.25",
                    "realtime_start": "2026-05-22",
                    "realtime_end": "2026-05-22",
                }
            ],
            "series_id": series_id,
        }


def test_provider_snapshot_writes_real_parquet_with_fake_providers(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    pipeline = MarketDataPipeline(
        storage=tmp_path,
        alpaca_client=_FakeAlpacaClient(),
        fred_client=_FakeFredClient(),
        bundle_config=ModelValidationBundleConfig(run_heston_smoke=False),
    )

    result = pipeline.snapshot(
        "SPY",
        asof=_ASOF,
        run_id="provider-parquet-smoke",
        curve_series_ids=(),
    )

    assert result.accepted_quote_count > 0
    assert result.silver_paths.market_inputs.exists()
    assert result.silver_paths.option_chain.exists()
    assert result.silver_paths.fred_series.exists()
    assert result.silver_paths.cleaned_quotes.exists()
    assert result.gold_paths.heston_quotes.exists()
    assert result.model_validation_bundle.cleaned_quotes.exists()

    market_inputs = pd.read_parquet(result.silver_paths.market_inputs)
    option_chain = pd.read_parquet(result.silver_paths.option_chain)
    fred_series = pd.read_parquet(result.silver_paths.fred_series)
    cleaned_quotes = pd.read_parquet(result.silver_paths.cleaned_quotes)
    heston_quotes = pd.read_parquet(result.gold_paths.heston_quotes)

    validate_dtypes(market_inputs, DatasetName.MARKET_INPUTS, allow_extra=False)
    validate_dtypes(option_chain, DatasetName.OPTION_CHAIN, allow_extra=False)
    validate_dtypes(fred_series, DatasetName.FRED_SERIES, allow_extra=False)
    validate_dtypes(cleaned_quotes, DatasetName.CLEANED_QUOTES, allow_extra=False)
    validate_dtypes(heston_quotes, DatasetName.HESTON_QUOTES, allow_extra=False)

    assert len(cleaned_quotes) == result.accepted_quote_count
    assert set(cleaned_quotes["underlying"].astype(str)) == {"SPY"}
    assert cleaned_quotes["mid"].astype(float).gt(0).all()


def test_live_provider_snapshot_smoke_optional(tmp_path: Path) -> None:
    secret_values = _skip_unless_live_provider_smoke_available()
    today = date.today()
    pipeline = MarketDataPipeline(
        PipelineConfig(
            alpaca=AlpacaConfig(),
            fred=FredConfig(),
            storage=StorageConfig(root=tmp_path),
        ),
        bundle_config=ModelValidationBundleConfig(run_heston_smoke=False),
    )

    result = pipeline.snapshot(
        "SPY",
        run_id="live-spy-provider-smoke",
        expiry_gte=today.isoformat(),
        expiry_lte=(today + timedelta(days=45)).isoformat(),
        option_type="call",
        curve_series_ids=(),
    )

    assert result.spot > 0.0
    assert math.isfinite(result.rate)
    assert result.accepted_quote_count > 0
    assert result.bronze_paths.manifest.exists()
    assert result.silver_paths.cleaned_quotes.exists()
    assert result.gold_paths.market_data.exists()
    assert result.gold_paths.heston_quotes.exists()
    assert result.model_validation_bundle.manifest_path.exists()
    _assert_no_secrets_in_text_artifacts(tmp_path, secret_values)


def _skip_unless_live_provider_smoke_available() -> tuple[str, ...]:
    missing_env = [name for name in _LIVE_CREDENTIAL_ENV_VARS if not os.getenv(name)]
    if missing_env:
        pytest.skip(
            "live provider smoke requires credentials: " + ", ".join(missing_env)
        )
    if importlib.util.find_spec("alpaca") is None:
        pytest.skip("live provider smoke requires alpaca-py")
    if importlib.util.find_spec("pyarrow") is None:
        pytest.skip("live provider smoke requires pyarrow")
    return tuple(str(os.environ[name]) for name in _LIVE_CREDENTIAL_ENV_VARS)


def _assert_no_secrets_in_text_artifacts(root: Path, secrets: Sequence[str]) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _TEXT_ARTIFACT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for secret in secrets:
            if secret and secret in text:
                pytest.fail(f"secret value leaked into text artifact {path}")


def _single_symbol(symbols: str | Sequence[str]) -> str:
    if isinstance(symbols, str):
        return symbols.upper()
    return str(next(iter(symbols))).upper()
