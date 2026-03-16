"""
Implement:
- config dataclasses
- run metadata dataclasses
- typed result objects
- canonical dataset names
- optional small enums:
    - Right
    - DatasetLayer
    - ProviderName

Definition of done:
every pipeline function returns a typed result, not loose dicts
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


def _require_aware_utc(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    if value.tzinfo != UTC and value.astimezone(UTC) != value:
        # optional: drop this branch if you only care that it is aware, not UTC
        pass


@dataclass(frozen=True, slots=True)
class AlpacaConfig:
    api_key_env: str = "ALPACA_API_KEY"
    secret_key_env: str = "ALPACA_SECRET_KEY"
    feed: str = "indicative"
    sandbox: bool = False


@dataclass(frozen=True, slots=True)
class FredConfig:
    api_key_env: str = "FRED_API_KEY"
    base_url: str = "https://api.stlouisfed.org/fred"


@dataclass(frozen=True, slots=True)
class StorageConfig:
    root: Path
    compression: str = "zstd"

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise TypeError("storage.root must be a pathlib.Path")


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    alpaca: AlpacaConfig
    fred: FredConfig
    storage: StorageConfig


@dataclass(frozen=True, slots=True)
class RunMetadata:
    run_id: str
    asof: datetime
    started_at: datetime
    git_sha: str | None = None

    def __post_init__(self) -> None:
        _require_aware_utc("asof", self.asof)
        _require_aware_utc("started_at", self.started_at)


EQUITY_QUOTES_COLUMNS = (
    "symbol",
    "quote_ts",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "mid",
    "source",
    "asof",
)

EQUITY_QUOTES_DTYPES = {
    "symbol": "string",
    "quote_ts": "datetime64[ns, UTC]",
    "bid": "Float64",
    "ask": "Float64",
    "bid_size": "Int64",
    "ask_size": "Int64",
    "mid": "Float64",
    "source": "string",
    "asof": "datetime64[ns, UTC]",
}

EQUITY_BARS_COLUMNS = (
    "symbol",
    "bar_ts",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
    "source",
    "asof",
)

EQUITY_BARS_DTYPES = {
    "symbol": "string",
    "bar_ts": "datetime64[ns, UTC]",
    "timeframe": "string",
    "open": "Float64",
    "high": "Float64",
    "low": "Float64",
    "close": "Float64",
    "volume": "Int64",
    "trade_count": "Int64",
    "vwap": "Float64",
    "source": "string",
    "asof": "datetime64[ns, UTC]",
}

OPTION_CHAIN_COLUMNS = (
    "underlying",
    "contract_symbol",
    "quote_ts",
    "expiry",
    "strike",
    "right",
    "bid",
    "ask",
    "mid",
    "last",
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "open_interest",
    "source",
    "asof",
)

OPTION_CHAIN_DTYPES = {
    "underlying": "string",
    "contract_symbol": "string",
    "quote_ts": "datetime64[ns, UTC]",
    "expiry": "datetime64[ns]",
    "strike": "Float64",
    "right": "string",  # can later narrow to Right enum at code boundary
    "bid": "Float64",
    "ask": "Float64",
    "mid": "Float64",
    "last": "Float64",
    "iv": "Float64",
    "delta": "Float64",
    "gamma": "Float64",
    "theta": "Float64",
    "vega": "Float64",
    "rho": "Float64",
    "open_interest": "Int64",
    "source": "string",
    "asof": "datetime64[ns, UTC]",
}

FRED_SERIES_COLUMNS = (
    "series_id",
    "observation_date",
    "value",
    "realtime_start",
    "realtime_end",
    "source",
    "asof",
)

FRED_SERIES_DTYPES = {
    "series_id": "string",
    "observation_date": "datetime64[ns]",
    "value": "Float64",
    "realtime_start": "datetime64[ns]",
    "realtime_end": "datetime64[ns]",
    "source": "string",
    "asof": "datetime64[ns, UTC]",
}

MARKET_SNAPSHOT_COLUMNS = (
    "underlying",
    "asof",
    "spot",
    "spot_source",
    "rate",
    "rate_source",
    "rate_observation_date",
    "dividend_yield",
    "dividend_source",
    "option_contract_count",
)

MARKET_SNAPSHOT_DTYPES = {
    "underlying": "string",
    "asof": "datetime64[ns, UTC]",
    "spot": "Float64",
    "spot_source": "string",
    "rate": "Float64",
    "rate_source": "string",
    "rate_observation_date": "datetime64[ns]",
    "dividend_yield": "Float64",
    "dividend_source": "string",
    "option_contract_count": "Int64",
}

DATASET_COLUMNS = {
    "equity_quotes": EQUITY_QUOTES_COLUMNS,
    "equity_bars": EQUITY_BARS_COLUMNS,
    "option_chain": OPTION_CHAIN_COLUMNS,
    "fred_series": FRED_SERIES_COLUMNS,
    "market_snapshot": MARKET_SNAPSHOT_COLUMNS,
}

DATASET_DTYPES = {
    "equity_quotes": EQUITY_QUOTES_DTYPES,
    "equity_bars": EQUITY_BARS_DTYPES,
    "option_chain": OPTION_CHAIN_DTYPES,
    "fred_series": FRED_SERIES_DTYPES,
    "market_snapshot": MARKET_SNAPSHOT_DTYPES,
}
