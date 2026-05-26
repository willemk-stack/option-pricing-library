"""Dataset schema registry for marketdata DataFrame contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

type PandasSchemaDtype = Literal[
    "datetime64[ns, UTC]",
    "datetime64[ns]",
    "Float64",
    "Int64",
    "string",
]

MARKET_INPUTS_SCHEMA_VERSION = "market_inputs.v1"
OPTION_CHAIN_SCHEMA_VERSION = "option_chain.v1"
CLEANED_QUOTES_SCHEMA_VERSION = "cleaned_quotes.v1"
REJECTED_QUOTES_SCHEMA_VERSION = "rejected_quotes.v1"
HESTON_QUOTES_SCHEMA_VERSION = "heston_quotes.v1"
SURFACE_INPUTS_SCHEMA_VERSION = "surface_inputs.v1"
MODEL_VALIDATION_BUNDLE_VERSION = "model_validation_bundle.v1"

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

EQUITY_QUOTES_DTYPES: dict[str, PandasSchemaDtype] = {
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

EQUITY_BARS_DTYPES: dict[str, PandasSchemaDtype] = {
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

OPTION_CHAIN_DTYPES: dict[str, PandasSchemaDtype] = {
    "underlying": "string",
    "contract_symbol": "string",
    "quote_ts": "datetime64[ns, UTC]",
    "expiry": "datetime64[ns]",
    "strike": "Float64",
    "right": "string",
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

CLEANED_QUOTES_COLUMNS = (
    "underlying",
    "contract_symbol",
    "quote_id",
    "quote_ts",
    "asof",
    "expiry",
    "expiry_years",
    "strike",
    "right",
    "bid",
    "ask",
    "mid",
    "iv",
    "vega",
    "delta",
    "gamma",
    "theta",
    "rho",
    "open_interest",
    "moneyness",
    "source",
    "cleaning_policy",
)

CLEANED_QUOTES_DTYPES: dict[str, PandasSchemaDtype] = {
    "underlying": "string",
    "contract_symbol": "string",
    "quote_id": "string",
    "quote_ts": "datetime64[ns, UTC]",
    "asof": "datetime64[ns, UTC]",
    "expiry": "datetime64[ns]",
    "expiry_years": "Float64",
    "strike": "Float64",
    "right": "string",
    "bid": "Float64",
    "ask": "Float64",
    "mid": "Float64",
    "iv": "Float64",
    "vega": "Float64",
    "delta": "Float64",
    "gamma": "Float64",
    "theta": "Float64",
    "rho": "Float64",
    "open_interest": "Int64",
    "moneyness": "Float64",
    "source": "string",
    "cleaning_policy": "string",
}

REJECTED_QUOTES_COLUMNS = (
    "underlying",
    "contract_symbol",
    "quote_id",
    "quote_ts",
    "asof",
    "expiry",
    "strike",
    "right",
    "bid",
    "ask",
    "mid",
    "iv",
    "vega",
    "source",
    "rejection_reason",
    "rejection_detail",
    "cleaning_policy",
)

REJECTED_QUOTES_DTYPES: dict[str, PandasSchemaDtype] = {
    "underlying": "string",
    "contract_symbol": "string",
    "quote_id": "string",
    "quote_ts": "datetime64[ns, UTC]",
    "asof": "datetime64[ns, UTC]",
    "expiry": "datetime64[ns]",
    "strike": "Float64",
    "right": "string",
    "bid": "Float64",
    "ask": "Float64",
    "mid": "Float64",
    "iv": "Float64",
    "vega": "Float64",
    "source": "string",
    "rejection_reason": "string",
    "rejection_detail": "string",
    "cleaning_policy": "string",
}

HESTON_QUOTES_COLUMNS = (
    "underlying",
    "contract_symbol",
    "quote_id",
    "asof",
    "expiry",
    "expiry_years",
    "strike",
    "right",
    "mid",
    "bid",
    "ask",
    "iv",
    "vega",
    "option_type",
    "label",
    "source",
    "cleaning_policy",
)

HESTON_QUOTES_DTYPES: dict[str, PandasSchemaDtype] = {
    "underlying": "string",
    "contract_symbol": "string",
    "quote_id": "string",
    "asof": "datetime64[ns, UTC]",
    "expiry": "datetime64[ns]",
    "expiry_years": "Float64",
    "strike": "Float64",
    "right": "string",
    "mid": "Float64",
    "bid": "Float64",
    "ask": "Float64",
    "iv": "Float64",
    "vega": "Float64",
    "option_type": "string",
    "label": "string",
    "source": "string",
    "cleaning_policy": "string",
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

FRED_SERIES_DTYPES: dict[str, PandasSchemaDtype] = {
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

MARKET_SNAPSHOT_DTYPES: dict[str, PandasSchemaDtype] = {
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

MARKET_INPUTS_COLUMNS = (
    "underlying",
    "asof",
    "spot",
    "spot_source",
    "rate",
    "rate_source",
    "rate_observation_date",
    "rate_compounding",
    "dividend_yield",
    "dividend_yield_source",
    "day_count",
)

MARKET_INPUTS_DTYPES: dict[str, PandasSchemaDtype] = {
    "underlying": "string",
    "asof": "datetime64[ns, UTC]",
    "spot": "Float64",
    "spot_source": "string",
    "rate": "Float64",
    "rate_source": "string",
    "rate_observation_date": "datetime64[ns]",
    "rate_compounding": "string",
    "dividend_yield": "Float64",
    "dividend_yield_source": "string",
    "day_count": "string",
}

SURFACE_INPUTS_COLUMNS = (
    "underlying",
    "quote_id",
    "asof",
    "expiry",
    "expiry_years",
    "strike",
    "right",
    "mid",
    "iv",
    "source",
    "cleaning_policy",
)

SURFACE_INPUTS_DTYPES: dict[str, PandasSchemaDtype] = {
    "underlying": "string",
    "quote_id": "string",
    "asof": "datetime64[ns, UTC]",
    "expiry": "datetime64[ns]",
    "expiry_years": "Float64",
    "strike": "Float64",
    "right": "string",
    "mid": "Float64",
    "iv": "Float64",
    "source": "string",
    "cleaning_policy": "string",
}

MODEL_VALIDATION_BUNDLE_COLUMNS: tuple[str, ...] = ()
MODEL_VALIDATION_BUNDLE_DTYPES: dict[str, PandasSchemaDtype] = {}


class DatasetName(StrEnum):
    """Enum object for safe internal handling of dataset_name calling."""

    EQUITY_QUOTES = "equity_quotes"
    EQUITY_BARS = "equity_bars"
    OPTION_CHAIN = "option_chain"
    CLEANED_QUOTES = "cleaned_quotes"
    REJECTED_QUOTES = "rejected_quotes"
    HESTON_QUOTES = "heston_quotes"
    FRED_SERIES = "fred_series"
    MARKET_SNAPSHOT = "market_snapshot"
    MARKET_INPUTS = "market_inputs"
    SURFACE_INPUTS = "surface_inputs"
    MODEL_VALIDATION_BUNDLE = "model_validation_bundle"


def parse_dataset_name(dataset_name: DatasetName | str) -> DatasetName:
    """Normalize a dataset name into a DatasetName enum."""

    if isinstance(dataset_name, DatasetName):
        return dataset_name

    if not isinstance(dataset_name, str):
        raise TypeError(
            f"dataset_name must be a DatasetName or str, got {type(dataset_name).__name__}"
        )

    try:
        return DatasetName(dataset_name.strip().lower())
    except ValueError as exc:
        known = ", ".join(item.value for item in DatasetName)
        raise ValueError(
            f"Unknown marketdata dataset_name {dataset_name!r}. "
            f"Expected one of: {known}"
        ) from exc


DATASET_COLUMNS: dict[DatasetName, tuple[str, ...]] = {
    DatasetName.EQUITY_QUOTES: EQUITY_QUOTES_COLUMNS,
    DatasetName.EQUITY_BARS: EQUITY_BARS_COLUMNS,
    DatasetName.OPTION_CHAIN: OPTION_CHAIN_COLUMNS,
    DatasetName.CLEANED_QUOTES: CLEANED_QUOTES_COLUMNS,
    DatasetName.REJECTED_QUOTES: REJECTED_QUOTES_COLUMNS,
    DatasetName.HESTON_QUOTES: HESTON_QUOTES_COLUMNS,
    DatasetName.FRED_SERIES: FRED_SERIES_COLUMNS,
    DatasetName.MARKET_SNAPSHOT: MARKET_SNAPSHOT_COLUMNS,
    DatasetName.MARKET_INPUTS: MARKET_INPUTS_COLUMNS,
    DatasetName.SURFACE_INPUTS: SURFACE_INPUTS_COLUMNS,
    DatasetName.MODEL_VALIDATION_BUNDLE: MODEL_VALIDATION_BUNDLE_COLUMNS,
}

DATASET_DTYPES: dict[DatasetName, dict[str, PandasSchemaDtype]] = {
    DatasetName.EQUITY_QUOTES: EQUITY_QUOTES_DTYPES,
    DatasetName.EQUITY_BARS: EQUITY_BARS_DTYPES,
    DatasetName.OPTION_CHAIN: OPTION_CHAIN_DTYPES,
    DatasetName.CLEANED_QUOTES: CLEANED_QUOTES_DTYPES,
    DatasetName.REJECTED_QUOTES: REJECTED_QUOTES_DTYPES,
    DatasetName.HESTON_QUOTES: HESTON_QUOTES_DTYPES,
    DatasetName.FRED_SERIES: FRED_SERIES_DTYPES,
    DatasetName.MARKET_SNAPSHOT: MARKET_SNAPSHOT_DTYPES,
    DatasetName.MARKET_INPUTS: MARKET_INPUTS_DTYPES,
    DatasetName.SURFACE_INPUTS: SURFACE_INPUTS_DTYPES,
    DatasetName.MODEL_VALIDATION_BUNDLE: MODEL_VALIDATION_BUNDLE_DTYPES,
}


__all__ = [
    "CLEANED_QUOTES_COLUMNS",
    "CLEANED_QUOTES_DTYPES",
    "CLEANED_QUOTES_SCHEMA_VERSION",
    "DATASET_COLUMNS",
    "DATASET_DTYPES",
    "EQUITY_BARS_COLUMNS",
    "EQUITY_BARS_DTYPES",
    "EQUITY_QUOTES_COLUMNS",
    "EQUITY_QUOTES_DTYPES",
    "FRED_SERIES_COLUMNS",
    "FRED_SERIES_DTYPES",
    "HESTON_QUOTES_COLUMNS",
    "HESTON_QUOTES_DTYPES",
    "HESTON_QUOTES_SCHEMA_VERSION",
    "MARKET_INPUTS_COLUMNS",
    "MARKET_INPUTS_DTYPES",
    "MARKET_INPUTS_SCHEMA_VERSION",
    "MARKET_SNAPSHOT_COLUMNS",
    "MARKET_SNAPSHOT_DTYPES",
    "MODEL_VALIDATION_BUNDLE_COLUMNS",
    "MODEL_VALIDATION_BUNDLE_DTYPES",
    "MODEL_VALIDATION_BUNDLE_VERSION",
    "OPTION_CHAIN_COLUMNS",
    "OPTION_CHAIN_DTYPES",
    "OPTION_CHAIN_SCHEMA_VERSION",
    "PandasSchemaDtype",
    "REJECTED_QUOTES_COLUMNS",
    "REJECTED_QUOTES_DTYPES",
    "REJECTED_QUOTES_SCHEMA_VERSION",
    "SURFACE_INPUTS_COLUMNS",
    "SURFACE_INPUTS_DTYPES",
    "SURFACE_INPUTS_SCHEMA_VERSION",
    "DatasetName",
    "parse_dataset_name",
]
