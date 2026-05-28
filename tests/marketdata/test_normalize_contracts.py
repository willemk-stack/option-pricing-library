from __future__ import annotations

from typing import get_type_hints

import pandas as pd

from option_pricing.marketdata.normalize import (
    normalize_market_inputs,
    normalize_option_chain,
)
from option_pricing.marketdata.schemas import (
    MARKET_INPUTS_DTYPES,
    OPTION_CHAIN_DTYPES,
    DatasetName,
)
from option_pricing.marketdata.validation import dataset_columns, dataset_dtypes


def test_normalize_market_inputs_signature() -> None:
    hints = get_type_hints(normalize_market_inputs)

    assert hints == {"frame": pd.DataFrame, "return": pd.DataFrame}


def test_normalize_option_chain_signature() -> None:
    hints = get_type_hints(normalize_option_chain)

    assert hints == {"frame": pd.DataFrame, "return": pd.DataFrame}


def test_normalization_contracts_reuse_existing_marketdata_schemas() -> None:
    assert DatasetName.MARKET_INPUTS.value == "market_inputs"
    assert DatasetName.OPTION_CHAIN.value == "option_chain"
    assert dataset_columns(DatasetName.MARKET_INPUTS) == (
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
    assert dataset_columns(DatasetName.OPTION_CHAIN) == (
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
    assert dataset_dtypes(DatasetName.MARKET_INPUTS) == MARKET_INPUTS_DTYPES
    assert dataset_dtypes(DatasetName.OPTION_CHAIN) == OPTION_CHAIN_DTYPES
