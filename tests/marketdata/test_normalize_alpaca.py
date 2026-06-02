from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from option_pricing.marketdata.normalize import normalize_alpaca_latest_quotes
from option_pricing.marketdata.schemas import (
    EQUITY_QUOTES_COLUMNS,
    EQUITY_QUOTES_DTYPES,
)
from option_pricing.marketdata.validation import validate_dtypes


@dataclass(frozen=True, slots=True)
class _ObjectQuote:
    symbol: str = "SPY"
    timestamp: str = "2026-05-22T11:30:00-04:00"
    bid_price: float = 499.0
    ask_price: float = 499.2
    bid_size: int = 3
    ask_size: int = 4


def _payload() -> dict[str, object]:
    return {
        "symbols": ("SPY",),
        "quotes": {"SPY": _ObjectQuote()},
        "source": "alpaca",
        "feed": "iex",
    }


def test_normalize_alpaca_latest_quotes_outputs_canonical_schema_order() -> None:
    normalized = normalize_alpaca_latest_quotes(
        _payload(),
        asof="2026-05-22T15:31:00Z",
    )

    assert tuple(normalized.columns) == EQUITY_QUOTES_COLUMNS
    assert {column: str(normalized[column].dtype) for column in normalized} == (
        EQUITY_QUOTES_DTYPES
    )
    validate_dtypes(normalized, "equity_quotes", allow_extra=False)


def test_normalize_alpaca_latest_quotes_computes_mid_from_bid_ask() -> None:
    normalized = normalize_alpaca_latest_quotes(
        {
            "quotes": {
                "SPY": _ObjectQuote(
                    bid_price=500.0,
                    ask_price=500.6,
                )
            }
        },
        asof="2026-05-22T15:31:00Z",
    )

    assert float(normalized.loc[0, "mid"]) == pytest.approx(500.3)


def test_normalize_alpaca_latest_quotes_normalizes_quote_ts_to_utc() -> None:
    normalized = normalize_alpaca_latest_quotes(
        _payload(),
        asof="2026-05-22T15:31:00Z",
    )

    assert normalized.loc[0, "quote_ts"] == pd.Timestamp("2026-05-22T15:30:00Z")


def test_normalize_alpaca_latest_quotes_normalizes_asof_to_utc() -> None:
    normalized = normalize_alpaca_latest_quotes(
        _payload(),
        asof=pd.Timestamp("2026-05-22T11:31:00-04:00"),
    )

    assert normalized.loc[0, "asof"] == pd.Timestamp("2026-05-22T15:31:00Z")


def test_normalize_alpaca_latest_quotes_ignores_extra_provider_fields() -> None:
    normalized = normalize_alpaca_latest_quotes(
        {
            "quotes": {
                "SPY": {
                    "symbol": "SPY",
                    "timestamp": "2026-05-22T15:30:00Z",
                    "bid_price": "499.00",
                    "ask_price": "499.20",
                    "bid_size": 3,
                    "ask_size": 4,
                    "exchange": "extra",
                }
            },
            "debug": "ignored",
        },
        asof="2026-05-22T15:31:00Z",
    )

    assert tuple(normalized.columns) == EQUITY_QUOTES_COLUMNS
    assert "debug" not in normalized.columns
    assert "exchange" not in normalized.columns


@pytest.mark.parametrize(
    ("quote", "message"),
    [
        (
            {
                "timestamp": "2026-05-22T15:30:00Z",
                "ask_price": 499.2,
                "bid_size": 3,
                "ask_size": 4,
            },
            "missing bid",
        ),
        (
            {
                "timestamp": "2026-05-22T15:30:00Z",
                "bid_price": 499.0,
                "bid_size": 3,
                "ask_size": 4,
            },
            "missing ask",
        ),
    ],
)
def test_normalize_alpaca_latest_quotes_missing_bid_or_ask_fails_clearly(
    quote: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_alpaca_latest_quotes(
            {"quotes": {"SPY": quote}},
            asof="2026-05-22T15:31:00Z",
        )


@pytest.mark.parametrize(
    ("quote", "message"),
    [
        (
            {
                "timestamp": "2026-05-22T15:30:00Z",
                "bid_price": float("inf"),
                "ask_price": 499.2,
            },
            "bid must be finite",
        ),
        (
            {
                "timestamp": "2026-05-22T15:30:00Z",
                "bid_price": 499.0,
                "ask_price": float("nan"),
            },
            "ask must be finite",
        ),
    ],
)
def test_normalize_alpaca_latest_quotes_non_finite_bid_or_ask_fails_clearly(
    quote: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_alpaca_latest_quotes(
            {"quotes": {"SPY": quote}},
            asof="2026-05-22T15:31:00Z",
        )


def test_normalize_alpaca_latest_quotes_multiple_symbols_are_deterministic() -> None:
    normalized = normalize_alpaca_latest_quotes(
        {
            "quotes": {
                "MSFT": _ObjectQuote(symbol="MSFT", bid_price=420.0, ask_price=420.2),
                "AAPL": _ObjectQuote(symbol="AAPL", bid_price=190.0, ask_price=190.4),
                "SPY": _ObjectQuote(symbol="SPY", bid_price=499.0, ask_price=499.2),
            }
        },
        asof="2026-05-22T15:31:00Z",
    )

    assert normalized["symbol"].astype(str).tolist() == ["AAPL", "MSFT", "SPY"]


def test_normalize_alpaca_latest_quotes_accepts_raw_abbreviated_quote_mapping() -> None:
    normalized = normalize_alpaca_latest_quotes(
        {
            "SPY": {
                "t": "2026-05-22T15:30:00Z",
                "bp": 499.0,
                "ap": 499.2,
                "bs": 3,
                "as": 4,
                "ignored": "yes",
            }
        },
        asof="2026-05-22T15:31:00Z",
    )

    assert normalized.loc[0, "symbol"] == "SPY"
    assert float(normalized.loc[0, "mid"]) == pytest.approx(499.1)
