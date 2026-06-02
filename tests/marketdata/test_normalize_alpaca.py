from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from option_pricing.marketdata.normalize import (
    normalize_alpaca_bars,
    normalize_alpaca_latest_quotes,
)
from option_pricing.marketdata.schemas import (
    EQUITY_BARS_COLUMNS,
    EQUITY_BARS_DTYPES,
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


@dataclass(frozen=True, slots=True)
class _ObjectBar:
    symbol: str = "SPY"
    timestamp: str = "2026-05-22T11:30:00-04:00"
    timeframe: str = "1Day"
    open: float = 499.0
    high: float = 501.0
    low: float = 498.5
    close: float = 500.5
    volume: int = 1000
    trade_count: int = 75
    vwap: float = 500.1


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


def _bars_payload() -> dict[str, object]:
    return {
        "symbols": ("SPY",),
        "bars": {"SPY": [_ObjectBar()]},
        "source": "alpaca",
        "feed": "iex",
        "timeframe": "1Day",
    }


def test_normalize_alpaca_bars_outputs_canonical_schema_order() -> None:
    normalized = normalize_alpaca_bars(
        _bars_payload(),
        asof="2026-05-22T15:31:00Z",
    )

    assert tuple(normalized.columns) == EQUITY_BARS_COLUMNS
    assert {column: str(normalized[column].dtype) for column in normalized} == (
        EQUITY_BARS_DTYPES
    )
    validate_dtypes(normalized, "equity_bars", allow_extra=False)


def test_normalize_alpaca_bars_accepts_object_style_records() -> None:
    normalized = normalize_alpaca_bars(
        {"bars": {"SPY": [_ObjectBar(close=501.25)]}},
        asof="2026-05-22T15:31:00Z",
    )

    assert normalized.loc[0, "symbol"] == "SPY"
    assert normalized.loc[0, "timeframe"] == "1Day"
    assert float(normalized.loc[0, "close"]) == pytest.approx(501.25)


def test_normalize_alpaca_bars_accepts_dict_raw_data_records() -> None:
    normalized = normalize_alpaca_bars(
        {
            "bars": {
                "SPY": [
                    {
                        "raw_data": {
                            "symbol": "SPY",
                            "timestamp": "2026-05-22T15:30:00Z",
                            "timeframe": "1Day",
                            "open": "499.0",
                            "high": "501.0",
                            "low": "498.5",
                            "close": "500.5",
                            "volume": "1000",
                            "trade_count": "75",
                            "vwap": "500.1",
                        },
                        "ignored": "yes",
                    }
                ]
            }
        },
        asof="2026-05-22T15:31:00Z",
    )

    assert normalized.loc[0, "symbol"] == "SPY"
    assert int(normalized.loc[0, "volume"]) == 1000
    assert int(normalized.loc[0, "trade_count"]) == 75
    assert float(normalized.loc[0, "vwap"]) == pytest.approx(500.1)


def test_normalize_alpaca_bars_accepts_short_alias_fields() -> None:
    normalized = normalize_alpaca_bars(
        {
            "bars": {
                "SPY": [
                    {
                        "S": "SPY",
                        "t": "2026-05-22T15:30:00Z",
                        "tf": "1Min",
                        "o": 499.0,
                        "h": 501.0,
                        "l": 498.5,
                        "c": 500.5,
                        "v": 1000,
                        "n": 75,
                        "vw": 500.1,
                        "exchange": "ignored",
                    }
                ]
            }
        },
        asof="2026-05-22T15:31:00Z",
    )

    assert normalized.loc[0, "timeframe"] == "1Min"
    assert float(normalized.loc[0, "open"]) == pytest.approx(499.0)
    assert float(normalized.loc[0, "vwap"]) == pytest.approx(500.1)


def test_normalize_alpaca_bars_multiple_symbols_are_sorted_deterministically() -> None:
    normalized = normalize_alpaca_bars(
        {
            "bars": {
                "SPY": [_ObjectBar(symbol="SPY", timestamp="2026-05-22T15:30:00Z")],
                "AAPL": [
                    _ObjectBar(symbol="AAPL", timestamp="2026-05-22T16:30:00Z"),
                    _ObjectBar(symbol="AAPL", timestamp="2026-05-22T15:30:00Z"),
                ],
                "MSFT": [_ObjectBar(symbol="MSFT", timestamp="2026-05-22T15:30:00Z")],
            },
            "timeframe": "1Day",
        },
        asof="2026-05-22T15:31:00Z",
    )

    assert normalized["symbol"].astype(str).tolist() == [
        "AAPL",
        "AAPL",
        "MSFT",
        "SPY",
    ]
    assert normalized.loc[0, "bar_ts"] == pd.Timestamp("2026-05-22T15:30:00Z")
    assert normalized.loc[1, "bar_ts"] == pd.Timestamp("2026-05-22T16:30:00Z")


def test_normalize_alpaca_bars_handles_single_letter_symbol_keys() -> None:
    normalized = normalize_alpaca_bars(
        {
            "bars": {
                "S": [
                    {
                        "timestamp": "2026-05-22T15:30:00Z",
                        "timeframe": "1Day",
                        "open": 20.0,
                        "high": 21.0,
                        "low": 19.5,
                        "close": 20.5,
                        "volume": 1000,
                    }
                ]
            }
        },
        asof="2026-05-22T15:31:00Z",
    )

    assert normalized.loc[0, "symbol"] == "S"


def test_normalize_alpaca_bars_missing_trade_count_and_vwap_are_nullable() -> None:
    normalized = normalize_alpaca_bars(
        {
            "bars": {
                "SPY": [
                    {
                        "timestamp": "2026-05-22T15:30:00Z",
                        "timeframe": "1Day",
                        "open": 499.0,
                        "high": 501.0,
                        "low": 498.5,
                        "close": 500.5,
                        "volume": 1000,
                    }
                ]
            }
        },
        asof="2026-05-22T15:31:00Z",
    )

    assert pd.isna(normalized.loc[0, "trade_count"])
    assert pd.isna(normalized.loc[0, "vwap"])
    assert str(normalized["trade_count"].dtype) == "Int64"
    assert str(normalized["vwap"].dtype) == "Float64"


@pytest.mark.parametrize(
    "missing_field",
    ["timestamp", "open", "high", "low", "close", "volume"],
)
def test_normalize_alpaca_bars_missing_required_fields_fail_clearly(
    missing_field: str,
) -> None:
    bar: dict[str, object] = {
        "timestamp": "2026-05-22T15:30:00Z",
        "timeframe": "1Day",
        "open": 499.0,
        "high": 501.0,
        "low": 498.5,
        "close": 500.5,
        "volume": 1000,
    }
    bar.pop(missing_field)

    with pytest.raises(ValueError, match=f"missing {missing_field}"):
        normalize_alpaca_bars(
            {"bars": {"SPY": [bar]}},
            asof="2026-05-22T15:31:00Z",
        )


def test_normalize_alpaca_bars_normalizes_asof_to_utc() -> None:
    normalized = normalize_alpaca_bars(
        _bars_payload(),
        asof=pd.Timestamp("2026-05-22T11:31:00-04:00"),
    )

    assert normalized.loc[0, "asof"] == pd.Timestamp("2026-05-22T15:31:00Z")


def test_normalize_alpaca_bars_normalizes_bar_ts_to_utc() -> None:
    normalized = normalize_alpaca_bars(
        _bars_payload(),
        asof="2026-05-22T15:31:00Z",
    )

    assert normalized.loc[0, "bar_ts"] == pd.Timestamp("2026-05-22T15:30:00Z")


def test_normalize_alpaca_bars_ignores_extra_provider_fields() -> None:
    normalized = normalize_alpaca_bars(
        {
            "bars": {
                "SPY": [
                    {
                        "timestamp": "2026-05-22T15:30:00Z",
                        "timeframe": "1Day",
                        "open": 499.0,
                        "high": 501.0,
                        "low": 498.5,
                        "close": 500.5,
                        "volume": 1000,
                        "trade_count": 75,
                        "vwap": 500.1,
                        "exchange": "ignored",
                    }
                ]
            },
            "debug": "ignored",
        },
        asof="2026-05-22T15:31:00Z",
    )

    assert tuple(normalized.columns) == EQUITY_BARS_COLUMNS
    assert "debug" not in normalized.columns
    assert "exchange" not in normalized.columns
