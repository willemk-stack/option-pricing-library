from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from option_pricing.marketdata.gold import (
    build_heston_quotes,
    heston_quote_set_from_frame,
)
from option_pricing.marketdata.schemas import (
    HESTON_QUOTES_COLUMNS,
    HESTON_QUOTES_SCHEMA_VERSION,
    DatasetName,
)
from option_pricing.marketdata.validation import coerce_frame, validate_dtypes
from option_pricing.types import MarketData


def _cleaned_quote_records() -> list[dict[str, object]]:
    return [
        {
            "underlying": "SYNTH",
            "contract_symbol": "SYNTH260619C00095000",
            "quote_id": "quote-call-95",
            "quote_ts": "2026-05-22T15:30:00Z",
            "asof": "2026-05-22T15:30:00Z",
            "expiry": "2026-06-19",
            "expiry_years": 0.123456789,
            "strike": 95.0,
            "right": "call",
            "bid": 7.05,
            "ask": 7.21,
            "mid": 7.13,
            "iv": 0.27,
            "vega": 0.089,
            "delta": 0.65,
            "gamma": 0.024,
            "theta": -0.028,
            "rho": 0.039,
            "open_interest": 390,
            "moneyness": 0.95,
            "source": "unit_test",
            "cleaning_policy": "quote_cleaning_policy.v1",
        },
        {
            "underlying": "SYNTH",
            "contract_symbol": "SYNTH260918P00105000",
            "quote_id": "quote-put-105",
            "quote_ts": "2026-05-22T15:30:00Z",
            "asof": "2026-05-22T15:30:00Z",
            "expiry": "2026-09-18",
            "expiry_years": 0.234567891,
            "strike": 105.0,
            "right": "put",
            "bid": 5.55,
            "ask": 5.75,
            "mid": 5.65,
            "iv": 0.28,
            "vega": 0.093,
            "delta": -0.62,
            "gamma": 0.024,
            "theta": -0.030,
            "rho": -0.040,
            "open_interest": 415,
            "moneyness": 1.05,
            "source": "unit_test",
            "cleaning_policy": "quote_cleaning_policy.v1",
        },
    ]


def _cleaned_quotes(
    row_overrides: Mapping[int, Mapping[str, object]] | None = None,
) -> pd.DataFrame:
    records = _cleaned_quote_records()
    for row_index, overrides in (row_overrides or {}).items():
        records[row_index].update(overrides)
    return coerce_frame(
        pd.DataFrame(records),
        DatasetName.CLEANED_QUOTES,
        allow_extra=False,
    )


def _market_data() -> MarketData:
    return MarketData(spot=100.0, rate=0.04, dividend_yield=0.01)


def _float_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame[column].to_numpy(dtype=np.float64, na_value=np.nan)


def test_build_heston_quotes_returns_exact_schema_and_valid_artifact() -> None:
    cleaned = _cleaned_quotes()

    result = build_heston_quotes(cleaned)
    heston_quotes = result.heston_quotes

    assert tuple(heston_quotes.columns) == HESTON_QUOTES_COLUMNS
    validate_dtypes(heston_quotes, DatasetName.HESTON_QUOTES, allow_extra=False)
    assert result.quote_count == len(cleaned)
    assert result.warnings == ()
    assert set(heston_quotes["right"].astype(str)) == {"call", "put"}
    assert set(heston_quotes["option_type"].astype(str)) == {"call", "put"}
    assert heston_quotes["option_type"].equals(heston_quotes["right"])
    assert heston_quotes["label"].equals(heston_quotes["contract_symbol"])

    source = cleaned.reset_index(drop=True)
    column_mapping = {
        "underlying": "underlying",
        "contract_symbol": "contract_symbol",
        "quote_id": "quote_id",
        "asof": "asof",
        "expiry": "expiry",
        "expiry_years": "expiry_years",
        "strike": "strike",
        "right": "right",
        "mid": "mid",
        "bid": "bid",
        "ask": "ask",
        "iv": "iv",
        "vega": "vega",
        "option_type": "right",
        "label": "contract_symbol",
        "source": "source",
        "cleaning_policy": "cleaning_policy",
    }
    for target_column, source_column in column_mapping.items():
        pd.testing.assert_series_equal(
            heston_quotes[target_column],
            source[source_column],
            check_names=False,
        )
    pd.testing.assert_series_equal(
        heston_quotes["expiry_years"],
        cleaned["expiry_years"].reset_index(drop=True),
        check_names=False,
    )


def test_build_heston_quotes_rejects_empty_cleaned_quotes() -> None:
    with pytest.raises(ValueError, match="cleaned_quotes.*at least one quote"):
        build_heston_quotes(_cleaned_quotes().iloc[0:0])


def test_build_heston_quotes_rejects_invalid_right() -> None:
    cleaned = _cleaned_quotes({0: {"right": "straddle"}})

    with pytest.raises(ValueError, match="right.*call.*put"):
        build_heston_quotes(cleaned)


def test_nullable_iv_and_vega_do_not_block_heston_quote_artifact() -> None:
    cleaned = _cleaned_quotes({0: {"iv": pd.NA}, 1: {"vega": pd.NA}})

    result = build_heston_quotes(cleaned)

    validate_dtypes(result.heston_quotes, DatasetName.HESTON_QUOTES, allow_extra=False)
    assert result.quote_count == len(cleaned)
    assert any(
        "IV" in warning and "reconstruction" in warning for warning in result.warnings
    )
    assert any(
        "vega" in warning and "reconstruction" in warning for warning in result.warnings
    )


def test_heston_quote_set_from_frame_reconstructs_valid_quote_set() -> None:
    heston_quotes = build_heston_quotes(_cleaned_quotes()).heston_quotes

    quote_set = heston_quote_set_from_frame(heston_quotes, _market_data())

    assert quote_set.n_quotes == len(heston_quotes)
    assert quote_set.labels == tuple(heston_quotes["contract_symbol"].astype(str))
    assert np.all(np.isfinite(quote_set.discount))
    assert np.all(quote_set.discount > 0.0)
    assert np.all(np.isfinite(quote_set.forward))
    assert np.all(quote_set.forward > 0.0)
    np.testing.assert_allclose(quote_set.mid, _float_array(heston_quotes, "mid"))
    assert quote_set.bid is not None
    assert quote_set.ask is not None
    np.testing.assert_allclose(quote_set.bid, _float_array(heston_quotes, "bid"))
    np.testing.assert_allclose(quote_set.ask, _float_array(heston_quotes, "ask"))
    assert quote_set.metadata is not None
    assert quote_set.metadata["schema_version"] == HESTON_QUOTES_SCHEMA_VERSION
    assert quote_set.metadata["quote_count"] == len(heston_quotes)
    assert quote_set.metadata["underlying"] == "SYNTH"
    assert quote_set.metadata["asof"] == "2026-05-22T15:30:00Z"
    assert quote_set.metadata["cleaning_policy"] == "quote_cleaning_policy.v1"


def test_complete_valid_iv_and_vega_are_passed_into_heston_quote_set() -> None:
    heston_quotes = build_heston_quotes(_cleaned_quotes()).heston_quotes

    quote_set = heston_quote_set_from_frame(heston_quotes, _market_data())

    assert quote_set.iv_mid is not None
    assert quote_set.bs_vega is not None
    np.testing.assert_allclose(quote_set.iv_mid, _float_array(heston_quotes, "iv"))
    np.testing.assert_allclose(quote_set.bs_vega, _float_array(heston_quotes, "vega"))
    assert quote_set.metadata is not None
    assert quote_set.metadata["iv_mid_included"] is True
    assert quote_set.metadata["bs_vega_included"] is True
    assert "optional_data_warnings" not in quote_set.metadata


def test_missing_or_invalid_iv_and_vega_are_omitted_from_quote_set() -> None:
    cleaned = _cleaned_quotes({0: {"iv": pd.NA}, 1: {"vega": -0.01}})

    result = build_heston_quotes(cleaned)
    quote_set = heston_quote_set_from_frame(result.heston_quotes, _market_data())

    assert any("IV" in warning for warning in result.warnings)
    assert any("vega" in warning for warning in result.warnings)
    assert quote_set.iv_mid is None
    assert quote_set.bs_vega is None
    assert quote_set.metadata is not None
    assert quote_set.metadata["iv_mid_included"] is False
    assert quote_set.metadata["bs_vega_included"] is False
    assert quote_set.metadata["optional_data_warnings"] == result.warnings


def test_heston_quote_set_from_frame_rejects_empty_heston_quotes() -> None:
    heston_quotes = build_heston_quotes(_cleaned_quotes()).heston_quotes.iloc[0:0]

    with pytest.raises(ValueError, match="heston_quotes.*at least one quote"):
        heston_quote_set_from_frame(heston_quotes, _market_data())


def test_heston_quote_set_from_frame_rejects_invalid_option_type() -> None:
    heston_quotes = build_heston_quotes(_cleaned_quotes()).heston_quotes
    heston_quotes.loc[0, "option_type"] = "C"

    with pytest.raises(ValueError, match="option_type.*call.*put"):
        heston_quote_set_from_frame(heston_quotes, _market_data())


def test_heston_quote_set_from_frame_rejects_option_type_mismatch() -> None:
    heston_quotes = build_heston_quotes(_cleaned_quotes()).heston_quotes
    heston_quotes.loc[0, "option_type"] = "put"

    with pytest.raises(ValueError, match="option_type must match right"):
        heston_quote_set_from_frame(heston_quotes, _market_data())


def test_heston_quote_set_from_frame_rejects_label_mismatch() -> None:
    heston_quotes = build_heston_quotes(_cleaned_quotes()).heston_quotes
    heston_quotes.loc[0, "label"] = "wrong-label"

    with pytest.raises(ValueError, match="label must match contract_symbol"):
        heston_quote_set_from_frame(heston_quotes, _market_data())
