from __future__ import annotations

from types import SimpleNamespace

import pytest

from option_pricing.marketdata.providers._alpaca_errors import (
    AlpacaDataUnavailableError,
)
from option_pricing.marketdata.providers._alpaca_payloads import _coerce_quote_mapping


def test_coerce_quote_mapping_accepts_response_data_mapping() -> None:
    response = SimpleNamespace(data={"AAPL": {"bid": 1.0}})

    assert _coerce_quote_mapping(response, ("AAPL",)) == {"AAPL": {"bid": 1.0}}


def test_coerce_quote_mapping_accepts_direct_mapping_and_stringifies_keys() -> None:
    quotes = _coerce_quote_mapping({123: "numeric-key", "MSFT": "quote"}, ("MSFT",))

    assert quotes["123"] == "numeric-key"
    assert quotes["MSFT"] == "quote"


def test_coerce_quote_mapping_rejects_non_mapping_payload() -> None:
    with pytest.raises(AlpacaDataUnavailableError, match="must be a mapping"):
        _coerce_quote_mapping(["not", "mapping"], ("AAPL",))


def test_coerce_quote_mapping_rejects_missing_symbols() -> None:
    with pytest.raises(AlpacaDataUnavailableError, match="missing symbols"):
        _coerce_quote_mapping({"AAPL": object()}, ("AAPL", "MSFT"))
