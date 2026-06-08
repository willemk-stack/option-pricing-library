"""Internal Alpaca response coercion helpers."""

from __future__ import annotations

from collections.abc import Mapping

from option_pricing.marketdata.providers._alpaca_errors import (
    AlpacaDataUnavailableError,
)


def _coerce_quote_mapping(
    response: object,
    symbols: tuple[str, ...],
) -> dict[str, object]:
    data = getattr(response, "data", response)
    if not isinstance(data, Mapping):
        raise AlpacaDataUnavailableError(
            "Alpaca latest equity quote response must be a mapping"
        )

    quotes = {str(symbol): quote for symbol, quote in data.items()}
    missing = [symbol for symbol in symbols if symbol not in quotes]
    if missing:
        raise AlpacaDataUnavailableError(
            f"Alpaca latest equity quote response is missing symbols: {missing}"
        )
    return quotes
