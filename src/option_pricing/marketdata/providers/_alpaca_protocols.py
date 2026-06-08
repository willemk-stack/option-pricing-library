"""Internal protocols for injectable Alpaca SDK clients."""

from __future__ import annotations

from typing import Protocol


class _StockDataClient(Protocol):
    def get_stock_latest_quote(self, request_params: object) -> object:
        """Return latest quote data for the request."""

    def get_stock_bars(self, request_params: object) -> object:
        """Return historical stock bars for the request."""


class _OptionDataClient(Protocol):
    def get_option_chain(self, request_params: object) -> object:
        """Return option chain snapshot data for the request."""
