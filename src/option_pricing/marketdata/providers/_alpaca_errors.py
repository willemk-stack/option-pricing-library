"""Alpaca-specific provider errors."""

from __future__ import annotations

from collections.abc import Sequence

from option_pricing.marketdata.errors import (
    MarketDataProviderError,
    MissingProviderCredentialError,
    ProviderDataUnavailableError,
    ProviderRequestError,
)


class AlpacaProviderError(MarketDataProviderError):
    """Base error for Alpaca provider failures."""


class AlpacaMissingCredentialsError(
    MissingProviderCredentialError,
    AlpacaProviderError,
):
    """Raised when an Alpaca API key or secret key is missing."""

    def __init__(
        self,
        env_var_name: str | None = None,
        *,
        credential_name: str = "credential",
    ) -> None:
        super().__init__(
            env_var_name,
            provider_name="Alpaca",
            credential_name=credential_name,
        )


class AlpacaRequestError(ProviderRequestError, AlpacaProviderError):
    """Raised when an Alpaca request fails."""

    def __init__(
        self,
        message: str,
        *,
        symbols: Sequence[str] | None = None,
    ) -> None:
        self.symbols = tuple(symbols) if symbols is not None else None
        suffix = f" (symbols={self.symbols!r})" if self.symbols is not None else ""
        super().__init__(f"{message}{suffix}")


class AlpacaDataUnavailableError(ProviderDataUnavailableError, AlpacaProviderError):
    """Raised when Alpaca returns no usable data for the requested quote."""
