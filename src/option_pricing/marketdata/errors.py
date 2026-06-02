"""Shared error hierarchy for marketdata providers."""

from __future__ import annotations


class MarketDataProviderError(RuntimeError):
    """Base error for marketdata provider failures."""


class MissingProviderCredentialError(MarketDataProviderError):
    """Raised when a required provider credential is unset or blank."""

    def __init__(
        self,
        env_var_name: str | None = None,
        *,
        provider_name: str = "marketdata provider",
        credential_name: str = "credential",
    ) -> None:
        self.env_var_name = env_var_name
        self.provider_name = provider_name
        self.credential_name = credential_name
        if env_var_name is None:
            message = (
                f"Missing {provider_name} {credential_name}: value is unset or blank."
            )
        else:
            message = (
                f"Missing {provider_name} {credential_name}: environment variable "
                f"{env_var_name!r} is unset or blank."
            )
        super().__init__(message)


class ProviderRequestError(MarketDataProviderError):
    """Raised when a provider request fails or returns an unusable response."""


class ProviderDataUnavailableError(MarketDataProviderError):
    """Raised when a provider has no usable data for the requested input."""


__all__ = [
    "MarketDataProviderError",
    "MissingProviderCredentialError",
    "ProviderDataUnavailableError",
    "ProviderRequestError",
]
