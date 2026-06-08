from __future__ import annotations

from option_pricing.marketdata.errors import (
    MarketDataProviderError,
    MissingProviderCredentialError,
    ProviderDataUnavailableError,
    ProviderRequestError,
)
from option_pricing.marketdata.providers.alpaca import (
    AlpacaDataUnavailableError,
    AlpacaMissingCredentialsError,
    AlpacaProviderError,
    AlpacaRequestError,
)
from option_pricing.marketdata.providers.fred import (
    FredMissingApiKeyError,
    FredProviderError,
    FredRateUnavailableError,
    FredRequestError,
)


def test_provider_specific_errors_inherit_shared_marketdata_errors() -> None:
    assert issubclass(FredProviderError, MarketDataProviderError)
    assert issubclass(FredMissingApiKeyError, MissingProviderCredentialError)
    assert issubclass(FredRequestError, ProviderRequestError)
    assert issubclass(FredRateUnavailableError, ProviderDataUnavailableError)

    assert issubclass(AlpacaProviderError, MarketDataProviderError)
    assert issubclass(AlpacaMissingCredentialsError, MissingProviderCredentialError)
    assert issubclass(AlpacaRequestError, ProviderRequestError)
    assert issubclass(AlpacaDataUnavailableError, ProviderDataUnavailableError)


def test_missing_provider_credential_error_identifies_env_var() -> None:
    error = MissingProviderCredentialError(
        "CUSTOM_API_KEY",
        provider_name="Example",
        credential_name="API key",
    )

    assert error.env_var_name == "CUSTOM_API_KEY"
    assert "CUSTOM_API_KEY" in str(error)
    assert "Example API key" in str(error)
