"""FRED market data provider."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from importlib import import_module
from typing import Any, Final, Literal, cast

from option_pricing.marketdata.config import FredConfig

_OBSERVATIONS_PATH: Final = "series/observations"


class FredProviderError(RuntimeError):
    """Base error for FRED provider failures."""


class FredMissingApiKeyError(FredProviderError):
    """Raised when the configured FRED API key environment variable is missing."""

    def __init__(self, env_var_name: str | None = None) -> None:
        self.env_var_name = env_var_name
        if env_var_name is None:
            message = "Missing FRED API key: api_key is unset or blank."
        else:
            message = (
                "Missing FRED API key: environment variable "
                f"{env_var_name!r} is unset or blank."
            )
        super().__init__(message)


class FredRequestError(FredProviderError):
    """Raised when FRED returns an unusable response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.endpoint = endpoint
        details: list[str] = []
        if status_code is not None:
            details.append(f"status_code={status_code}")
        if endpoint is not None:
            details.append(f"endpoint={endpoint}")
        suffix = f" ({', '.join(details)})" if details else ""
        super().__init__(f"{message}{suffix}")


class FredRateUnavailableError(FredProviderError):
    """Raised when a FRED rate series has no usable observation for an asof."""


@dataclass(frozen=True, slots=True)
class _RequestSpec:
    endpoint: str
    params: dict[str, object]


class FredClient:
    """Small client for the FRED ``fred/series/observations`` endpoint."""

    def __init__(
        self,
        api_key: str,
        *,
        config: FredConfig | None = None,
        session: object | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.config = config or FredConfig()
        self._api_key = _clean_api_key(api_key)
        self._session = session
        self._timeout = _validate_timeout(timeout)

    @classmethod
    def from_env(
        cls,
        config: FredConfig | None = None,
        *,
        session: object | None = None,
        timeout: float = 10.0,
    ) -> FredClient:
        """Create a client from the environment variable named by ``FredConfig``."""

        resolved_config = config or FredConfig()
        api_key = os.environ.get(resolved_config.api_key_env)
        if api_key is None or not api_key.strip():
            raise FredMissingApiKeyError(resolved_config.api_key_env)
        return cls(
            api_key,
            config=resolved_config,
            session=session,
            timeout=timeout,
        )

    @property
    def base_url(self) -> str:
        """Configured FRED API base URL without a trailing slash."""

        return self.config.base_url.rstrip("/")

    @property
    def timeout(self) -> float:
        """Request timeout in seconds."""

        return self._timeout

    def fetch_observations(
        self,
        series_id: str,
        *,
        observation_start: date | datetime | str | None = None,
        observation_end: date | datetime | str | None = None,
        realtime_start: date | datetime | str | None = None,
        realtime_end: date | datetime | str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        sort_order: Literal["asc", "desc"] | str | None = None,
        units: str | None = None,
        frequency: str | None = None,
        aggregation_method: str | None = None,
    ) -> dict[str, Any]:
        """Fetch raw FRED observations for one series as JSON."""

        request = self._observations_request(
            series_id,
            observation_start=observation_start,
            observation_end=observation_end,
            realtime_start=realtime_start,
            realtime_end=realtime_end,
            limit=limit,
            offset=offset,
            sort_order=sort_order,
            units=units,
            frequency=frequency,
            aggregation_method=aggregation_method,
        )
        return self._get_json(request)

    def _observations_request(
        self,
        series_id: str,
        *,
        observation_start: date | datetime | str | None,
        observation_end: date | datetime | str | None,
        realtime_start: date | datetime | str | None,
        realtime_end: date | datetime | str | None,
        limit: int | None,
        offset: int | None,
        sort_order: str | None,
        units: str | None,
        frequency: str | None,
        aggregation_method: str | None,
    ) -> _RequestSpec:
        endpoint = f"{self.base_url}/{_OBSERVATIONS_PATH}"
        params: dict[str, object] = {
            "api_key": self._api_key,
            "file_type": "json",
            "series_id": _clean_required_text(series_id, "series_id"),
        }
        _set_date_param(params, "observation_start", observation_start)
        _set_date_param(params, "observation_end", observation_end)
        _set_date_param(params, "realtime_start", realtime_start)
        _set_date_param(params, "realtime_end", realtime_end)
        _set_int_param(params, "limit", limit)
        _set_int_param(params, "offset", offset)
        _set_text_param(params, "sort_order", sort_order)
        _set_text_param(params, "units", units)
        _set_text_param(params, "frequency", frequency)
        _set_text_param(params, "aggregation_method", aggregation_method)
        return _RequestSpec(endpoint=endpoint, params=params)

    def _get_json(self, request: _RequestSpec) -> dict[str, Any]:
        safe_endpoint = _endpoint_label(request.endpoint)
        getter = self._request_getter(safe_endpoint)

        try:
            response = getter(
                request.endpoint,
                params=request.params,
                timeout=self._timeout,
            )
        except Exception:
            raise FredRequestError(
                "FRED request failed",
                endpoint=safe_endpoint,
            ) from None

        status_code = int(getattr(response, "status_code", 0))
        if status_code != 200:
            raise FredRequestError(
                "FRED request returned a non-200 response",
                status_code=status_code,
                endpoint=safe_endpoint,
            )

        json_method = getattr(response, "json", None)
        if not callable(json_method):
            raise FredRequestError(
                "FRED response did not provide JSON decoding",
                status_code=status_code,
                endpoint=safe_endpoint,
            )

        try:
            payload = json_method()
        except Exception:
            raise FredRequestError(
                "FRED response was not valid JSON",
                status_code=status_code,
                endpoint=safe_endpoint,
            ) from None

        if not isinstance(payload, Mapping):
            raise FredRequestError(
                "FRED response JSON must be an object",
                status_code=status_code,
                endpoint=safe_endpoint,
            )

        return cast(dict[str, Any], dict(payload))

    def _request_getter(self, endpoint: str) -> Callable[..., object]:
        if self._session is None:
            try:
                requests_module = import_module("requests")
            except ModuleNotFoundError:
                raise FredRequestError(
                    "requests is required to fetch FRED observations",
                    endpoint=endpoint,
                ) from None
            getter = getattr(requests_module, "get", None)
        else:
            getter = getattr(self._session, "get", None)

        if not callable(getter):
            raise TypeError("FRED HTTP session must provide a callable get method")

        return cast(Callable[..., object], getter)

    def __repr__(self) -> str:
        return (
            "FredClient("
            f"config={self.config!r}, "
            "api_key=<redacted>, "
            f"timeout={self._timeout!r}"
            ")"
        )


def _clean_api_key(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("api_key must be a string")
    if not value.strip():
        raise FredMissingApiKeyError()
    return value


def _clean_required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    return cleaned


def _clean_optional_text(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _clean_required_text(value, field_name)


def _format_date(value: date | datetime | str, field_name: str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return _clean_required_text(value, field_name)
    raise TypeError(f"{field_name} must be a date, datetime, or string")


def _set_date_param(
    params: dict[str, object],
    name: str,
    value: date | datetime | str | None,
) -> None:
    if value is not None:
        params[name] = _format_date(value, name)


def _set_int_param(
    params: dict[str, object],
    name: str,
    value: int | None,
) -> None:
    if value is not None:
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        params[name] = value


def _set_text_param(
    params: dict[str, object],
    name: str,
    value: str | None,
) -> None:
    cleaned = _clean_optional_text(value, name)
    if cleaned is not None:
        params[name] = cleaned


def _validate_timeout(value: float) -> float:
    if not isinstance(value, int | float):
        raise TypeError("timeout must be numeric")
    timeout = float(value)
    if timeout <= 0:
        raise ValueError("timeout must be > 0")
    return timeout


def _endpoint_label(endpoint: str) -> str:
    return endpoint.split("?", maxsplit=1)[0]


__all__ = [
    "FredClient",
    "FredMissingApiKeyError",
    "FredProviderError",
    "FredRateUnavailableError",
    "FredRequestError",
]
