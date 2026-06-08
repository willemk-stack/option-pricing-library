from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any

import pytest

from option_pricing.marketdata.config import FredConfig
from option_pricing.marketdata.providers.fred import (
    FredClient,
    FredMissingApiKeyError,
    FredRequestError,
)


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: object | None = None,
        json_error: Exception | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self.payload = {} if payload is None else payload
        self.json_error = json_error
        self.text = text

    def json(self) -> object:
        if self.json_error is not None:
            raise self.json_error
        return self.payload


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, object],
        timeout: float,
    ) -> _FakeResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return self.response


def test_fred_client_from_env_missing_key_raises_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    with pytest.raises(FredMissingApiKeyError, match="FRED_API_KEY") as excinfo:
        FredClient.from_env()

    assert excinfo.value.env_var_name == "FRED_API_KEY"


def test_fred_client_from_env_uses_configured_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUSTOM_FRED_API_KEY", "custom-secret-key")

    client = FredClient.from_env(FredConfig(api_key_env="CUSTOM_FRED_API_KEY"))

    assert "custom-secret-key" not in repr(client)
    assert "api_key=<redacted>" in repr(client)


def test_fred_fetch_observations_sends_expected_request_params() -> None:
    session = _FakeSession(_FakeResponse(payload={"observations": []}))
    client = FredClient(
        "fred-secret",
        config=FredConfig(base_url="https://example.test/fred/"),
        session=session,
        timeout=3.5,
    )

    payload = client.fetch_observations(
        "DGS3MO",
        observation_start=date(2026, 1, 1),
        observation_end="2026-01-31",
        realtime_start=datetime(2026, 1, 2, 12, 0, 0),
        realtime_end="2026-02-01",
        limit=100,
        offset=5,
        sort_order="asc",
        units="lin",
        frequency="d",
        aggregation_method="avg",
    )

    assert payload == {"observations": []}
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == "https://example.test/fred/series/observations"
    assert call["timeout"] == pytest.approx(3.5)
    assert call["params"] == {
        "api_key": "fred-secret",
        "file_type": "json",
        "series_id": "DGS3MO",
        "observation_start": "2026-01-01",
        "observation_end": "2026-01-31",
        "realtime_start": "2026-01-02",
        "realtime_end": "2026-02-01",
        "limit": 100,
        "offset": 5,
        "sort_order": "asc",
        "units": "lin",
        "frequency": "d",
        "aggregation_method": "avg",
    }


def test_fred_non_200_response_raises_without_leaking_secret() -> None:
    secret = "fred-secret-value"
    session = _FakeSession(
        _FakeResponse(
            status_code=429,
            text=f"rate limited; secret was {secret}",
        )
    )
    client = FredClient(secret, session=session)

    with pytest.raises(FredRequestError) as excinfo:
        client.fetch_observations("DGS3MO")

    assert excinfo.value.status_code == 429
    details = f"{excinfo.value!s} {excinfo.value!r} {client!r}"
    assert "non-200" in str(excinfo.value)
    assert secret not in details


def test_fred_malformed_json_raises_without_leaking_secret() -> None:
    secret = "fred-secret-value"
    session = _FakeSession(
        _FakeResponse(
            json_error=ValueError(f"invalid json near {secret}"),
        )
    )
    client = FredClient(secret, session=session)

    with pytest.raises(FredRequestError, match="not valid JSON") as excinfo:
        client.fetch_observations("DGS3MO")

    details = f"{excinfo.value!s} {excinfo.value!r} {client!r}"
    assert secret not in details


@pytest.mark.skipif(
    not os.environ.get("FRED_API_KEY"),
    reason="FRED_API_KEY is not set",
)
def test_fred_live_fetch_observations_smoke() -> None:
    payload = FredClient.from_env().fetch_observations("DGS3MO", limit=1)

    assert "observations" in payload
