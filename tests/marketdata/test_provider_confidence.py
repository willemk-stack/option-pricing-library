from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from option_pricing.marketdata.gold import build_heston_quotes
from option_pricing.marketdata.provider_confidence import (
    ProviderSnapshotBundleValidationResult,
    validate_provider_snapshot_bundle,
)
from option_pricing.marketdata.schemas import DatasetName
from option_pricing.marketdata.validation import coerce_frame

ASOF = "2026-05-22T15:30:00Z"


@pytest.fixture
def fake_parquet(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_to_parquet(
        self: pd.DataFrame,
        path: str | Path,
        compression: str | None = None,
        index: bool = False,
    ) -> None:
        del compression
        payload = self if index else self.reset_index(drop=True)
        payload.to_pickle(path)

    def _fake_read_parquet(
        path: str | Path,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        frame = cast(pd.DataFrame, pd.read_pickle(path))
        if columns is None:
            return frame
        return cast(pd.DataFrame, frame.loc[:, columns])

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet)
    monkeypatch.setattr(pd, "read_parquet", _fake_read_parquet)


def _market_data_payload() -> dict[str, object]:
    return {
        "schema_version": "gold_market_data.v1",
        "underlying": "SPY",
        "valuation_timestamp_utc": ASOF,
        "run_id": "provider-confidence",
        "snapshot_id": "provider-confidence:SPY:2026-05-22",
        "market_data": {
            "spot": 500.0,
            "rate": 0.04162167469081947,
            "dividend_yield": 0.0,
        },
        "sources": {
            "spot_source": "fake_alpaca",
            "rate_source": "fake_fred",
            "dividend_yield_source": "zero_assumption",
        },
        "rate_compounding": "continuous",
        "day_count": "ACT/365",
        "quote_cleaning_policy": "quote_cleaning_policy.v1",
        "library_commit": "abc123",
    }


def _cleaned_quotes(
    *,
    underlying: str = "SPY",
    row_overrides: dict[str, object] | None = None,
) -> pd.DataFrame:
    row: dict[str, object] = {
        "underlying": underlying,
        "contract_symbol": f"{underlying}260619C00500000",
        "quote_id": "quote-call-500",
        "quote_ts": ASOF,
        "asof": ASOF,
        "expiry": "2026-06-19",
        "expiry_years": 28.0 / 365.0,
        "strike": 500.0,
        "right": "call",
        "bid": 10.0,
        "ask": 10.5,
        "mid": 10.25,
        "spread": 0.5,
        "relative_spread": 0.5 / 10.25,
        "iv": 0.2,
        "vega": 0.11,
        "delta": 0.51,
        "gamma": 0.02,
        "theta": -0.03,
        "rho": 0.04,
        "open_interest": 250,
        "moneyness": 1.0,
        "log_moneyness": 0.0,
        "time_to_expiry_years": 28.0 / 365.0,
        "option_price_for_model": 10.25,
        "mid_computed": False,
        "time_to_expiry_computed": True,
        "moneyness_computed": True,
        "provider_iv_available": True,
        "provider_greeks_available": True,
        "model_price_available": True,
        "model_validation_ready": True,
        "iv_validation_ready": True,
        "greek_validation_ready": True,
        "source": "fake_alpaca",
        "cleaning_policy": "quote_cleaning_policy.v1",
    }
    row.update(row_overrides or {})
    return coerce_frame(pd.DataFrame([row]), DatasetName.CLEANED_QUOTES)


def _heston_quotes(cleaned_quotes: pd.DataFrame | None = None) -> pd.DataFrame:
    return build_heston_quotes(
        _cleaned_quotes() if cleaned_quotes is None else cleaned_quotes
    ).heston_quotes


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_provider_artifacts(
    root: Path,
    *,
    market_payload: object | None = None,
    cleaned_quotes: pd.DataFrame | None = None,
    heston_quotes: pd.DataFrame | None = None,
) -> dict[str, Path]:
    root.mkdir(parents=True)
    paths = {
        "market_data": root / "market_data.json",
        "cleaned_quotes": root / "cleaned_quotes.parquet",
        "heston_quotes": root / "heston_quotes.parquet",
    }
    cleaned = _cleaned_quotes() if cleaned_quotes is None else cleaned_quotes
    _write_json(
        paths["market_data"],
        _market_data_payload() if market_payload is None else market_payload,
    )
    cleaned.to_parquet(paths["cleaned_quotes"], index=False)
    (_heston_quotes(cleaned) if heston_quotes is None else heston_quotes).to_parquet(
        paths["heston_quotes"],
        index=False,
    )
    return paths


def _validate(paths: dict[str, Path]) -> ProviderSnapshotBundleValidationResult:
    return validate_provider_snapshot_bundle(
        market_data_path=paths["market_data"],
        cleaned_quotes_path=paths["cleaned_quotes"],
        heston_quotes_path=paths["heston_quotes"],
    )


def test_provider_snapshot_bundle_validation_result_shape_is_stable() -> None:
    assert tuple(
        field.name for field in fields(ProviderSnapshotBundleValidationResult)
    ) == (
        "underlying",
        "cleaned_quote_count",
        "heston_quote_count",
        "spot",
        "rate",
        "dividend_yield",
    )


def test_validate_provider_snapshot_bundle_success_on_synthetic_artifacts(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    paths = _write_provider_artifacts(tmp_path / "bundle")

    result = _validate(paths)

    assert isinstance(result, ProviderSnapshotBundleValidationResult)
    assert result.underlying == "SPY"
    assert result.cleaned_quote_count == 1
    assert result.heston_quote_count == 1
    assert result.spot == pytest.approx(500.0)
    assert result.rate == pytest.approx(0.04162167469081947)
    assert result.dividend_yield == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("unreadable_market_data", "Unable to read provider market_data.json"),
        ("invalid_market_data", "provider market_data.json is not compatible"),
        ("unreadable_cleaned_quotes", "Unable to read provider cleaned_quotes"),
        ("unreadable_heston_quotes", "Unable to read provider heston_quotes"),
        ("schema_mismatch", "provider cleaned_quotes artifact does not match"),
        ("underlying_mismatch", "cleaned_quotes and heston_quotes underlyings differ"),
        ("empty_cleaned_quotes", "provider cleaned_quotes artifact is empty"),
        ("empty_heston_quotes", "provider heston_quotes artifact is empty"),
        ("quote_set_reconstruction_failure", "could not be consumed"),
    ],
)
def test_validate_provider_snapshot_bundle_failure_modes_are_clear(
    tmp_path: Path,
    fake_parquet: None,
    case: str,
    match: str,
) -> None:
    paths = _write_provider_artifacts(tmp_path / "bundle")

    if case == "unreadable_market_data":
        paths["market_data"].write_text("{", encoding="utf-8")
    elif case == "invalid_market_data":
        _write_json(paths["market_data"], {"schema_version": "not_gold"})
    elif case == "unreadable_cleaned_quotes":
        paths["cleaned_quotes"].write_text("not a parquet artifact", encoding="utf-8")
    elif case == "unreadable_heston_quotes":
        paths["heston_quotes"].write_text("not a parquet artifact", encoding="utf-8")
    elif case == "schema_mismatch":
        pd.DataFrame({"underlying": ["SPY"]}).to_parquet(
            paths["cleaned_quotes"],
            index=False,
        )
    elif case == "underlying_mismatch":
        _heston_quotes(_cleaned_quotes(underlying="QQQ")).to_parquet(
            paths["heston_quotes"],
            index=False,
        )
    elif case == "empty_cleaned_quotes":
        _cleaned_quotes().iloc[0:0].to_parquet(paths["cleaned_quotes"], index=False)
    elif case == "empty_heston_quotes":
        _heston_quotes().iloc[0:0].to_parquet(paths["heston_quotes"], index=False)
    elif case == "quote_set_reconstruction_failure":
        heston = _heston_quotes()
        heston.loc[0, "option_type"] = "put"
        heston.to_parquet(paths["heston_quotes"], index=False)
    else:  # pragma: no cover - parametrization guard
        raise AssertionError(f"unhandled case {case!r}")

    with pytest.raises(ValueError, match=match):
        _validate(paths)
