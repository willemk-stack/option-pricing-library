from __future__ import annotations

import pandas as pd
import pytest

from option_pricing.marketdata.config import StorageConfig
from option_pricing.marketdata.schemas import MODEL_VALIDATION_BUNDLE_VERSION
from option_pricing.marketdata.storage import LocalStorage
from option_pricing.marketdata.validation import coerce_frame, validate_columns


@pytest.fixture
def fake_parquet(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_to_parquet(
        self: pd.DataFrame,
        path: str,
        compression: str | None = None,
        index: bool = False,
    ) -> None:
        del compression
        payload = self if index else self.reset_index(drop=True)
        payload.to_pickle(path)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet)


def _cleaned_quotes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "underlying": "SPY",
                "contract_symbol": "SPY260619C00400000",
                "quote_id": "quote-001",
                "quote_ts": "2026-05-22T14:30:00Z",
                "asof": "2026-05-22T14:31:00Z",
                "expiry": "2026-06-19",
                "expiry_years": "0.0767",
                "strike": "400",
                "right": "call",
                "bid": "12.30",
                "ask": "12.50",
                "mid": "12.40",
                "iv": "0.21",
                "vega": "0.35",
                "delta": "0.52",
                "gamma": "0.03",
                "theta": "-0.02",
                "rho": "0.01",
                "open_interest": "42",
                "moneyness": "1.01",
                "source": "fixture",
                "cleaning_policy": "phase_a_default",
            }
        ]
    )


def _rejected_quotes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "underlying": "SPY",
                "contract_symbol": "SPY260619P00300000",
                "quote_id": "quote-002",
                "quote_ts": "2026-05-22T14:30:00Z",
                "asof": "2026-05-22T14:31:00Z",
                "expiry": "2026-06-19",
                "strike": "300",
                "right": "put",
                "bid": "0.00",
                "ask": "0.01",
                "mid": "0.005",
                "iv": "0.95",
                "vega": "0.01",
                "source": "fixture",
                "rejection_reason": "wide_or_stale",
                "rejection_detail": "synthetic rejection for contract test",
                "cleaning_policy": "phase_a_default",
            }
        ]
    )


def _heston_quotes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "underlying": "SPY",
                "contract_symbol": "SPY260619C00400000",
                "quote_id": "quote-001",
                "asof": "2026-05-22T14:31:00Z",
                "expiry": "2026-06-19",
                "expiry_years": "0.0767",
                "strike": "400",
                "right": "call",
                "mid": "12.40",
                "bid": "12.30",
                "ask": "12.50",
                "iv": "0.21",
                "vega": "0.35",
                "option_type": "call",
                "label": "SPY 2026-06-19 400C",
                "source": "fixture",
                "cleaning_policy": "phase_a_default",
            }
        ]
    )


def test_a1_contracts_and_storage_support_model_validation_layout(
    tmp_path,
    fake_parquet: None,
) -> None:
    storage = LocalStorage(StorageConfig(root=tmp_path))
    partitions = {
        "underlying": "SPY",
        "date": "2026-05-22",
        "run_id": "test-run",
    }

    cleaned = coerce_frame(_cleaned_quotes(), "cleaned_quotes")
    rejected = coerce_frame(_rejected_quotes(), "rejected_quotes")
    heston = coerce_frame(_heston_quotes(), "heston_quotes")

    validate_columns(cleaned, "cleaned_quotes")
    validate_columns(rejected, "rejected_quotes")
    validate_columns(heston, "heston_quotes")

    cleaned_path = storage.write_frame(
        cleaned,
        dataset="cleaned_quotes",
        layer="silver",
        partitions=partitions,
        filename="cleaned_quotes",
    )
    rejected_path = storage.write_frame(
        rejected,
        dataset="rejected_quotes",
        layer="silver",
        partitions=partitions,
        filename="rejected_quotes",
    )
    heston_path = storage.write_frame(
        heston,
        dataset="heston_quotes",
        layer="gold",
        partitions=partitions,
        filename="heston_quotes",
    )

    manifest_path = storage.write_manifest(
        {
            "artifact_schema_version": MODEL_VALIDATION_BUNDLE_VERSION,
            "run_id": "test-run",
            "created_at_utc": "2026-05-22T14:35:00Z",
            "library_commit": "abc123",
            "underlying": "SPY",
            "valuation_timestamp_utc": "2026-05-22T14:31:00Z",
            "spot_source": "fixture",
            "rate_source": "fixture",
            "rate_compounding": "continuous",
            "dividend_yield_source": "fixture",
            "day_count": "ACT/365",
            "quote_cleaning_policy": "phase_a_default",
            "rows": {
                "cleaned_quotes": int(len(cleaned)),
                "rejected_quotes": int(len(rejected)),
                "heston_quotes": int(len(heston)),
            },
            "warnings": [],
            "artifacts": {
                "cleaned_quotes": cleaned_path.as_posix(),
                "rejected_quotes": rejected_path.as_posix(),
                "heston_quotes": heston_path.as_posix(),
            },
        },
        dataset="model_validation_bundle",
        layer="gold",
        partitions=partitions,
    )

    assert cleaned_path.exists()
    assert rejected_path.exists()
    assert heston_path.exists()
    assert manifest_path.exists()
    assert cleaned_path.parent.parts[-3:] == (
        "underlying=SPY",
        "date=2026-05-22",
        "run_id=test-run",
    )
    assert heston_path.parent.parts[-3:] == (
        "underlying=SPY",
        "date=2026-05-22",
        "run_id=test-run",
    )
    assert manifest_path.name == "manifest.json"
