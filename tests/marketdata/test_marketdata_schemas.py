from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from option_pricing.marketdata.contracts import RunMetadata
from option_pricing.marketdata.manifests import (
    MODEL_VALIDATION_MANIFEST_REQUIRED_FIELDS,
    validate_model_validation_manifest,
)
from option_pricing.marketdata.schemas import (
    CLEANED_QUOTES_COLUMNS,
    MODEL_VALIDATION_BUNDLE_VERSION,
    DatasetName,
)
from option_pricing.marketdata.validation import (
    coerce_frame,
    dataset_columns,
    dataset_dtypes,
    validate_columns,
    validate_dtypes,
)


def _cleaned_quote_record() -> dict[str, object]:
    return {
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


def _valid_manifest() -> dict[str, object]:
    return {
        "artifact_schema_version": MODEL_VALIDATION_BUNDLE_VERSION,
        "run_id": "test-run",
        "snapshot_id": "snapshot-001",
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
        "rows": {"cleaned_quotes": 1},
        "reason_counts": {},
        "warnings": [],
        "artifacts": {"cleaned_quotes": "silver/cleaned_quotes"},
        "heston_smoke": {
            "status": "skipped",
            "message": "not run in A5-S1",
            "objective_type": "price_rmse",
            "quote_count": 1,
        },
    }


def test_phase_a_dataset_contracts_are_registered() -> None:
    expected_names = {
        "market_inputs",
        "cleaned_quotes",
        "rejected_quotes",
        "heston_quotes",
        "surface_inputs",
        "model_validation_bundle",
    }

    assert expected_names.issubset({dataset.value for dataset in DatasetName})
    assert dataset_columns("cleaned_quotes") == CLEANED_QUOTES_COLUMNS
    assert dataset_columns("model_validation_bundle") == ()
    assert dataset_dtypes("model_validation_bundle") == {}
    assert dataset_dtypes("market_inputs")["rate_compounding"] == "string"


def test_unknown_dataset_name_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown marketdata dataset_name"):
        dataset_columns("made_up_dataset")


def test_validate_columns_reports_missing_and_rejects_extra_when_strict() -> None:
    frame = pd.DataFrame([_cleaned_quote_record()])

    validate_columns(frame, "cleaned_quotes")
    validate_columns(frame.assign(extra_field="ok"), "cleaned_quotes")

    with pytest.raises(ValueError, match="unexpected extra columns"):
        validate_columns(
            frame.assign(extra_field="nope"),
            "cleaned_quotes",
            allow_extra=False,
        )

    with pytest.raises(ValueError, match="missing required columns"):
        validate_columns(frame.drop(columns=["quote_id"]), "cleaned_quotes")


def test_coerce_frame_handles_phase_a_dtypes() -> None:
    frame = pd.DataFrame([_cleaned_quote_record()])

    coerced = coerce_frame(frame, "cleaned_quotes")

    validate_dtypes(coerced, "cleaned_quotes")
    assert str(coerced["quote_ts"].dtype) == "datetime64[ns, UTC]"
    assert str(coerced["expiry"].dtype) == "datetime64[ns]"
    assert str(coerced["open_interest"].dtype) == "Int64"
    assert str(coerced["mid"].dtype) == "Float64"


def test_run_metadata_requires_aware_datetimes() -> None:
    with pytest.raises(ValueError, match="asof must be timezone-aware"):
        RunMetadata(
            run_id="test-run",
            asof=datetime(2026, 5, 22, 14, 31),
            started_at=datetime(2026, 5, 22, 14, 32, tzinfo=UTC),
        )


def test_model_validation_manifest_contract() -> None:
    validate_model_validation_manifest(_valid_manifest())

    missing = _valid_manifest()
    missing.pop("artifacts")
    with pytest.raises(ValueError, match="missing required fields"):
        validate_model_validation_manifest(missing)

    wrong_version = _valid_manifest()
    wrong_version["artifact_schema_version"] = "model_validation_bundle.v0"
    with pytest.raises(ValueError, match="artifact_schema_version"):
        validate_model_validation_manifest(wrong_version)

    secret = _valid_manifest()
    secret["ALPACA_API_KEY"] = "not-for-manifests"
    with pytest.raises(ValueError, match="secret-looking keys"):
        validate_model_validation_manifest(secret)

    assert MODEL_VALIDATION_MANIFEST_REQUIRED_FIELDS == (
        "artifact_schema_version",
        "run_id",
        "snapshot_id",
        "created_at_utc",
        "library_commit",
        "underlying",
        "valuation_timestamp_utc",
        "spot_source",
        "rate_source",
        "rate_compounding",
        "dividend_yield_source",
        "day_count",
        "quote_cleaning_policy",
        "rows",
        "reason_counts",
        "warnings",
        "artifacts",
        "heston_smoke",
    )
