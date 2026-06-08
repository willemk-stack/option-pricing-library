from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from option_pricing.marketdata import (
    LoadedModelValidationBundle,
    load_model_validation_bundle,
)
from option_pricing.marketdata.bundles import (
    HestonSmokeResult,
    build_model_validation_manifest,
)
from option_pricing.marketdata.gold import GoldMarketDataSnapshot
from option_pricing.types import MarketData

EXPECTED_ARTIFACTS = {
    "market_data": "market_data.json",
    "cleaned_quotes": "cleaned_quotes.parquet",
    "rejected_quotes": "rejected_quotes.parquet",
    "heston_quotes": "heston_quotes.parquet",
    "surface_inputs": "surface_inputs.parquet",
    "heston_fit_summary": "heston_fit_summary.csv",
    "warnings": "warnings.json",
}


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
        "underlying": "SYNTH",
        "valuation_timestamp_utc": "2026-05-22T15:30:00Z",
        "run_id": "test-run",
        "snapshot_id": "snapshot-001",
        "market_data": {
            "spot": 100.0,
            "rate": 0.04,
            "dividend_yield": 0.01,
        },
        "sources": {
            "spot_source": "local_fixture",
            "rate_source": "local_fixture",
            "dividend_yield_source": "assumption",
        },
        "rate_compounding": "continuous",
        "day_count": "ACT/365",
        "quote_cleaning_policy": "quote_cleaning_policy.v1",
        "library_commit": "abc123",
    }


def _manifest() -> dict[str, object]:
    return build_model_validation_manifest(
        run_id="test-run",
        snapshot_id="snapshot-001",
        underlying="SYNTH",
        valuation_timestamp_utc="2026-05-22T15:30:00Z",
        market_data_payload=_market_data_payload(),
        rows={
            "market_inputs": 1,
            "cleaned_quotes": 1,
            "rejected_quotes": 1,
            "heston_quotes": 1,
            "surface_inputs": 1,
        },
        reason_counts={"wide_spread": 1},
        warnings=["synthetic warning"],
        artifacts=EXPECTED_ARTIFACTS,
        heston_smoke=HestonSmokeResult(
            status="skipped",
            message="not run in loader test",
            objective_type="price_rmse",
            quote_count=1,
        ),
        library_commit="abc123",
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_complete_bundle(root: Path) -> Path:
    root.mkdir(parents=True)
    _write_json(root / "manifest.json", _manifest())
    _write_json(root / "market_data.json", _market_data_payload())
    _write_json(
        root / "warnings.json",
        {
            "warnings": ["synthetic warning"],
            "data_quality": ["synthetic data quality"],
            "heston_smoke": ["not run in loader test"],
        },
    )

    frame = pd.DataFrame({"quote_id": ["q1"], "mid": [1.25]})
    for filename in (
        "cleaned_quotes.parquet",
        "rejected_quotes.parquet",
        "heston_quotes.parquet",
        "surface_inputs.parquet",
    ):
        frame.to_parquet(root / filename, index=False)
    pd.DataFrame({"status": ["skipped"]}).to_csv(
        root / "heston_fit_summary.csv",
        index=False,
    )
    return root


def test_load_model_validation_bundle_from_bundle_root(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    root = _write_complete_bundle(tmp_path / "bundle")

    loaded = load_model_validation_bundle(root)

    assert isinstance(loaded, LoadedModelValidationBundle)
    assert loaded.root == root
    assert loaded.manifest_path == root / "manifest.json"
    assert loaded.manifest["run_id"] == "test-run"
    assert loaded.warnings["warnings"] == ["synthetic warning"]
    assert isinstance(loaded.market_snapshot, GoldMarketDataSnapshot)
    assert isinstance(loaded.market_data, MarketData)
    assert loaded.market_data is loaded.market_snapshot.market_data
    assert loaded.market_data.spot == 100.0
    assert loaded.cleaned_quotes.equals(
        pd.DataFrame({"quote_id": ["q1"], "mid": [1.25]})
    )
    assert loaded.rejected_quotes.equals(loaded.cleaned_quotes)
    assert loaded.heston_quotes.equals(loaded.cleaned_quotes)
    assert loaded.surface_inputs.equals(loaded.cleaned_quotes)
    assert loaded.heston_fit_summary.equals(pd.DataFrame({"status": ["skipped"]}))


def test_load_model_validation_bundle_from_manifest_path(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    root = _write_complete_bundle(tmp_path / "bundle")

    loaded = load_model_validation_bundle(root / "manifest.json")

    assert loaded.root == root
    assert loaded.manifest_path == root / "manifest.json"


def test_missing_manifest_raises_clear_error(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()

    with pytest.raises(FileNotFoundError) as exc_info:
        load_model_validation_bundle(root)

    message = str(exc_info.value)
    assert "manifest.json" in message
    assert "complete model-validation bundle" in message
    assert "load_model_validation_bundle(path)" in message


def test_missing_required_artifact_raises_clear_error(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    root = _write_complete_bundle(tmp_path / "bundle")
    (root / "surface_inputs.parquet").unlink()

    with pytest.raises(FileNotFoundError) as exc_info:
        load_model_validation_bundle(root)

    message = str(exc_info.value)
    assert "surface_inputs.parquet" in message
    assert "complete model-validation bundle" in message
    assert "load_model_validation_bundle(path)" in message


def test_invalid_json_raises_clear_error(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    root = _write_complete_bundle(tmp_path / "bundle")
    (root / "manifest.json").write_text("{", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"manifest\.json.*complete model-validation bundle.*load_model_validation_bundle",
    ):
        load_model_validation_bundle(root)


def test_market_data_json_rehydrates_snapshot_and_marketdata(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    root = _write_complete_bundle(tmp_path / "bundle")

    loaded = load_model_validation_bundle(root)

    assert isinstance(loaded.market_snapshot, GoldMarketDataSnapshot)
    assert loaded.market_snapshot.metadata["schema_version"] == "gold_market_data.v1"
    assert loaded.market_snapshot.metadata["underlying"] == "SYNTH"
    assert loaded.market_data == MarketData(
        spot=100.0,
        rate=0.04,
        dividend_yield=0.01,
    )
