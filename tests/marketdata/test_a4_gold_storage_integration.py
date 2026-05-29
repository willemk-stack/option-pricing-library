from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from option_pricing.marketdata.cleaning import clean_option_quotes
from option_pricing.marketdata.config import StorageConfig
from option_pricing.marketdata.gold import (
    GOLD_CONVERSION_MANIFEST_VERSION,
    GOLD_MARKET_DATA_SCHEMA_VERSION,
    build_heston_quotes,
    heston_quote_set_from_frame,
    market_data_snapshot_from_json,
    write_gold_artifacts,
    write_heston_quotes_gold,
)
from option_pricing.marketdata.normalize import (
    normalize_market_inputs,
    normalize_option_chain,
)
from option_pricing.marketdata.schemas import (
    HESTON_QUOTES_COLUMNS,
    HESTON_QUOTES_SCHEMA_VERSION,
    DatasetName,
)
from option_pricing.marketdata.storage import LocalStorage
from option_pricing.marketdata.validation import validate_dtypes
from option_pricing.types import MarketData

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "local_snapshot_synth_schema_v1"
FIXTURE_NAME = "local_snapshot_synth_schema_v1"


@dataclass(frozen=True, slots=True)
class _LocalSnapshotStub:
    fixture_name: str
    snapshot_id: str
    run_id: str | None
    underlying: str
    asof: pd.Timestamp


@dataclass(frozen=True, slots=True)
class _A3Outputs:
    local_snapshot: _LocalSnapshotStub
    market_inputs: pd.DataFrame
    cleaned_quotes: pd.DataFrame
    rejected_quotes: pd.DataFrame
    reason_counts: dict[str, int]
    warnings: tuple[str, ...]


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

    def _fake_read_parquet(
        path: str,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        frame = pd.read_pickle(path)
        if columns is None:
            return frame
        return frame.loc[:, columns]

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet)
    monkeypatch.setattr(pd, "read_parquet", _fake_read_parquet)


def _a3_outputs(run_id: str | None = "test-run") -> _A3Outputs:
    market_inputs = normalize_market_inputs(
        pd.read_csv(FIXTURE_ROOT / "market_inputs.csv")
    )
    option_chain = normalize_option_chain(
        pd.read_csv(FIXTURE_ROOT / "option_chain.csv")
    )
    result = clean_option_quotes(option_chain, market_inputs)
    asof = pd.Timestamp(market_inputs.iloc[0]["asof"])
    local_snapshot = _LocalSnapshotStub(
        fixture_name=FIXTURE_NAME,
        snapshot_id=f"{FIXTURE_NAME}:SYNTH:{asof.isoformat()}",
        run_id=run_id,
        underlying="SYNTH",
        asof=asof,
    )
    return _A3Outputs(
        local_snapshot=local_snapshot,
        market_inputs=market_inputs,
        cleaned_quotes=result.cleaned_quotes,
        rejected_quotes=result.rejected_quotes,
        reason_counts=result.reason_counts,
        warnings=result.warnings,
    )


def _storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(StorageConfig(root=tmp_path))


def _partitions() -> dict[str, str | date]:
    return {
        "underlying": "SYNTH",
        "date": date(2026, 5, 22),
        "run_id": "test-run",
    }


def _write_gold(
    storage: LocalStorage,
    outputs: _A3Outputs,
    *,
    overwrite: bool = False,
    library_commit: str | None = "abc123",
):
    return write_gold_artifacts(
        storage,
        local_snapshot=outputs.local_snapshot,
        market_inputs=outputs.market_inputs,
        cleaned_quotes=outputs.cleaned_quotes,
        rejected_quotes=outputs.rejected_quotes,
        reason_counts=outputs.reason_counts,
        warnings=outputs.warnings,
        overwrite=overwrite,
        library_commit=library_commit,
    )


def _gold_path(
    root: Path,
    dataset: str,
    filename: str,
) -> Path:
    return (
        root
        / "gold"
        / dataset
        / "underlying=SYNTH"
        / "date=2026-05-22"
        / "run_id=test-run"
        / filename
    )


def _read_gold_json(
    storage: LocalStorage,
    *,
    dataset: DatasetName,
    filename: str,
) -> dict[str, object]:
    return storage.read_json(
        dataset=dataset.value,
        layer="gold",
        partitions=_partitions(),
        filename=filename,
    )


def test_write_heston_quotes_gold_writes_expected_path(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    storage = _storage(tmp_path)
    heston_quotes = build_heston_quotes(_a3_outputs().cleaned_quotes).heston_quotes

    path = write_heston_quotes_gold(
        storage,
        heston_quotes=heston_quotes,
        partitions=_partitions(),
    )

    assert path == _gold_path(
        tmp_path,
        DatasetName.HESTON_QUOTES.value,
        "heston_quotes.parquet",
    )
    assert path.exists()

    read_back = storage.read_frame(
        dataset=DatasetName.HESTON_QUOTES.value,
        layer="gold",
        partitions=_partitions(),
        columns=list(HESTON_QUOTES_COLUMNS),
    )
    pd.testing.assert_frame_equal(
        read_back.reset_index(drop=True),
        heston_quotes.reset_index(drop=True),
    )


def test_write_heston_quotes_gold_rejects_non_reconstructable_conventions(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    storage = _storage(tmp_path)
    heston_quotes = build_heston_quotes(_a3_outputs().cleaned_quotes).heston_quotes
    heston_quotes.loc[0, "label"] = "wrong-label"

    with pytest.raises(ValueError, match="label must match contract_symbol"):
        write_heston_quotes_gold(
            storage,
            heston_quotes=heston_quotes,
            partitions=_partitions(),
        )


def test_write_gold_artifacts_writes_paths_and_readback_is_usable(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    storage = _storage(tmp_path)
    outputs = _a3_outputs()

    paths = _write_gold(storage, outputs)

    assert paths.market_data == _gold_path(
        tmp_path,
        DatasetName.MARKET_SNAPSHOT.value,
        "market_data.json",
    )
    assert paths.market_manifest == _gold_path(
        tmp_path,
        DatasetName.MARKET_SNAPSHOT.value,
        "manifest.json",
    )
    assert paths.heston_quotes == _gold_path(
        tmp_path,
        DatasetName.HESTON_QUOTES.value,
        "heston_quotes.parquet",
    )
    assert paths.heston_manifest == _gold_path(
        tmp_path,
        DatasetName.HESTON_QUOTES.value,
        "manifest.json",
    )
    assert all(
        path.exists()
        for path in (
            paths.market_data,
            paths.market_manifest,
            paths.heston_quotes,
            paths.heston_manifest,
        )
    )

    market_payload = _read_gold_json(
        storage,
        dataset=DatasetName.MARKET_SNAPSHOT,
        filename="market_data.json",
    )
    snapshot = market_data_snapshot_from_json(market_payload)
    assert snapshot.market_data == MarketData(
        spot=100.0,
        rate=0.04,
        dividend_yield=0.0,
    )
    context = snapshot.market_data.to_context()
    assert context.spot == pytest.approx(100.0)
    assert context.df(1.0) == pytest.approx(0.9607894391523232)

    heston_quotes = storage.read_frame(
        dataset=DatasetName.HESTON_QUOTES.value,
        layer="gold",
        partitions=_partitions(),
        columns=list(HESTON_QUOTES_COLUMNS),
    )
    assert tuple(heston_quotes.columns) == HESTON_QUOTES_COLUMNS
    validate_dtypes(heston_quotes, DatasetName.HESTON_QUOTES, allow_extra=False)
    quote_set = heston_quote_set_from_frame(heston_quotes, snapshot.market_data)
    assert quote_set.n_quotes == len(heston_quotes)
    assert quote_set.metadata is not None
    assert quote_set.metadata["schema_version"] == HESTON_QUOTES_SCHEMA_VERSION


def test_gold_manifests_summarize_outputs_without_rejected_rows(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    storage = _storage(tmp_path)
    outputs = _a3_outputs()

    _write_gold(storage, outputs)

    market_manifest = _read_gold_json(
        storage,
        dataset=DatasetName.MARKET_SNAPSHOT,
        filename="manifest.json",
    )
    heston_manifest = _read_gold_json(
        storage,
        dataset=DatasetName.HESTON_QUOTES,
        filename="manifest.json",
    )

    assert market_manifest["conversion_manifest_version"] == (
        GOLD_CONVERSION_MANIFEST_VERSION
    )
    assert heston_manifest["conversion_manifest_version"] == (
        GOLD_CONVERSION_MANIFEST_VERSION
    )
    assert market_manifest["artifact_schema_version"] == GOLD_MARKET_DATA_SCHEMA_VERSION
    assert heston_manifest["artifact_schema_version"] == HESTON_QUOTES_SCHEMA_VERSION
    assert market_manifest["artifact"] == "market_data"
    assert heston_manifest["artifact"] == "heston_quotes"
    assert market_manifest["library_commit"] == "abc123"
    assert heston_manifest["library_commit"] == "abc123"
    assert market_manifest["quote_cleaning_policy"] == "quote_cleaning_policy.v1"
    assert heston_manifest["quote_cleaning_policy"] == "quote_cleaning_policy.v1"
    assert market_manifest["rate_compounding"] == "continuous"
    assert market_manifest["day_count"] == "ACT/365"
    for manifest in (market_manifest, heston_manifest):
        assert manifest["run_id"] == "test-run"
        assert manifest["snapshot_id"] == outputs.local_snapshot.snapshot_id
        assert manifest["underlying"] == "SYNTH"
        assert manifest["valuation_timestamp_utc"] == "2026-05-22T15:30:00Z"
        assert manifest["reason_counts"] == outputs.reason_counts
        assert manifest["warnings"] == list(outputs.warnings)
        assert manifest["source"] == {
            "source_type": "local_fixture",
            "fixture_name": FIXTURE_NAME,
        }
        serialized = json.dumps(manifest, sort_keys=True)
        assert "rejection_detail" not in serialized
        assert "rejected_quote_rows" not in serialized

    assert market_manifest["row_counts"] == {
        "market_inputs": len(outputs.market_inputs),
        "cleaned_quotes": len(outputs.cleaned_quotes),
        "rejected_quotes": len(outputs.rejected_quotes),
    }
    assert heston_manifest["row_counts"] == {
        "cleaned_quotes": len(outputs.cleaned_quotes),
        "rejected_quotes": len(outputs.rejected_quotes),
        "heston_quotes": len(outputs.cleaned_quotes),
    }
    assert market_manifest["artifacts"] == {"market_data": "market_data.json"}
    assert heston_manifest["artifacts"] == {"heston_quotes": "heston_quotes.parquet"}
    assert "rejected_quotes" not in market_manifest["artifacts"]
    assert "rejected_quotes" not in heston_manifest["artifacts"]
    assert market_manifest["spot"] == 100.0
    assert market_manifest["rate"] == 0.04
    assert market_manifest["dividend_yield"] == 0.0
    assert market_manifest["sources"] == {
        "spot_source": "local_fixture",
        "rate_source": "local_fixture",
        "dividend_yield_source": "assumption",
    }
    assert heston_manifest["optional_data_warnings"] == []
    assert "every IV is finite and > 0" in heston_manifest["iv_mid_policy"]
    assert "every vega is finite and >= 0" in heston_manifest["bs_vega_policy"]


def test_write_gold_artifacts_requires_run_id(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    storage = _storage(tmp_path)
    outputs = _a3_outputs(run_id=None)

    with pytest.raises(ValueError, match="run_id"):
        _write_gold(storage, outputs)


def test_write_gold_artifacts_requires_matching_market_inputs_asof(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    storage = _storage(tmp_path)
    outputs = _a3_outputs()
    market_inputs = outputs.market_inputs.copy()
    market_inputs.loc[0, "asof"] = pd.Timestamp("2026-05-23T15:30:00Z")

    with pytest.raises(ValueError, match="asof.*match"):
        write_gold_artifacts(
            storage,
            local_snapshot=outputs.local_snapshot,
            market_inputs=market_inputs,
            cleaned_quotes=outputs.cleaned_quotes,
            rejected_quotes=outputs.rejected_quotes,
            reason_counts=outputs.reason_counts,
            warnings=outputs.warnings,
        )


def test_write_gold_artifacts_requires_matching_underlying(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    storage = _storage(tmp_path)
    outputs = _a3_outputs()
    local_snapshot = replace(outputs.local_snapshot, underlying="OTHER")

    with pytest.raises(ValueError, match="underlying.*match"):
        write_gold_artifacts(
            storage,
            local_snapshot=local_snapshot,
            market_inputs=outputs.market_inputs,
            cleaned_quotes=outputs.cleaned_quotes,
            rejected_quotes=outputs.rejected_quotes,
            reason_counts=outputs.reason_counts,
            warnings=outputs.warnings,
        )


@pytest.mark.parametrize(
    ("dataset", "filename"),
    [
        (DatasetName.MARKET_SNAPSHOT.value, "market_data.json"),
        (DatasetName.MARKET_SNAPSHOT.value, "manifest.json"),
        (DatasetName.HESTON_QUOTES.value, "heston_quotes.parquet"),
        (DatasetName.HESTON_QUOTES.value, "manifest.json"),
    ],
)
def test_write_gold_artifacts_preflights_all_targets(
    tmp_path: Path,
    fake_parquet: None,
    dataset: str,
    filename: str,
) -> None:
    storage = _storage(tmp_path)
    outputs = _a3_outputs()
    existing_target = _gold_path(tmp_path, dataset, filename)
    all_targets = (
        _gold_path(tmp_path, DatasetName.MARKET_SNAPSHOT.value, "market_data.json"),
        _gold_path(tmp_path, DatasetName.MARKET_SNAPSHOT.value, "manifest.json"),
        _gold_path(tmp_path, DatasetName.HESTON_QUOTES.value, "heston_quotes.parquet"),
        _gold_path(tmp_path, DatasetName.HESTON_QUOTES.value, "manifest.json"),
    )
    existing_target.parent.mkdir(parents=True)
    existing_target.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        _write_gold(storage, outputs)

    assert existing_target.read_text(encoding="utf-8") == "existing"
    for target in all_targets:
        if target == existing_target:
            continue
        assert not target.exists()
    assert not (tmp_path / "_meta" / "artifacts.jsonl").exists()


def test_write_gold_artifacts_overwrite_true_replaces_full_output_set(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    storage = _storage(tmp_path)
    first_outputs = _a3_outputs()
    first_paths = _write_gold(storage, first_outputs, library_commit="first")

    replacement_quotes = first_outputs.cleaned_quotes.copy()
    replacement_quotes.loc[0, "mid"] = 9.99
    replacement_outputs = _A3Outputs(
        local_snapshot=first_outputs.local_snapshot,
        market_inputs=first_outputs.market_inputs,
        cleaned_quotes=replacement_quotes,
        rejected_quotes=first_outputs.rejected_quotes,
        reason_counts=first_outputs.reason_counts,
        warnings=first_outputs.warnings,
    )

    replacement_paths = _write_gold(
        storage,
        replacement_outputs,
        overwrite=True,
        library_commit="replacement",
    )

    assert replacement_paths == first_paths
    market_payload = _read_gold_json(
        storage,
        dataset=DatasetName.MARKET_SNAPSHOT,
        filename="market_data.json",
    )
    assert market_payload["library_commit"] == "replacement"
    market_manifest = _read_gold_json(
        storage,
        dataset=DatasetName.MARKET_SNAPSHOT,
        filename="manifest.json",
    )
    heston_manifest = _read_gold_json(
        storage,
        dataset=DatasetName.HESTON_QUOTES,
        filename="manifest.json",
    )
    assert market_manifest["library_commit"] == "replacement"
    assert heston_manifest["library_commit"] == "replacement"
    heston_quotes = storage.read_frame(
        dataset=DatasetName.HESTON_QUOTES.value,
        layer="gold",
        partitions=_partitions(),
        columns=list(HESTON_QUOTES_COLUMNS),
    )
    assert float(heston_quotes.loc[0, "mid"]) == pytest.approx(9.99)
