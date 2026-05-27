from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from option_pricing.marketdata.config import StorageConfig
from option_pricing.marketdata.providers.local import (
    LOCAL_SNAPSHOT_BRONZE_SCHEMA_VERSION,
    LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1,
    LocalSnapshotConfig,
    LocalSnapshotProvider,
    LocalSnapshotResult,
    write_local_snapshot_bronze,
)
from option_pricing.marketdata.storage import LocalStorage

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
LIVE_PROVIDER_MODULES = (
    "alpaca",
    "fredapi",
    "option_pricing.marketdata.providers.alpaca",
    "option_pricing.marketdata.providers.fred",
    "option_pricing.marketdata.providers.yahoo",
    "requests",
    "yfinance",
)


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


def _load_fixture(run_id: str | None = "test-run") -> LocalSnapshotResult:
    config = LocalSnapshotConfig(
        fixture_root=FIXTURE_ROOT,
        fixture_name=LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1,
        expected_underlying="SYNTH",
        run_id=run_id,
    )
    return LocalSnapshotProvider(config).load_snapshot()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_utc(value: object) -> datetime:
    assert isinstance(value, str)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    return parsed


def test_a2_local_snapshot_loads_and_writes_bronze(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    for module_name in LIVE_PROVIDER_MODULES:
        sys.modules.pop(module_name, None)

    storage = LocalStorage(StorageConfig(root=tmp_path))
    result = _load_fixture(run_id="test-run")

    paths = write_local_snapshot_bronze(storage, result, overwrite=False)

    expected_root = (
        tmp_path
        / "bronze"
        / "local_snapshot"
        / "underlying=SYNTH"
        / "date=2026-05-22"
        / "run_id=test-run"
    )
    assert paths.root == expected_root
    assert paths.manifest == expected_root / "manifest.json"
    assert paths.market_inputs == expected_root / "market_inputs.parquet"
    assert paths.option_chain == expected_root / "option_chain.parquet"
    assert paths.manifest.exists()
    assert paths.market_inputs.exists()
    assert paths.option_chain.exists()

    manifest = _read_json(paths.manifest)
    assert manifest["local_snapshot_schema_version"] == (
        LOCAL_SNAPSHOT_BRONZE_SCHEMA_VERSION
    )
    assert (
        manifest["fixture_schema_version"] == result.manifest["fixture_schema_version"]
    )
    assert manifest["fixture_name"] == result.fixture_name
    assert manifest["source_type"] == "local_fixture"
    assert manifest["underlying"] == "SYNTH"
    assert manifest["run_id"] == result.run_id
    assert manifest["snapshot_id"] == result.snapshot_id
    assert manifest["library_commit"] is None
    assert manifest["rows"] == {
        "market_inputs_raw": len(result.market_inputs_raw),
        "option_chain_raw": len(result.option_chain_raw),
    }
    assert manifest["artifacts"] == {
        "market_inputs": "market_inputs.parquet",
        "option_chain": "option_chain.parquet",
    }
    assert manifest["warnings"] == []
    assert _parse_utc(manifest["created_at_utc"])
    assert _parse_utc(manifest["valuation_timestamp_utc"]) == (
        result.asof.to_pydatetime()
    )

    market_inputs = pd.read_parquet(paths.market_inputs)
    option_chain = pd.read_parquet(paths.option_chain)
    assert len(market_inputs) == len(result.market_inputs_raw)
    assert len(option_chain) == len(result.option_chain_raw)
    assert not (tmp_path / "silver").exists()
    assert not (tmp_path / "gold").exists()

    for module_name in LIVE_PROVIDER_MODULES:
        assert module_name not in sys.modules


def test_local_snapshot_bronze_overwrite_policy(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    storage = LocalStorage(StorageConfig(root=tmp_path))
    result = _load_fixture(run_id="test-run")

    paths = write_local_snapshot_bronze(storage, result)

    with pytest.raises(FileExistsError, match="overwrite=True"):
        write_local_snapshot_bronze(storage, result)

    replacement_paths = write_local_snapshot_bronze(
        storage,
        result,
        overwrite=True,
        library_commit="abc123",
    )
    assert replacement_paths == paths
    assert _read_json(paths.manifest)["library_commit"] == "abc123"


def test_local_snapshot_bronze_requires_run_id(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    storage = LocalStorage(StorageConfig(root=tmp_path))
    result = _load_fixture(run_id=None)

    with pytest.raises(ValueError, match="run_id is required"):
        write_local_snapshot_bronze(storage, result)
