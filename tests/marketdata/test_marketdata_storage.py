from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from option_pricing.marketdata.schemas import RunMetadata, StorageConfig
from option_pricing.marketdata.storage import LocalStorage


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


def test_write_and_read_frame_round_trips_partitioned_data(
    tmp_path, fake_parquet: None
) -> None:
    storage = LocalStorage(StorageConfig(root=tmp_path))
    frame = pd.DataFrame({"open": [100.0], "close": [101.0]})

    path = storage.write_frame(
        frame,
        dataset="equity_bars",
        partitions={
            "date": "2026-03-16",
            "timeframe": "1Min",
            "symbol": "SPY",
        },
        filename="bars",
    )

    expected_path = (
        tmp_path
        / "bronze"
        / "equity_bars"
        / "symbol=SPY"
        / "timeframe=1Min"
        / "date=2026-03-16"
        / "bars.parquet"
    )
    assert path == expected_path

    loaded = storage.read_frame(
        dataset="equity_bars",
        partitions={
            "timeframe": "1Min",
            "symbol": "SPY",
            "date": "2026-03-16",
        },
    )

    assert loaded.to_dict(orient="records") == [
        {
            "open": 100.0,
            "close": 101.0,
            "symbol": "SPY",
            "timeframe": "1Min",
            "date": "2026-03-16",
        }
    ]

    artifact_lines = (
        (tmp_path / "_meta" / "artifacts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(artifact_lines) == 1
    artifact = json.loads(artifact_lines[0])
    assert artifact["artifact_type"] == "frame"
    assert artifact["rows"] == 1


def test_manifest_run_registry_and_checkpoints_are_persisted(tmp_path) -> None:
    storage = LocalStorage(tmp_path)

    manifest_path = storage.write_manifest(
        {"row_count": 2},
        dataset="market_snapshots",
        layer="gold",
        partitions={
            "date": date(2026, 3, 16),
            "underlying": "SPY",
        },
    )

    expected_manifest_path = (
        tmp_path
        / "gold"
        / "market_snapshot"
        / "underlying=SPY"
        / "date=2026-03-16"
        / "manifest.json"
    )
    assert manifest_path == expected_manifest_path

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["dataset"] == "market_snapshot"
    assert manifest_payload["layer"] == "gold"
    assert manifest_payload["row_count"] == 2

    runs_path = storage.record_run(
        RunMetadata(
            run_id="run-001",
            asof=datetime(2026, 3, 16, 14, 0, tzinfo=UTC),
            started_at=datetime(2026, 3, 16, 14, 1, tzinfo=UTC),
            git_sha="abc123",
        ),
        artifacts=[manifest_path],
        details={"provider": "yahoo"},
    )

    run_lines = runs_path.read_text(encoding="utf-8").splitlines()
    assert len(run_lines) == 1
    run_payload = json.loads(run_lines[0])
    assert run_payload["artifacts"] == [
        "gold/market_snapshot/underlying=SPY/date=2026-03-16/manifest.json"
    ]
    assert run_payload["details"] == {"provider": "yahoo"}

    checkpoints_path = storage.set_checkpoint(
        "yahoo/equity_quotes/SPY",
        {"last_trade_date": "2026-03-16"},
    )
    assert checkpoints_path == tmp_path / "_meta" / "checkpoints.json"
    assert storage.get_checkpoint("yahoo/equity_quotes/SPY") == {
        "last_trade_date": "2026-03-16"
    }

    checkpoints_payload = json.loads(checkpoints_path.read_text(encoding="utf-8"))
    assert checkpoints_payload["checkpoints"]["yahoo/equity_quotes/SPY"] == {
        "updated_at": checkpoints_payload["updated_at"],
        "value": {"last_trade_date": "2026-03-16"},
    }


def test_write_frame_raises_helpful_error_when_parquet_engine_is_missing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = LocalStorage(tmp_path)
    frame = pd.DataFrame({"bid": [1.0], "ask": [1.2]})

    def _missing_engine(
        self: pd.DataFrame,
        path: str,
        compression: str | None = None,
        index: bool = False,
    ) -> None:
        del self, path, compression, index
        raise ImportError("missing parquet engine")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _missing_engine)

    with pytest.raises(RuntimeError, match="pyarrow|fastparquet"):
        storage.write_frame(
            frame,
            dataset="equity_quotes",
            partitions={"date": "2026-03-16"},
        )
