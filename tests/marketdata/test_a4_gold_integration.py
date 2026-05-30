from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from option_pricing.marketdata.cleaning import clean_option_quotes
from option_pricing.marketdata.config import StorageConfig
from option_pricing.marketdata.gold import (
    GOLD_CONVERSION_MANIFEST_VERSION,
    heston_quote_set_from_frame,
    market_data_snapshot_from_json,
    write_gold_artifacts,
)
from option_pricing.marketdata.normalize import (
    normalize_market_inputs,
    normalize_option_chain,
)
from option_pricing.marketdata.providers.local import (
    LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1,
    LocalSnapshotConfig,
    LocalSnapshotProvider,
)
from option_pricing.marketdata.schemas import HESTON_QUOTES_COLUMNS, DatasetName
from option_pricing.marketdata.silver import write_cleaned_quotes_silver
from option_pricing.marketdata.storage import LocalStorage
from option_pricing.marketdata.validation import validate_dtypes
from option_pricing.types import MarketData


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


def _partitions() -> dict[str, str | date]:
    return {
        "underlying": "SYNTH",
        "date": date(2026, 5, 22),
        "run_id": "test-run",
    }


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


def test_a4_silver_outputs_convert_to_gold_artifacts(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    local = LocalSnapshotProvider(
        LocalSnapshotConfig(
            fixture_name=LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1,
            run_id="test-run",
        )
    ).load_snapshot()

    market_inputs = normalize_market_inputs(local.market_inputs_raw)
    option_chain = normalize_option_chain(local.option_chain_raw)
    cleaning = clean_option_quotes(option_chain, market_inputs)

    storage = LocalStorage(StorageConfig(root=tmp_path))

    silver_paths = write_cleaned_quotes_silver(
        storage,
        local_snapshot=local,
        market_inputs=market_inputs,
        result=cleaning,
    )

    gold_paths = write_gold_artifacts(
        storage,
        local_snapshot=local,
        market_inputs=market_inputs,
        cleaned_quotes=cleaning.cleaned_quotes,
        rejected_quotes=cleaning.rejected_quotes,
        reason_counts=cleaning.reason_counts,
        warnings=cleaning.warnings,
    )

    assert silver_paths.market_inputs.exists()
    assert silver_paths.cleaned_quotes.exists()
    assert silver_paths.rejected_quotes.exists()
    assert silver_paths.manifest.exists()

    assert gold_paths.market_data.exists()
    assert gold_paths.market_manifest.exists()
    assert gold_paths.heston_quotes.exists()
    assert gold_paths.heston_manifest.exists()
    assert not (tmp_path / "gold" / DatasetName.REJECTED_QUOTES.value).exists()
    assert not list(
        (tmp_path / "gold").rglob(f"{DatasetName.SURFACE_INPUTS.value}.parquet")
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

    ctx = snapshot.market_data.to_context()
    assert ctx.spot == pytest.approx(100.0)
    assert ctx.df(1.0) == pytest.approx(math.exp(-0.04))
    assert ctx.fwd(1.0) == pytest.approx(100.0 * math.exp(0.04))
    assert math.isfinite(ctx.df(1.0)) and ctx.df(1.0) > 0.0
    assert math.isfinite(ctx.fwd(1.0)) and ctx.fwd(1.0) > 0.0

    heston_quotes = storage.read_frame(
        dataset=DatasetName.HESTON_QUOTES.value,
        layer="gold",
        partitions=_partitions(),
        columns=list(HESTON_QUOTES_COLUMNS),
    )
    validate_dtypes(heston_quotes, DatasetName.HESTON_QUOTES, allow_extra=False)
    assert tuple(heston_quotes.columns) == HESTON_QUOTES_COLUMNS
    pd.testing.assert_series_equal(
        heston_quotes["expiry_years"],
        cleaning.cleaned_quotes["expiry_years"].reset_index(drop=True),
        check_names=False,
    )

    quote_set = heston_quote_set_from_frame(heston_quotes, snapshot.market_data)
    assert quote_set.n_quotes == len(heston_quotes)
    np.testing.assert_array_equal(
        quote_set.is_call,
        (heston_quotes["right"] == "call").to_numpy(dtype=np.bool_),
    )
    assert np.all(np.isfinite(quote_set.discount))
    assert np.all(quote_set.discount > 0.0)
    assert np.all(np.isfinite(quote_set.forward))
    assert np.all(quote_set.forward > 0.0)
    assert quote_set.labels == tuple(heston_quotes["contract_symbol"].astype(str))
    assert all(label.strip() for label in quote_set.labels)

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

    required_manifest_keys = {
        "conversion_manifest_version",
        "run_id",
        "snapshot_id",
        "underlying",
        "valuation_timestamp_utc",
        "library_commit",
        "row_counts",
        "reason_counts",
        "warnings",
        "artifacts",
        "source",
    }
    for manifest in (market_manifest, heston_manifest):
        assert required_manifest_keys.issubset(manifest)
        assert manifest["conversion_manifest_version"] == (
            GOLD_CONVERSION_MANIFEST_VERSION
        )
        assert manifest["run_id"] == "test-run"
        assert manifest["snapshot_id"] == local.snapshot_id
        assert manifest["underlying"] == "SYNTH"
        assert manifest["valuation_timestamp_utc"] == "2026-05-22T15:30:00Z"
        assert manifest["library_commit"] is None
        assert manifest["reason_counts"] == cleaning.reason_counts
        assert manifest["warnings"] == list(cleaning.warnings)
        assert manifest["source"] == {
            "source_type": "local_fixture",
            "fixture_name": LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1,
        }
        manifest_text = json.dumps(manifest, sort_keys=True)
        assert "rejection_detail" not in manifest_text
        assert "rejected_quote_rows" not in manifest_text
        assert "contract_symbol" not in manifest_text
        assert "quote_id" not in manifest_text
        assert "rejected_quotes" not in manifest["artifacts"]

    assert market_manifest["row_counts"] == {
        "market_inputs": len(market_inputs),
        "cleaned_quotes": len(cleaning.cleaned_quotes),
        "rejected_quotes": len(cleaning.rejected_quotes),
    }
    assert market_manifest["artifacts"] == {"market_data": "market_data.json"}

    assert heston_manifest["row_counts"] == {
        "cleaned_quotes": len(cleaning.cleaned_quotes),
        "rejected_quotes": len(cleaning.rejected_quotes),
        "heston_quotes": len(heston_quotes),
    }
    assert heston_manifest["artifacts"] == {"heston_quotes": "heston_quotes.parquet"}
    assert "every IV is finite and > 0" in str(heston_manifest["iv_mid_policy"])
    assert "every vega is finite and >= 0" in str(heston_manifest["bs_vega_policy"])
