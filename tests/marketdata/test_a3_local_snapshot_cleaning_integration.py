from __future__ import annotations

import ast
import json
from dataclasses import fields
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from option_pricing.marketdata.cleaning import (
    QuoteCleaningResult,
    QuoteRejectionReason,
    clean_option_quotes,
)
from option_pricing.marketdata.config import StorageConfig
from option_pricing.marketdata.normalize import (
    normalize_market_inputs,
    normalize_option_chain,
)
from option_pricing.marketdata.providers.local import (
    LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1,
    LocalSnapshotConfig,
    LocalSnapshotProvider,
    LocalSnapshotResult,
)
from option_pricing.marketdata.schemas import (
    CLEANED_QUOTES_SCHEMA_VERSION,
    MARKET_INPUTS_SCHEMA_VERSION,
    REJECTED_QUOTES_SCHEMA_VERSION,
    DatasetName,
)
from option_pricing.marketdata.silver import (
    SILVER_CLEANING_SCHEMA_VERSION,
    SilverCleaningPaths,
    write_cleaned_quotes_silver,
)
from option_pricing.marketdata.storage import LocalStorage

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
DOCS_GUIDE = Path("docs/user_guides/market_snapshot_validation.md")
S4_FILES = (
    Path("src/option_pricing/marketdata/silver.py"),
    Path("tests/marketdata/test_a3_local_snapshot_cleaning_integration.py"),
)
A3_DOCS_NON_GOALS = (
    "no live providers",
    "no credentials",
    "no CLI",
    "no Gold",
    "no Heston",
    "no MarketData/PricingContext construction",
    "no model-validation bundle",
    "no research exports",
)
DISALLOWED_IMPORT_ROOTS = {
    "alpaca",
    "fredapi",
    "requests",
    "yfinance",
}
DISALLOWED_IMPORT_NAMES = {
    "Heston",
    "MarketData",
    "PricingContext",
    "option_pricing.marketdata.providers.alpaca",
    "option_pricing.marketdata.providers.fred",
    "option_pricing.marketdata.providers.yahoo",
}
DISALLOWED_IMPORT_PARTS = {
    "calibration",
    "cli",
    "heston",
    "pricers",
    "pricing",
}


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


def _clean_fixture(
    local_snapshot: LocalSnapshotResult,
) -> tuple[pd.DataFrame, pd.DataFrame, QuoteCleaningResult]:
    market_inputs = normalize_market_inputs(local_snapshot.market_inputs_raw)
    option_chain = normalize_option_chain(local_snapshot.option_chain_raw)
    result = clean_option_quotes(option_chain, market_inputs)
    return market_inputs, option_chain, result


def _storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(StorageConfig(root=tmp_path))


def _partitions() -> dict[str, str | date]:
    return {
        "underlying": "SYNTH",
        "date": date(2026, 5, 22),
        "run_id": "test-run",
    }


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_fixture_silver(
    storage: LocalStorage,
    *,
    local_snapshot: LocalSnapshotResult | None = None,
    overwrite: bool = False,
    library_commit: str | None = None,
) -> tuple[
    LocalSnapshotResult,
    pd.DataFrame,
    pd.DataFrame,
    QuoteCleaningResult,
    SilverCleaningPaths,
]:
    local = local_snapshot or _load_fixture()
    market_inputs, option_chain, result = _clean_fixture(local)
    paths = write_cleaned_quotes_silver(
        storage,
        local_snapshot=local,
        market_inputs=market_inputs,
        result=result,
        overwrite=overwrite,
        library_commit=library_commit,
    )
    return local, market_inputs, option_chain, result, paths


def test_silver_cleaning_paths_contract() -> None:
    assert tuple(field.name for field in fields(SilverCleaningPaths)) == (
        "market_inputs",
        "cleaned_quotes",
        "rejected_quotes",
        "manifest",
    )


def test_a3_local_fixture_writes_cleaning_outputs_to_silver(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    storage = _storage(tmp_path)
    local, market_inputs, option_chain, result, paths = _write_fixture_silver(
        storage,
        library_commit="abc123",
    )

    assert paths.market_inputs.exists()
    assert paths.cleaned_quotes.exists()
    assert paths.rejected_quotes.exists()
    assert paths.manifest.exists()

    read_market_inputs = storage.read_frame(
        dataset=DatasetName.MARKET_INPUTS.value,
        layer="silver",
        partitions=_partitions(),
    )
    read_cleaned_quotes = storage.read_frame(
        dataset=DatasetName.CLEANED_QUOTES.value,
        layer="silver",
        partitions=_partitions(),
    )
    read_rejected_quotes = storage.read_frame(
        dataset=DatasetName.REJECTED_QUOTES.value,
        layer="silver",
        partitions=_partitions(),
    )

    assert len(read_market_inputs) == len(market_inputs)
    assert len(read_cleaned_quotes) == len(result.cleaned_quotes)
    assert len(read_rejected_quotes) == len(result.rejected_quotes)
    assert len(result.cleaned_quotes) + len(result.rejected_quotes) == len(option_chain)

    manifest = _read_json(paths.manifest)
    assert manifest["silver_schema_version"] == SILVER_CLEANING_SCHEMA_VERSION
    assert manifest["cleaning_policy"] == "quote_cleaning_policy.v1"
    assert manifest["market_inputs_schema_version"] == MARKET_INPUTS_SCHEMA_VERSION
    assert manifest["cleaned_quotes_schema_version"] == CLEANED_QUOTES_SCHEMA_VERSION
    assert manifest["rejected_quotes_schema_version"] == REJECTED_QUOTES_SCHEMA_VERSION
    assert manifest["fixture_name"] == LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1
    assert manifest["snapshot_id"] == local.snapshot_id
    assert manifest["run_id"] == "test-run"
    assert manifest["source_type"] == "local_fixture"
    assert manifest["underlying"] == "SYNTH"
    assert manifest["valuation_timestamp_utc"] == "2026-05-22T15:30:00Z"
    assert manifest["spot"] == 100.0
    assert manifest["rate"] == 0.04
    assert manifest["dividend_yield"] == 0.0
    assert manifest["day_count"] == "ACT/365"
    assert manifest["rows"] == {
        "market_inputs": len(market_inputs),
        "option_chain_input": len(option_chain),
        "accepted": len(result.cleaned_quotes),
        "rejected": len(result.rejected_quotes),
    }
    assert manifest["reason_counts"] == result.reason_counts
    assert manifest["warnings"] == list(result.warnings)
    assert manifest["artifacts"] == {
        "market_inputs": "market_inputs.parquet",
        "cleaned_quotes": "cleaned_quotes.parquet",
        "rejected_quotes": "rejected_quotes.parquet",
    }
    assert manifest["library_commit"] == "abc123"


def test_silver_cleaning_overwrite_false_preflights_all_targets(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    storage = _storage(tmp_path)
    _, _, _, _, paths = _write_fixture_silver(storage)

    with pytest.raises(FileExistsError, match="overwrite=True"):
        _write_fixture_silver(storage)

    paths.market_inputs.unlink()
    artifact_lines_before = (
        (tmp_path / "_meta" / "artifacts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    with pytest.raises(FileExistsError, match="overwrite=True"):
        _write_fixture_silver(storage)

    assert not paths.market_inputs.exists()
    assert paths.cleaned_quotes.exists()
    assert paths.rejected_quotes.exists()
    assert paths.manifest.exists()
    assert (tmp_path / "_meta" / "artifacts.jsonl").read_text(
        encoding="utf-8"
    ).splitlines() == artifact_lines_before


def test_silver_cleaning_overwrite_true_replaces_existing_outputs(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    storage = _storage(tmp_path)
    _, _, _, _, first_paths = _write_fixture_silver(storage)
    _, market_inputs, _, result, second_paths = _write_fixture_silver(
        storage,
        overwrite=True,
        library_commit="replacement",
    )

    assert second_paths == first_paths
    assert all(
        path.exists()
        for path in (
            second_paths.market_inputs,
            second_paths.cleaned_quotes,
            second_paths.rejected_quotes,
            second_paths.manifest,
        )
    )
    assert len(
        storage.read_frame(
            dataset=DatasetName.MARKET_INPUTS.value,
            layer="silver",
            partitions=_partitions(),
        )
    ) == len(market_inputs)
    assert len(
        storage.read_frame(
            dataset=DatasetName.CLEANED_QUOTES.value,
            layer="silver",
            partitions=_partitions(),
        )
    ) == len(result.cleaned_quotes)
    assert len(
        storage.read_frame(
            dataset=DatasetName.REJECTED_QUOTES.value,
            layer="silver",
            partitions=_partitions(),
        )
    ) == len(result.rejected_quotes)
    assert _read_json(second_paths.manifest)["library_commit"] == "replacement"


def test_silver_cleaning_requires_run_id(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    storage = _storage(tmp_path)
    local = _load_fixture(run_id=None)
    market_inputs, _, result = _clean_fixture(local)

    with pytest.raises(ValueError, match="run_id"):
        write_cleaned_quotes_silver(
            storage,
            local_snapshot=local,
            market_inputs=market_inputs,
            result=result,
        )


def test_silver_cleaning_validates_input_schemas(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    storage = _storage(tmp_path)
    local = _load_fixture()
    market_inputs, _, result = _clean_fixture(local)

    with pytest.raises(ValueError, match="market_inputs"):
        write_cleaned_quotes_silver(
            storage,
            local_snapshot=local,
            market_inputs=market_inputs.drop(columns=["spot"]),
            result=result,
        )

    with pytest.raises(ValueError, match="cleaned_quotes"):
        write_cleaned_quotes_silver(
            storage,
            local_snapshot=local,
            market_inputs=market_inputs,
            result=QuoteCleaningResult(
                cleaned_quotes=result.cleaned_quotes.drop(columns=["quote_id"]),
                rejected_quotes=result.rejected_quotes,
                reason_counts=result.reason_counts,
                warnings=result.warnings,
            ),
        )

    with pytest.raises(ValueError, match="rejected_quotes"):
        write_cleaned_quotes_silver(
            storage,
            local_snapshot=local,
            market_inputs=market_inputs,
            result=QuoteCleaningResult(
                cleaned_quotes=result.cleaned_quotes,
                rejected_quotes=result.rejected_quotes.drop(columns=["quote_id"]),
                reason_counts=result.reason_counts,
                warnings=result.warnings,
            ),
        )


def test_silver_cleaning_uses_expected_partition_layout(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    storage = _storage(tmp_path)
    _, _, _, _, paths = _write_fixture_silver(storage)

    assert (
        "silver/market_inputs/underlying=SYNTH/date=2026-05-22/run_id=test-run/"
        in paths.market_inputs.as_posix()
    )
    assert (
        "silver/cleaned_quotes/underlying=SYNTH/date=2026-05-22/run_id=test-run/"
        in paths.cleaned_quotes.as_posix()
    )
    assert (
        "silver/rejected_quotes/underlying=SYNTH/date=2026-05-22/run_id=test-run/"
        in paths.rejected_quotes.as_posix()
    )


def test_a3_docs_cover_rejection_reasons_and_non_goal_boundaries() -> None:
    docs = DOCS_GUIDE.read_text(encoding="utf-8")

    assert [
        reason.value
        for reason in QuoteRejectionReason
        if docs.count(f"`{reason.value}`") != 1
    ] == []
    assert [boundary for boundary in A3_DOCS_NON_GOALS if boundary not in docs] == []


def _import_root(name: str) -> str:
    return name.split(".", maxsplit=1)[0]


def _import_leaf(name: str) -> str:
    return name.rsplit(".", maxsplit=1)[-1]


def _imported_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    names: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.append(node.module)
            names.extend(
                f"{node.module}.{alias.name}"
                for alias in node.names
                if alias.name != "*"
            )

    return names


def _is_disallowed_import(name: str) -> bool:
    lowered_parts = {part.lower() for part in name.split(".")}
    if _import_root(name) in DISALLOWED_IMPORT_ROOTS:
        return True
    if name in DISALLOWED_IMPORT_NAMES or _import_leaf(name) in DISALLOWED_IMPORT_NAMES:
        return True
    if name.startswith("option_pricing.marketdata.providers") and not name.startswith(
        "option_pricing.marketdata.providers.local"
    ):
        return True
    if name.startswith("option_pricing.") and lowered_parts.intersection(
        DISALLOWED_IMPORT_PARTS
    ):
        return True
    return False


def test_s4_files_do_not_import_live_providers_pricing_heston_or_cli() -> None:
    forbidden: dict[str, list[str]] = {
        path.as_posix(): [
            name for name in _imported_names(path) if _is_disallowed_import(name)
        ]
        for path in S4_FILES
    }

    assert forbidden == {path.as_posix(): [] for path in S4_FILES}
