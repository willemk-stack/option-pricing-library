from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import pandas as pd

from option_pricing.marketdata.providers.local import (
    LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1,
    LocalSnapshotProvider,
    LocalSnapshotResult,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
A2_TEST_FILES = (
    Path(__file__),
    Path(__file__).with_name("test_local_snapshot_provider.py"),
    Path(__file__).with_name("test_local_snapshot_storage.py"),
)
DISALLOWED_A2_IMPORTS = {
    "alpaca",
    "fredapi",
    "option_pricing.marketdata.providers.alpaca",
    "option_pricing.marketdata.providers.fred",
    "option_pricing.marketdata.providers.yahoo",
    "requests",
    "yfinance",
}
STABLE_HANDOFF_FIELDS = (
    "fixture_name",
    "snapshot_id",
    "run_id",
    "underlying",
    "asof",
    "manifest",
    "market_inputs_raw",
    "option_chain_raw",
    "metadata",
    "row_counts",
    "warnings",
)
DOWNSTREAM_OUTPUT_FIELDS = (
    "cleaned_quotes",
    "rejected_quotes",
    "heston_quotes",
    "market_snapshot",
    "pricing",
    "pricing_outputs",
)


def _load_fixture() -> LocalSnapshotResult:
    provider = LocalSnapshotProvider(FIXTURE_ROOT)
    return provider.load_snapshot(LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1)


def _import_root(name: str) -> str:
    return name.split(".", maxsplit=1)[0]


def test_a2_local_snapshot_result_is_stable_a3_handoff_contract() -> None:
    """A3 consumes raw-ish local inputs; normalized and cleaned outputs come later."""

    result = _load_fixture()

    assert tuple(field.name for field in fields(LocalSnapshotResult)) == (
        STABLE_HANDOFF_FIELDS
    )
    assert result.fixture_name == LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1
    assert result.snapshot_id
    assert result.run_id is None
    assert result.underlying == "SYNTH"
    assert result.asof == pd.Timestamp("2026-05-22T15:30:00Z")
    assert result.manifest["fixture_name"] == result.fixture_name
    assert result.warnings == ()

    assert isinstance(result.market_inputs_raw, pd.DataFrame)
    assert isinstance(result.option_chain_raw, pd.DataFrame)
    assert not result.market_inputs_raw.empty
    assert not result.option_chain_raw.empty
    assert result.row_counts == {
        "market_inputs": len(result.market_inputs_raw),
        "option_chain": len(result.option_chain_raw),
    }

    assert result.metadata["source"] == "local_fixture"
    assert result.metadata["scope"] == "schema_only"
    assert result.metadata["synthetic"] is True
    assert result.metadata["provider_neutral"] is True

    for field_name in DOWNSTREAM_OUTPUT_FIELDS:
        assert not hasattr(result, field_name)


def test_a2_local_snapshot_tests_do_not_import_live_or_network_providers() -> None:
    for path in A2_TEST_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported = [node.module]
                imported.extend(
                    f"{node.module}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )
            else:
                continue

            forbidden = [
                name
                for name in imported
                if name in DISALLOWED_A2_IMPORTS
                or _import_root(name) in DISALLOWED_A2_IMPORTS
            ]
            assert forbidden == []
