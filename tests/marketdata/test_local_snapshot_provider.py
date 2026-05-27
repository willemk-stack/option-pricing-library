from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from option_pricing.marketdata.providers.local import (
    LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1,
    LocalSnapshotConfig,
    LocalSnapshotProvider,
    LocalSnapshotResult,
)
from option_pricing.marketdata.validation import (
    dataset_columns,
    validate_columns,
    validate_dtypes,
)

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
PROVIDER_CREDENTIAL_ENV_VARS = (
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "FRED_API_KEY",
)


def _load_fixture() -> LocalSnapshotResult:
    provider = LocalSnapshotProvider(FIXTURE_ROOT)
    return provider.load_snapshot(LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1)


def _copy_fixture(tmp_path: Path) -> Path:
    target = tmp_path / LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1
    shutil.copytree(FIXTURE_ROOT / LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1, target)
    return target


def _read_manifest(path: Path) -> dict[str, object]:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    (path / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_local_snapshot_fixture_loads_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for env_var in PROVIDER_CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    for module_name in LIVE_PROVIDER_MODULES:
        sys.modules.pop(module_name, None)

    result = _load_fixture()

    assert result.fixture_name == LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1
    assert result.name == LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1
    assert result.underlying == "SYNTH"
    assert result.asof == pd.Timestamp("2026-05-22T15:30:00Z")
    assert result.run_id is None
    assert result.snapshot_id == (
        "local_snapshot_synth_schema_v1:SYNTH:2026-05-22T15:30:00+00:00"
    )
    assert result.row_counts == {"market_inputs": 1, "option_chain": 18}
    assert result.market_inputs is result.market_inputs_raw
    assert result.option_chain is result.option_chain_raw
    assert not result.market_inputs_raw.empty
    assert not result.option_chain_raw.empty
    assert result.warnings == ()
    for module_name in LIVE_PROVIDER_MODULES:
        assert module_name not in sys.modules


def test_local_snapshot_config_loads_by_path_with_run_id() -> None:
    config = LocalSnapshotConfig(
        fixture_path=FIXTURE_ROOT / LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1,
        expected_underlying="SYNTH",
        run_id="unit-test-run",
    )

    result = LocalSnapshotProvider(config).load_snapshot()

    assert result.fixture_name == LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1
    assert result.run_id == "unit-test-run"
    assert result.underlying == "SYNTH"
    assert result.metadata["source"] == "local_fixture"


def test_local_snapshot_fixture_has_expected_metadata() -> None:
    result = _load_fixture()

    assert result.manifest["fixture_name"] == LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1
    assert result.manifest["synthetic"] is True
    assert result.manifest["provider_neutral"] is True
    assert result.manifest["scope"] == "schema_only"
    assert result.manifest["underlying"] == "SYNTH"
    assert result.manifest["source"] == "local_fixture"

    assumptions = result.manifest["assumptions"]
    assert assumptions["rate"] == 0.04
    assert assumptions["rate_units"] == "annualized_decimal"
    assert assumptions["rate_compounding"] == "continuous"
    assert assumptions["dividend_yield"] == 0.0
    assert assumptions["dividend_yield_units"] == "annualized_decimal"
    assert assumptions["dividend_yield_source"] == "assumption"
    assert assumptions["day_count"] == "ACT/365"


def test_local_snapshot_market_inputs_validate_against_schema() -> None:
    result = _load_fixture()
    market_inputs = result.market_inputs_raw

    validate_columns(market_inputs, "market_inputs", allow_extra=False)
    validate_dtypes(market_inputs, "market_inputs", allow_extra=False)

    assert tuple(market_inputs.columns) == dataset_columns("market_inputs")
    assert len(market_inputs) == 1

    row = market_inputs.iloc[0]
    assert row["underlying"] == "SYNTH"
    assert row["asof"] == pd.Timestamp("2026-05-22T15:30:00Z")
    assert float(row["spot"]) == pytest.approx(100.00)
    assert row["spot_source"] == "local_fixture"
    assert float(row["rate"]) == pytest.approx(0.04)
    assert row["rate_source"] == "local_fixture"
    assert row["rate_observation_date"] == pd.Timestamp("2026-05-22")
    assert row["rate_compounding"] == "continuous"
    assert float(row["dividend_yield"]) == pytest.approx(0.00)
    assert row["dividend_yield_source"] == "assumption"
    assert row["day_count"] == "ACT/365"


def test_local_snapshot_option_chain_validate_against_schema() -> None:
    result = _load_fixture()
    option_chain = result.option_chain_raw

    validate_columns(option_chain, "option_chain", allow_extra=False)
    validate_dtypes(option_chain, "option_chain", allow_extra=False)

    assert tuple(option_chain.columns) == dataset_columns("option_chain")
    assert len(option_chain) == 18
    assert set(option_chain["underlying"].astype(str)) == {"SYNTH"}
    assert set(option_chain["source"].astype(str)) == {"local_fixture"}
    assert set(option_chain["right"].astype(str)) == {"call", "put"}
    assert sorted(option_chain["strike"].astype(float).unique().tolist()) == [
        80.0,
        90.0,
        95.0,
        100.0,
        105.0,
        110.0,
        120.0,
    ]
    assert sorted(option_chain["expiry"].dt.strftime("%Y-%m-%d").unique().tolist()) == [
        "2026-06-19",
        "2026-09-18",
        "2026-12-18",
    ]

    asof = result.market_inputs["asof"].iloc[0].tz_convert(None)
    assert bool((option_chain["expiry"] > asof).all())
    assert bool((option_chain["strike"] > 0).all())
    assert bool((option_chain["bid"] > 0).all())
    assert bool((option_chain["ask"] > 0).all())
    assert bool((option_chain["bid"] <= option_chain["ask"]).all())
    assert bool((option_chain["mid"] >= option_chain["bid"]).all())
    assert bool((option_chain["mid"] <= option_chain["ask"]).all())


def test_unknown_local_snapshot_name_fails_clearly() -> None:
    provider = LocalSnapshotProvider(FIXTURE_ROOT)

    with pytest.raises(FileNotFoundError, match="Unknown local snapshot fixture"):
        provider.load_snapshot("does_not_exist")


@pytest.mark.parametrize(
    "name",
    ["", ".", "..", "../outside", r"..\outside", "/absolute"],
)
def test_fixture_name_path_traversal_is_rejected(name: str) -> None:
    provider = LocalSnapshotProvider(FIXTURE_ROOT)

    with pytest.raises(ValueError, match="simple directory name"):
        provider.load_snapshot(name)


def test_config_fixture_name_path_traversal_is_rejected() -> None:
    provider = LocalSnapshotProvider(
        LocalSnapshotConfig(
            fixture_path=FIXTURE_ROOT / LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1,
            fixture_name="../outside",
        )
    )

    with pytest.raises(ValueError, match="simple directory name"):
        provider.load_snapshot()


@pytest.mark.parametrize(
    "missing_file",
    ["manifest.json", "market_inputs.csv", "option_chain.csv"],
)
def test_missing_fixture_file_fails_clearly(
    tmp_path: Path,
    missing_file: str,
) -> None:
    target = _copy_fixture(tmp_path)
    (target / missing_file).unlink()

    provider = LocalSnapshotProvider(tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match=f"missing required file: {re.escape(missing_file)}",
    ):
        provider.load_snapshot(LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1)


def test_missing_manifest_key_fails_clearly(tmp_path: Path) -> None:
    target = _copy_fixture(tmp_path)
    manifest = _read_manifest(target)
    del manifest["underlying"]
    _write_manifest(target, manifest)

    provider = LocalSnapshotProvider(tmp_path)

    with pytest.raises(ValueError, match=r"missing required keys: \['underlying'\]"):
        provider.load_snapshot(LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1)


def test_manifest_files_section_must_match_actual_files(tmp_path: Path) -> None:
    target = _copy_fixture(tmp_path)
    manifest = _read_manifest(target)
    files = manifest["files"]
    assert isinstance(files, dict)
    files["market_inputs"] = "renamed_market_inputs.csv"
    _write_manifest(target, manifest)

    provider = LocalSnapshotProvider(tmp_path)

    with pytest.raises(
        ValueError, match="files section must match actual fixture files"
    ):
        provider.load_snapshot(LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1)


def test_underlying_mismatch_fails_clearly(tmp_path: Path) -> None:
    target = _copy_fixture(tmp_path)
    market_inputs = pd.read_csv(target / "market_inputs.csv")
    market_inputs.loc[0, "underlying"] = "OTHER"
    market_inputs.to_csv(target / "market_inputs.csv", index=False)

    provider = LocalSnapshotProvider(tmp_path)

    with pytest.raises(ValueError, match="market_inputs underlying .* manifest"):
        provider.load_snapshot(LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1)


def test_expected_underlying_mismatch_fails_clearly() -> None:
    config = LocalSnapshotConfig(
        fixture_root=FIXTURE_ROOT,
        fixture_name=LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1,
        expected_underlying="OTHER",
    )

    with pytest.raises(ValueError, match="expected 'OTHER'"):
        LocalSnapshotProvider(config).load_snapshot()


def test_market_inputs_must_have_exactly_one_row(tmp_path: Path) -> None:
    target = _copy_fixture(tmp_path)
    market_inputs = pd.read_csv(target / "market_inputs.csv")
    market_inputs = pd.concat([market_inputs, market_inputs], ignore_index=True)
    market_inputs.to_csv(target / "market_inputs.csv", index=False)

    provider = LocalSnapshotProvider(tmp_path)

    with pytest.raises(ValueError, match="exactly one market_inputs row; found 2"):
        provider.load_snapshot(LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1)


def test_duplicate_contract_symbol_fails_clearly(tmp_path: Path) -> None:
    target = _copy_fixture(tmp_path)
    option_chain = pd.read_csv(target / "option_chain.csv")
    option_chain.loc[1, "contract_symbol"] = option_chain.loc[0, "contract_symbol"]
    option_chain.to_csv(target / "option_chain.csv", index=False)

    provider = LocalSnapshotProvider(tmp_path)

    with pytest.raises(ValueError, match="duplicate contract_symbol"):
        provider.load_snapshot(LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1)


def test_repeated_local_snapshot_loads_are_deterministic() -> None:
    provider = LocalSnapshotProvider(FIXTURE_ROOT)

    first = provider.load_snapshot(LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1)
    second = provider.load_snapshot(LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1)

    assert first == second
    pd.testing.assert_frame_equal(first.market_inputs_raw, second.market_inputs_raw)
    pd.testing.assert_frame_equal(first.option_chain_raw, second.option_chain_raw)

    sorted_option_chain = first.option_chain_raw.sort_values(
        ["expiry", "strike", "right", "contract_symbol"],
        kind="mergesort",
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(first.option_chain_raw, sorted_option_chain)
