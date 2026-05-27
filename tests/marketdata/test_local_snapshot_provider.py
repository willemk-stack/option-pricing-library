from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from option_pricing.marketdata.providers.local import (
    LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1,
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
    "option_pricing.marketdata.providers.alpaca",
    "option_pricing.marketdata.providers.fred",
    "option_pricing.marketdata.providers.yahoo",
)
PROVIDER_CREDENTIAL_ENV_VARS = (
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "FRED_API_KEY",
)


def _load_fixture() -> LocalSnapshotResult:
    provider = LocalSnapshotProvider(FIXTURE_ROOT)
    return provider.load_snapshot(LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1)


def test_local_snapshot_fixture_loads_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for env_var in PROVIDER_CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    for module_name in LIVE_PROVIDER_MODULES:
        sys.modules.pop(module_name, None)

    result = _load_fixture()

    assert result.name == LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1
    assert not result.market_inputs.empty
    assert not result.option_chain.empty
    assert result.warnings == ()
    for module_name in LIVE_PROVIDER_MODULES:
        assert module_name not in sys.modules


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
    market_inputs = result.market_inputs

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
    option_chain = result.option_chain

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
    "missing_file",
    ["manifest.json", "market_inputs.csv", "option_chain.csv"],
)
def test_missing_fixture_file_fails_clearly(
    tmp_path: Path,
    missing_file: str,
) -> None:
    target = tmp_path / LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1
    shutil.copytree(FIXTURE_ROOT / LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1, target)
    (target / missing_file).unlink()

    provider = LocalSnapshotProvider(tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match=f"missing required file: {re.escape(missing_file)}",
    ):
        provider.load_snapshot(LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1)
