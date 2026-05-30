from __future__ import annotations

import ast
import json
import math
from collections.abc import Callable
from dataclasses import fields
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import option_pricing.marketdata.gold as gold_module
from option_pricing.marketdata.config import StorageConfig
from option_pricing.marketdata.gold import (
    GOLD_CONVERSION_MANIFEST_VERSION,
    GOLD_MARKET_DATA_SCHEMA_VERSION,
    GoldConversionPaths,
    GoldHestonQuotesResult,
    GoldMarketDataSnapshot,
    build_market_data_snapshot,
    market_data_snapshot_from_json,
    market_data_snapshot_to_json,
    write_market_data_gold,
)
from option_pricing.marketdata.normalize import normalize_market_inputs
from option_pricing.marketdata.storage import LocalStorage
from option_pricing.types import MarketData

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "local_snapshot_synth_schema_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
GOLD_FILES = (
    REPO_ROOT / "src/option_pricing/marketdata/gold.py",
    Path(__file__).resolve(),
    REPO_ROOT / "tests/marketdata/test_heston_gold_conversions.py",
    REPO_ROOT / "tests/marketdata/test_a4_gold_integration.py",
    REPO_ROOT / "tests/marketdata/test_a4_gold_storage_integration.py",
)
DISALLOWED_IMPORT_ROOTS = {
    "alpaca",
    "argparse",
    "click",
    "fredapi",
    "requests",
    "yfinance",
}
DISALLOWED_IMPORT_PARTS = {
    "calibration",
    "cli",
    "research",
}
ALLOWED_HESTON_QUOTESET_IMPORTS = {
    "option_pricing.models.heston.calibration.heston_types",
    "option_pricing.models.heston.calibration.heston_types.HestonQuoteSet",
}
ALLOWED_LOCAL_PROVIDER_IMPORTS = {
    "option_pricing.marketdata.providers.local",
    "option_pricing.marketdata.providers.local.LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1",
    "option_pricing.marketdata.providers.local.LocalSnapshotConfig",
    "option_pricing.marketdata.providers.local.LocalSnapshotProvider",
}
DISALLOWED_SOURCE_STRINGS = frozenset(
    {
        "".join(("api", "_key")),
        "".join(("secret", "_key")),
        "".join(("credential", "s")),
        "".join(("model_validation", "_bundle")),
        "".join(("research", "_export")),
        "".join(("surface_inputs", ".parquet")),
    }
)
DISALLOWED_CALL_NAMES = frozenset(
    {
        "".join(("build_model_validation", "_bundle")),
        "".join(("calibr", "ate")),
        "".join(("refresh", "_providers")),
        "".join(("run_calibr", "ation")),
        "".join(("write_surface", "_inputs")),
    }
)


def _fixture_market_inputs() -> pd.DataFrame:
    return normalize_market_inputs(pd.read_csv(FIXTURE_ROOT / "market_inputs.csv"))


def _snapshot(
    frame: pd.DataFrame | None = None,
    *,
    library_commit: str | None = "abc123",
) -> GoldMarketDataSnapshot:
    return build_market_data_snapshot(
        _fixture_market_inputs() if frame is None else frame,
        run_id="test-run",
        snapshot_id="snapshot-001",
        cleaning_policy="quote_cleaning_policy.v1",
        library_commit=library_commit,
    )


def _gold_partitions() -> dict[str, str | date]:
    return {
        "run_id": "test-run",
        "date": date(2026, 5, 22),
        "underlying": "SYNTH",
    }


def _with_value(column: str, value: object) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def _mutate(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out.loc[0, column] = value
        return out

    return _mutate


def test_gold_public_contracts_exist() -> None:
    assert GOLD_MARKET_DATA_SCHEMA_VERSION == "gold_market_data.v1"
    assert GOLD_CONVERSION_MANIFEST_VERSION == "gold_conversion_manifest.v1"
    assert tuple(gold_module.__all__) == (
        "GOLD_CONVERSION_MANIFEST_VERSION",
        "GOLD_MARKET_DATA_SCHEMA_VERSION",
        "GoldConversionPaths",
        "GoldHestonQuotesResult",
        "GoldMarketDataSnapshot",
        "build_heston_quotes",
        "build_market_data_snapshot",
        "heston_quote_set_from_frame",
        "market_data_snapshot_from_json",
        "market_data_snapshot_to_json",
        "write_gold_artifacts",
        "write_heston_quotes_gold",
        "write_market_data_gold",
    )
    assert tuple(field.name for field in fields(GoldMarketDataSnapshot)) == (
        "market_data",
        "metadata",
    )
    assert tuple(field.name for field in fields(GoldHestonQuotesResult)) == (
        "heston_quotes",
        "quote_count",
        "warnings",
    )
    assert tuple(field.name for field in fields(GoldConversionPaths)) == (
        "market_data",
        "market_manifest",
        "heston_quotes",
        "heston_manifest",
    )


def test_market_inputs_build_gold_market_data_snapshot() -> None:
    snapshot = _snapshot()

    assert snapshot.market_data == MarketData(
        spot=100.0,
        rate=0.04,
        dividend_yield=0.0,
    )
    assert snapshot.metadata == {
        "schema_version": GOLD_MARKET_DATA_SCHEMA_VERSION,
        "underlying": "SYNTH",
        "valuation_timestamp_utc": "2026-05-22T15:30:00Z",
        "run_id": "test-run",
        "snapshot_id": "snapshot-001",
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


def test_market_data_json_round_trips_and_context_is_usable() -> None:
    snapshot = _snapshot()
    payload = market_data_snapshot_to_json(snapshot)

    assert payload == {
        "schema_version": GOLD_MARKET_DATA_SCHEMA_VERSION,
        "underlying": "SYNTH",
        "valuation_timestamp_utc": "2026-05-22T15:30:00Z",
        "run_id": "test-run",
        "snapshot_id": "snapshot-001",
        "market_data": {
            "spot": 100.0,
            "rate": 0.04,
            "dividend_yield": 0.0,
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
    json.loads(json.dumps(payload))

    loaded = market_data_snapshot_from_json(payload)

    assert loaded == snapshot
    ctx = loaded.market_data.to_context()
    assert ctx.spot == pytest.approx(100.0)
    assert ctx.df(1.0) == pytest.approx(math.exp(-0.04))
    assert ctx.fwd(1.0) == pytest.approx(100.0 * math.exp(0.04))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.iloc[0:0], "exactly one row"),
        (lambda frame: pd.concat([frame, frame], ignore_index=True), "exactly one row"),
        (_with_value("spot", 0.0), "spot"),
        (_with_value("rate", float("inf")), "rate must be finite"),
        (
            _with_value("dividend_yield", float("nan")),
            "dividend_yield must be finite",
        ),
        (
            _with_value("rate_compounding", "annual"),
            "rate_compounding must be 'continuous'",
        ),
        (
            _with_value("day_count", "ACT/360"),
            "day_count must be 'ACT/365'",
        ),
        (
            lambda frame: frame.assign(unexpected_column=1),
            "unexpected extra columns",
        ),
    ],
)
def test_build_market_data_snapshot_rejects_invalid_market_inputs(
    mutate: Callable[[pd.DataFrame], pd.DataFrame],
    message: str,
) -> None:
    frame = mutate(_fixture_market_inputs())

    with pytest.raises((ValueError, TypeError), match=message):
        _snapshot(frame)


def test_build_market_data_snapshot_requires_dataframe() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        build_market_data_snapshot(  # type: ignore[arg-type]
            [],
            run_id="test-run",
            snapshot_id="snapshot-001",
            cleaning_policy="quote_cleaning_policy.v1",
        )


def test_market_data_snapshot_requires_gold_schema_version() -> None:
    payload = market_data_snapshot_to_json(_snapshot())
    payload["schema_version"] = "gold_market_data.v0"

    with pytest.raises(ValueError, match=GOLD_MARKET_DATA_SCHEMA_VERSION):
        market_data_snapshot_from_json(payload)


def test_market_data_snapshot_rejects_invalid_loaded_conventions() -> None:
    payload = market_data_snapshot_to_json(_snapshot())
    payload["day_count"] = "ACT/360"

    with pytest.raises(ValueError, match="ACT/365"):
        market_data_snapshot_from_json(payload)


@pytest.mark.parametrize(
    ("market_data", "message"),
    [
        (MarketData(spot=0.0, rate=0.04, dividend_yield=0.0), "spot"),
        (MarketData(spot=100.0, rate=float("inf"), dividend_yield=0.0), "rate"),
        (
            MarketData(spot=100.0, rate=0.04, dividend_yield=float("nan")),
            "dividend_yield",
        ),
    ],
)
def test_market_data_snapshot_to_json_rejects_invalid_market_data(
    market_data: MarketData,
    message: str,
) -> None:
    snapshot = GoldMarketDataSnapshot(
        market_data=market_data,
        metadata=_snapshot().metadata,
    )

    with pytest.raises(ValueError, match=message):
        market_data_snapshot_to_json(snapshot)


def test_market_data_snapshot_from_json_requires_market_data_object() -> None:
    payload = market_data_snapshot_to_json(_snapshot())
    payload.pop("market_data")

    with pytest.raises(ValueError, match="market_data object"):
        market_data_snapshot_from_json(payload)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("spot", -1.0, "spot"),
        ("rate", float("inf"), "rate"),
        ("dividend_yield", float("nan"), "dividend_yield"),
    ],
)
def test_market_data_snapshot_from_json_rejects_invalid_market_data_values(
    key: str,
    value: object,
    message: str,
) -> None:
    payload = market_data_snapshot_to_json(_snapshot())
    market_data = payload["market_data"]
    assert isinstance(market_data, dict)
    market_data[key] = value

    with pytest.raises(ValueError, match=message):
        market_data_snapshot_from_json(payload)


def test_write_market_data_gold_persists_reloadable_market_data_json(
    tmp_path: Path,
) -> None:
    storage = LocalStorage(StorageConfig(root=tmp_path))
    snapshot = _snapshot()

    path = write_market_data_gold(
        storage,
        snapshot=snapshot,
        partitions=_gold_partitions(),
    )

    assert path == (
        tmp_path
        / "gold"
        / "market_snapshot"
        / "underlying=SYNTH"
        / "date=2026-05-22"
        / "run_id=test-run"
        / "market_data.json"
    )
    assert path.exists()
    assert "run_id=test-run" in path.as_posix()

    payload = storage.read_json(
        dataset="market_snapshot",
        layer="gold",
        partitions=_gold_partitions(),
        filename="market_data.json",
    )
    loaded = market_data_snapshot_from_json(payload)

    assert loaded == snapshot
    assert loaded.market_data == MarketData(
        spot=100.0,
        rate=0.04,
        dividend_yield=0.0,
    )


def test_write_market_data_gold_respects_overwrite_policy(tmp_path: Path) -> None:
    storage = LocalStorage(StorageConfig(root=tmp_path))
    partitions = _gold_partitions()
    first = _snapshot(library_commit="first")
    replacement = _snapshot(library_commit="replacement")

    first_path = write_market_data_gold(
        storage,
        snapshot=first,
        partitions=partitions,
    )

    with pytest.raises(FileExistsError, match="overwrite=True"):
        write_market_data_gold(
            storage,
            snapshot=replacement,
            partitions=partitions,
        )

    replacement_path = write_market_data_gold(
        storage,
        snapshot=replacement,
        partitions=partitions,
        overwrite=True,
    )

    assert replacement_path == first_path
    loaded = market_data_snapshot_from_json(
        storage.read_json(
            dataset="market_snapshot",
            layer="gold",
            partitions=partitions,
            filename="market_data.json",
        )
    )
    assert loaded == replacement


def _import_root(name: str) -> str:
    return name.split(".", maxsplit=1)[0]


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
    if (
        name in ALLOWED_HESTON_QUOTESET_IMPORTS
        or name in ALLOWED_LOCAL_PROVIDER_IMPORTS
    ):
        return False
    lowered_parts = {part.lower() for part in name.split(".")}
    if _import_root(name) in DISALLOWED_IMPORT_ROOTS:
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


def test_gold_files_do_not_import_live_providers_cli_research_or_calibration() -> None:
    forbidden: dict[str, list[str]] = {
        path.as_posix(): [
            name for name in _imported_names(path) if _is_disallowed_import(name)
        ]
        for path in GOLD_FILES
    }

    assert forbidden == {path.as_posix(): [] for path in GOLD_FILES}


def _string_constants(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _called_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    return [
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and (name := _call_name(node.func)) is not None
    ]


def test_gold_files_do_not_write_out_of_scope_artifacts_or_run_calibration() -> None:
    forbidden_strings: dict[str, list[str]] = {
        path.as_posix(): [
            value
            for value in _string_constants(path)
            if value in DISALLOWED_SOURCE_STRINGS
        ]
        for path in GOLD_FILES
    }
    forbidden_calls: dict[str, list[str]] = {
        path.as_posix(): [
            name for name in _called_names(path) if name in DISALLOWED_CALL_NAMES
        ]
        for path in GOLD_FILES
    }

    assert forbidden_strings == {path.as_posix(): [] for path in GOLD_FILES}
    assert forbidden_calls == {path.as_posix(): [] for path in GOLD_FILES}
