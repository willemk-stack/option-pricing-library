from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from option_pricing.marketdata.normalize import (
    normalize_market_inputs,
    normalize_option_chain,
)
from option_pricing.marketdata.schemas import (
    MARKET_INPUTS_COLUMNS,
    MARKET_INPUTS_DTYPES,
    OPTION_CHAIN_COLUMNS,
    OPTION_CHAIN_DTYPES,
    DatasetName,
)
from option_pricing.marketdata.validation import validate_dtypes

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "local_snapshot_synth_schema_v1"
S2_FILES = (
    Path("src/option_pricing/marketdata/normalize.py"),
    Path("tests/marketdata/test_normalize.py"),
)
DISALLOWED_IMPORT_ROOTS = {
    "alpaca",
    "fredapi",
    "requests",
    "yfinance",
}
DISALLOWED_IMPORT_NAMES = {
    "MarketData",
    "PricingContext",
}
DISALLOWED_IMPORT_PARTS = {
    "calibration",
    "cli",
    "heston",
    "providers",
    "pricing",
    "pricers",
}


def _fixture_market_inputs() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_ROOT / "market_inputs.csv")


def _fixture_option_chain() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_ROOT / "option_chain.csv")


def _market_inputs_frame(**overrides: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "underlying": "SYNTH",
        "asof": "2026-05-22T15:30:00Z",
        "spot": 100.0,
        "spot_source": "unit_test",
        "rate": 0.04,
        "rate_source": "unit_test",
        "rate_observation_date": "2026-05-22",
        "rate_compounding": "continuous",
        "dividend_yield": 0.0,
        "dividend_yield_source": "unit_test",
        "day_count": "ACT/365",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _option_chain_frame(**overrides: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "underlying": "SYNTH",
        "contract_symbol": "SYNTH260619C00100000",
        "quote_ts": "2026-05-22T15:30:00Z",
        "expiry": "2026-06-19",
        "strike": 100.0,
        "right": "call",
        "bid": 4.15,
        "ask": 4.31,
        "mid": 4.23,
        "last": 4.24,
        "iv": 0.25,
        "delta": 0.52,
        "gamma": 0.028,
        "theta": -0.031,
        "vega": 0.102,
        "rho": 0.030,
        "open_interest": 520,
        "source": "unit_test",
        "asof": "2026-05-22T15:30:00Z",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _assert_market_inputs_schema(frame: pd.DataFrame) -> None:
    assert tuple(frame.columns) == MARKET_INPUTS_COLUMNS
    assert {
        column: str(frame[column].dtype) for column in frame
    } == MARKET_INPUTS_DTYPES
    validate_dtypes(frame, DatasetName.MARKET_INPUTS, allow_extra=False)


def _assert_option_chain_schema(frame: pd.DataFrame) -> None:
    assert tuple(frame.columns) == OPTION_CHAIN_COLUMNS
    assert {column: str(frame[column].dtype) for column in frame} == OPTION_CHAIN_DTYPES
    validate_dtypes(frame, DatasetName.OPTION_CHAIN, allow_extra=False)


def test_normalize_market_inputs_fixture_normalizes_successfully() -> None:
    normalized = normalize_market_inputs(_fixture_market_inputs())

    _assert_market_inputs_schema(normalized)
    assert len(normalized) == 1
    assert normalized.loc[0, "asof"] == pd.Timestamp("2026-05-22T15:30:00Z")
    assert normalized.loc[0, "rate_observation_date"] == pd.Timestamp("2026-05-22")


def test_normalize_market_inputs_tolerates_and_omits_extra_columns() -> None:
    frame = _market_inputs_frame(extra_column="discard me")

    normalized = normalize_market_inputs(frame)

    _assert_market_inputs_schema(normalized)
    assert "extra_column" not in normalized.columns


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame(columns=MARKET_INPUTS_COLUMNS), "exactly one row; found 0"),
        (
            pd.concat(
                [_market_inputs_frame(), _market_inputs_frame()], ignore_index=True
            ),
            "exactly one row; found 2",
        ),
        (_market_inputs_frame(spot=0.0), "spot"),
        (_market_inputs_frame(spot=-1.0), "spot"),
        (_market_inputs_frame(rate=float("inf")), "rate must be finite"),
        (
            _market_inputs_frame(dividend_yield=float("nan")),
            "dividend_yield must be finite",
        ),
        (
            _market_inputs_frame(rate_compounding="annual"),
            "rate_compounding must be 'continuous'",
        ),
        (_market_inputs_frame(day_count="ACT/360"), "day_count must be 'ACT/365'"),
    ],
)
def test_normalize_market_inputs_rejects_invalid_inputs(
    frame: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        normalize_market_inputs(frame)


def test_normalize_option_chain_fixture_normalizes_successfully() -> None:
    normalized = normalize_option_chain(_fixture_option_chain())

    _assert_option_chain_schema(normalized)
    assert len(normalized) == 18
    assert set(normalized["right"].astype(str)) == {"call", "put"}


def test_normalize_option_chain_tolerates_and_omits_extra_columns() -> None:
    frame = _option_chain_frame(extra_column="discard me")

    normalized = normalize_option_chain(frame)

    _assert_option_chain_schema(normalized)
    assert "extra_column" not in normalized.columns


@pytest.mark.parametrize("raw_right", ["C", "CALL", "Call", "call"])
def test_normalize_option_chain_right_call_aliases(raw_right: str) -> None:
    normalized = normalize_option_chain(_option_chain_frame(right=raw_right))

    assert normalized.loc[0, "right"] == "call"


@pytest.mark.parametrize("raw_right", ["P", "PUT", "Put", "put"])
def test_normalize_option_chain_right_put_aliases(raw_right: str) -> None:
    normalized = normalize_option_chain(
        _option_chain_frame(
            contract_symbol=f"SYNTH260619{raw_right.upper()[0]}00100000",
            right=raw_right,
        )
    )

    assert normalized.loc[0, "right"] == "put"


def test_normalize_option_chain_invalid_right_fails_clearly() -> None:
    with pytest.raises(ValueError, match="invalid right values"):
        normalize_option_chain(_option_chain_frame(right="straddle"))


def test_normalize_option_chain_missing_mid_is_computed_from_bid_ask() -> None:
    frame = _option_chain_frame(bid=4.0, ask=4.4).drop(columns=["mid"])

    normalized = normalize_option_chain(frame)

    assert float(normalized.loc[0, "mid"]) == pytest.approx(4.2)


def test_normalize_option_chain_existing_mid_is_preserved() -> None:
    normalized = normalize_option_chain(_option_chain_frame(bid=4.0, ask=4.4, mid=9.9))

    assert float(normalized.loc[0, "mid"]) == pytest.approx(9.9)


def test_normalize_option_chain_duplicate_contract_symbol_fails_clearly() -> None:
    first = _option_chain_frame()
    second = _option_chain_frame(right="put", bid=3.0, ask=3.2, mid=3.1)
    frame = pd.concat([first, second], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate contract_symbol"):
        normalize_option_chain(frame)


def test_normalize_option_chain_output_order_is_deterministic() -> None:
    rows: list[dict[str, Any]] = [
        _option_chain_frame(
            contract_symbol="SYNTH260918P00100000",
            expiry="2026-09-18",
            strike=100.0,
            right="put",
        )
        .iloc[0]
        .to_dict(),
        _option_chain_frame(
            contract_symbol="SYNTH260619P00100000",
            expiry="2026-06-19",
            strike=100.0,
            right="put",
        )
        .iloc[0]
        .to_dict(),
        _option_chain_frame(
            contract_symbol="SYNTH260619C00090000",
            expiry="2026-06-19",
            strike=90.0,
            right="call",
        )
        .iloc[0]
        .to_dict(),
        _option_chain_frame(
            contract_symbol="SYNTH260619C00100000",
            expiry="2026-06-19",
            strike=100.0,
            right="call",
        )
        .iloc[0]
        .to_dict(),
    ]

    normalized = normalize_option_chain(pd.DataFrame(rows))

    assert normalized["contract_symbol"].astype(str).tolist() == [
        "SYNTH260619C00090000",
        "SYNTH260619C00100000",
        "SYNTH260619P00100000",
        "SYNTH260918P00100000",
    ]


def test_normalize_option_chain_does_not_reject_crossed_market_shaped_rows() -> None:
    frame = _option_chain_frame(bid=5.0, ask=4.0).drop(columns=["mid"])

    normalized = normalize_option_chain(frame)

    _assert_option_chain_schema(normalized)
    assert float(normalized.loc[0, "bid"]) == pytest.approx(5.0)
    assert float(normalized.loc[0, "ask"]) == pytest.approx(4.0)
    assert float(normalized.loc[0, "mid"]) == pytest.approx(4.5)


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
    if _import_root(name) in DISALLOWED_IMPORT_ROOTS:
        return True
    if name in DISALLOWED_IMPORT_NAMES or _import_leaf(name) in DISALLOWED_IMPORT_NAMES:
        return True
    if name.startswith("option_pricing.") and any(
        part in DISALLOWED_IMPORT_PARTS for part in name.split(".")
    ):
        return True
    return False


def test_a3_s2_files_do_not_import_providers_network_pricing_heston_or_cli() -> None:
    forbidden: dict[str, list[str]] = {
        path.as_posix(): [
            name for name in _imported_names(path) if _is_disallowed_import(name)
        ]
        for path in S2_FILES
    }

    assert forbidden == {path.as_posix(): [] for path in S2_FILES}
