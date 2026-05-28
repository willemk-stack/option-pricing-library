from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from option_pricing.marketdata.cleaning import (
    QuoteCleaningPolicyV1,
    QuoteRejectionReason,
    clean_option_quotes,
)
from option_pricing.marketdata.normalize import (
    normalize_market_inputs,
    normalize_option_chain,
)
from option_pricing.marketdata.schemas import (
    CLEANED_QUOTES_COLUMNS,
    CLEANED_QUOTES_DTYPES,
    OPTION_CHAIN_COLUMNS,
    REJECTED_QUOTES_COLUMNS,
    REJECTED_QUOTES_DTYPES,
    DatasetName,
)
from option_pricing.marketdata.validation import validate_dtypes

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "local_snapshot_synth_schema_v1"
S3_FILES = (
    Path("src/option_pricing/marketdata/cleaning.py"),
    Path("tests/marketdata/test_quote_cleaning.py"),
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
    "pricing",
    "pricers",
}


def _fixture_market_inputs() -> pd.DataFrame:
    return normalize_market_inputs(pd.read_csv(FIXTURE_ROOT / "market_inputs.csv"))


def _fixture_option_chain() -> pd.DataFrame:
    return normalize_option_chain(pd.read_csv(FIXTURE_ROOT / "option_chain.csv"))


def _market_inputs() -> pd.DataFrame:
    return normalize_market_inputs(
        pd.DataFrame(
            [
                {
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
            ]
        )
    )


def _option_chain(**overrides: object) -> pd.DataFrame:
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
    return normalize_option_chain(pd.DataFrame([row]))


def _assert_cleaned_schema(frame: pd.DataFrame) -> None:
    assert tuple(frame.columns) == CLEANED_QUOTES_COLUMNS
    assert {column: str(frame[column].dtype) for column in frame} == (
        CLEANED_QUOTES_DTYPES
    )
    validate_dtypes(frame, DatasetName.CLEANED_QUOTES, allow_extra=False)


def _assert_rejected_schema(frame: pd.DataFrame) -> None:
    assert tuple(frame.columns) == REJECTED_QUOTES_COLUMNS
    assert {column: str(frame[column].dtype) for column in frame} == (
        REJECTED_QUOTES_DTYPES
    )
    validate_dtypes(frame, DatasetName.REJECTED_QUOTES, allow_extra=False)


def _assert_single_rejection(
    option_chain: pd.DataFrame,
    reason: QuoteRejectionReason,
    *,
    policy: QuoteCleaningPolicyV1 = QuoteCleaningPolicyV1(),  # noqa: B008
) -> None:
    result = clean_option_quotes(option_chain, _market_inputs(), policy=policy)

    assert result.cleaned_quotes.empty
    assert len(result.rejected_quotes) == 1
    _assert_rejected_schema(result.rejected_quotes)
    assert result.rejected_quotes.loc[0, "rejection_reason"] == reason.value
    assert result.reason_counts == {reason.value: 1}
    assert all(isinstance(key, str) for key in result.reason_counts)


def test_clean_fixture_path_accepts_all_local_snapshot_rows() -> None:
    market_inputs = _fixture_market_inputs()
    option_chain = _fixture_option_chain()

    result = clean_option_quotes(option_chain, market_inputs)

    assert len(result.cleaned_quotes) == len(option_chain)
    assert result.rejected_quotes.empty
    assert result.reason_counts == {}
    assert result.warnings == ()
    assert set(result.cleaned_quotes["cleaning_policy"].astype(str)) == {
        "quote_cleaning_policy.v1"
    }


def test_output_cleaned_quotes_has_exact_schema_and_dtypes() -> None:
    result = clean_option_quotes(_fixture_option_chain(), _fixture_market_inputs())

    _assert_cleaned_schema(result.cleaned_quotes)


def test_output_rejected_quotes_has_exact_schema_and_dtypes() -> None:
    result = clean_option_quotes(_fixture_option_chain(), _fixture_market_inputs())

    _assert_rejected_schema(result.rejected_quotes)


def test_quote_id_is_deterministic_and_stable() -> None:
    option_chain = _option_chain()
    market_inputs = _market_inputs()

    first = clean_option_quotes(option_chain, market_inputs)
    second = clean_option_quotes(option_chain, market_inputs)

    expected = (
        "SYNTH|SYNTH260619C00100000|"
        "2026-05-22T15:30:00+00:00|2026-05-22T15:30:00+00:00"
    )
    assert first.cleaned_quotes["quote_id"].astype(str).tolist() == [expected]
    assert first.cleaned_quotes["quote_id"].equals(second.cleaned_quotes["quote_id"])


def test_expiry_years_is_act_365_and_positive_for_valid_rows() -> None:
    result = clean_option_quotes(_option_chain(), _market_inputs())

    expiry_utc = pd.Timestamp("2026-06-19").tz_localize("UTC")
    asof_utc = pd.Timestamp("2026-05-22T15:30:00Z")
    expected = (expiry_utc - asof_utc).total_seconds() / (365 * 24 * 3600)

    assert float(result.cleaned_quotes.loc[0, "expiry_years"]) == pytest.approx(
        expected
    )
    assert (result.cleaned_quotes["expiry_years"] > 0).all()


def test_moneyness_equals_strike_divided_by_spot() -> None:
    result = clean_option_quotes(_fixture_option_chain(), _fixture_market_inputs())
    expected = result.cleaned_quotes["strike"] / 100.0

    pd.testing.assert_series_equal(
        result.cleaned_quotes["moneyness"],
        expected.astype(pd.Float64Dtype()),
        check_names=False,
    )


def test_reason_counts_and_warnings_are_empty_when_all_rows_are_accepted() -> None:
    result = clean_option_quotes(_fixture_option_chain(), _fixture_market_inputs())

    assert result.reason_counts == {}
    assert result.warnings == ()


@pytest.mark.parametrize(
    ("option_chain", "reason"),
    [
        (_option_chain(bid=-1.0), QuoteRejectionReason.NEGATIVE_BID),
        (_option_chain(bid=0.0, ask=0.0), QuoteRejectionReason.NONPOSITIVE_ASK),
        (
            _option_chain(bid=5.0, ask=4.0, mid=4.5),
            QuoteRejectionReason.CROSSED_MARKET,
        ),
        (
            _option_chain(expiry="2026-05-22"),
            QuoteRejectionReason.EXPIRED_CONTRACT,
        ),
        (_option_chain(strike=0.0), QuoteRejectionReason.NONPOSITIVE_STRIKE),
        (_option_chain(bid=pd.NA), QuoteRejectionReason.MISSING_REQUIRED_PRICE),
        (_option_chain(mid=0.0), QuoteRejectionReason.INVALID_MID),
        (
            _option_chain(strike=90.0, bid=0.9, ask=1.1, mid=1.0),
            QuoteRejectionReason.BELOW_INTRINSIC_TOLERANCE,
        ),
        (
            _option_chain(bid=1.0, ask=5.0, mid=2.0),
            QuoteRejectionReason.SPREAD_TOO_WIDE,
        ),
    ],
)
def test_price_and_contract_rejection_reasons(
    option_chain: pd.DataFrame,
    reason: QuoteRejectionReason,
) -> None:
    _assert_single_rejection(option_chain, reason)


def test_missing_iv_rejects_when_iv_is_required() -> None:
    _assert_single_rejection(
        _option_chain(iv=pd.NA),
        QuoteRejectionReason.MISSING_IV_FOR_IV_REQUIRED_WORKFLOW,
        policy=QuoteCleaningPolicyV1(require_iv=True),
    )


def test_missing_vega_rejects_when_vega_is_required() -> None:
    _assert_single_rejection(
        _option_chain(vega=pd.NA),
        QuoteRejectionReason.MISSING_VEGA_FOR_WEIGHTED_CALIBRATION,
        policy=QuoteCleaningPolicyV1(require_vega=True),
    )


def test_missing_iv_is_accepted_when_iv_is_not_required() -> None:
    result = clean_option_quotes(_option_chain(iv=pd.NA), _market_inputs())

    assert len(result.cleaned_quotes) == 1
    assert result.rejected_quotes.empty


def test_missing_vega_is_accepted_when_vega_is_not_required() -> None:
    result = clean_option_quotes(_option_chain(vega=pd.NA), _market_inputs())

    assert len(result.cleaned_quotes) == 1
    assert result.rejected_quotes.empty


@pytest.mark.parametrize(
    ("option_chain", "reason"),
    [
        (
            _option_chain(strike=0.0, bid=-1.0, ask=0.0, mid=0.0),
            QuoteRejectionReason.NONPOSITIVE_STRIKE,
        ),
        (
            _option_chain(expiry="2026-05-22", bid=-1.0, ask=0.0, mid=0.0),
            QuoteRejectionReason.EXPIRED_CONTRACT,
        ),
        (
            _option_chain(bid=pd.NA, mid=0.0),
            QuoteRejectionReason.MISSING_REQUIRED_PRICE,
        ),
    ],
)
def test_primary_reason_priority(
    option_chain: pd.DataFrame,
    reason: QuoteRejectionReason,
) -> None:
    _assert_single_rejection(option_chain, reason)


def test_all_quotes_rejected_warning_when_every_input_row_is_rejected() -> None:
    result = clean_option_quotes(_option_chain(bid=-1.0), _market_inputs())

    assert result.warnings == ("all_quotes_rejected",)


def test_empty_option_chain_returns_empty_canonical_frames() -> None:
    empty_option_chain = normalize_option_chain(
        pd.DataFrame(columns=OPTION_CHAIN_COLUMNS)
    )

    result = clean_option_quotes(empty_option_chain, _market_inputs())

    assert result.cleaned_quotes.empty
    assert result.rejected_quotes.empty
    _assert_cleaned_schema(result.cleaned_quotes)
    _assert_rejected_schema(result.rejected_quotes)
    assert result.reason_counts == {}
    assert result.warnings == ()


def test_invalid_policy_day_count_raises_value_error() -> None:
    with pytest.raises(ValueError, match="ACT/365"):
        clean_option_quotes(
            _option_chain(),
            _market_inputs(),
            policy=QuoteCleaningPolicyV1(day_count="ACT/360"),
        )


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
    if name.startswith("option_pricing.marketdata.providers"):
        return True
    if name.startswith("option_pricing.") and any(
        part in DISALLOWED_IMPORT_PARTS for part in name.split(".")
    ):
        return True
    return False


def test_s3_files_do_not_import_providers_network_pricing_heston_or_cli() -> None:
    forbidden: dict[str, list[str]] = {
        path.as_posix(): [
            name for name in _imported_names(path) if _is_disallowed_import(name)
        ]
        for path in S3_FILES
    }

    assert forbidden == {path.as_posix(): [] for path in S3_FILES}
