from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pandas as pd
import pytest

from option_pricing.marketdata.cleaning import (
    QuoteCleaningPolicyV1,
    QuoteCleaningResult,
    QuoteRejectionReason,
)

A3_SOURCE_FILES = (
    Path("src/option_pricing/marketdata/cleaning.py"),
    Path("src/option_pricing/marketdata/normalize.py"),
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
    "option_pricing.marketdata.providers.alpaca",
    "option_pricing.marketdata.providers.fred",
    "option_pricing.marketdata.providers.yahoo",
}
DISALLOWED_IMPORT_PARTS = {
    "calibration",
    "heston",
    "pricing",
    "pricers",
}


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


def test_quote_rejection_reason_values_match_contract() -> None:
    assert [reason.value for reason in QuoteRejectionReason] == [
        "negative_bid",
        "nonpositive_ask",
        "crossed_market",
        "expired_contract",
        "nonpositive_strike",
        "missing_required_price",
        "invalid_mid",
        "below_intrinsic_tolerance",
        "spread_too_wide",
        "missing_iv_for_iv_required_workflow",
        "missing_vega_for_weighted_calibration",
    ]


def test_quote_cleaning_policy_v1_defaults_match_contract() -> None:
    policy = QuoteCleaningPolicyV1()

    assert tuple(field.name for field in fields(QuoteCleaningPolicyV1)) == (
        "max_relative_spread",
        "intrinsic_tolerance",
        "require_iv",
        "require_vega",
        "day_count",
    )
    assert policy.max_relative_spread == pytest.approx(1.00)
    assert policy.intrinsic_tolerance == pytest.approx(1e-8)
    assert policy.require_iv is False
    assert policy.require_vega is False
    assert policy.day_count == "ACT/365"


def test_quote_cleaning_policy_v1_is_frozen() -> None:
    policy = QuoteCleaningPolicyV1()

    with pytest.raises(FrozenInstanceError):
        policy.require_iv = True


def test_quote_cleaning_result_accepts_empty_contract_frames() -> None:
    cleaned_quotes = pd.DataFrame()
    rejected_quotes = pd.DataFrame()

    result = QuoteCleaningResult(
        cleaned_quotes=cleaned_quotes,
        rejected_quotes=rejected_quotes,
        reason_counts={},
        warnings=(),
    )

    assert result.cleaned_quotes is cleaned_quotes
    assert result.rejected_quotes is rejected_quotes
    assert result.reason_counts == {}
    assert result.warnings == ()


def test_a3_s1_files_do_not_import_providers_network_pricing_or_heston() -> None:
    for path in A3_SOURCE_FILES:
        forbidden = [
            name for name in _imported_names(path) if _is_disallowed_import(name)
        ]

        assert forbidden == []
