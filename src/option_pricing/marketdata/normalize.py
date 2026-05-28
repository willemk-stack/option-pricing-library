"""Normalization contracts for marketdata DataFrame inputs."""

from __future__ import annotations

import math

import pandas as pd

from option_pricing.marketdata.schemas import (
    MARKET_INPUTS_COLUMNS,
    OPTION_CHAIN_COLUMNS,
    DatasetName,
)
from option_pricing.marketdata.validation import (
    coerce_frame,
    order_columns,
    validate_dtypes,
)

_OPTION_RIGHT_ALIASES = {
    "c": "call",
    "call": "call",
    "p": "put",
    "put": "put",
}


def normalize_market_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw market inputs into the existing ``market_inputs`` schema."""

    _require_frame(frame, "market_inputs")
    if len(frame) != 1:
        raise ValueError(
            f"market_inputs must contain exactly one row; found {len(frame)}"
        )

    coerced = coerce_frame(frame, DatasetName.MARKET_INPUTS, allow_extra=True)
    _validate_market_inputs_values(coerced)

    out = order_columns(coerced, DatasetName.MARKET_INPUTS).loc[
        :, list(MARKET_INPUTS_COLUMNS)
    ]
    validate_dtypes(out, DatasetName.MARKET_INPUTS, allow_extra=False)
    return out.reset_index(drop=True)


def normalize_option_chain(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw option chains into the existing ``option_chain`` schema."""

    _require_frame(frame, "option_chain")

    with_mid = _fill_missing_mid(frame)
    coerced = coerce_frame(with_mid, DatasetName.OPTION_CHAIN, allow_extra=True)
    coerced["right"] = _normalize_option_rights(coerced["right"])
    _validate_unique_contract_symbols(coerced)

    out = (
        order_columns(coerced, DatasetName.OPTION_CHAIN)
        .loc[:, list(OPTION_CHAIN_COLUMNS)]
        .sort_values(
            ["expiry", "strike", "right", "contract_symbol"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    validate_dtypes(out, DatasetName.OPTION_CHAIN, allow_extra=False)
    return out


def _require_frame(frame: pd.DataFrame, dataset_name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"{dataset_name} input must be a pandas DataFrame, "
            f"got {type(frame).__name__}"
        )


def _validate_market_inputs_values(frame: pd.DataFrame) -> None:
    row = frame.iloc[0]

    _require_not_missing(row, "market_inputs", "asof")
    _require_not_missing(row, "market_inputs", "rate_observation_date")

    spot = _required_float(row, "market_inputs", "spot")
    if not math.isfinite(spot) or spot <= 0:
        raise ValueError("market_inputs spot must be finite and > 0")

    rate = _required_float(row, "market_inputs", "rate")
    if not math.isfinite(rate):
        raise ValueError("market_inputs rate must be finite")

    dividend_yield = _required_float(row, "market_inputs", "dividend_yield")
    if not math.isfinite(dividend_yield):
        raise ValueError("market_inputs dividend_yield must be finite")

    rate_compounding = _required_text(row, "market_inputs", "rate_compounding")
    if rate_compounding != "continuous":
        raise ValueError(
            "market_inputs rate_compounding must be 'continuous'; "
            f"got {rate_compounding!r}"
        )

    day_count = _required_text(row, "market_inputs", "day_count")
    if day_count != "ACT/365":
        raise ValueError(
            f"market_inputs day_count must be 'ACT/365'; got {day_count!r}"
        )


def _required_float(row: pd.Series, dataset_name: str, column: str) -> float:
    value = row[column]
    if pd.isna(value):
        return math.nan

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{dataset_name} {column} must be numeric") from exc


def _required_text(row: pd.Series, dataset_name: str, column: str) -> str:
    value = row[column]
    if pd.isna(value):
        raise ValueError(f"{dataset_name} {column} must not be missing")
    return str(value)


def _require_not_missing(row: pd.Series, dataset_name: str, column: str) -> None:
    if pd.isna(row[column]):
        raise ValueError(f"{dataset_name} {column} must not be missing")


def _fill_missing_mid(frame: pd.DataFrame) -> pd.DataFrame:
    if "mid" not in frame.columns:
        if "bid" not in frame.columns or "ask" not in frame.columns:
            return frame

        out = frame.copy()
        out["mid"] = _compute_mid_from_bid_ask(out)
        return out

    missing_mid = frame["mid"].isna()
    if (
        "bid" not in frame.columns
        or "ask" not in frame.columns
        or not bool(missing_mid.any())
    ):
        return frame

    out = frame.copy()
    out.loc[missing_mid, "mid"] = _compute_mid_from_bid_ask(out).loc[missing_mid]
    return out


def _compute_mid_from_bid_ask(frame: pd.DataFrame) -> pd.Series:
    try:
        bid = pd.to_numeric(frame["bid"], errors="raise")
        ask = pd.to_numeric(frame["ask"], errors="raise")
    except Exception as exc:
        raise TypeError(
            "Could not compute missing option_chain mid from bid/ask"
        ) from exc

    return (bid + ask) / 2


def _normalize_option_rights(rights: pd.Series) -> pd.Series:
    normalized = rights.astype("string").str.strip().str.lower()
    if bool(normalized.isna().any()):
        raise ValueError("option_chain right values must not be missing")

    invalid = sorted(
        normalized[~normalized.isin(tuple(_OPTION_RIGHT_ALIASES))]
        .dropna()
        .unique()
        .tolist()
    )
    if invalid:
        raise ValueError(
            "option_chain has invalid right values: "
            f"{invalid}. Expected call/put or C/P aliases."
        )

    return normalized.map(_OPTION_RIGHT_ALIASES).astype("string")


def _validate_unique_contract_symbols(frame: pd.DataFrame) -> None:
    duplicated = frame["contract_symbol"].duplicated(keep=False)
    if not bool(duplicated.any()):
        return

    symbols = sorted(
        frame.loc[duplicated, "contract_symbol"]
        .astype("string")
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    raise ValueError(f"option_chain has duplicate contract_symbol values: {symbols}")


__all__ = [
    "normalize_market_inputs",
    "normalize_option_chain",
]
