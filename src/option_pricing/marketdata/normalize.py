"""Normalization contracts for marketdata DataFrame inputs."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, cast

import pandas as pd

from option_pricing.marketdata.schemas import (
    EQUITY_BARS_COLUMNS,
    EQUITY_QUOTES_COLUMNS,
    FRED_SERIES_COLUMNS,
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
_MISSING = object()

_ALPACA_QUOTE_TS_ALIASES = ("timestamp", "t")
_ALPACA_BID_ALIASES = ("bid_price", "bp", "bid")
_ALPACA_ASK_ALIASES = ("ask_price", "ap", "ask")
_ALPACA_BID_SIZE_ALIASES = ("bid_size", "bs", "bidsize")
_ALPACA_ASK_SIZE_ALIASES = ("ask_size", "as", "asksize")
_ALPACA_SYMBOL_ALIASES = ("symbol", "S")
_ALPACA_BAR_TS_ALIASES = ("timestamp", "t")
_ALPACA_BAR_OPEN_ALIASES = ("open", "o")
_ALPACA_BAR_HIGH_ALIASES = ("high", "h")
_ALPACA_BAR_LOW_ALIASES = ("low", "l")
_ALPACA_BAR_CLOSE_ALIASES = ("close", "c")
_ALPACA_BAR_VOLUME_ALIASES = ("volume", "v")
_ALPACA_BAR_TRADE_COUNT_ALIASES = ("trade_count", "n", "tradecount")
_ALPACA_BAR_VWAP_ALIASES = ("vwap", "vw")
_ALPACA_BAR_TIMEFRAME_ALIASES = ("timeframe", "tf")
_ALPACA_OPTION_CONTRACT_SYMBOL_ALIASES = (
    "contract_symbol",
    "contractSymbol",
    "symbol",
    "option_symbol",
    "optionSymbol",
)
_ALPACA_OPTION_EXPIRY_ALIASES = (
    "expiry",
    "expiration",
    "expiration_date",
    "expirationDate",
)
_ALPACA_OPTION_STRIKE_ALIASES = ("strike", "strike_price", "strikePrice")
_ALPACA_OPTION_RIGHT_ALIASES = ("right", "type", "option_type", "optionType")
_ALPACA_OPTION_CONTRACT_ALIASES = ("contract", "option_contract", "optionContract")
_ALPACA_OPTION_LATEST_QUOTE_ALIASES = ("latest_quote", "latestQuote", "quote")
_ALPACA_OPTION_LATEST_TRADE_ALIASES = ("latest_trade", "latestTrade", "trade")
_ALPACA_OPTION_TRADE_PRICE_ALIASES = ("price", "p", "trade_price", "tradePrice")
_ALPACA_OPTION_IV_ALIASES = ("implied_volatility", "impliedVolatility", "iv")
_ALPACA_OPTION_GREEKS_ALIASES = ("greeks", "greek")
_ALPACA_OPTION_DELTA_ALIASES = ("delta",)
_ALPACA_OPTION_GAMMA_ALIASES = ("gamma",)
_ALPACA_OPTION_THETA_ALIASES = ("theta",)
_ALPACA_OPTION_VEGA_ALIASES = ("vega",)
_ALPACA_OPTION_RHO_ALIASES = ("rho",)
_ALPACA_OPTION_OPEN_INTEREST_ALIASES = (
    "open_interest",
    "openInterest",
    "oi",
)
_OCC_CONTRACT_SYMBOL_RE = re.compile(r"^([A-Z0-9.]+?)(\d{6})([CP])(\d{8})$")
_PROVIDER_REJECTED_CONTRACT_COLUMNS = (
    "underlying",
    "contract_symbol",
    "payload_contract_key",
    "asof",
    "source",
    "feed",
    "rejection_stage",
    "reason",
    "rejection_detail",
    "raw_quote_timestamp",
    "raw_bid",
    "raw_ask",
    "raw_expiry",
    "raw_strike",
    "raw_right",
)

_ALPACA_BAR_FIELD_ALIASES = (
    *_ALPACA_BAR_TS_ALIASES,
    *_ALPACA_BAR_OPEN_ALIASES,
    *_ALPACA_BAR_HIGH_ALIASES,
    *_ALPACA_BAR_LOW_ALIASES,
    *_ALPACA_BAR_CLOSE_ALIASES,
    *_ALPACA_BAR_VOLUME_ALIASES,
    *_ALPACA_BAR_TRADE_COUNT_ALIASES,
    *_ALPACA_BAR_VWAP_ALIASES,
    *_ALPACA_BAR_TIMEFRAME_ALIASES,
)


@dataclass(frozen=True, slots=True)
class _AlpacaOptionContractMetadata:
    contract_symbol: str
    expiry: date
    strike: float
    right: str


@dataclass(frozen=True, slots=True)
class AlpacaOptionChainNormalizationAudit:
    option_chain: pd.DataFrame
    rejected_contracts: pd.DataFrame


def normalize_alpaca_latest_quotes(
    payload: Mapping[str, Any],
    *,
    asof: str | pd.Timestamp,
) -> pd.DataFrame:
    """Normalize Alpaca latest equity quotes into the ``equity_quotes`` schema."""

    if _is_missing_value(asof):
        raise ValueError("alpaca latest quotes asof must not be missing")

    rows = [
        _alpaca_latest_quote_row(symbol, quote, asof=asof)
        for symbol, quote in _alpaca_latest_quote_items(payload)
    ]
    frame = pd.DataFrame(rows, columns=list(EQUITY_QUOTES_COLUMNS))
    coerced = coerce_frame(frame, DatasetName.EQUITY_QUOTES, allow_extra=False)
    out = (
        order_columns(coerced, DatasetName.EQUITY_QUOTES)
        .loc[:, list(EQUITY_QUOTES_COLUMNS)]
        .sort_values(["symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    validate_dtypes(out, DatasetName.EQUITY_QUOTES, allow_extra=False)
    return out


def normalize_alpaca_bars(
    payload: Mapping[str, Any],
    *,
    asof: str | pd.Timestamp,
) -> pd.DataFrame:
    """Normalize Alpaca historical equity bars into the ``equity_bars`` schema."""

    if not isinstance(payload, Mapping):
        raise TypeError(
            "alpaca equity bars payload must be a mapping, "
            f"got {type(payload).__name__}"
        )
    if _is_missing_value(asof):
        raise ValueError("alpaca equity bars asof must not be missing")

    payload_timeframe = _optional_payload_text(payload, "timeframe")
    rows = [
        _alpaca_bar_row(
            symbol,
            bar,
            asof=asof,
            payload_timeframe=payload_timeframe,
        )
        for symbol, bar in _alpaca_bar_items(payload)
    ]
    frame = pd.DataFrame(rows, columns=list(EQUITY_BARS_COLUMNS))
    coerced = coerce_frame(frame, DatasetName.EQUITY_BARS, allow_extra=False)
    out = (
        order_columns(coerced, DatasetName.EQUITY_BARS)
        .loc[:, list(EQUITY_BARS_COLUMNS)]
        .sort_values(["symbol", "bar_ts"], kind="mergesort")
        .reset_index(drop=True)
    )
    validate_dtypes(out, DatasetName.EQUITY_BARS, allow_extra=False)
    return out


def normalize_alpaca_option_chain(
    payload: Mapping[str, Any],
    *,
    underlying: str | None = None,
    asof: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Normalize Alpaca option chain snapshots into the ``option_chain`` schema."""

    return normalize_alpaca_option_chain_with_audit(
        payload,
        underlying=underlying,
        asof=asof,
    ).option_chain


def normalize_alpaca_option_chain_with_audit(
    payload: Mapping[str, Any],
    *,
    underlying: str | None = None,
    asof: str | pd.Timestamp | None = None,
    feed: str | None = None,
) -> AlpacaOptionChainNormalizationAudit:
    """Normalize Alpaca option chains and retain pre-cleaning contract rejections."""

    if not isinstance(payload, Mapping):
        raise TypeError(
            "alpaca option chain payload must be a mapping, "
            f"got {type(payload).__name__}"
        )

    resolved_asof: object = asof
    if _is_missing_value(resolved_asof):
        resolved_asof = payload.get("asof", _MISSING)
    if _is_missing_value(resolved_asof):
        raise ValueError("alpaca option chain asof must not be missing")

    resolved_underlying: object = underlying
    if _is_missing_value(resolved_underlying):
        resolved_underlying = payload.get("underlying", _MISSING)
    cleaned_underlying = _clean_alpaca_option_underlying(resolved_underlying)
    source = _optional_diagnostic_text(payload.get("source", "alpaca"))
    resolved_feed = feed
    if _is_missing_value(resolved_feed):
        resolved_feed = payload.get("feed", _MISSING)
    cleaned_feed = _optional_diagnostic_text(resolved_feed)

    rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    for default_symbol, contract in _alpaca_option_contract_items(payload):
        row, rejected = _alpaca_option_chain_row_with_audit(
            cleaned_underlying,
            default_symbol,
            contract,
            asof=resolved_asof,
            source=source,
            feed=cleaned_feed,
        )
        if row is not None:
            rows.append(row)
        elif rejected is not None:
            rejected_rows.append(rejected)

    if not rows:
        raise ValueError(
            "alpaca option chain has no contracts with usable latest quote bid/ask"
        )

    frame = pd.DataFrame(rows, columns=list(OPTION_CHAIN_COLUMNS))
    coerced = coerce_frame(frame, DatasetName.OPTION_CHAIN, allow_extra=False)
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
    return AlpacaOptionChainNormalizationAudit(
        option_chain=out,
        rejected_contracts=_provider_rejected_contracts_frame(rejected_rows),
    )


def normalize_fred_observations(
    payload: Mapping[str, Any],
    *,
    series_id: str,
    asof: str | pd.Timestamp,
) -> pd.DataFrame:
    """Normalize raw FRED observations into the existing ``fred_series`` schema."""

    if not isinstance(payload, Mapping):
        raise TypeError(
            "fred_series payload must be a mapping, " f"got {type(payload).__name__}"
        )

    rows = [
        {
            "series_id": series_id,
            "observation_date": _fred_required_observation_text(
                observation,
                "date",
            ),
            "value": _normalize_fred_value(observation.get("value")),
            "realtime_start": _fred_required_observation_text(
                observation,
                "realtime_start",
            ),
            "realtime_end": _fred_required_observation_text(
                observation,
                "realtime_end",
            ),
            "source": "fred",
            "asof": asof,
        }
        for observation in _fred_observations(payload)
    ]
    frame = pd.DataFrame(rows, columns=list(FRED_SERIES_COLUMNS))
    coerced = coerce_frame(frame, DatasetName.FRED_SERIES, allow_extra=False)
    out = order_columns(coerced, DatasetName.FRED_SERIES).loc[
        :, list(FRED_SERIES_COLUMNS)
    ]
    validate_dtypes(out, DatasetName.FRED_SERIES, allow_extra=False)
    return out.reset_index(drop=True)


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


def _alpaca_option_contract_items(payload: Mapping[str, Any]) -> list[tuple[Any, Any]]:
    if "contracts" in payload:
        contracts_payload = payload["contracts"]
    elif "snapshots" in payload:
        contracts_payload = payload["snapshots"]
    else:
        raise ValueError(
            "alpaca option chain payload must contain a contracts mapping or list"
        )

    contracts_payload = getattr(contracts_payload, "data", contracts_payload)
    if isinstance(contracts_payload, Mapping):
        if not contracts_payload:
            raise ValueError("alpaca option chain payload must contain contracts")
        return list(contracts_payload.items())
    if isinstance(contracts_payload, list | tuple):
        if not contracts_payload:
            raise ValueError("alpaca option chain payload must contain contracts")
        return [(_MISSING, contract) for contract in contracts_payload]

    raise ValueError(
        "alpaca option chain contracts must be provided as a mapping or list"
    )


def _alpaca_option_chain_row(
    underlying: str,
    default_symbol: Any,
    contract: Any,
    *,
    asof: object,
) -> dict[str, object] | None:
    row, _ = _alpaca_option_chain_row_with_audit(
        underlying,
        default_symbol,
        contract,
        asof=asof,
        source="alpaca",
        feed=pd.NA,
    )
    return row


def _alpaca_option_chain_row_with_audit(
    underlying: str,
    default_symbol: Any,
    contract: Any,
    *,
    asof: object,
    source: object,
    feed: object,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    raw_contract_symbol = _option_metadata_value(
        contract,
        _ALPACA_OPTION_CONTRACT_SYMBOL_ALIASES,
        default=default_symbol,
    )
    raw_expiry = _option_metadata_value(contract, _ALPACA_OPTION_EXPIRY_ALIASES)
    raw_strike = _option_metadata_value(contract, _ALPACA_OPTION_STRIKE_ALIASES)
    raw_right = _option_metadata_value(contract, _ALPACA_OPTION_RIGHT_ALIASES)
    quote = _option_snapshot_value(contract, _ALPACA_OPTION_LATEST_QUOTE_ALIASES)
    raw_quote_ts = _MISSING
    raw_bid = _MISSING
    raw_ask = _MISSING
    if not _is_missing_value(quote):
        raw_quote_ts = _quote_value(quote, _ALPACA_QUOTE_TS_ALIASES)
        raw_bid = _quote_value(quote, _ALPACA_BID_ALIASES)
        raw_ask = _quote_value(quote, _ALPACA_ASK_ALIASES)

    try:
        metadata = _alpaca_option_contract_metadata(default_symbol, contract)
    except Exception as exc:
        return None, _provider_rejected_contract_row(
            underlying=underlying,
            contract_symbol=raw_contract_symbol,
            payload_contract_key=default_symbol,
            asof=asof,
            source=source,
            feed=feed,
            reason="invalid_contract_metadata",
            rejection_detail=str(exc),
            raw_quote_timestamp=raw_quote_ts,
            raw_bid=raw_bid,
            raw_ask=raw_ask,
            raw_expiry=raw_expiry,
            raw_strike=raw_strike,
            raw_right=raw_right,
        )

    if _is_missing_value(quote):
        return None, _provider_rejected_contract_row(
            underlying=underlying,
            contract_symbol=metadata.contract_symbol,
            payload_contract_key=default_symbol,
            asof=asof,
            source=source,
            feed=feed,
            reason="missing_required_price",
            rejection_detail="latest_quote is missing",
            raw_quote_timestamp=raw_quote_ts,
            raw_bid=raw_bid,
            raw_ask=raw_ask,
            raw_expiry=raw_expiry,
            raw_strike=raw_strike,
            raw_right=raw_right,
        )

    if _is_missing_value(raw_quote_ts):
        return None, _provider_rejected_contract_row(
            underlying=underlying,
            contract_symbol=metadata.contract_symbol,
            payload_contract_key=default_symbol,
            asof=asof,
            source=source,
            feed=feed,
            reason="missing_quote_timestamp",
            rejection_detail="latest_quote timestamp is missing",
            raw_quote_timestamp=raw_quote_ts,
            raw_bid=raw_bid,
            raw_ask=raw_ask,
            raw_expiry=raw_expiry,
            raw_strike=raw_strike,
            raw_right=raw_right,
        )

    missing_price_fields = [
        field_name
        for field_name, value in (("bid", raw_bid), ("ask", raw_ask))
        if _is_missing_value(value)
    ]
    if missing_price_fields:
        return None, _provider_rejected_contract_row(
            underlying=underlying,
            contract_symbol=metadata.contract_symbol,
            payload_contract_key=default_symbol,
            asof=asof,
            source=source,
            feed=feed,
            reason="missing_required_price",
            rejection_detail=(
                "latest_quote is missing " + ", ".join(missing_price_fields)
            ),
            raw_quote_timestamp=raw_quote_ts,
            raw_bid=raw_bid,
            raw_ask=raw_ask,
            raw_expiry=raw_expiry,
            raw_strike=raw_strike,
            raw_right=raw_right,
        )

    metadata = _alpaca_option_contract_metadata(default_symbol, contract)
    bid = _optional_finite_option_quote_number(
        quote,
        _ALPACA_BID_ALIASES,
    )
    ask = _optional_finite_option_quote_number(
        quote,
        _ALPACA_ASK_ALIASES,
    )
    quote_ts = raw_quote_ts
    if bid is None or ask is None:
        return None, _provider_rejected_contract_row(
            underlying=underlying,
            contract_symbol=metadata.contract_symbol,
            payload_contract_key=default_symbol,
            asof=asof,
            source=source,
            feed=feed,
            reason="unusable_bid_ask",
            rejection_detail="latest_quote bid/ask must be finite numeric values",
            raw_quote_timestamp=raw_quote_ts,
            raw_bid=raw_bid,
            raw_ask=raw_ask,
            raw_expiry=raw_expiry,
            raw_strike=raw_strike,
            raw_right=raw_right,
        )

    return {
        "underlying": underlying,
        "contract_symbol": metadata.contract_symbol,
        "quote_ts": quote_ts,
        "expiry": metadata.expiry,
        "strike": metadata.strike,
        "right": metadata.right,
        "bid": bid,
        "ask": ask,
        "mid": (bid + ask) / 2,
        "last": _alpaca_option_last_price(contract),
        "iv": _optional_option_snapshot_value(contract, _ALPACA_OPTION_IV_ALIASES),
        "delta": _alpaca_option_greek_value(contract, _ALPACA_OPTION_DELTA_ALIASES),
        "gamma": _alpaca_option_greek_value(contract, _ALPACA_OPTION_GAMMA_ALIASES),
        "theta": _alpaca_option_greek_value(contract, _ALPACA_OPTION_THETA_ALIASES),
        "vega": _alpaca_option_greek_value(contract, _ALPACA_OPTION_VEGA_ALIASES),
        "rho": _alpaca_option_greek_value(contract, _ALPACA_OPTION_RHO_ALIASES),
        "open_interest": _optional_option_snapshot_value(
            contract,
            _ALPACA_OPTION_OPEN_INTEREST_ALIASES,
        ),
        "source": "alpaca",
        "asof": asof,
    }, None


def _provider_rejected_contract_row(
    *,
    underlying: str,
    contract_symbol: object,
    payload_contract_key: object,
    asof: object,
    source: object,
    feed: object,
    reason: str,
    rejection_detail: str,
    raw_quote_timestamp: object,
    raw_bid: object,
    raw_ask: object,
    raw_expiry: object,
    raw_strike: object,
    raw_right: object,
) -> dict[str, object]:
    return {
        "underlying": underlying,
        "contract_symbol": _optional_diagnostic_text(contract_symbol),
        "payload_contract_key": _optional_diagnostic_text(payload_contract_key),
        "asof": asof,
        "source": _optional_diagnostic_text(source),
        "feed": _optional_diagnostic_text(feed),
        "rejection_stage": "provider_normalization",
        "reason": reason,
        "rejection_detail": rejection_detail,
        "raw_quote_timestamp": _optional_diagnostic_text(raw_quote_timestamp),
        "raw_bid": _optional_diagnostic_text(raw_bid),
        "raw_ask": _optional_diagnostic_text(raw_ask),
        "raw_expiry": _optional_diagnostic_text(raw_expiry),
        "raw_strike": _optional_diagnostic_text(raw_strike),
        "raw_right": _optional_diagnostic_text(raw_right),
    }


def _provider_rejected_contracts_frame(
    rows: list[dict[str, object]],
) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=list(_PROVIDER_REJECTED_CONTRACT_COLUMNS))
    for column in _PROVIDER_REJECTED_CONTRACT_COLUMNS:
        if column == "asof":
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
        else:
            frame[column] = frame[column].astype("string")
    return frame.reset_index(drop=True)


def _optional_diagnostic_text(value: object) -> object:
    if _is_missing_value(value):
        return pd.NA
    raw_value = getattr(value, "value", value)
    if isinstance(raw_value, pd.Timestamp):
        return raw_value.isoformat()
    if isinstance(raw_value, date):
        return raw_value.isoformat()
    text = str(raw_value).strip()
    return text if text else pd.NA


def _alpaca_option_contract_metadata(
    default_symbol: Any,
    contract: Any,
) -> _AlpacaOptionContractMetadata:
    contract_symbol = _clean_alpaca_option_contract_symbol(
        _option_metadata_value(
            contract,
            _ALPACA_OPTION_CONTRACT_SYMBOL_ALIASES,
            default=default_symbol,
        )
    )
    expiry_value = _option_metadata_value(contract, _ALPACA_OPTION_EXPIRY_ALIASES)
    strike_value = _option_metadata_value(contract, _ALPACA_OPTION_STRIKE_ALIASES)
    right_value = _option_metadata_value(contract, _ALPACA_OPTION_RIGHT_ALIASES)

    parsed: _AlpacaOptionContractMetadata | None = None
    if (
        _is_missing_value(expiry_value)
        or _is_missing_value(strike_value)
        or _is_missing_value(right_value)
    ):
        parsed = _parse_occ_contract_symbol(contract_symbol)

    expiry = (
        parsed.expiry
        if _is_missing_value(expiry_value) and parsed is not None
        else _normalize_option_expiry(expiry_value, contract_symbol)
    )
    strike = (
        parsed.strike
        if _is_missing_value(strike_value) and parsed is not None
        else _required_positive_option_strike(strike_value, contract_symbol)
    )
    right = (
        parsed.right
        if _is_missing_value(right_value) and parsed is not None
        else _normalize_alpaca_option_right_value(right_value, contract_symbol)
    )

    return _AlpacaOptionContractMetadata(
        contract_symbol=contract_symbol,
        expiry=expiry,
        strike=strike,
        right=right,
    )


def _option_metadata_value(
    contract: Any,
    aliases: tuple[str, ...],
    *,
    default: object = _MISSING,
) -> Any:
    value = _option_snapshot_value(contract, aliases)
    if value is not _MISSING:
        return value

    nested_contract = _option_snapshot_value(contract, _ALPACA_OPTION_CONTRACT_ALIASES)
    if nested_contract is not _MISSING:
        value = _provider_value(nested_contract, aliases)
        if value is not _MISSING:
            return value

    return default


def _parse_occ_contract_symbol(symbol: str) -> _AlpacaOptionContractMetadata:
    normalized_symbol = symbol.replace(" ", "").upper()
    match = _OCC_CONTRACT_SYMBOL_RE.fullmatch(normalized_symbol)
    if match is None:
        raise ValueError(
            f"alpaca option chain could not parse OCC contract symbol {symbol!r}"
        )

    _, expiry_raw, right_raw, strike_raw = match.groups()
    year = 2000 + int(expiry_raw[:2])
    month = int(expiry_raw[2:4])
    day = int(expiry_raw[4:6])
    try:
        expiry = date(year, month, day)
    except ValueError as exc:
        raise ValueError(
            f"alpaca option chain contract symbol {symbol!r} has invalid expiry"
        ) from exc

    strike = int(strike_raw) / 1000
    if strike <= 0:
        raise ValueError(
            f"alpaca option chain contract symbol {symbol!r} has nonpositive strike"
        )

    return _AlpacaOptionContractMetadata(
        contract_symbol=normalized_symbol,
        expiry=expiry,
        strike=strike,
        right=_OPTION_RIGHT_ALIASES[right_raw.lower()],
    )


def _normalize_option_expiry(value: Any, contract_symbol: str) -> date:
    if _is_missing_value(value):
        raise ValueError(
            f"alpaca option chain contract {contract_symbol!r} is missing expiry"
        )
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(
            f"alpaca option chain contract {contract_symbol!r} expiry is invalid"
        ) from exc
    if pd.isna(timestamp):
        raise ValueError(
            f"alpaca option chain contract {contract_symbol!r} expiry is invalid"
        )
    return timestamp.date()


def _required_positive_option_strike(value: Any, contract_symbol: str) -> float:
    if _is_missing_value(value):
        raise ValueError(
            f"alpaca option chain contract {contract_symbol!r} is missing strike"
        )
    try:
        strike = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"alpaca option chain contract {contract_symbol!r} strike must be numeric"
        ) from exc
    if not math.isfinite(strike) or strike <= 0:
        raise ValueError(
            f"alpaca option chain contract {contract_symbol!r} strike must be > 0"
        )
    return strike


def _normalize_alpaca_option_right_value(value: Any, contract_symbol: str) -> str:
    if _is_missing_value(value):
        raise ValueError(
            f"alpaca option chain contract {contract_symbol!r} is missing right"
        )
    raw_value = getattr(value, "value", value)
    text = str(raw_value).strip().lower()
    if text.endswith(".call"):
        text = "call"
    elif text.endswith(".put"):
        text = "put"

    normalized = _OPTION_RIGHT_ALIASES.get(text)
    if normalized is None:
        raise ValueError(
            f"alpaca option chain contract {contract_symbol!r} has invalid right "
            f"{value!r}; expected call/put or C/P aliases"
        )
    return normalized


def _optional_finite_option_quote_number(
    quote: Any,
    aliases: tuple[str, ...],
) -> float | None:
    value = _quote_value(quote, aliases)
    if _is_missing_value(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _optional_option_snapshot_value(contract: Any, aliases: tuple[str, ...]) -> object:
    value = _option_snapshot_value(contract, aliases)
    if _is_missing_value(value):
        return pd.NA
    return value


def _alpaca_option_greek_value(contract: Any, aliases: tuple[str, ...]) -> object:
    value = _option_snapshot_value(contract, aliases)
    if not _is_missing_value(value):
        return value

    greeks = _option_snapshot_value(contract, _ALPACA_OPTION_GREEKS_ALIASES)
    if _is_missing_value(greeks):
        return pd.NA

    value = _provider_value(greeks, aliases)
    if _is_missing_value(value):
        return pd.NA
    return value


def _alpaca_option_last_price(contract: Any) -> object:
    trade = _option_snapshot_value(contract, _ALPACA_OPTION_LATEST_TRADE_ALIASES)
    if _is_missing_value(trade):
        return pd.NA

    value = _provider_value(trade, _ALPACA_OPTION_TRADE_PRICE_ALIASES)
    if _is_missing_value(value):
        return pd.NA
    return value


def _option_snapshot_value(
    contract: Any,
    aliases: tuple[str, ...],
    *,
    default: object = _MISSING,
) -> Any:
    return _provider_value(contract, aliases, default=default)


def _clean_alpaca_option_underlying(value: object) -> str:
    return _required_text_value(value, "alpaca option chain", "underlying").upper()


def _clean_alpaca_option_contract_symbol(value: object) -> str:
    return (
        _required_text_value(value, "alpaca option chain", "contract_symbol")
        .replace(" ", "")
        .upper()
    )


def _alpaca_bar_items(payload: Mapping[str, Any]) -> list[tuple[str, Any]]:
    if not isinstance(payload, Mapping):
        raise TypeError(
            "alpaca equity bars payload must be a mapping, "
            f"got {type(payload).__name__}"
        )

    bars_payload = payload.get("bars", payload.get("bar", payload))
    bars_payload = getattr(bars_payload, "data", bars_payload)

    items: list[tuple[str, Any]] = []
    if _is_bar_record(bars_payload):
        symbol = _payload_or_bar_symbol(payload, bars_payload)
        items.append((symbol, bars_payload))
    elif isinstance(bars_payload, Mapping):
        for symbol, records in bars_payload.items():
            cleaned_symbol = _clean_alpaca_bar_symbol(symbol)
            items.extend(
                (cleaned_symbol, record)
                for record in _bar_records(records, symbol=cleaned_symbol)
            )
    elif isinstance(bars_payload, list | tuple):
        default_symbol = _single_payload_symbol(payload)
        for record in bars_payload:
            symbol = _clean_alpaca_bar_symbol(
                _bar_value(record, _ALPACA_SYMBOL_ALIASES, default=default_symbol)
            )
            items.append((symbol, record))
    else:
        raise ValueError("alpaca equity bars payload must contain bars")

    if not items:
        raise ValueError("alpaca equity bars payload must contain at least one bar")
    return items


def _bar_records(records: Any, *, symbol: str) -> list[Any]:
    records = getattr(records, "data", records)
    if _is_bar_record(records):
        return [records]
    if isinstance(records, list | tuple):
        return list(records)
    raise ValueError(f"alpaca equity bars for {symbol!r} must be bar records")


def _alpaca_bar_row(
    symbol: str,
    bar: Any,
    *,
    asof: str | pd.Timestamp,
    payload_timeframe: str | None,
) -> dict[str, object]:
    bar_symbol = _clean_alpaca_bar_symbol(
        _bar_value(bar, _ALPACA_SYMBOL_ALIASES, default=symbol)
    )
    timeframe = payload_timeframe
    if timeframe is None:
        timeframe = _required_bar_text_value(
            bar,
            "timeframe",
            _ALPACA_BAR_TIMEFRAME_ALIASES,
            symbol=bar_symbol,
        )

    return {
        "symbol": bar_symbol,
        "bar_ts": _required_bar_value(
            bar,
            "timestamp",
            _ALPACA_BAR_TS_ALIASES,
            symbol=bar_symbol,
        ),
        "timeframe": timeframe,
        "open": _required_bar_value(
            bar,
            "open",
            _ALPACA_BAR_OPEN_ALIASES,
            symbol=bar_symbol,
        ),
        "high": _required_bar_value(
            bar,
            "high",
            _ALPACA_BAR_HIGH_ALIASES,
            symbol=bar_symbol,
        ),
        "low": _required_bar_value(
            bar,
            "low",
            _ALPACA_BAR_LOW_ALIASES,
            symbol=bar_symbol,
        ),
        "close": _required_bar_value(
            bar,
            "close",
            _ALPACA_BAR_CLOSE_ALIASES,
            symbol=bar_symbol,
        ),
        "volume": _required_bar_value(
            bar,
            "volume",
            _ALPACA_BAR_VOLUME_ALIASES,
            symbol=bar_symbol,
        ),
        "trade_count": _optional_bar_value(bar, _ALPACA_BAR_TRADE_COUNT_ALIASES),
        "vwap": _optional_bar_value(bar, _ALPACA_BAR_VWAP_ALIASES),
        "source": "alpaca",
        "asof": asof,
    }


def _required_bar_text_value(
    bar: Any,
    field_label: str,
    aliases: tuple[str, ...],
    *,
    symbol: str,
) -> str:
    value = _required_bar_value(bar, field_label, aliases, symbol=symbol)
    return _required_text_value(value, "alpaca equity bar", field_label)


def _required_bar_value(
    bar: Any,
    field_label: str,
    aliases: tuple[str, ...],
    *,
    symbol: str,
) -> Any:
    value = _bar_value(bar, aliases)
    if _is_missing_value(value):
        raise ValueError(f"alpaca equity bar for {symbol!r} is missing {field_label}")
    return value


def _optional_bar_value(bar: Any, aliases: tuple[str, ...]) -> object:
    value = _bar_value(bar, aliases)
    if _is_missing_value(value):
        return pd.NA
    return value


def _payload_or_bar_symbol(payload: Mapping[str, Any], bar: Any) -> str:
    payload_symbol = payload.get("symbol", _MISSING)
    value = _bar_value(bar, _ALPACA_SYMBOL_ALIASES, default=payload_symbol)
    return _clean_alpaca_bar_symbol(value)


def _single_payload_symbol(payload: Mapping[str, Any]) -> str:
    value = payload.get("symbol", _MISSING)
    if not _is_missing_value(value):
        return _clean_alpaca_bar_symbol(value)

    symbols = payload.get("symbols", _MISSING)
    if isinstance(symbols, str):
        return _clean_alpaca_bar_symbol(symbols)
    if isinstance(symbols, list | tuple) and len(symbols) == 1:
        return _clean_alpaca_bar_symbol(symbols[0])

    raise ValueError(
        "alpaca equity bars payload with a bar list must include one symbol"
    )


def _optional_payload_text(payload: Mapping[str, Any], field_name: str) -> str | None:
    value = payload.get(field_name, _MISSING)
    if _is_missing_value(value):
        return None
    return _required_text_value(value, "alpaca equity bars", field_name)


def _is_bar_record(value: object) -> bool:
    if isinstance(value, Mapping):
        if _mapping_value(value, _ALPACA_BAR_FIELD_ALIASES) is not _MISSING:
            return True
        raw_data = value.get("raw_data")
        return (
            isinstance(raw_data, Mapping)
            and _mapping_value(raw_data, _ALPACA_BAR_FIELD_ALIASES) is not _MISSING
        )

    return any(hasattr(value, alias) for alias in _ALPACA_BAR_FIELD_ALIASES)


def _alpaca_latest_quote_items(
    payload: Mapping[str, Any],
) -> list[tuple[str, Any]]:
    if not isinstance(payload, Mapping):
        raise TypeError(
            "alpaca latest quotes payload must be a mapping, "
            f"got {type(payload).__name__}"
        )

    quotes_payload: object
    if "quotes" in payload:
        quotes_payload = payload["quotes"]
    elif "quote" in payload:
        symbol = _required_text_value(
            payload.get("symbol", _MISSING),
            "alpaca latest quote",
            "symbol",
        )
        quotes_payload = {symbol: payload["quote"]}
    else:
        quotes_payload = payload

    quotes_payload = getattr(quotes_payload, "data", quotes_payload)
    if not isinstance(quotes_payload, Mapping):
        raise ValueError("alpaca latest quotes payload must contain a quotes mapping")
    if not quotes_payload:
        raise ValueError("alpaca latest quotes payload must contain at least one quote")

    return [
        (_clean_alpaca_symbol(symbol), quote)
        for symbol, quote in quotes_payload.items()
    ]


def _alpaca_latest_quote_row(
    symbol: str,
    quote: Any,
    *,
    asof: str | pd.Timestamp,
) -> dict[str, object]:
    quote_symbol = _clean_alpaca_symbol(
        _quote_value(quote, _ALPACA_SYMBOL_ALIASES, default=symbol)
    )
    quote_ts = _required_quote_value(
        quote,
        "quote timestamp",
        _ALPACA_QUOTE_TS_ALIASES,
        symbol=quote_symbol,
    )
    bid = _required_finite_quote_number(
        quote,
        "bid",
        _ALPACA_BID_ALIASES,
        symbol=quote_symbol,
    )
    ask = _required_finite_quote_number(
        quote,
        "ask",
        _ALPACA_ASK_ALIASES,
        symbol=quote_symbol,
    )
    bid_size = _optional_quote_value(quote, _ALPACA_BID_SIZE_ALIASES)
    ask_size = _optional_quote_value(quote, _ALPACA_ASK_SIZE_ALIASES)

    return {
        "symbol": quote_symbol,
        "quote_ts": quote_ts,
        "bid": bid,
        "ask": ask,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "mid": (bid + ask) / 2,
        "source": "alpaca",
        "asof": asof,
    }


def _required_quote_value(
    quote: Any,
    field_label: str,
    aliases: tuple[str, ...],
    *,
    symbol: str,
) -> Any:
    value = _quote_value(quote, aliases)
    if _is_missing_value(value):
        raise ValueError(f"alpaca latest quote for {symbol!r} is missing {field_label}")
    return value


def _optional_quote_value(quote: Any, aliases: tuple[str, ...]) -> object:
    value = _quote_value(quote, aliases)
    if _is_missing_value(value):
        return pd.NA
    return value


def _required_finite_quote_number(
    quote: Any,
    field_label: str,
    aliases: tuple[str, ...],
    *,
    symbol: str,
) -> float:
    value = _quote_value(quote, aliases)
    if value is _MISSING or value is None or value is pd.NA:
        raise ValueError(f"alpaca latest quote for {symbol!r} is missing {field_label}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"alpaca latest quote for {symbol!r} {field_label} must be numeric"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(
            f"alpaca latest quote for {symbol!r} {field_label} must be finite"
        )
    return number


def _quote_value(
    quote: Any,
    aliases: tuple[str, ...],
    *,
    default: object = _MISSING,
) -> Any:
    return _provider_value(quote, aliases, default=default)


def _bar_value(
    bar: Any,
    aliases: tuple[str, ...],
    *,
    default: object = _MISSING,
) -> Any:
    return _provider_value(bar, aliases, default=default)


def _provider_value(
    record: Any,
    aliases: tuple[str, ...],
    *,
    default: object = _MISSING,
) -> Any:
    if isinstance(record, Mapping):
        value = _mapping_value(record, aliases)
        if value is not _MISSING:
            return value

        raw_data = record.get("raw_data")
        if isinstance(raw_data, Mapping):
            value = _mapping_value(raw_data, aliases)
            if value is not _MISSING:
                return value
    else:
        for alias in aliases:
            try:
                return getattr(record, alias)
            except AttributeError:
                continue

        raw_data = getattr(record, "raw_data", None)
        if isinstance(raw_data, Mapping):
            value = _mapping_value(raw_data, aliases)
            if value is not _MISSING:
                return value

    return default


def _mapping_value(mapping: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        if alias in mapping:
            return mapping[alias]
    return _MISSING


def _clean_alpaca_symbol(value: object) -> str:
    return _required_text_value(value, "alpaca latest quote", "symbol").upper()


def _clean_alpaca_bar_symbol(value: object) -> str:
    return _required_text_value(value, "alpaca equity bar", "symbol").upper()


def _required_text_value(value: object, dataset_name: str, column: str) -> str:
    if _is_missing_value(value):
        raise ValueError(f"{dataset_name} {column} must not be missing")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{dataset_name} {column} must be a non-empty string")
    return text


def _is_missing_value(value: object) -> bool:
    if value is _MISSING or value is None:
        return True
    try:
        return bool(pd.isna(cast(Any, value)))
    except (TypeError, ValueError):
        return False


def _fred_observations(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise ValueError("fred_series payload must contain an observations list")

    out: list[Mapping[str, Any]] = []
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            raise ValueError(
                "fred_series observations must be objects; "
                f"observation {index} has type {type(observation).__name__}"
            )
        out.append(cast(Mapping[str, Any], observation))
    return out


def _fred_required_observation_text(
    observation: Mapping[str, Any],
    field_name: str,
) -> Any:
    value = observation.get(field_name)
    if value is None:
        raise ValueError(f"fred_series observation is missing {field_name!r}")
    return value


def _normalize_fred_value(value: Any) -> Any:
    if isinstance(value, str) and value.strip() == ".":
        return pd.NA
    return value


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
    "AlpacaOptionChainNormalizationAudit",
    "normalize_alpaca_bars",
    "normalize_alpaca_latest_quotes",
    "normalize_alpaca_option_chain",
    "normalize_alpaca_option_chain_with_audit",
    "normalize_fred_observations",
    "normalize_market_inputs",
    "normalize_option_chain",
]
