from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from option_pricing.marketdata.gold import (
    heston_quote_set_from_frame,
    market_data_snapshot_from_json,
)
from option_pricing.marketdata.schemas import DatasetName
from option_pricing.marketdata.validation import validate_dtypes


@dataclass(frozen=True, slots=True)
class ProviderSnapshotBundleValidationResult:
    """Summary from reading provider-backed model-consumption artifacts."""

    underlying: str
    cleaned_quote_count: int
    heston_quote_count: int
    spot: float
    rate: float
    dividend_yield: float


def validate_provider_snapshot_bundle(
    *,
    market_data_path: Path,
    cleaned_quotes_path: Path,
    heston_quotes_path: Path,
) -> ProviderSnapshotBundleValidationResult:
    """Read provider-backed artifacts and verify model-facing compatibility."""

    market_snapshot = _read_market_data_snapshot(market_data_path)
    cleaned_quotes = _read_frame(
        cleaned_quotes_path,
        DatasetName.CLEANED_QUOTES,
        frame_name="cleaned_quotes",
    )
    heston_quotes = _read_frame(
        heston_quotes_path,
        DatasetName.HESTON_QUOTES,
        frame_name="heston_quotes",
    )
    try:
        quote_set = heston_quote_set_from_frame(
            heston_quotes,
            market_snapshot.market_data,
        )
    except Exception as exc:
        raise ValueError(
            "provider snapshot heston_quotes.parquet could not be consumed by "
            "the Heston quote contract"
        ) from exc

    cleaned_underlyings = set(cleaned_quotes["underlying"].astype(str))
    heston_underlyings = set(heston_quotes["underlying"].astype(str))
    if not cleaned_underlyings:
        raise ValueError("provider snapshot cleaned_quotes.parquet is empty")
    if cleaned_underlyings != heston_underlyings:
        raise ValueError(
            "provider snapshot cleaned_quotes and heston_quotes underlyings differ: "
            f"cleaned={sorted(cleaned_underlyings)!r}, "
            f"heston={sorted(heston_underlyings)!r}"
        )
    if quote_set.mid.size != len(heston_quotes):
        raise ValueError(
            "provider snapshot Heston quote-set size does not match "
            "heston_quotes.parquet rows"
        )

    return ProviderSnapshotBundleValidationResult(
        underlying=next(iter(cleaned_underlyings)),
        cleaned_quote_count=int(len(cleaned_quotes)),
        heston_quote_count=int(len(heston_quotes)),
        spot=float(market_snapshot.market_data.spot),
        rate=float(market_snapshot.market_data.rate),
        dividend_yield=float(market_snapshot.market_data.dividend_yield),
    )


def _read_market_data_snapshot(path: Path) -> Any:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Unable to read provider market_data.json at {path}") from exc
    try:
        return market_data_snapshot_from_json(payload)
    except Exception as exc:
        raise ValueError(
            f"provider market_data.json is not compatible with MarketData: {path}"
        ) from exc


def _read_frame(
    path: Path,
    dataset_name: DatasetName,
    *,
    frame_name: str,
) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise ValueError(
            f"Unable to read provider {frame_name} artifact at {path}"
        ) from exc
    try:
        validate_dtypes(frame, dataset_name, allow_extra=False)
    except Exception as exc:
        raise ValueError(
            f"provider {frame_name} artifact does not match {dataset_name.value} schema"
        ) from exc
    if frame.empty:
        raise ValueError(f"provider {frame_name} artifact is empty")
    return frame.reset_index(drop=True)


__all__ = [
    "ProviderSnapshotBundleValidationResult",
    "validate_provider_snapshot_bundle",
]
