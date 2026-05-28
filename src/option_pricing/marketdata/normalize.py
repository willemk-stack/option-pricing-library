"""Normalization contracts for marketdata DataFrame inputs."""

from __future__ import annotations

import pandas as pd


def normalize_market_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw market inputs into the existing ``market_inputs`` schema."""

    raise NotImplementedError(
        "Phase A3-S1 defines the market_inputs normalization contract only; "
        "implementation is deferred."
    )


def normalize_option_chain(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw option chains into the existing ``option_chain`` schema."""

    raise NotImplementedError(
        "Phase A3-S1 defines the option_chain normalization contract only; "
        "implementation is deferred."
    )


__all__ = [
    "normalize_market_inputs",
    "normalize_option_chain",
]
