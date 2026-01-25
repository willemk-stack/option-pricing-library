"""Metadata extraction utilities for PDE diagnostics.

The diagnostics modules typically work on :class:`~option_pricing.types.PricingInputs`
and want a stable, notebook-friendly set of scalar columns (spot, strike, tau, etc.).

This module aims to be tolerant: if a field is missing, it falls back to NaN.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from option_pricing import PricingInputs


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _kind_str(p: PricingInputs) -> str:
    k = getattr(getattr(p, "spec", None), "kind", None)
    if k is None:
        return ""
    return str(getattr(k, "value", k))


def meta_from_inputs(p: PricingInputs) -> dict[str, object]:
    """Extract a stable set of columns from a PricingInputs instance."""

    market = getattr(p, "market", None)
    spec = getattr(p, "spec", None)

    r = getattr(market, "rate", getattr(p, "r", np.nan))
    q = getattr(market, "dividend_yield", getattr(p, "q", np.nan))

    payout = getattr(spec, "payout", np.nan)

    return {
        "tau": _safe_float(getattr(p, "tau", np.nan)),
        "spot": _safe_float(getattr(p, "S", np.nan)),
        "strike": _safe_float(getattr(p, "K", np.nan)),
        "kind": _kind_str(p),
        "payout": _safe_float(payout) if payout is not np.nan else np.nan,
        "sigma": _safe_float(getattr(p, "sigma", np.nan)),
        "r": _safe_float(r),
        "q": _safe_float(q),
    }


__all__ = [
    "meta_from_inputs",
]
