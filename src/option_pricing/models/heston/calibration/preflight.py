from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .heston_types import HestonQuoteSet

type HestonQuoteRecommendation = Literal["ok", "review", "block"]


@dataclass(frozen=True, slots=True)
class HestonQuotePreflight:
    quote_count: int
    price_bound_violation_count: int
    mid_outside_bid_ask_count: int
    recommendation: HestonQuoteRecommendation
    messages: tuple[str, ...]


def _validate_tolerance(name: str, value: float) -> float:
    tol = float(value)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return tol


def _format_indices(indices: np.ndarray, *, limit: int = 5) -> str:
    shown = ", ".join(str(int(index)) for index in indices[:limit])
    if indices.size <= limit:
        return shown
    return f"{shown}, ..."


def preflight_heston_quotes(
    quotes: HestonQuoteSet,
    *,
    price_bound_tol: float = 1e-10,
    bid_ask_tol: float = 1e-12,
    raise_on_block: bool = False,
) -> HestonQuotePreflight:
    """Run calibration-specific economic checks on a Heston quote set.

    This helper is intentionally separate from ``HestonQuoteSet`` structural
    validation. It adds opt-in economic checks that are useful immediately
    before calibration: vanilla no-arbitrage price bounds and, when both are
    present, whether each mid lies inside its bid/ask interval.

    Recommendation policy is deliberately simple:

    - ``"ok"`` when there are no violations;
    - ``"block"`` when any price-bound violation is found;
    - ``"block"`` when any mid lies outside bid/ask because the quote set is
      internally inconsistent for calibration.

    The ``"review"`` variant remains available in the result type for future
    softer warning policies, but is not currently emitted.
    """

    price_bound_tol = _validate_tolerance("price_bound_tol", price_bound_tol)
    bid_ask_tol = _validate_tolerance("bid_ask_tol", bid_ask_tol)
    discount = quotes.discount
    forward = quotes.forward

    lower_bounds = discount * np.where(
        quotes.is_call,
        np.maximum(forward - quotes.strike, 0.0),
        np.maximum(quotes.strike - forward, 0.0),
    )
    upper_bounds = discount * np.where(
        quotes.is_call,
        forward,
        quotes.strike,
    )

    price_bound_mask = (quotes.mid < (lower_bounds - price_bound_tol)) | (
        quotes.mid > (upper_bounds + price_bound_tol)
    )
    price_bound_indices = np.flatnonzero(price_bound_mask)

    mid_outside_bid_ask_mask = np.zeros(quotes.n_quotes, dtype=np.bool_)
    if quotes.bid is not None and quotes.ask is not None:
        mid_outside_bid_ask_mask = (quotes.mid < (quotes.bid - bid_ask_tol)) | (
            quotes.mid > (quotes.ask + bid_ask_tol)
        )
    mid_outside_bid_ask_indices = np.flatnonzero(mid_outside_bid_ask_mask)

    messages: list[str] = []
    if price_bound_indices.size:
        messages.append(
            f"{price_bound_indices.size} quote(s) violate vanilla no-arbitrage "
            f"price bounds at indices [{_format_indices(price_bound_indices)}]."
        )
    if mid_outside_bid_ask_indices.size:
        messages.append(
            f"{mid_outside_bid_ask_indices.size} quote(s) have mid outside "
            f"bid/ask at indices [{_format_indices(mid_outside_bid_ask_indices)}]."
        )
    if not messages:
        messages.append("All quotes passed Heston calibration preflight.")

    recommendation: HestonQuoteRecommendation = "ok"
    if price_bound_indices.size or mid_outside_bid_ask_indices.size:
        recommendation = "block"

    result = HestonQuotePreflight(
        quote_count=quotes.n_quotes,
        price_bound_violation_count=int(price_bound_indices.size),
        mid_outside_bid_ask_count=int(mid_outside_bid_ask_indices.size),
        recommendation=recommendation,
        messages=tuple(messages),
    )

    if raise_on_block and result.recommendation == "block":
        raise ValueError(
            "Heston quote preflight blocked calibration: " + " ".join(result.messages)
        )

    return result
