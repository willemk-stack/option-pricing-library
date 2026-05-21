"""Deterministic Heston capstone quote fixtures."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ...models.heston.calibration.heston_types import HestonQuoteSet
from ...pricers.black_scholes import bs_greeks_from_ctx, bs_price_from_ctx
from ...types import MarketData, OptionType
from ...typing import FloatArray

MARKET_LIKE_SYNTHETIC_FIXTURE_LABEL = "deterministic market-like synthetic fixture"


def _default_market_like_expiries() -> FloatArray:
    return np.array([0.25, 0.50, 1.00, 1.50, 2.00], dtype=np.float64)


def _default_market_like_log_moneyness() -> FloatArray:
    return np.array([-0.20, -0.10, 0.0, 0.10, 0.20], dtype=np.float64)


def _coerce_finite_vector(
    values: Sequence[float] | np.ndarray | None,
    *,
    default: FloatArray,
    label: str,
) -> FloatArray:
    arr = default.copy() if values is None else np.asarray(values, dtype=np.float64)
    arr = arr.reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{label} must contain at least one value.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must be finite.")
    return np.asarray(arr, dtype=np.float64)


def _market_like_equity_index_iv(
    log_moneyness: FloatArray, expiry: float
) -> FloatArray:
    """Smooth deterministic equity-index-style skew/smile fixture surface."""
    y = np.asarray(log_moneyness, dtype=np.float64)
    T = float(expiry)
    atm = 0.170 + 0.035 * (1.0 - np.exp(-T / 1.50)) + 0.004 * np.sqrt(T)
    skew = -0.105 * np.exp(-T / 1.80) - 0.025
    curvature = 0.280 + 0.060 * np.exp(-T / 2.00)
    iv = atm + skew * y + curvature * y * y
    return np.asarray(np.maximum(iv, 0.05), dtype=np.float64)


def build_market_like_heston_quote_set(
    *,
    market: MarketData | None = None,
    expiries: Sequence[float] | np.ndarray | None = None,
    log_moneyness: Sequence[float] | np.ndarray | None = None,
) -> HestonQuoteSet:
    """Build the fixed quote target for Heston-vs-local-vol comparison.

    The fixture is synthetic and deterministic. It is intentionally not sampled
    from a Heston model, which keeps the final model-comparison target from
    advantaging Heston through a Heston-generated quote target.
    """
    resolved_market = (
        MarketData(spot=100.0, rate=0.020, dividend_yield=0.010)
        if market is None
        else market
    )
    if resolved_market.spot <= 0.0 or not np.isfinite(resolved_market.spot):
        raise ValueError("market spot must be positive and finite.")

    expiry_grid = _coerce_finite_vector(
        expiries,
        default=_default_market_like_expiries(),
        label="expiries",
    )
    if np.any(expiry_grid <= 0.0):
        raise ValueError("expiries must be positive.")

    log_mny_grid = _coerce_finite_vector(
        log_moneyness,
        default=_default_market_like_log_moneyness(),
        label="log_moneyness",
    )

    ctx = resolved_market.to_context()
    strikes: list[float] = []
    expiry_values: list[float] = []
    mid_values: list[float] = []
    iv_values: list[float] = []
    vega_values: list[float] = []
    labels: list[str] = []

    for tau in expiry_grid:
        forward = float(ctx.fwd(float(tau)))
        strike_slice = np.asarray(forward * np.exp(log_mny_grid), dtype=np.float64)
        iv_slice = _market_like_equity_index_iv(log_mny_grid, float(tau))
        for point_idx, (strike, log_m, iv) in enumerate(
            zip(strike_slice, log_mny_grid, iv_slice, strict=True)
        ):
            price = bs_price_from_ctx(
                kind=OptionType.CALL,
                strike=float(strike),
                sigma=float(iv),
                tau=float(tau),
                ctx=ctx,
            )
            greeks = bs_greeks_from_ctx(
                kind=OptionType.CALL,
                strike=float(strike),
                sigma=float(iv),
                tau=float(tau),
                ctx=ctx,
            )
            strikes.append(float(strike))
            expiry_values.append(float(tau))
            mid_values.append(float(price))
            iv_values.append(float(iv))
            vega_values.append(float(greeks["vega"]))
            labels.append(
                f"market-like-call-T{float(tau):.6g}-y{float(log_m):+.6g}-{point_idx}"
            )

    return HestonQuoteSet(
        ctx=ctx,
        strike=np.asarray(strikes, dtype=np.float64),
        expiry=np.asarray(expiry_values, dtype=np.float64),
        is_call=np.ones(len(strikes), dtype=np.bool_),
        mid=np.asarray(mid_values, dtype=np.float64),
        bs_vega=np.asarray(vega_values, dtype=np.float64),
        iv_mid=np.asarray(iv_values, dtype=np.float64),
        labels=tuple(labels),
        metadata={
            "fixture_label": MARKET_LIKE_SYNTHETIC_FIXTURE_LABEL,
            "construction": (
                "Black-76 calls from a deterministic equity-index-style "
                "implied-vol skew/smile and flat forward/discount assumptions."
            ),
            "data_source": "synthetic_not_market_data",
            "generated_from_heston": False,
        },
    )


__all__ = [
    "MARKET_LIKE_SYNTHETIC_FIXTURE_LABEL",
    "build_market_like_heston_quote_set",
]
