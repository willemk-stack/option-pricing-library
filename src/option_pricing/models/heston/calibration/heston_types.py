from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ....types import MarketData, PricingContext
from ....typing import BoolArray, FloatArray


@dataclass(frozen=True, slots=True)
class HestonQuoteSet:
    ctx: PricingContext
    strike: FloatArray
    expiry: FloatArray
    is_call: BoolArray
    mid: FloatArray
    bs_vega: FloatArray | None = None
    sqrt_weights: FloatArray | None = None
    bid: FloatArray | None = None
    ask: FloatArray | None = None
    iv_mid: FloatArray | None = None
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        n = self.strike.size

        arrays = {
            "strike": self.strike,
            "expiry": self.expiry,
            "is_call": self.is_call,
            "mid": self.mid,
        }

        for name, arr in arrays.items():
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")

        term_arrays = {
            "discount": self.discount,
            "forward": self.forward,
        }

        for name, arr in term_arrays.items():
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")

        optional_arrays = {
            "bs_vega": self.bs_vega,
            "bid": self.bid,
            "ask": self.ask,
            "iv_mid": self.iv_mid,
            "sqrt_weights": self.sqrt_weights,
        }

        for name, optional_arr in optional_arrays.items():
            if optional_arr is not None and optional_arr.shape != (n,):
                raise ValueError(
                    f"{name} must have shape ({n},), got {optional_arr.shape}"
                )

        if not np.isfinite(self.ctx.spot) or self.ctx.spot <= 0.0:
            raise ValueError("spot must be positive and finite")

        for name, arr in {**arrays, **term_arrays}.items():
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be finite")

        if np.any(self.strike <= 0.0):
            raise ValueError("strike values must be positive")
        if np.any(self.expiry <= 0.0):
            raise ValueError("expiry values must be positive")
        if np.any(self.discount <= 0.0):
            raise ValueError("discount values must be positive")
        if np.any(self.forward <= 0.0):
            raise ValueError("forward values must be positive")
        if np.any(self.mid < 0.0):
            raise ValueError("mid prices must be nonnegative")

        if self.bid is not None and self.ask is not None:
            if np.any(self.bid < 0.0):
                raise ValueError("bid prices must be nonnegative")
            if np.any(self.ask < self.bid):
                raise ValueError("ask must be >= bid")

        if self.bs_vega is not None:
            if not np.all(np.isfinite(self.bs_vega)):
                raise ValueError("bs_vega must be finite")
            if np.any(self.bs_vega < 0.0):
                raise ValueError("bs_vega must be nonnegative")

        if self.iv_mid is not None:
            if not np.all(np.isfinite(self.iv_mid)):
                raise ValueError("iv_mid must be finite")
            if np.any(self.iv_mid <= 0.0):
                raise ValueError("iv_mid must be positive")

        if self.sqrt_weights is not None:
            if not np.all(np.isfinite(self.sqrt_weights)):
                raise ValueError("sqrt_weights must be finite")
            if np.any(self.sqrt_weights < 0.0):
                raise ValueError("sqrt_weights must be nonnegative")

        if self.labels is not None and len(self.labels) != n:
            raise ValueError(f"labels must have length {n}")

    @property
    def n_quotes(self) -> int:
        return int(self.strike.size)

    @property
    def discount(self) -> FloatArray:
        return np.asarray(
            [self.ctx.df(float(tau)) for tau in self.expiry],
            dtype=np.float64,
        )

    @property
    def forward(self) -> FloatArray:
        return np.asarray(
            [self.ctx.fwd(float(tau)) for tau in self.expiry],
            dtype=np.float64,
        )

    @property
    def log_moneyness(self) -> FloatArray:
        return np.log(self.strike / self.forward)

    @property
    def base_sqrt_weights(self) -> FloatArray:
        if self.sqrt_weights is None:
            return np.ones(self.n_quotes, dtype=np.float64)
        return self.sqrt_weights

    def vega_price_scales(self, vega_floor: float = 1e-6) -> FloatArray:
        if self.bs_vega is None:
            raise ValueError(
                "bs_vega is required for objective='vega_price'. "
                "Compute it once from market IVs before calibration."
            )
        return np.maximum(self.bs_vega, float(vega_floor))

    @classmethod
    def from_flat_market(
        cls,
        *,
        market: MarketData,
        strike: FloatArray,
        expiry: FloatArray,
        is_call: BoolArray,
        mid: FloatArray,
        bs_vega: FloatArray | None = None,
        bid: FloatArray | None = None,
        ask: FloatArray | None = None,
        iv_mid: FloatArray | None = None,
        sqrt_weights: FloatArray | None = None,
        labels: tuple[str, ...] | None = None,
    ) -> HestonQuoteSet:
        strike = np.asarray(strike, dtype=np.float64).reshape(-1)
        expiry = np.asarray(expiry, dtype=np.float64).reshape(-1)

        return cls(
            ctx=market.to_context(),
            strike=strike,
            expiry=expiry,
            is_call=np.asarray(is_call, dtype=np.bool_).reshape(-1),
            mid=np.asarray(mid, dtype=np.float64).reshape(-1),
            bs_vega=(
                None
                if bs_vega is None
                else np.asarray(bs_vega, dtype=np.float64).reshape(-1)
            ),
            bid=None if bid is None else np.asarray(bid, dtype=np.float64).reshape(-1),
            ask=None if ask is None else np.asarray(ask, dtype=np.float64).reshape(-1),
            iv_mid=(
                None
                if iv_mid is None
                else np.asarray(iv_mid, dtype=np.float64).reshape(-1)
            ),
            sqrt_weights=(
                None
                if sqrt_weights is None
                else np.asarray(sqrt_weights, dtype=np.float64).reshape(-1)
            ),
            labels=labels,
        )


@dataclass(frozen=True, slots=True)
class HestonRegConfig: ...
