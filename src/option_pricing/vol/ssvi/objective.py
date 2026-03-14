from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...market.curves import PricingContext
from ...types import MarketData, OptionType
from .math import essvi_implied_price
from .models import ESSVITermStructures


def _as_float_vector(
    name: str, value: NDArray[np.float64] | float
) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    return arr


def _to_ctx(market: MarketData | PricingContext) -> PricingContext:
    if isinstance(market, PricingContext):
        return market
    return market.to_context()


@dataclass(slots=True)
class ESSVIPriceObjective:
    """Price-space least-squares objective for eSSVI calibration.

    The repository convention is ``y = ln(K / F(T))``. If ``is_call`` is not
    provided, market prices are interpreted as OTM quotes, so puts are used for
    ``y < 0`` and calls are used for ``y >= 0``.
    """

    y: NDArray[np.float64]
    T: NDArray[np.float64]
    price_mkt: NDArray[np.float64]
    market: MarketData | PricingContext
    sqrt_weights: NDArray[np.float64] | None = None
    is_call: NDArray[np.bool_] | None = None
    weights: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        y = _as_float_vector("y", self.y)
        T = _as_float_vector("T", self.T)
        price_mkt = _as_float_vector("price_mkt", self.price_mkt)
        sqrt_weights = self._resolve_sqrt_weights(y)

        if not (y.size == T.size == price_mkt.size == sqrt_weights.size):
            raise ValueError(
                "y, T, price_mkt, and sqrt_weights must have the same size."
            )
        if np.any(T <= 0.0):
            raise ValueError("T must be > 0.")
        if np.any(price_mkt < 0.0):
            raise ValueError("price_mkt must be >= 0.")
        if np.any(sqrt_weights < 0.0):
            raise ValueError("sqrt_weights must be >= 0.")

        self.y = y
        self.T = T
        self.price_mkt = price_mkt
        self.sqrt_weights = sqrt_weights

        if self.is_call is not None:
            is_call = np.asarray(self.is_call, dtype=np.bool_).reshape(-1)
            if is_call.size != y.size:
                raise ValueError("is_call must have the same size as y.")
            self.is_call = is_call

    def _resolve_sqrt_weights(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.sqrt_weights is not None and self.weights is not None:
            raise ValueError("Pass either sqrt_weights or weights, not both.")
        if self.sqrt_weights is None and self.weights is None:
            return np.ones_like(y, dtype=np.float64)
        if self.sqrt_weights is not None:
            return _as_float_vector("sqrt_weights", self.sqrt_weights)
        assert self.weights is not None
        warnings.warn(
            "'weights' is deprecated for ESSVIPriceObjective; use 'sqrt_weights' "
            "because the residual is sqrt_weights * price_error.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return _as_float_vector("weights", self.weights)

    def _ctx(self) -> PricingContext:
        return _to_ctx(self.market)

    def forward_and_df(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        ctx = self._ctx()
        forward = np.fromiter(
            (ctx.fwd(float(tau)) for tau in self.T),
            dtype=np.float64,
            count=self.T.size,
        )
        df = np.fromiter(
            (ctx.df(float(tau)) for tau in self.T),
            dtype=np.float64,
            count=self.T.size,
        )
        return forward, df

    def strikes(self) -> NDArray[np.float64]:
        forward, _ = self.forward_and_df()
        return np.asarray(forward * np.exp(self.y), dtype=np.float64)

    def call_mask(self) -> NDArray[np.bool_]:
        if self.is_call is not None:
            return self.is_call
        return np.asarray(self.y >= 0.0, dtype=np.bool_)

    def model_prices(self, params: ESSVITermStructures) -> NDArray[np.float64]:
        forward, df = self.forward_and_df()
        strike = np.asarray(forward * np.exp(self.y), dtype=np.float64)
        call_model = essvi_implied_price(
            kind=OptionType.CALL,
            strike=strike,
            forward=forward,
            df=df,
            params=params,
            T=self.T,
        )
        put_model = np.asarray(call_model - df * (forward - strike), dtype=np.float64)
        return np.asarray(
            np.where(self.call_mask(), call_model, put_model), dtype=np.float64
        )

    def residual(self, params: ESSVITermStructures) -> NDArray[np.float64]:
        model = self.model_prices(params)
        assert self.sqrt_weights is not None
        return np.asarray(
            self.sqrt_weights * (model - self.price_mkt), dtype=np.float64
        )


SSVIObjective = ESSVIPriceObjective
