from __future__ import annotations

from dataclasses import dataclass
from typing import overload

import numpy as np

from ...market.curves import PricingContext
from ...types import MarketData, OptionType
from ...typing import ArrayLike
from .interp import interpolate_nodes
from .math import (
    essvi_implied_price,
    essvi_total_variance,
    essvi_total_variance_dk,
    essvi_total_variance_dk_dT,
    essvi_total_variance_dkk,
    essvi_total_variance_dT,
    essvi_w_and_derivs,
)
from .models import (
    ESSVINodeSet,
    ESSVITermStructures,
    EtaTermStructure,
    PsiTermStructure,
    ThetaTermStructure,
)


def _constant_slice_params(
    *,
    theta: float,
    psi: float,
    eta: float,
    eps: float,
) -> ESSVITermStructures:
    return ESSVITermStructures(
        theta_term=ThetaTermStructure.constant(theta),
        psi_term=PsiTermStructure.constant(psi),
        eta_term=EtaTermStructure.constant(eta),
        eps=eps,
    )


@dataclass(frozen=True, slots=True)
class ESSVISmileSlice:
    """Single-expiry analytic eSSVI smile slice in total-variance space."""

    T: float
    params: ESSVITermStructures
    y_min: float = -2.5
    y_max: float = 2.5

    def __post_init__(self) -> None:
        self.params.validate(self.T)

    def w_at(self, yq: ArrayLike) -> np.ndarray:
        return essvi_total_variance(y=yq, params=self.params, T=self.T)

    def iv_at(self, yq: ArrayLike) -> np.ndarray:
        w = self.w_at(yq)
        return np.asarray(
            np.sqrt(np.maximum(w / np.float64(self.T), np.float64(0.0))),
            dtype=np.float64,
        )

    @overload
    def price_at(
        self,
        *,
        kind: OptionType,
        strike: ArrayLike,
        forward: ArrayLike,
        df: ArrayLike = 1.0,
        market: None = None,
    ) -> np.ndarray: ...

    @overload
    def price_at(
        self,
        *,
        kind: OptionType,
        strike: ArrayLike,
        market: MarketData | PricingContext,
        forward: None = None,
        df: None = None,
    ) -> np.ndarray: ...

    def price_at(
        self,
        *,
        kind: OptionType,
        strike: ArrayLike,
        forward: ArrayLike | None = None,
        df: ArrayLike | None = 1.0,
        market: MarketData | PricingContext | None = None,
    ) -> np.ndarray:
        if market is not None:
            return essvi_implied_price(
                kind=kind,
                strike=strike,
                params=self.params,
                T=self.T,
                market=market,
            )

        if forward is None or df is None:
            raise ValueError(
                "Either market or both forward and df must be provided to price_at."
            )

        return essvi_implied_price(
            kind=kind,
            strike=strike,
            forward=forward,
            df=df,
            params=self.params,
            T=self.T,
        )

    def dw_dy(self, yq: ArrayLike) -> np.ndarray:
        return essvi_total_variance_dk(y=yq, params=self.params, T=self.T)

    def d2w_dy2(self, yq: ArrayLike) -> np.ndarray:
        return essvi_total_variance_dkk(y=yq, params=self.params, T=self.T)


@dataclass(frozen=True, slots=True)
class ESSVINodalSmileSlice:
    """Single-expiry slice from exact Mingone nodal interpolation."""

    T: float
    nodes: ESSVINodeSet
    y_min: float = -2.5
    y_max: float = 2.5

    def _params(self) -> ESSVITermStructures:
        state = interpolate_nodes(self.nodes, float(self.T), eps=self.nodes.eps)
        return _constant_slice_params(
            theta=float(np.asarray(state.theta)),
            psi=float(np.asarray(state.psi)),
            eta=float(np.asarray(state.eta)),
            eps=self.nodes.eps,
        )

    def w_at(self, yq: ArrayLike) -> np.ndarray:
        return essvi_total_variance(y=yq, params=self._params(), T=self.T)

    def iv_at(self, yq: ArrayLike) -> np.ndarray:
        w = self.w_at(yq)
        return np.asarray(
            np.sqrt(np.maximum(w / np.float64(self.T), np.float64(0.0))),
            dtype=np.float64,
        )

    @overload
    def price_at(
        self,
        *,
        kind: OptionType,
        strike: ArrayLike,
        forward: ArrayLike,
        df: ArrayLike = 1.0,
        market: None = None,
    ) -> np.ndarray: ...

    @overload
    def price_at(
        self,
        *,
        kind: OptionType,
        strike: ArrayLike,
        market: MarketData | PricingContext,
        forward: None = None,
        df: None = None,
    ) -> np.ndarray: ...

    def price_at(
        self,
        *,
        kind: OptionType,
        strike: ArrayLike,
        forward: ArrayLike | None = None,
        df: ArrayLike | None = 1.0,
        market: MarketData | PricingContext | None = None,
    ) -> np.ndarray:
        params = self._params()
        if market is not None:
            return essvi_implied_price(
                kind=kind,
                strike=strike,
                params=params,
                T=self.T,
                market=market,
            )

        if forward is None or df is None:
            raise ValueError(
                "Either market or both forward and df must be provided to price_at."
            )

        return essvi_implied_price(
            kind=kind,
            strike=strike,
            forward=forward,
            df=df,
            params=params,
            T=self.T,
        )

    def dw_dy(self, yq: ArrayLike) -> np.ndarray:
        return essvi_total_variance_dk(y=yq, params=self._params(), T=self.T)

    def d2w_dy2(self, yq: ArrayLike) -> np.ndarray:
        return essvi_total_variance_dkk(y=yq, params=self._params(), T=self.T)


@dataclass(frozen=True, slots=True)
class ESSVIImpliedSurface:
    """Continuous analytic implied surface driven by an eSSVI parameter surface."""

    params: ESSVITermStructures
    y_min: float = -2.5
    y_max: float = 2.5

    def slice(self, T: float) -> ESSVISmileSlice:
        return ESSVISmileSlice(
            T=float(T),
            params=self.params,
            y_min=self.y_min,
            y_max=self.y_max,
        )

    def w(self, y: ArrayLike, T: float) -> np.ndarray:
        return essvi_total_variance(y=y, params=self.params, T=T)

    def iv(self, y: ArrayLike, T: float) -> np.ndarray:
        T = float(T)
        if T <= 0.0:
            raise ValueError("T must be > 0.")
        w = self.w(y=y, T=T)
        return np.asarray(
            np.sqrt(np.maximum(w / np.float64(T), np.float64(0.0))),
            dtype=np.float64,
        )

    def dw_dy(self, y: ArrayLike, T: float) -> np.ndarray:
        return essvi_total_variance_dk(y=y, params=self.params, T=T)

    def d2w_dy2(self, y: ArrayLike, T: float) -> np.ndarray:
        return essvi_total_variance_dkk(y=y, params=self.params, T=T)

    def dw_dT(self, y: ArrayLike, T: float) -> np.ndarray:
        return essvi_total_variance_dT(y=y, params=self.params, T=T)

    def d2w_dy_dT(self, y: ArrayLike, T: float) -> np.ndarray:
        return essvi_total_variance_dk_dT(y=y, params=self.params, T=T)

    @overload
    def price(
        self,
        *,
        kind: OptionType,
        strike: ArrayLike,
        forward: ArrayLike,
        T: ArrayLike,
        df: ArrayLike = 1.0,
        market: None = None,
    ) -> np.ndarray: ...

    @overload
    def price(
        self,
        *,
        kind: OptionType,
        strike: ArrayLike,
        T: ArrayLike,
        market: MarketData | PricingContext,
        forward: None = None,
        df: None = None,
    ) -> np.ndarray: ...

    def price(
        self,
        *,
        kind: OptionType,
        strike: ArrayLike,
        forward: ArrayLike | None = None,
        T: ArrayLike,
        df: ArrayLike | None = 1.0,
        market: MarketData | PricingContext | None = None,
    ) -> np.ndarray:
        if market is not None:
            return essvi_implied_price(
                kind=kind,
                strike=strike,
                params=self.params,
                T=T,
                market=market,
            )

        if forward is None or df is None:
            raise ValueError(
                "Either market or both forward and df must be provided to price."
            )

        return essvi_implied_price(
            kind=kind,
            strike=strike,
            forward=forward,
            df=df,
            params=self.params,
            T=T,
        )

    def w_and_derivs(
        self,
        y: np.ndarray,
        T: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return essvi_w_and_derivs(y=y, params=self.params, T=T)


@dataclass(frozen=True, slots=True)
class ESSVINodalSurface:
    """Exact arbitrage-safe nodal eSSVI surface from Mingone interpolation."""

    nodes: ESSVINodeSet
    y_min: float = -2.5
    y_max: float = 2.5

    def slice(self, T: float) -> ESSVINodalSmileSlice:
        return ESSVINodalSmileSlice(
            T=float(T),
            nodes=self.nodes,
            y_min=self.y_min,
            y_max=self.y_max,
        )

    def w(self, y: ArrayLike, T: float) -> np.ndarray:
        return self.slice(float(T)).w_at(y)

    def iv(self, y: ArrayLike, T: float) -> np.ndarray:
        return self.slice(float(T)).iv_at(y)

    def dw_dy(self, y: ArrayLike, T: float) -> np.ndarray:
        return self.slice(float(T)).dw_dy(y)

    def d2w_dy2(self, y: ArrayLike, T: float) -> np.ndarray:
        return self.slice(float(T)).d2w_dy2(y)

    @overload
    def price(
        self,
        *,
        kind: OptionType,
        strike: ArrayLike,
        forward: ArrayLike,
        T: ArrayLike,
        df: ArrayLike = 1.0,
        market: None = None,
    ) -> np.ndarray: ...

    @overload
    def price(
        self,
        *,
        kind: OptionType,
        strike: ArrayLike,
        T: ArrayLike,
        market: MarketData | PricingContext,
        forward: None = None,
        df: None = None,
    ) -> np.ndarray: ...

    def price(
        self,
        *,
        kind: OptionType,
        strike: ArrayLike,
        forward: ArrayLike | None = None,
        T: ArrayLike,
        df: ArrayLike | None = 1.0,
        market: MarketData | PricingContext | None = None,
    ) -> np.ndarray:
        T_arr = np.asarray(T, dtype=np.float64)
        strike_arr = np.asarray(strike, dtype=np.float64)
        if market is None:
            if forward is None or df is None:
                raise ValueError(
                    "Either market or both forward and df must be provided to price."
                )
            strike_arr, T_arr, forward_arr, df_arr = np.broadcast_arrays(
                strike_arr,
                T_arr,
                np.asarray(forward, dtype=np.float64),
                np.asarray(df, dtype=np.float64),
            )
        else:
            strike_arr, T_arr = np.broadcast_arrays(strike_arr, T_arr)

        out = np.empty_like(T_arr, dtype=np.float64)
        for idx, tau in np.ndenumerate(T_arr):
            smile = self.slice(float(tau))
            strike_value = float(strike_arr[idx])
            if market is not None:
                price = smile.price_at(
                    kind=kind,
                    strike=strike_value,
                    market=market,
                )
            else:
                price = smile.price_at(
                    kind=kind,
                    strike=strike_value,
                    forward=float(forward_arr[idx]),
                    df=float(df_arr[idx]),
                )
            out[idx] = float(np.asarray(price))
        return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ESSVISmoothedSurface(ESSVIImpliedSurface):
    """Dupire-oriented continuous surface built from a smooth node projection."""
