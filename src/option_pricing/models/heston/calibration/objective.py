from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ....pricers.heston import heston_price_from_ctx
from ....types import OptionType
from ....typing import FloatArray
from ..fourier import HestonBackend, QuadratureConfig
from ..params import HestonParams
from .heston_types import HestonQuoteSet, HestonRegConfig


def _price_heston_quotes(
    quotes: HestonQuoteSet,
    params: HestonParams,
    *,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
) -> FloatArray:
    prices = np.empty(quotes.n_quotes, dtype=np.float64)

    for T in np.unique(quotes.expiry):
        idx_T = np.flatnonzero(quotes.expiry == T)

        for is_call_value, kind in (
            (True, OptionType.CALL),
            (False, OptionType.PUT),
        ):
            idx = idx_T[quotes.is_call[idx_T] == is_call_value]
            if idx.size == 0:
                continue

            prices[idx] = heston_price_from_ctx(
                kind=kind,
                strike=quotes.strike[idx],
                tau=float(T),
                ctx=quotes.ctx,  # or quotes.market.to_context()
                params=params,
                backend=backend,
                quad_cfg=quad_cfg,
            )

    return prices


@dataclass(frozen=True, slots=True)
class HestonObjective:
    quotes: HestonQuoteSet
    sqrt_weights: FloatArray | None = None
    vega_floor: float = 1e-6
    price_floor: float = 1e-10
    spread_floor: float = 1e-6
    backend: HestonBackend = "gauss_legendre"
    quad_cfg: QuadratureConfig | None = None
    reg: HestonRegConfig | None = None

    def __post_init__(self) -> None:
        if self.sqrt_weights is not None:
            if self.sqrt_weights.shape != (self.quotes.n_quotes,):
                raise ValueError(
                    "sqrt_weights must have shape "
                    f"({self.quotes.n_quotes},), got {self.sqrt_weights.shape}"
                )
            if not np.all(np.isfinite(self.sqrt_weights)):
                raise ValueError("sqrt_weights must be finite")
            if np.any(self.sqrt_weights < 0.0):
                raise ValueError("sqrt_weights must be nonnegative")

    @property
    def effective_sqrt_weights(self) -> FloatArray:
        if self.sqrt_weights is None:
            return self.quotes.base_sqrt_weights
        return self.sqrt_weights

    def residual(self, u: FloatArray) -> FloatArray:
        params = HestonParams.transform_to_constrained(u)

        model = _price_heston_quotes(
            self.quotes,
            params,
            backend=self.backend,
            quad_cfg=self.quad_cfg,
        )

        scale = self.quotes.vega_price_scales(self.vega_floor)
        return self.effective_sqrt_weights * (model - self.quotes.mid) / scale

    def jac(self, u: FloatArray) -> FloatArray:
        raise NotImplementedError(
            "Analytic Jacobian is not implemented yet; use finite differences."
        )
        # theta = self.transform.decode(u)
        # prices, dC_dtheta = heston_prices_and_jac(theta, self.quotes)
        # dtheta_du = self.transform.jacobian(u, theta)
        # J_raw = dC_dtheta @ dtheta_du
        # return self._scale_jacobian(J_raw, prices, theta)
