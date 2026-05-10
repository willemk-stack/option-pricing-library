from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from ....pricers.heston import (
    heston_price_and_param_jac_from_ctx,
    heston_price_from_ctx,
)
from ....types import OptionType
from ....typing import FloatArray
from ..charfunc import HESTON_ANALYTIC_JAC_ETA_MIN
from ..fourier import HestonBackend, QuadratureConfig
from ..params import HESTON_PARAM_NAMES, HestonParams
from .bounds import (
    HestonCalibrationBounds,
    bounded_transform_jac_diag_from_raw,
    transform_to_bounded_constrained,
)
from .heston_types import (
    HESTON_OBJECTIVE_TYPES,
    HESTON_PARAMETER_TRANSFORMS,
    HestonObjectiveType,
    HestonParameterTransform,
    HestonQuoteSet,
    HestonRegConfig,
)
from .regulate import heston_regularization_jacobian, heston_regularization_residuals


def _validate_analytic_jacobian_params(
    params: HestonParams,
    bounds: HestonCalibrationBounds,
) -> None:
    """Validate the supported analytic-Jacobian calibration domain."""
    if params.eta < HESTON_ANALYTIC_JAC_ETA_MIN:
        raise ValueError(
            "Analytic Heston calibration Jacobians require eta >= "
            f"{HESTON_ANALYTIC_JAC_ETA_MIN:g}. Price-only deterministic-limit "
            "pricing near eta=0 is separate; use finite-difference calibration "
            "or choose an eta floor above the analytic-Jacobian threshold."
        )

    lower = bounds.lower_array()
    upper = bounds.upper_array()
    values = params.as_array()
    outside = (values < lower) | (values > upper)
    if np.any(outside):
        idx = int(np.flatnonzero(outside)[0])
        name = HESTON_PARAM_NAMES[idx]
        raise ValueError(
            "Analytic Heston calibration Jacobians are supported only inside "
            f"the documented calibration bounds. {name}={values[idx]} is outside "
            f"[{lower[idx]}, {upper[idx]}]."
        )


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


def _price_and_jac_heston_quotes(
    quotes: HestonQuoteSet,
    params: HestonParams,
    *,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
) -> tuple[FloatArray, FloatArray]:
    prices = np.empty(quotes.n_quotes, dtype=np.float64)
    dprice_dtheta = np.empty((quotes.n_quotes, 5), dtype=np.float64)

    for T in np.unique(quotes.expiry):
        idx_T = np.flatnonzero(quotes.expiry == T)

        for is_call_value, kind in (
            (True, OptionType.CALL),
            (False, OptionType.PUT),
        ):
            idx = idx_T[quotes.is_call[idx_T] == is_call_value]
            if idx.size == 0:
                continue

            price, jac = heston_price_and_param_jac_from_ctx(
                kind=kind,
                strike=quotes.strike[idx],
                tau=float(T),
                ctx=quotes.ctx,
                params=params,
                backend=backend,
                quad_cfg=quad_cfg,
            )
            prices[idx] = np.asarray(price, dtype=np.float64)
            dprice_dtheta[idx, :] = np.asarray(jac, dtype=np.float64)

    return prices, dprice_dtheta


@dataclass(frozen=True, slots=True)
class HestonObjective:
    """Least-squares Heston price-residual objective.

    Supported objective types are all price-residual-family objectives:
    ``price_rmse``, ``relative_price_rmse``, ``vega_scaled_price``, and
    ``bid_ask_normalized``. ``vega_scaled_price`` approximates IV-error
    behavior without repeated implied-vol inversion. Calibration optimizes
    vega-scaled price residuals as a robust proxy for IV error; IV RMSE is
    reported diagnostically after repricing and Black inversion, not optimized
    directly. ``iv_rmse`` is intentionally not implemented because it requires
    implied-vol inversion and additional Jacobian and robustness handling.
    """

    quotes: HestonQuoteSet
    sqrt_weights: FloatArray | None = None
    vega_floor: float = 1e-6
    price_floor: float = 1e-10
    spread_floor: float = 1e-6
    backend: HestonBackend = "gauss_legendre"
    quad_cfg: QuadratureConfig | None = None
    reg: HestonRegConfig | None = None
    # NOTE: vega_scaled_price remains the default to preserve legacy
    # calibration behavior while keeping optimization in price-residual space.
    # IV RMSE remains a reported diagnostic, not a direct optimizer target.
    objective_type: HestonObjectiveType = "vega_scaled_price"
    parameter_transform: HestonParameterTransform = "unconstrained"
    bounds: HestonCalibrationBounds | None = None

    def __post_init__(self) -> None:
        if self.objective_type not in HESTON_OBJECTIVE_TYPES:
            supported = ", ".join(repr(value) for value in HESTON_OBJECTIVE_TYPES)
            raise ValueError(
                f"objective_type must be one of {supported}; "
                f"got {self.objective_type!r}"
            )
        if self.parameter_transform not in HESTON_PARAMETER_TRANSFORMS:
            supported = ", ".join(repr(value) for value in HESTON_PARAMETER_TRANSFORMS)
            raise ValueError(
                f"parameter_transform must be one of {supported}; "
                f"got {self.parameter_transform!r}"
            )
        if self.parameter_transform == "unconstrained" and self.bounds is not None:
            raise ValueError("bounds are only used with parameter_transform='bounded'.")
        for name, value in (
            ("vega_floor", self.vega_floor),
            ("price_floor", self.price_floor),
            ("spread_floor", self.spread_floor),
        ):
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
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

    @property
    def supports_analytic_jac(self) -> bool:
        return (
            self.backend == "gauss_legendre"
            and self.objective_type in HESTON_OBJECTIVE_TYPES
        )

    def _resolved_bounds(self) -> HestonCalibrationBounds:
        if self.bounds is None:
            return HestonCalibrationBounds()
        return self.bounds

    def _params_from_raw(self, u: FloatArray) -> HestonParams:
        if self.parameter_transform == "unconstrained":
            return HestonParams.transform_to_constrained(u)
        if self.parameter_transform == "bounded":
            return transform_to_bounded_constrained(u, self._resolved_bounds())
        raise ValueError(
            f"Unsupported parameter_transform {self.parameter_transform!r}"
        )

    def _transform_jac_diag_from_raw(self, u: FloatArray) -> np.ndarray:
        if self.parameter_transform == "unconstrained":
            return HestonParams.transform_jac_diag_from_raw(u)
        if self.parameter_transform == "bounded":
            return bounded_transform_jac_diag_from_raw(u, self._resolved_bounds())
        raise ValueError(
            f"Unsupported parameter_transform {self.parameter_transform!r}"
        )

    def _price_residual_scale(self) -> FloatArray:
        if self.objective_type == "price_rmse":
            return np.ones(self.quotes.n_quotes, dtype=np.float64)

        if self.objective_type == "relative_price_rmse":
            # NOTE: Relative price residuals use market-mid normalization and
            # intentionally avoid model-dependent denominator terms.
            return np.asarray(
                np.maximum(np.abs(self.quotes.mid), float(self.price_floor)),
                dtype=np.float64,
            )

        if self.objective_type == "vega_scaled_price":
            if self.quotes.bs_vega is None:
                raise ValueError(
                    "objective_type='vega_scaled_price' requires quotes.bs_vega"
                )
            return self.quotes.vega_price_scales(self.vega_floor)

        if self.objective_type == "bid_ask_normalized":
            if self.quotes.bid is None or self.quotes.ask is None:
                raise ValueError(
                    "objective_type='bid_ask_normalized' requires "
                    "quotes.bid and quotes.ask"
                )
            spread = self.quotes.ask - self.quotes.bid
            if not np.all(np.isfinite(spread)):
                raise ValueError("bid/ask spread must be finite")
            if np.any(spread <= 0.0):
                raise ValueError("bid/ask spread must be strictly positive")
            # NOTE: Bid/ask-normalized residuals treat the quoted spread as
            # the residual scale and do not otherwise model quote reliability.
            return np.asarray(
                np.maximum(spread, float(self.spread_floor)), dtype=np.float64
            )

        raise ValueError(f"Unsupported objective_type {self.objective_type!r}")

    def residual(self, u: FloatArray) -> FloatArray:
        scale = self._price_residual_scale()
        params = self._params_from_raw(u)

        model = _price_heston_quotes(
            self.quotes,
            params,
            backend=self.backend,
            quad_cfg=self.quad_cfg,
        )

        quote_residual = self.effective_sqrt_weights * (model - self.quotes.mid) / scale
        reg_residual = heston_regularization_residuals(params, self.reg)

        full_residual = np.concatenate([quote_residual, reg_residual])
        return np.asarray(full_residual, dtype=np.float64)

    def jac(self, u: FloatArray) -> FloatArray:
        scale = self._price_residual_scale()
        params = self._params_from_raw(u)
        bounds = self._resolved_bounds()
        bounds.require_analytic_jacobian_compatible()
        _validate_analytic_jacobian_params(params, bounds)

        _, dprice_dtheta = _price_and_jac_heston_quotes(
            self.quotes,
            params,
            backend=self.backend,
            quad_cfg=self.quad_cfg,
        )
        dtheta_du = self._transform_jac_diag_from_raw(u)
        dprice_du = dprice_dtheta * dtheta_du[None, :]

        scaled_jac = np.asarray(
            self.effective_sqrt_weights[:, None] * dprice_du / scale[:, None],
            dtype=np.float64,
        )
        reg_jac = heston_regularization_jacobian(params, self.reg)
        if reg_jac.size == 0:
            return cast(FloatArray, scaled_jac)

        reg_jac_raw = np.asarray(reg_jac * dtheta_du[None, :], dtype=np.float64)
        full_jac = np.vstack([scaled_jac, reg_jac_raw])
        return cast(FloatArray, full_jac)
