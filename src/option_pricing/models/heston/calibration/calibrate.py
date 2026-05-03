"""
Calibration helpers and main object
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from ....exceptions import NoConvergenceError
from ....typing import FloatArray
from ..fourier import HestonBackend, QuadratureConfig
from ..params import HestonParams
from .bounds import HestonCalibrationBounds, transform_to_bounded_constrained
from .heston_types import (
    HESTON_PARAMETER_TRANSFORMS,
    HestonObjectiveType,
    HestonParameterTransform,
    HestonQuoteSet,
    HestonRegConfig,
)
from .objective import HestonObjective


def calibrate_heston(
    quotes: HestonQuoteSet,
    sqrt_weights: FloatArray | None = None,
    objective_type: HestonObjectiveType = "vega_scaled_price",
    x0_params: HestonParams | None = None,
    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "soft_l1",
    x_scale: Literal["jac"] | float | FloatArray | None = "jac",
    parameter_transform: HestonParameterTransform = "unconstrained",
    vega_floor: float | None = None,
    price_floor: float | None = None,
    spread_floor: float | None = None,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    reg: HestonRegConfig | None = None,
    bounds: HestonCalibrationBounds | None = None,
    use_analytic_jac: bool = True,
    method: Literal["trf", "dogbox", "lm"] = "trf",
    max_nfev: int | None = None,
    ftol: float | None = None,
    xtol: float | None = None,
    gtol: float | None = None,
    return_result: bool = False,
) -> HestonParams | tuple[HestonParams, OptimizeResult]:
    if parameter_transform not in HESTON_PARAMETER_TRANSFORMS:
        supported = ", ".join(repr(value) for value in HESTON_PARAMETER_TRANSFORMS)
        raise ValueError(
            f"parameter_transform must be one of {supported}; "
            f"got {parameter_transform!r}"
        )
    if parameter_transform == "unconstrained" and bounds is not None:
        raise ValueError("bounds are only used with parameter_transform='bounded'.")
    if x0_params is None:
        raise ValueError(
            "x0_params must be provided until a Heston calibration seed heuristic "
            "is implemented."
        )
    # REVIEW: Keep the default optimizer at "trf" because the default
    # loss="soft_l1" is invalid for SciPy's method="lm"; explicit LM is still
    # allowed with loss="linear".
    if method == "lm" and loss != "linear":
        raise ValueError(
            "method='lm' requires loss='linear' in scipy.optimize.least_squares."
        )

    obj = HestonObjective(
        quotes=quotes,
        objective_type=objective_type,
        sqrt_weights=sqrt_weights,
        vega_floor=1e-6 if vega_floor is None else float(vega_floor),
        price_floor=1e-10 if price_floor is None else float(price_floor),
        spread_floor=1e-6 if spread_floor is None else float(spread_floor),
        backend=backend,
        quad_cfg=quad_cfg,
        reg=reg,
        parameter_transform=parameter_transform,
        bounds=bounds,
    )

    def analytic_jac(u: np.ndarray, *_args: object, **_kwargs: object) -> FloatArray:
        return obj.jac(np.asarray(u, dtype=np.float64))

    if parameter_transform == "bounded":
        resolved_bounds = bounds if bounds is not None else HestonCalibrationBounds()
        x0 = x0_params.transform_to_bounded_unconstrained(resolved_bounds)
    else:
        resolved_bounds = None
        x0 = x0_params.transform_to_unconstrained()
    if any(tol is not None for tol in (ftol, xtol, gtol)):
        if use_analytic_jac:
            res = least_squares(
                fun=obj.residual,
                jac=analytic_jac,
                x0=x0,
                loss=loss,
                x_scale=x_scale,
                method=method,
                max_nfev=max_nfev,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
            )
        else:
            res = least_squares(
                fun=obj.residual,
                jac="2-point",
                x0=x0,
                loss=loss,
                x_scale=x_scale,
                method=method,
                max_nfev=max_nfev,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
            )
    else:
        if use_analytic_jac:
            res = least_squares(
                fun=obj.residual,
                jac=analytic_jac,
                x0=x0,
                loss=loss,
                x_scale=x_scale,
                method=method,
                max_nfev=max_nfev,
            )
        else:
            res = least_squares(
                fun=obj.residual,
                jac="2-point",
                x0=x0,
                loss=loss,
                x_scale=x_scale,
                method=method,
                max_nfev=max_nfev,
            )

    if not res.success or not np.all(np.isfinite(res.x)):
        raise NoConvergenceError(f"Heston calibration failed: {res.message}")

    raw_fit = np.asarray(res.x, dtype=np.float64)
    if parameter_transform == "bounded":
        assert resolved_bounds is not None
        fitted_params = transform_to_bounded_constrained(raw_fit, resolved_bounds)
    else:
        fitted_params = HestonParams.transform_to_constrained(raw_fit)
    if return_result:
        return fitted_params, res
    return fitted_params
