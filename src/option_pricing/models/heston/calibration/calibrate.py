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
from .heston_types import HestonQuoteSet, HestonRegConfig
from .objective import HestonObjective


def calibrate_heston(
    quotes: HestonQuoteSet,
    sqrt_weights: FloatArray | None = None,
    x0_params: HestonParams | None = None,
    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "soft_l1",
    x_scale: Literal["jac"] | float | FloatArray | None = "jac",
    parameter_transform: str = "unconstrained",
    vega_floor: float | None = None,
    price_floor: float | None = None,
    spread_floor: float | None = None,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    reg: HestonRegConfig | None = None,
    use_analytic_jac: bool = True,
    method: Literal["trf", "dogbox", "lm"] = "trf",
    max_nfev: int | None = None,
    ftol: float | None = None,
    xtol: float | None = None,
    gtol: float | None = None,
    return_result: bool = False,
) -> HestonParams | tuple[HestonParams, OptimizeResult]:
    if parameter_transform != "unconstrained":
        raise ValueError(
            "Only parameter_transform='unconstrained' is currently supported."
        )
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
        sqrt_weights=sqrt_weights,
        vega_floor=1e-6 if vega_floor is None else float(vega_floor),
        price_floor=1e-10 if price_floor is None else float(price_floor),
        spread_floor=1e-6 if spread_floor is None else float(spread_floor),
        backend=backend,
        quad_cfg=quad_cfg,
        reg=reg,
    )

    def analytic_jac(u: np.ndarray, *_args: object, **_kwargs: object) -> FloatArray:
        return obj.jac(np.asarray(u, dtype=np.float64))

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

    fitted_params = HestonParams.transform_to_constrained(
        np.asarray(res.x, dtype=np.float64)
    )
    if return_result:
        return fitted_params, res
    return fitted_params
