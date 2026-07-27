"""Authoritative numerical validation for selected Heston calibrations."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ....typing import FloatArray
from ..fourier import HestonBackend, QuadratureConfig
from ..numerical import (
    HestonNumericalEvaluationError,
    numerical_error_from_floating_point,
    require_finite,
)
from ..params import HESTON_PARAM_NAMES, HestonParams
from .bounds import HestonCalibrationBounds
from .heston_types import HestonQuoteSet
from .objective import (
    _price_and_jac_heston_quotes,
    _price_heston_quotes,
    _validate_analytic_jacobian_params,
)


@dataclass(frozen=True, slots=True)
class HestonCalibrationValidation:
    """Finite selected-fit outputs established by authoritative re-evaluation."""

    model_prices: FloatArray
    price_jacobian: FloatArray | None
    price_rmse: float
    max_abs_price_error: float
    analytic_jacobian_validated: bool


def _validate_selected_parameters(
    params: HestonParams,
    bounds: HestonCalibrationBounds | None,
    *,
    require_analytic_jacobian: bool,
) -> None:
    parameter_vector = params.as_array()
    require_finite(
        parameter_vector,
        evaluation_stage="selected_solution_validation",
        parameter_vector=parameter_vector,
        maturity=None,
        probability_index=None,
        failing_expression="selected_parameters",
    )
    if bounds is not None:
        lower = bounds.lower_array()
        upper = bounds.upper_array()
        outside = (parameter_vector < lower) | (parameter_vector > upper)
        if np.any(outside):
            idx = int(np.flatnonzero(outside)[0])
            name = HESTON_PARAM_NAMES[idx]
            raise HestonNumericalEvaluationError(
                evaluation_stage="selected_solution_validation",
                category="outside_bounds",
                message=(
                    f"selected {name}={parameter_vector[idx]} is outside "
                    f"[{lower[idx]}, {upper[idx]}]"
                ),
                parameter_vector=parameter_vector,
                failing_expression=f"bounds[{name}]",
            )
    if require_analytic_jacobian:
        if bounds is None:
            raise ValueError(
                "bounds are required when authoritative validation requires "
                "the analytic Jacobian"
            )
        try:
            bounds.require_analytic_jacobian_compatible()
            _validate_analytic_jacobian_params(params, bounds)
        except ValueError as exc:
            raise HestonNumericalEvaluationError(
                evaluation_stage="selected_solution_validation",
                category="unsupported_domain",
                message=str(exc),
                parameter_vector=parameter_vector,
                failing_expression="analytic_jacobian_domain",
            ) from exc


def validate_heston_calibration_solution(
    quotes: HestonQuoteSet,
    params: HestonParams,
    *,
    bounds: HestonCalibrationBounds | None = None,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    require_analytic_jacobian: bool = True,
    required_diagnostics: Mapping[str, float | None] | None = None,
) -> HestonCalibrationValidation:
    """Strictly re-evaluate prices, Jacobian, and selected-fit diagnostics.

    Runtime warnings are promoted locally to a structured failure. Nothing is
    globally suppressed, and no value is clipped or replaced. The analytic
    Jacobian requirement can be disabled only for an explicitly requested
    finite-difference calibration mode.
    """
    _validate_selected_parameters(
        params,
        bounds,
        require_analytic_jacobian=require_analytic_jacobian,
    )
    parameter_vector = params.as_array()
    stage = "selected_solution_validation"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                if require_analytic_jacobian:
                    model_prices, price_jacobian = _price_and_jac_heston_quotes(
                        quotes,
                        params,
                        backend=backend,
                        quad_cfg=quad_cfg,
                    )
                    require_finite(
                        price_jacobian,
                        evaluation_stage=stage,
                        parameter_vector=parameter_vector,
                        maturity=None,
                        probability_index=None,
                        failing_expression="selected_price_jacobian",
                    )
                else:
                    model_prices = _price_heston_quotes(
                        quotes,
                        params,
                        backend=backend,
                        quad_cfg=quad_cfg,
                    )
                    price_jacobian = None

                require_finite(
                    model_prices,
                    evaluation_stage=stage,
                    parameter_vector=parameter_vector,
                    maturity=None,
                    probability_index=None,
                    failing_expression="selected_model_prices",
                )
                price_errors = np.asarray(
                    model_prices - quotes.mid,
                    dtype=np.float64,
                )
                price_rmse = float(np.sqrt(np.mean(price_errors * price_errors)))
                max_abs_price_error = float(np.max(np.abs(price_errors)))
                require_finite(
                    (price_rmse, max_abs_price_error),
                    evaluation_stage=stage,
                    parameter_vector=parameter_vector,
                    maturity=None,
                    probability_index=None,
                    failing_expression="post_fit_price_diagnostics",
                )

                for name, value in (required_diagnostics or {}).items():
                    if value is None:
                        continue
                    require_finite(
                        value,
                        evaluation_stage=stage,
                        parameter_vector=parameter_vector,
                        maturity=None,
                        probability_index=None,
                        failing_expression=f"post_fit_diagnostic[{name}]",
                    )

                return HestonCalibrationValidation(
                    model_prices=np.asarray(model_prices, dtype=np.float64),
                    price_jacobian=(
                        None
                        if price_jacobian is None
                        else np.asarray(price_jacobian, dtype=np.float64)
                    ),
                    price_rmse=price_rmse,
                    max_abs_price_error=max_abs_price_error,
                    analytic_jacobian_validated=require_analytic_jacobian,
                )
    except HestonNumericalEvaluationError as exc:
        raise exc.contextualized(
            evaluation_stage=stage,
            parameter_vector=parameter_vector,
        ) from exc
    except (FloatingPointError, RuntimeWarning) as exc:
        floating_error = (
            exc if isinstance(exc, FloatingPointError) else FloatingPointError(str(exc))
        )
        raise numerical_error_from_floating_point(
            floating_error,
            evaluation_stage=stage,
            parameter_vector=parameter_vector,
            failing_expression="selected_solution_re_evaluation",
        ) from exc
