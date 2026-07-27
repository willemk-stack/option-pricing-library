"""Structured failures and guards for Heston numerical evaluation."""

from __future__ import annotations

from typing import Literal

import numpy as np

type HestonNumericalFailureCategory = Literal[
    "divide_by_zero",
    "invalid_floating_point_operation",
    "nonfinite_result",
    "outside_bounds",
    "overflow",
    "singular_denominator",
    "unsupported_domain",
    "unsafe_exponent",
]

_FLOAT64_LOG_MAX = float(np.log(np.finfo(np.float64).max))


class HestonNumericalEvaluationError(FloatingPointError):
    """A local, contextual failure of a Heston price or Jacobian evaluation."""

    def __init__(
        self,
        *,
        evaluation_stage: str,
        category: HestonNumericalFailureCategory,
        message: str,
        seed_index: int | None = None,
        maturity: float | None = None,
        probability_index: int | None = None,
        parameter_vector: object | None = None,
        failing_expression: str | None = None,
    ) -> None:
        self.evaluation_stage = str(evaluation_stage)
        self.seed_index = None if seed_index is None else int(seed_index)
        self.maturity = None if maturity is None else float(maturity)
        self.probability_index = (
            None if probability_index is None else int(probability_index)
        )
        self.parameter_vector = (
            None
            if parameter_vector is None
            else tuple(
                float(value)
                for value in np.asarray(parameter_vector, dtype=np.float64).reshape(-1)
            )
        )
        self.failing_expression = failing_expression
        self.category: HestonNumericalFailureCategory = category
        self.detail = str(message)
        super().__init__(self.__str__())

    def contextualized(
        self,
        *,
        evaluation_stage: str | None = None,
        seed_index: int | None = None,
        maturity: float | None = None,
        probability_index: int | None = None,
        parameter_vector: object | None = None,
        failing_expression: str | None = None,
    ) -> HestonNumericalEvaluationError:
        """Return a copy with newly available outer-boundary context."""
        return type(self)(
            evaluation_stage=(
                self.evaluation_stage
                if evaluation_stage is None
                else str(evaluation_stage)
            ),
            seed_index=self.seed_index if seed_index is None else seed_index,
            maturity=self.maturity if maturity is None else maturity,
            probability_index=(
                self.probability_index
                if probability_index is None
                else probability_index
            ),
            parameter_vector=(
                self.parameter_vector if parameter_vector is None else parameter_vector
            ),
            failing_expression=(
                self.failing_expression
                if failing_expression is None
                else failing_expression
            ),
            category=self.category,
            message=self.detail,
        )

    def __str__(self) -> str:
        context = [f"stage={self.evaluation_stage}", f"category={self.category}"]
        if self.seed_index is not None:
            context.append(f"seed_index={self.seed_index}")
        if self.maturity is not None:
            context.append(f"maturity={self.maturity:g}")
        if self.probability_index is not None:
            context.append(f"probability_index={self.probability_index}")
        if self.failing_expression is not None:
            context.append(f"expression={self.failing_expression}")
        if self.parameter_vector is not None:
            values = ", ".join(f"{value:.8g}" for value in self.parameter_vector)
            context.append(f"parameters=[{values}]")
        return (
            f"Heston numerical evaluation failed ({', '.join(context)}): {self.detail}"
        )


def _floating_point_category(
    exc: FloatingPointError,
) -> HestonNumericalFailureCategory:
    message = str(exc).lower()
    if "overflow" in message:
        return "overflow"
    if "divide by zero" in message:
        return "divide_by_zero"
    return "invalid_floating_point_operation"


def numerical_error_from_floating_point(
    exc: FloatingPointError,
    *,
    evaluation_stage: str,
    parameter_vector: object | None = None,
    maturity: float | None = None,
    probability_index: int | None = None,
    failing_expression: str | None = None,
) -> HestonNumericalEvaluationError:
    """Convert a NumPy floating-point exception into a contextual Heston error."""
    return HestonNumericalEvaluationError(
        evaluation_stage=evaluation_stage,
        category=_floating_point_category(exc),
        message=str(exc),
        maturity=maturity,
        probability_index=probability_index,
        parameter_vector=parameter_vector,
        failing_expression=failing_expression,
    )


def require_finite(
    value: object,
    *,
    evaluation_stage: str,
    parameter_vector: object | None,
    maturity: float | None,
    probability_index: int | None,
    failing_expression: str,
) -> None:
    """Reject a nonfinite real or complex scalar/array."""
    values = np.asarray(value)
    if np.all(np.isfinite(values)):
        return
    raise HestonNumericalEvaluationError(
        evaluation_stage=evaluation_stage,
        category="nonfinite_result",
        message="expression produced a nonfinite value",
        maturity=maturity,
        probability_index=probability_index,
        parameter_vector=parameter_vector,
        failing_expression=failing_expression,
    )


def require_nonzero(
    value: object,
    *,
    evaluation_stage: str,
    parameter_vector: object | None,
    maturity: float | None,
    probability_index: int | None,
    failing_expression: str,
) -> None:
    """Reject an exact zero divisor without changing the mathematics."""
    require_finite(
        value,
        evaluation_stage=evaluation_stage,
        parameter_vector=parameter_vector,
        maturity=maturity,
        probability_index=probability_index,
        failing_expression=failing_expression,
    )
    if not np.any(np.asarray(value) == 0):
        return
    raise HestonNumericalEvaluationError(
        evaluation_stage=evaluation_stage,
        category="singular_denominator",
        message="expression is an exact zero divisor",
        maturity=maturity,
        probability_index=probability_index,
        parameter_vector=parameter_vector,
        failing_expression=failing_expression,
    )


def require_safe_exp_argument(
    value: object,
    *,
    evaluation_stage: str,
    parameter_vector: object | None,
    maturity: float | None,
    probability_index: int | None,
    failing_expression: str,
) -> None:
    """Reject nonfinite or overflowing float64 exponential arguments."""
    require_finite(
        value,
        evaluation_stage=evaluation_stage,
        parameter_vector=parameter_vector,
        maturity=maturity,
        probability_index=probability_index,
        failing_expression=failing_expression,
    )
    real_part = np.real(np.asarray(value))
    if not np.any(real_part > _FLOAT64_LOG_MAX):
        return
    max_real = float(np.max(real_part))
    raise HestonNumericalEvaluationError(
        evaluation_stage=evaluation_stage,
        category="unsafe_exponent",
        message=(
            f"real exponent {max_real:.8g} exceeds the finite float64 exp range "
            f"({_FLOAT64_LOG_MAX:.8g})"
        ),
        maturity=maturity,
        probability_index=probability_index,
        parameter_vector=parameter_vector,
        failing_expression=failing_expression,
    )
