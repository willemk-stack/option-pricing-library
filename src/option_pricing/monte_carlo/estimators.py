"""Monte Carlo estimators and variance-reduction utilities.

Responsibilities:
- compute sample means and standard errors
- discount payoff samples into price estimates
- aggregate antithetic payoff pairs
- apply control variates when an auxiliary variable with known expectation is
  available

This module should contain statistical estimation logic, not model path
simulation or instrument-specific payoff definitions.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..typing import FloatArray, FloatDType


@dataclass(frozen=True, slots=True)
class ControlVariate:
    """
    Control variate specification for variance reduction.

    A control variate is a random variable Y that is correlated with the target
    payoff X and has a known (or analytically computable) expectation E[Y]. The
    Monte Carlo estimator can be adjusted using Y to reduce variance.

    Attributes
    ----------
    values
        Function mapping terminal prices ``ST`` (shape ``(n_paths,)``) to control
        variate samples ``Y(ST)`` (same shape).
    mean
        The known expectation of the control variate under the pricing measure,
        i.e. ``E[Y]``.
    """

    values: Callable[[FloatArray], FloatArray]
    mean: float


def apply_control_variate(X: FloatArray, Y: FloatArray, EY: float) -> FloatArray:
    # Guard against degenerate controls
    var_y = float(np.var(Y, ddof=1)) if Y.size > 1 else 0.0
    if var_y <= 0.0:
        return X

    cov = float(np.cov(X, Y, ddof=1)[0, 1])
    b = cov / var_y
    return np.asarray(X - b * (Y - float(EY)), dtype=FloatDType)


def estimate_mean_stderr(
    samples: FloatArray,
    *,
    discount: float = 1.0,
) -> tuple[float, float]:
    x = np.asarray(samples, dtype=float)

    if x.ndim != 1:
        raise ValueError("samples must be one-dimensional")
    if len(x) == 0:
        raise ValueError("at least one sample is required")

    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

    price = discount * mean
    stderr = discount * std / float(np.sqrt(len(x)))

    return price, stderr


def estimate_discounted_payoff(
    payoff_values: FloatArray,
    *,
    discount: float,
) -> tuple[float, float]:
    """Estimate discounted expected payoff and standard error."""
    return estimate_mean_stderr(payoff_values, discount=float(discount))


def pair_antithetic(pos: FloatArray, neg: FloatArray) -> FloatArray:
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)

    if pos.shape != neg.shape:
        raise ValueError("pos and neg must have the same shape")

    return 0.5 * (pos + neg)
