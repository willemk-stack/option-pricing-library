import math

import numpy as np
import pytest

from option_pricing.monte_carlo.estimators import (
    apply_control_variate,
    estimate_discounted_payoff,
    estimate_mean_stderr,
    pair_antithetic,
)


def test_estimate_mean_stderr_applies_discount() -> None:
    samples = np.array([1.0, 2.0, 4.0])

    price, stderr = estimate_mean_stderr(samples, discount=0.5)

    assert math.isclose(
        price, 0.5 * float(np.mean(samples)), rel_tol=0.0, abs_tol=1e-12
    )
    assert math.isclose(
        stderr,
        0.5 * float(np.std(samples, ddof=1)) / math.sqrt(len(samples)),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_estimate_discounted_payoff_matches_mean_stderr() -> None:
    samples = np.array([0.5, 1.5, 3.0, 5.0])

    assert estimate_discounted_payoff(samples, discount=0.97) == estimate_mean_stderr(
        samples,
        discount=0.97,
    )


def test_apply_control_variate_collapses_exact_linear_control() -> None:
    control = np.array([-1.0, 0.0, 2.0, 3.0])
    payoff = 5.0 + 2.0 * control

    adjusted = apply_control_variate(payoff, control, EY=1.5)

    np.testing.assert_allclose(adjusted, np.full_like(control, 8.0))


def test_pair_antithetic_requires_matching_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        pair_antithetic(np.array([1.0, 2.0]), np.array([1.0]))
