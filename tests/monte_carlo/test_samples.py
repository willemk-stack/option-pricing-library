import numpy as np
import pytest

from option_pricing.monte_carlo.estimators import ControlVariate
from option_pricing.monte_carlo.samples import (
    as_1d_samples,
    effective_payoff_samples,
    effective_samples_from_payoffs,
)


def test_as_1d_samples_accepts_expected_shape() -> None:
    samples = as_1d_samples(
        np.array([1.0, 2.0, 3.0]),
        name="terminal",
        expected_shape=(3,),
    )

    np.testing.assert_allclose(samples, np.array([1.0, 2.0, 3.0]))


def test_as_1d_samples_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match=r"terminal samples must have shape \(3,\)"):
        as_1d_samples(
            np.array([[1.0, 2.0, 3.0]]),
            name="terminal",
            expected_shape=(3,),
        )


def test_effective_samples_from_payoffs_pairs_antithetic_samples() -> None:
    payoff_values = np.array([1.0, 3.0, 5.0, 7.0])

    effective = effective_samples_from_payoffs(payoff_values, antithetic=True)

    np.testing.assert_allclose(effective, np.array([3.0, 5.0]))


def test_effective_payoff_samples_applies_control_variate_after_pairing() -> None:
    terminal = np.array([95.0, 105.0, 105.0, 95.0])
    payoff_values = terminal.copy()
    control = ControlVariate(values=lambda samples: samples, mean=100.0)

    effective = effective_payoff_samples(
        terminal=terminal,
        payoff_values=payoff_values,
        antithetic=True,
        control=control,
    )

    np.testing.assert_allclose(effective, np.array([100.0, 100.0]))
