import math

import numpy as np
import pytest

from option_pricing.monte_carlo import MonteCarloResult
from option_pricing.monte_carlo.results import monte_carlo_result_from_samples


def test_monte_carlo_result_from_samples_populates_fields() -> None:
    metadata = {"n_steps": 8}
    result = monte_carlo_result_from_samples(
        np.array([2.0, 4.0, 6.0]),
        discount=0.95,
        n_paths=6,
        antithetic=True,
        seed=17,
        metadata=metadata,
    )

    assert isinstance(result, MonteCarloResult)
    assert math.isclose(result.price, 0.95 * 4.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(
        result.stderr,
        0.95 * float(np.std([2.0, 4.0, 6.0], ddof=1)) / math.sqrt(3),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert result.n_paths == 6
    assert result.effective_n == 3
    assert result.discount == 0.95
    assert result.sample_mean == 4.0
    assert result.sample_std is not None
    assert result.seed == 17
    assert result.antithetic is True
    assert result.metadata == metadata
    assert result.as_tuple() == (result.price, result.stderr)


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        (np.array([]), "at least one sample is required"),
        (np.array([[1.0, 2.0]]), "samples must be one-dimensional"),
    ],
)
def test_monte_carlo_result_from_samples_validates_input(
    samples: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        monte_carlo_result_from_samples(
            samples,
            discount=1.0,
            n_paths=1,
            antithetic=False,
        )


def test_monte_carlo_result_confidence_interval_95_percent():
    result = MonteCarloResult(
        price=10.0,
        stderr=0.5,
        n_paths=10_000,
        effective_n=10_000,
        discount=1.0,
        sample_mean=10.0,
        sample_std=50.0,
    )

    ci = result.confidence_interval(0.95)

    assert ci.level == 0.95
    assert math.isclose(ci.half_width, 1.959963984540054 * 0.5)
    assert math.isclose(ci.low, 10.0 - ci.half_width)
    assert math.isclose(ci.high, 10.0 + ci.half_width)


def test_monte_carlo_result_confidence_interval_validates_level():
    result = MonteCarloResult(
        price=10.0,
        stderr=0.5,
        n_paths=10_000,
        effective_n=10_000,
        discount=1.0,
        sample_mean=10.0,
    )

    for bad_level in [0.0, 1.0, -0.1, 1.1]:
        try:
            result.confidence_interval(bad_level)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")
