import numpy as np
import pytest

from option_pricing.monte_carlo import MCConfig, RandomConfig
from option_pricing.monte_carlo.rng import (
    correlated_normals,
    make_rng,
    rng_from_random_config,
    standard_normals,
)


@pytest.mark.parametrize("rng_type", ["pcg64", "mt19937"])
def test_rng_from_random_config_is_reproducible(rng_type: str) -> None:
    cfg = RandomConfig(seed=123, rng_type=rng_type)

    draws_a = rng_from_random_config(cfg).standard_normal(8)
    draws_b = rng_from_random_config(cfg).standard_normal(8)

    np.testing.assert_allclose(draws_a, draws_b)


def test_rng_from_random_config_rejects_sobol() -> None:
    with pytest.raises(NotImplementedError, match="Sobol sequences"):
        rng_from_random_config(RandomConfig(seed=0, rng_type="sobol"))


def test_make_rng_prefers_explicit_generator() -> None:
    rng = np.random.default_rng(7)
    cfg = MCConfig(n_paths=4, rng=rng)

    assert make_rng(cfg) is rng


def test_standard_normals_builds_antithetic_pairs() -> None:
    draws = standard_normals(
        np.random.default_rng(11),
        n_paths=6,
        sample_shape=(2,),
        antithetic=True,
    )

    assert draws.shape == (6, 2)
    np.testing.assert_allclose(draws[:3], -draws[3:])


def test_correlated_normals_matches_identity_correlation_draws() -> None:
    corr = np.eye(2)
    rng_a = np.random.default_rng(19)
    rng_b = np.random.default_rng(19)

    expected = standard_normals(rng_a, n_paths=5, sample_shape=(3, 2))
    actual = correlated_normals(rng_b, n_paths=5, corr=corr, sample_shape=(3,))

    assert actual.shape == (5, 3, 2)
    np.testing.assert_allclose(actual, expected)


def test_correlated_normals_rejects_invalid_correlation_matrix() -> None:
    with pytest.raises(ValueError, match="corr must have ones on the diagonal"):
        correlated_normals(
            np.random.default_rng(0),
            n_paths=4,
            corr=np.array([[1.0, 0.1], [0.1, 0.5]]),
        )
