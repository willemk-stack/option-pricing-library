import numpy as np
import pytest

from option_pricing.models.gbm import (
    GBMParams,
    simulate_gbm_paths,
    simulate_gbm_terminal,
)


def test_gbm_log_return_mean_and_variance_sanity():
    """Log-returns of GBM are normal with known mean/var."""
    S0 = 100.0
    mu = 0.07
    sigma = 0.2
    T = 1.5
    n_paths = 50_000
    seed = 123

    rng = np.random.default_rng(seed)
    normals = rng.standard_normal(n_paths)
    params = GBMParams(spot=S0, drift=mu, sigma=sigma)
    ST = simulate_gbm_terminal(params=params, tau=T, normals=normals)

    logR = np.log(ST / S0)

    theo_mean = (mu - 0.5 * sigma * sigma) * T
    theo_var = (sigma * sigma) * T

    sample_mean = float(np.mean(logR))
    sample_var = float(np.var(logR, ddof=1))

    # Standard errors for mean and variance (for normal) to set stable, scale-aware tolerances
    se_mean = (theo_var / n_paths) ** 0.5
    se_var = (2.0 / (n_paths - 1)) ** 0.5 * theo_var

    assert abs(sample_mean - theo_mean) <= 5.0 * se_mean
    assert abs(sample_var - theo_var) <= 5.0 * se_var


def test_gbm_paths_match_exact_log_increment_construction() -> None:
    params = GBMParams(spot=100.0, drift=0.03, sigma=0.2)
    time_grid = np.array([0.0, 0.25, 1.0])
    normals = np.array([[1.0, -0.5], [0.0, 0.0]])

    paths = simulate_gbm_paths(params=params, time_grid=time_grid, normals=normals)

    dt = np.diff(time_grid)
    log_increments = (params.drift - 0.5 * params.sigma**2) * dt[
        None, :
    ] + params.sigma * np.sqrt(dt)[None, :] * normals
    expected = np.empty((2, 3), dtype=np.float64)
    expected[:, 0] = params.spot
    expected[:, 1:] = params.spot * np.exp(np.cumsum(log_increments, axis=1))

    np.testing.assert_allclose(paths, expected)

    tau = float(time_grid[-1] - time_grid[0])
    collapsed_normals = np.sum(np.sqrt(dt)[None, :] * normals, axis=1) / np.sqrt(tau)
    terminal = simulate_gbm_terminal(params=params, tau=tau, normals=collapsed_normals)
    np.testing.assert_allclose(paths[:, -1], terminal)


def test_gbm_paths_promotes_one_dimensional_normals() -> None:
    params = GBMParams(spot=90.0, drift=0.01, sigma=0.15)
    time_grid = np.array([0.0, 0.5, 1.0])

    paths = simulate_gbm_paths(
        params=params,
        time_grid=time_grid,
        normals=np.array([0.0, 1.0]),
    )

    assert paths.shape == (1, 3)
    assert np.all(paths > 0.0)


def test_gbm_paths_rejects_invalid_time_grid() -> None:
    params = GBMParams(spot=100.0, drift=0.03, sigma=0.2)

    with pytest.raises(ValueError, match="strictly increasing"):
        simulate_gbm_paths(
            params=params,
            time_grid=np.array([0.0, 0.5, 0.5]),
            normals=np.zeros((2, 2)),
        )
