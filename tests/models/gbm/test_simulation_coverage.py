from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from option_pricing.models.gbm.simulation import (  # noqa: E402
    GBMParams,
    plot_sample_paths,
    sim_brownian,
    simulate_gbm_paths,
    simulate_gbm_terminal,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spot": 0.0, "drift": 0.0, "sigma": 0.2},
        {"spot": 1.0, "drift": float("nan"), "sigma": 0.2},
        {"spot": 1.0, "drift": 0.0, "sigma": -0.1},
    ],
)
def test_gbm_params_validate_inputs(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        GBMParams(**kwargs)


def test_simulate_gbm_terminal_tau_zero_returns_spot() -> None:
    params = GBMParams(spot=100.0, drift=0.03, sigma=0.2)

    out = simulate_gbm_terminal(params, 0.0, np.array([-2.0, 0.0, 2.0]))

    np.testing.assert_allclose(out, np.array([100.0, 100.0, 100.0]))


@pytest.mark.parametrize("tau", [float("nan"), -0.1])
def test_simulate_gbm_terminal_rejects_bad_tau(tau: float) -> None:
    with pytest.raises(ValueError):
        simulate_gbm_terminal(GBMParams(100.0, 0.0, 0.2), tau, np.array([0.0]))


def test_simulate_gbm_paths_promotes_one_dimensional_normals() -> None:
    params = GBMParams(spot=100.0, drift=0.0, sigma=0.0)

    paths = simulate_gbm_paths(params, np.array([0.0, 0.5, 1.0]), np.array([1.0, -1.0]))

    assert paths.shape == (1, 3)
    np.testing.assert_allclose(paths, np.array([[100.0, 100.0, 100.0]]))


@pytest.mark.parametrize(
    ("time_grid", "normals", "match"),
    [
        (np.zeros((2, 2)), np.zeros((1, 1)), "one-dimensional"),
        (np.array([0.0]), np.zeros((1, 0)), "at least two"),
        (np.array([0.0, 0.5, 0.5]), np.zeros((1, 2)), "strictly increasing"),
        (np.array([0.0, 1.0]), np.zeros((1, 1, 1)), "one- or two-dimensional"),
        (np.array([0.0, 0.5, 1.0]), np.zeros((1, 1)), "normals must have shape"),
    ],
)
def test_simulate_gbm_paths_validates_shapes(
    time_grid: np.ndarray,
    normals: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        simulate_gbm_paths(GBMParams(100.0, 0.0, 0.2), time_grid, normals)


def test_sim_brownian_is_seeded_and_prepends_zero() -> None:
    t, paths = sim_brownian(n_paths=2, T=1.0, dt=0.5, rng=np.random.default_rng(123))

    assert t.tolist() == [0.0, 0.5, 1.0]
    assert paths.shape == (2, 3)
    np.testing.assert_allclose(paths[:, 0], np.zeros(2))


def test_plot_sample_paths_smoke() -> None:
    plot_sample_paths(np.array([0.0, 1.0]), np.array([[1.0, 2.0]]), n_plot=1)
