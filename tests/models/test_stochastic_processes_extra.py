import numpy as np
import pytest

from option_pricing.models.stochastic_processes import (
    plot_sample_paths,
    sim_brownian,
    sim_gbm_terminal,
)


def test_sim_brownian_shape_and_starts_at_zero(rng):
    g = rng(0)
    t, W = sim_brownian(n_paths=5, T=1.0, dt=0.2, rng=g)
    assert W.shape == (5, t.size)
    assert np.allclose(W[:, 0], 0.0)


def test_sim_gbm_terminal_deterministic_with_seed(rng):
    a = sim_gbm_terminal(
        n_paths=1000, T=1.0, mu=0.03, sigma=0.2, S0=100.0, rng=rng(123)
    )
    b = sim_gbm_terminal(
        n_paths=1000, T=1.0, mu=0.03, sigma=0.2, S0=100.0, rng=rng(123)
    )
    np.testing.assert_allclose(a, b)
    assert np.all(a > 0.0)


def test_plot_sample_paths_smoke_no_gui(monkeypatch):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg", force=True)

    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    t = np.linspace(0.0, 1.0, 6)
    paths = np.vstack([t, t * 0.0])
    plot_sample_paths(t, paths, n_plot=1, title="smoke")
