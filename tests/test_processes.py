# tests/test_processes.py
import numpy as np
import pytest

from option_pricing.models.stochastic_processes import sim_brownian, sim_gbm  # adapt this import if needed (check level in __init__.py)


def test_brownian_terminal_mean_and_variance():
    """
    For standard Brownian motion W_t:
      W_T ~ N(0, T)
      => E[W_T] = 0, Var[W_T] = T (approximately)
    """
    np.random.seed(123)

    n_paths = 100_000
    T = 1.0
    dt = 0.01

    t, W = sim_brownian(n_paths=n_paths, T=T, dt=dt)
    T_eff = t[-1]               # in case T/dt was rounded
    terminal = W[:, -1]

    sample_mean = np.mean(terminal)
    sample_var = np.var(terminal)

    # Allow for Monte Carlo noise
    assert sample_mean == pytest.approx(0.0, abs=0.02)
    assert sample_var == pytest.approx(T_eff, rel=0.03)


def test_gbm_terminal_mean_and_variance():
    """
    For GBM:
        S_t = S0 * exp((mu - 0.5*sigma^2)t + sigma * W_t)
    we know:
        E[S_T]   = S0 * exp(mu * T)
        Var[S_T] = S0^2 * exp(2*mu*T) * (exp(sigma^2*T) - 1)
    """
    np.random.seed(456)

    n_paths = 100_000
    T = 1.0
    dt = 0.01
    S0 = 1.0
    mu = 0.05
    sigma = 0.2

    t, S = sim_gbm(
        n_paths=n_paths,
        T=T,
        dt=dt,
        mu=mu,
        sigma=sigma,
        S0=S0,
    )
    T_eff = t[-1]
    terminal = S[:, -1]

    sample_mean = np.mean(terminal)
    sample_var = np.var(terminal)

    theo_mean = S0 * np.exp(mu * T_eff)
    theo_var = (S0 ** 2) * np.exp(2 * mu * T_eff) * (np.exp(sigma ** 2 * T_eff) - 1.0)

    # Tolerances chosen for large n_paths; tweak if needed
    assert sample_mean == pytest.approx(theo_mean, rel=0.02)
    assert sample_var == pytest.approx(theo_var, rel=0.05)


def test_brownian_starts_at_zero():
    """
    Sanity check: W_0 should be exactly 0 for all paths.
    """
    np.random.seed(789)

    n_paths = 1_000
    T = 1.0
    dt = 0.1

    t, W = sim_brownian(n_paths=n_paths, T=T, dt=dt)

    assert t[0] == 0.0
    assert np.allclose(W[:, 0], 0.0)
