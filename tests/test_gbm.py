import numpy as np

from option_pricing.models.stochastic_processes import sim_gbm_terminal


def test_gbm_log_return_mean_and_variance_sanity():
    """Log-returns of GBM are normal with known mean/var."""
    S0 = 100.0
    mu = 0.07
    sigma = 0.2
    T = 1.5
    n_paths = 50_000
    seed = 123

    rng = np.random.default_rng(seed)
    ST = sim_gbm_terminal(n_paths=n_paths, T=T, mu=mu, sigma=sigma, S0=S0, rng=rng)

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
