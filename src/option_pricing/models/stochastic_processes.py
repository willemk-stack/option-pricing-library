from __future__ import annotations

import numpy as np


def sim_brownian(
    n_paths: int,
    T: float,
    dt: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate n_paths of Brownian motion on [0, T] with step size dt.

    Parameters
    ----------
    n_paths : int
        Number of independent Brownian paths.
    T : float
        Time horizon.
    dt : float
        Time step size.
    rng : np.random.Generator, optional
        Random number generator. If None, a new default_rng() is created.

    Returns
    -------
    t : ndarray, shape (n_steps + 1,)
        Time grid.
    W : ndarray, shape (n_paths, n_steps + 1)
        Simulated Brownian paths.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_steps = int(np.round(T / dt))

    Z = rng.standard_normal((n_paths, n_steps))
    dW = np.sqrt(dt) * Z
    W_incr = np.cumsum(dW, axis=1)

    # prepend W0 = 0
    W = np.hstack([np.zeros((n_paths, 1)), W_incr])

    t = np.arange(n_steps + 1) * dt
    return t, W


def sim_gbm(
    n_paths: int,
    T: float,
    dt: float,
    mu: float = 0.0,
    sigma: float = 1.0,
    S0: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate n_paths of Geometric Brownian Motion on [0, T] with step size dt.

    Parameters
    ----------
    n_paths : int
        Number of independent GBM paths.
    T : float
        Time horizon.
    dt : float
        Time step size.
    mu : float, optional
        Drift.
    sigma : float, optional
        Volatility.
    S0 : float, optional
        Initial level.
    rng : np.random.Generator, optional
        Random number generator. If None, a new default_rng() is created.

    Returns
    -------
    t : ndarray, shape (n_steps + 1,)
        Time grid.
    S : ndarray, shape (n_paths, n_steps + 1)
        Simulated GBM paths.
    """
    # pass rng through so Brownian & GBM share the same generator
    t, B = sim_brownian(n_paths, T, dt, rng=rng)

    S = S0 * np.exp((mu - 0.5 * sigma**2) * t[None, :] + sigma * B)
    return t, S


def sim_gbm_terminal(
    n_paths: int,
    T: float,
    mu: float = 0.0,
    sigma: float = 1.0,
    S0: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Simulate terminal values S_T of a GBM on [0, T].

    Parameters
    ----------
    n_paths : int
        Number of independent terminal samples.
    T : float
        Time horizon.
    mu : float, optional
        Drift.
    sigma : float, optional
        Volatility.
    S0 : float, optional
        Initial level.
    rng : np.random.Generator, optional
        Random number generator. If None, a new default_rng() is created.

    Returns
    -------
    ST : ndarray, shape (n_paths,)
        Terminal values at time T.
    """
    if rng is None:
        rng = np.random.default_rng()

    Z = rng.standard_normal(n_paths)
    return S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def plot_sample_paths(
    t: np.ndarray, paths: np.ndarray, n_plot: int = 10, title: str = "Sample paths"
) -> None:
    """
    Plot up to n_plot sample paths against the time grid t.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "plot_sample_paths requires matplotlib. Install it with: pip install matplotlib"
        ) from e

    plt.figure(figsize=(10, 5))
    for i in range(min(n_plot, len(paths))):
        plt.plot(t, paths[i], lw=0.8)
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("Value")
    plt.show()
