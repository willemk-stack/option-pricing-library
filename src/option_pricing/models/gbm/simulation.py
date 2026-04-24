from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...typing import FloatArray


@dataclass(frozen=True, slots=True)
class GBMParams:
    """Parameters for a geometric Brownian motion process.

    Attributes
    ----------
    spot : float
        Initial spot price. Must be positive.
    drift : float
        Constant drift rate.
    sigma : float
        Constant volatility. Must be non-negative.
    """

    spot: float
    drift: float
    sigma: float

    def __post_init__(self) -> None:
        spot = float(self.spot)
        drift = float(self.drift)
        sigma = float(self.sigma)

        if not np.isfinite(spot):
            raise ValueError("spot must be finite")
        if not np.isfinite(drift):
            raise ValueError("drift must be finite")
        if not np.isfinite(sigma):
            raise ValueError("sigma must be finite")
        if spot <= 0.0:
            raise ValueError("spot must be positive")
        if sigma < 0.0:
            raise ValueError("sigma must be non-negative")


def simulate_gbm_terminal(
    params: GBMParams,
    tau: float,
    normals: FloatArray,
) -> FloatArray:
    """Simulate terminal values of a geometric Brownian motion.

    Uses the exact GBM transition:

        S_tau = S0 * exp((mu - 0.5 * sigma**2) * tau
                         + sigma * sqrt(tau) * Z)

    Parameters
    ----------
    params : GBMParams
        Parameters of the GBM process. The drift is interpreted as ``mu``.
    tau : float
        Time horizon for the simulation. Must be finite and non-negative.
    normals : array_like
        Standard normal variates ``Z``. The returned array has the same shape
        as ``normals``.

    Returns
    -------
    ndarray
        Simulated terminal spot values with dtype ``float64``.

    Raises
    ------
    ValueError
        If ``tau`` is not finite or is negative.

    Notes
    -----
    This is exact terminal simulation for GBM, not an Euler discretization.
    """
    tau = float(tau)
    if not np.isfinite(tau):
        raise ValueError("tau must be finite")
    if tau < 0.0:
        raise ValueError("tau must be non-negative")

    z = np.asarray(normals, dtype=np.float64)
    S0, mu, sigma = params.spot, params.drift, params.sigma

    out = S0 * np.exp((mu - 0.5 * sigma**2) * tau + sigma * np.sqrt(tau) * z)
    return np.asarray(out, dtype=np.float64)


def simulate_gbm_paths(
    params: GBMParams,
    time_grid: FloatArray,
    normals: FloatArray,
) -> FloatArray:
    """Simulate geometric Brownian motion paths on a time grid.

    Uses independent Brownian increments over each interval in ``time_grid``.
    For interval ``[t_j, t_{j+1}]``:

        dlogS_j = (mu - 0.5 * sigma**2) * dt_j
                  + sigma * sqrt(dt_j) * Z_j

    The log increments are accumulated with ``cumsum`` to obtain exact GBM
    values at the requested grid points.

    Parameters
    ----------
    params : GBMParams
        Parameters of the GBM process. The drift is interpreted as ``mu``.
    time_grid : array_like of shape (n_steps + 1,)
        Strictly increasing simulation times. Adjacent entries define the
        simulation intervals.
    normals : array_like of shape (n_paths, n_steps) or (n_steps,)
        Independent standard normal variates for the Brownian increments.
        If one-dimensional, the input is interpreted as a single path and is
        promoted internally to shape ``(1, n_steps)``.

    Returns
    -------
    ndarray of shape (n_paths, n_steps + 1)
        Simulated GBM paths with dtype ``float64``. The first column is
        ``params.spot`` and subsequent columns correspond to ``time_grid[1:]``.

    Raises
    ------
    ValueError
        If ``time_grid`` is not one-dimensional, has fewer than two points, or
        is not strictly increasing.
    ValueError
        If ``normals`` is not one- or two-dimensional, or if its final dimension
        is not equal to ``len(time_grid) - 1``.

    Notes
    -----
    This is exact grid-point simulation for GBM because the log process has
    independent Gaussian increments. It is not an Euler approximation.
    """
    times = np.asarray(time_grid, dtype=np.float64)
    z = np.asarray(normals, dtype=np.float64)

    if times.ndim != 1:
        raise ValueError("time_grid must be one-dimensional")
    if len(times) < 2:
        raise ValueError("time_grid must contain at least two points")

    dt = np.diff(times)
    if np.any(dt <= 0.0):
        raise ValueError("time_grid must be strictly increasing")

    n_steps = len(dt)

    if z.ndim == 1:
        z = z[None, :]

    if z.ndim != 2:
        raise ValueError("normals must be one- or two-dimensional")

    if z.shape[1] != n_steps:
        raise ValueError(f"normals must have shape (n_paths, {n_steps}); got {z.shape}")

    S0, mu, sigma = params.spot, params.drift, params.sigma

    log_increments = (mu - 0.5 * sigma**2) * dt[None, :] + sigma * np.sqrt(dt)[
        None, :
    ] * z

    log_paths = np.log(S0) + np.cumsum(log_increments, axis=1)

    paths = np.empty((z.shape[0], n_steps + 1), dtype=np.float64)
    paths[:, 0] = S0
    paths[:, 1:] = np.exp(log_paths)

    return paths


# To be removed....
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
