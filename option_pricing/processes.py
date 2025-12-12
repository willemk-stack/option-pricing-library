import numpy as np
import matplotlib.pyplot as plt

def sim_brownian(n_paths, T, dt):
    """
    Simulate n_paths of Brownian motion on [0, T] with step size dt.

    Returns:
        t : (n_steps + 1,)
        W : (n_paths, n_steps + 1)
    """
    n_steps = int(np.round(T / dt))

    Z = np.random.normal(0, 1, (n_paths, n_steps))
    dW = np.sqrt(dt) * Z
    W_incr = np.cumsum(dW, axis=1)

    # prepend W0 = 0
    W = np.hstack([np.zeros((n_paths, 1)), W_incr])

    t = np.arange(n_steps + 1) * dt
    return t, W


def sim_gbm(n_paths, T, dt, mu=0.0, sigma=1.0, S0=1.0):
    """
    Simulate n_paths of Geometric Brownian Motion on [0, T] with step size dt.

    Returns:
        t : (n_steps + 1,)
        S : (n_paths, n_steps + 1)
    """
    t, B = sim_brownian(n_paths, T, dt)  # unpack t and Brownian paths

    S = S0 * np.exp((mu - 0.5 * sigma**2) * t[None, :] + sigma * B)
    return t, S

def sim_gbm_terminal(n_paths, T, mu=0.0, sigma=1.0, S0=1.0):
    """
    Simulate terminal values S_T of a GBM on [0, T].

    Parameters
    ----------
    n_paths : int
        Number of independent paths.
    T : float
        Time horizon.
    mu : float
        Drift.
    sigma : float
        Volatility.
    S0 : float
        Initial level.
        
    Returns
    -------
    ST : (n_paths,)
        Terminal values at time T.
    """
    
    Z = np.random.normal(size=n_paths)
    return S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def plot_sample_paths(t, paths, n_plot=10, title="Sample paths"):
    plt.figure(figsize=(10,5))
    for i in range(min(n_plot, len(paths))):
        plt.plot(t, paths[i], lw=0.8)
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("Value")
    plt.show()
