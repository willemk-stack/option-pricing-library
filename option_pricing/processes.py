import numpy as np
import matplotlib.pyplot as plt

def sim_brownian(n_paths, n_steps, dt):
    """Simulate n_paths of Brownian motion with n_steps each."""
    Z = np.random.normal(0, 1, (n_paths, n_steps))
    dW = np.sqrt(dt) * Z
    W = np.cumsum(dW, axis=1)
    W = np.hstack([np.zeros((n_paths, 1)), W])  # prepend W0=0
    return W

def plot_sample_paths(t, paths, n_plot=10, title="Sample paths"):
    plt.figure(figsize=(10,5))
    for i in range(min(n_plot, len(paths))):
        plt.plot(t, paths[i], lw=0.8)
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("Value")
    plt.show()

def sim_gbm(n_paths, n_steps, dt, mu=0.0, sigma=1.0, S0=1.0):
    """Simulate n_paths of Geometric Brownian Motion using Brownian motion."""
    # Simulate underlying Brownian motion
    B = sim_brownian(n_paths, n_steps, dt)
    # Compute GBM paths
    S = S0 * np.exp((mu - 0.5 * sigma**2) * np.linspace(0, n_steps*dt, n_steps+1)[None, :] + sigma * B)
    return S, B