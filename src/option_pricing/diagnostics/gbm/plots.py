from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .._mpl import get_plt, pretty_ax

if TYPE_CHECKING:
    pass


def plot_gbm_terminal_dists(
    S_T_values: Sequence[float],
    *,
    S0: float,
    r: float,
    sigma: float,
    T: float,
    q: float = 0.0,
    bins: int = 60,
    show_mean: bool = True,
    show_median: bool = True,
):
    """Compare empirical terminal distribution vs theoretical (if SciPy available).

    Produces two panels:
      1) Histogram of S_T with optional lognormal pdf overlay
      2) Histogram of log-returns with optional normal pdf overlay
    """
    S_T = np.asarray(list(S_T_values), dtype=float)
    if np.any(S_T <= 0):
        raise ValueError("S_T_values must be positive for log-return diagnostics")

    S0 = float(S0)
    r = float(r)
    q = float(q)
    sigma = float(sigma)
    T = float(T)

    plt = get_plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Panel 1: S_T
    ax1.hist(S_T, bins=bins, density=True, alpha=0.6, label="MC histogram")
    ax1.set_title(r"Terminal prices $S_T$")
    ax1.set_xlabel(r"$S_T$")
    ax1.set_ylabel("Density")

    # Panel 2: log returns
    logR = np.log(S_T / S0)
    ax2.hist(logR, bins=bins, density=True, alpha=0.6, label="MC histogram")
    ax2.set_title(r"Log returns $\ln(S_T/S_0)$")
    ax2.set_xlabel(r"$\ln(S_T/S_0)$")
    ax2.set_ylabel("Density")

    # Theoretical overlays if SciPy present
    try:
        from scipy.stats import lognorm, norm  # type: ignore

        # Under risk-neutral drift for prices:
        mu = (r - q - 0.5 * sigma**2) * T
        sig = sigma * np.sqrt(T)

        # lognormal params: shape = sig, scale = S0*exp((r-q-0.5*sigma^2)T)
        x_grid = np.linspace(np.min(S_T), np.max(S_T), 400)
        ax1.plot(
            x_grid,
            lognorm.pdf(x_grid, s=sig, scale=S0 * np.exp((r - q - 0.5 * sigma**2) * T)),
            label="Lognormal PDF",
        )

        r_grid = np.linspace(np.min(logR), np.max(logR), 400)
        ax2.plot(r_grid, norm.pdf(r_grid, loc=mu, scale=sig), label="Normal PDF")
    except Exception:
        pass

    if show_mean:
        ax1.axvline(np.mean(S_T), ls="--", label="mean")
        ax2.axvline(np.mean(logR), ls="--", label="mean")
    if show_median:
        ax1.axvline(np.median(S_T), ls=":", label="median")
        ax2.axvline(np.median(logR), ls=":", label="median")

    ax1.legend()
    ax2.legend()
    pretty_ax(ax1)
    pretty_ax(ax2)
    return fig, (ax1, ax2)
