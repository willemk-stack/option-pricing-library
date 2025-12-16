from __future__ import annotations

import numpy as np

from option_pricing import DistSpec, plot_specs


def _pretty_ax(ax, _samples) -> None:
    """Consistent, clean axis styling for diagnostics plots."""
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, framealpha=0.95)


def plot_gbm_terminal_dists(
    S_T_values: np.ndarray,
    *,
    S0: float,
    r: float,
    sigma: float,
    T: float,
    bins: int | str = 50,
    show_mean: bool = True,
    show_median: bool = True,
):
    """
    Two-panel diagnostic:
      1) log returns ln(S_T / S0) vs theoretical Normal
      2) terminal prices S_T vs theoretical Lognormal

    Requires SciPy for the theoretical PDFs.
    """
    try:
        from scipy.stats import lognorm, norm
    except ImportError as e:
        raise ImportError(
            "plot_gbm_terminal_dists requires scipy. Install with: pip install scipy"
        ) from e

    S_T_values = np.asarray(S_T_values)
    log_returns = np.log(S_T_values / S0)

    # Theoretical parameters under GBM
    mu_R = (r - 0.5 * sigma**2) * T
    sigma_R = sigma * np.sqrt(T)

    # Lognormal parameterization used by scipy.stats.lognorm:
    # s = sigma of underlying normal, scale = exp(mean of underlying normal)
    s_lognorm = sigma_R
    scale_lognorm = S0 * np.exp(mu_R)

    specs = [
        DistSpec(
            name="Log Returns",
            samples=log_returns,
            curve_fn=lambda x: norm.pdf(x, loc=mu_R, scale=sigma_R),
            bins=bins,
            density=True,
            title=r"Log Returns: $\ln(S_T/S_0)$ vs Normal PDF",
            xlabel="Log Return",
            show_mean=show_mean,
            show_median=show_median,
            hist_kwargs={"label": "MC histogram"},
            curve_kwargs={"label": "Theoretical Normal"},
            post_ax=_pretty_ax,
        ),
        DistSpec(
            name="Terminal Prices",
            samples=S_T_values,
            curve_fn=lambda x: lognorm.pdf(x, s=s_lognorm, scale=scale_lognorm),
            bins=bins,
            density=True,
            title=r"Terminal Prices: $S_T$ vs Lognormal PDF",
            xlabel=rf"$S_T$ at T={T}",
            show_mean=show_mean,
            show_median=show_median,
            hist_kwargs={"label": "MC histogram"},
            curve_kwargs={"label": "Theoretical Lognormal"},
            post_ax=_pretty_ax,
        ),
    ]

    fig, axes = plot_specs(
        specs,
        ncols=2,
        figsize=(15, 6),
        suptitle="Monte Carlo Simulation Histograms vs Theoretical Densities",
        share_ylim=False,
    )

    return fig, axes
