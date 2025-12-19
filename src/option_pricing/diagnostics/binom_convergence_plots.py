"""Binomial-tree convergence plotting helpers.

This module is meant to live under: option_pricing/diagnostics/

It factors out the binomial convergence plot used in demo 03:
  - left panel: price vs number of steps N
  - right panel: absolute error vs N (typically log-scale)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

# NOTE: absolute imports so the module works both when vendored into the
# package (option_pricing/diagnostics/) and when copied into notebooks.
from option_pricing.pricers.black_scholes import bs_price
from option_pricing.pricers.tree import binom_price
from option_pricing.types import PricingInputs

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "This plotting function requires matplotlib. Install it with: pip install matplotlib"
        ) from e
    return plt


def _pretty_ax(ax: Axes) -> None:
    ax.grid(axis="both", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    leg = ax.get_legend()
    if leg is not None:
        leg.set_frame_on(True)
        frame = leg.get_frame()
        if frame is not None:
            frame.set_alpha(0.95)


def binom_convergence_series(
    p: PricingInputs,
    *,
    n_steps_max: int = 500,
    step: int = 5,
    method: Literal["tree", "closed_form"] = "tree",
) -> dict[str, np.ndarray]:
    """Compute binomial prices over a grid of N and compare to BS.

    Returns a dict with keys:
      - n_steps:    (m,) int
      - binom:      (m,) float
      - bs:         (1,) float
      - abs_error:  (m,) float
    """
    if n_steps_max <= 0:
        raise ValueError("n_steps_max must be > 0")
    if step <= 0:
        raise ValueError("step must be > 0")
    if n_steps_max < step:
        raise ValueError("n_steps_max must be >= step")

    n_steps_vals = np.arange(step, n_steps_max + 1, step=step, dtype=int)
    binom_vals = np.asarray(
        [float(binom_price(p, int(n), method=method)) for n in n_steps_vals],
        dtype=float,
    )
    bs_val = float(bs_price(p))
    abs_err = np.abs(bs_val - binom_vals)

    return {
        "n_steps": n_steps_vals,
        "binom": binom_vals,
        "bs": np.asarray([bs_val], dtype=float),
        "abs_error": abs_err,
    }


def plot_binom_convergence_to_bs(
    p: PricingInputs,
    *,
    n_steps_max: int = 500,
    step: int = 5,
    method: Literal["tree", "closed_form"] = "tree",
    tol: float | None = 1e-2,
    err_scale: Literal["log", "linear"] = "log",
    ax_price: Axes | None = None,
    ax_err: Axes | None = None,
    figsize: tuple[float, float] = (12, 4),
) -> tuple[Figure, tuple[Axes, Axes], dict[str, np.ndarray]]:
    """Plot binomial convergence to Black–Scholes (price + absolute error).

    Parameters
    ----------
    p:
        PricingInputs for a *European* option.
    tol:
        Optional horizontal tolerance line on the error plot.
        Pass None to disable.
    err_scale:
        "log" (default) or "linear" for the error panel y-axis.

    Returns
    -------
    fig, (ax_price, ax_err), data
    """
    data = binom_convergence_series(
        p,
        n_steps_max=n_steps_max,
        step=step,
        method=method,
    )

    n_steps_vals = data["n_steps"]
    binom_price_vals = data["binom"]
    bs_val = float(data["bs"][0])
    abs_error_vals = data["abs_error"]

    # Avoid log-scale issues when error hits 0 exactly (can happen for some N).
    if err_scale == "log":
        abs_error_plot = np.maximum(abs_error_vals, np.finfo(float).tiny)
    else:
        abs_error_plot = abs_error_vals

    plt = _get_plt()

    if ax_price is None or ax_err is None:
        fig, (ax_price2, ax_err2) = plt.subplots(
            1, 2, figsize=figsize, constrained_layout=True
        )
        ax_price = ax_price or ax_price2
        ax_err = ax_err or ax_err2
    else:
        fig = ax_price.figure

    # --- left: price vs N ---
    ax_price.plot(n_steps_vals, binom_price_vals, label="Binomial (CRR tree)")
    ax_price.hlines(
        bs_val,
        float(np.min(n_steps_vals)),
        float(np.max(n_steps_vals)),
        linestyles="dashed",
        label="Black–Scholes",
    )
    ax_price.set_xlabel("Number of steps $N$")
    ax_price.set_ylabel("Option price")
    ax_price.set_title("Convergence of binomial price to Black–Scholes")
    ax_price.legend()
    _pretty_ax(ax_price)

    # --- right: error vs N ---
    ax_err.plot(n_steps_vals, abs_error_plot, label="Absolute error")
    if tol is not None and not (err_scale == "log" and tol <= 0):
        ax_err.hlines(
            float(tol),
            float(np.min(n_steps_vals)),
            float(np.max(n_steps_vals)),
            linestyles="dashed",
            label=f"Tolerance = {tol:.2g}",
        )
    ax_err.set_xlabel("Number of steps $N$")
    ax_err.set_ylabel("Absolute error")
    ax_err.set_yscale(err_scale)
    ax_err.set_title("Error decay with $N$")
    ax_err.legend()
    ax_err.grid(True, which="both", ls=":", alpha=0.35)
    ax_err.spines["top"].set_visible(False)
    ax_err.spines["right"].set_visible(False)

    return fig, (ax_price, ax_err), data
