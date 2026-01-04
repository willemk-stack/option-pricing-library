from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._mpl import get_plt, pretty_ax
from .compute import binom_convergence_series

if TYPE_CHECKING:
    pass


def plot_binom_convergence_to_bs(
    p: Any,
    n_steps: int | Sequence[int],
    *,
    method: str = "crr",
    err_scale: Literal["linear", "log"] = "log",
    tol: float | None = None,
    figsize=(12, 5),
):
    """Plot binomial price vs N and absolute error vs N (vs BS)."""
    data = binom_convergence_series(p, n_steps, method=method)

    n = data["n_steps"]
    binom_vals = data["binom"]
    bs_val = float(data["bs"][0])
    abs_err = data["abs_error"]

    plt = get_plt()
    fig, (ax_price, ax_err) = plt.subplots(
        1, 2, figsize=figsize, constrained_layout=True
    )

    ax_price.plot(n, binom_vals, marker="o", label=f"Binomial ({method})")
    ax_price.axhline(bs_val, ls="--", label="BS benchmark")
    ax_price.set_xlabel("Number of steps N")
    ax_price.set_ylabel("Price")
    ax_price.set_title("Price convergence")
    ax_price.legend()
    pretty_ax(ax_price)

    ax_err.plot(n, abs_err, marker="o", label="|BS - binom|")
    if tol is not None:
        ax_err.hlines(
            float(tol),
            float(np.min(n)),
            float(np.max(n)),
            linestyles="dashed",
            label=f"tol={tol:g}",
        )
    ax_err.set_xlabel("Number of steps N")
    ax_err.set_ylabel("Absolute error")
    ax_err.set_yscale(err_scale)
    ax_err.set_title("Error decay")
    ax_err.legend()
    pretty_ax(ax_err)

    return fig, (ax_price, ax_err), data
