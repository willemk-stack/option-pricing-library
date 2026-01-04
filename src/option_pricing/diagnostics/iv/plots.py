from __future__ import annotations

from typing import Any

import numpy as np


def _get_plt():
    try:
        from .._mpl import get_plt  # type: ignore

        return get_plt()
    except Exception:
        import matplotlib.pyplot as plt

        return plt


def _pretty_ax(ax):
    try:
        from .._mpl import pretty_ax  # type: ignore

        pretty_ax(ax)
    except Exception:
        ax.grid(True, alpha=0.3)


def plot_iv_smile(
    data: Any,
    *,
    x: str = "K",
    y: str = "implied_vol",
    ax: Any | None = None,
    title: str = "Implied volatility smile",
    xlabel: str = "Strike K",
    ylabel: str = "Implied vol",
):
    """
    Plot an IV smile.

    Parameters
    ----------
    data :
        Either a pandas DataFrame containing columns `x` and `y`,
        or a pair of sequences (K, iv) passed as a 2-tuple.
    x, y :
        Column names when `data` is a DataFrame.
    ax :
        Optional matplotlib axes.
    """
    plt = _get_plt()

    if isinstance(data, tuple) and len(data) == 2:
        K = np.asarray(data[0], dtype=float)
        v = np.asarray(data[1], dtype=float)
    else:
        # assume DataFrame-like
        K = np.asarray(data[x], dtype=float)
        v = np.asarray(data[y], dtype=float)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
    else:
        fig = ax.figure

    ax.plot(K, v, marker="o", linewidth=1, label=y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    _pretty_ax(ax)
    return fig, ax
