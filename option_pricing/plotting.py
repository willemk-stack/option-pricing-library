from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence
from typing import Any

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


CurveFn = Callable[[np.ndarray], np.ndarray]
PostAxFn = Callable[[Axes, np.ndarray], None]


@dataclass(frozen=True)
class DistSpec:
    """
    Declarative spec for a single distribution panel.

    The key idea: this stays *generic*. `curve_fn` can be a PDF, KDE, fitted curve,
    or any function y = f(x).
    """
    name: str
    samples: np.ndarray
    curve_fn: CurveFn | None = None

    bins: int | str = 50          # can be int, or e.g. "fd" (numpy-supported strings)
    density: bool = True
    n_grid: int = 400
    xlim_quantiles: tuple[float, float] = (0.001, 0.999)

    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = "Probability Density"

    xscale: str = "linear"        # "linear" or "log"
    yscale: str = "linear"        # "linear" or "log"

    show_mean: bool = False
    show_median: bool = False

    hist_kwargs: dict[str, Any] | None = None
    curve_kwargs: dict[str, Any] | None = None

    post_ax: PostAxFn | None = None

def hist_with_curve(
    ax: Axes,
    samples: np.ndarray,
    curve_fn: CurveFn | None = None,
    *,
    bins: int | str = 50,
    density: bool = True,
    n_grid: int = 400,
    xlim_quantiles: tuple[float, float] = (0.001, 0.999),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = "Probability Density",
    xscale: str = "linear",
    yscale: str = "linear",
    show_mean: bool = False,
    show_median: bool = False,
    hist_kwargs: dict[str, Any] | None = None,
    curve_kwargs: dict[str, Any] | None = None,
    post_ax: PostAxFn | None = None,
) -> Axes:
    """
    Generic primitive: histogram + optional overlay curve.
    """

    # --- Style defaults (overrideable by user kwargs) ---
    default_hist = {
        "color": "#4C78A8",        # muted blue
        "alpha": 0.85,
        "edgecolor": "white",
        "linewidth": 0.4,
        "zorder": 1,
    }
    default_curve = {
        "color": "#F58518",        # orange
        "linewidth": 2.5,
        "zorder": 3,
    }
    mean_style = {
    "color": "#111111",
    "linestyle": "--",
    "linewidth": 2.4,     # <- thicker
    "alpha": 1.0,
    "zorder": 6,          # <- foreground
    "label": "Sample mean",
    }
    med_style = {
        "color": "#7A5195",
        "linestyle": ":",
        "linewidth": 2.4,     # <- thicker
        "alpha": 1.0,
        "zorder": 7,          # <- foreground
        "label": "Sample median",
    }
    
    hist_kwargs = {**default_hist, **(hist_kwargs or {})}
    curve_kwargs = {**default_curve, **(curve_kwargs or {})}

    x = np.asarray(samples)
    x = x[np.isfinite(x)]

    if x.size == 0:
        ax.set_title((title or "") + " (empty)")
        ax.axis("off")
        return ax

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    # Histogram first (so we can reuse bin edges if we need scaling)
    _, edges, _ = ax.hist(x, bins=bins, density=density, **hist_kwargs)

    # Robust x-range (avoid a single outlier stretching the axis)
    lo, hi = np.quantile(x, xlim_quantiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.min(x)), float(np.max(x))

    # Overlay curve
    if curve_fn is not None:
        xs = np.linspace(lo, hi, n_grid)
        ys = curve_fn(xs)

        if not density:
            bin_width = float(np.mean(np.diff(edges))) if len(edges) > 1 else 1.0
            ys = ys * x.size * bin_width

        ax.plot(xs, ys, **curve_kwargs)

    # Mean/median reference lines (now styled + legend-ready)
    if show_mean:
        ax.axvline(float(np.mean(x)), **mean_style)
    if show_median:
        ax.axvline(float(np.median(x)), **med_style)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if post_ax:
        post_ax(ax, x)

    return ax



def plot_specs(
    specs: Sequence[DistSpec],
    *,
    ncols: int = 2,
    figsize: tuple[float, float] = (15, 6),
    suptitle: str | None = None,
    share_xlim: bool = False,
    share_ylim: bool = False,
) -> tuple[Figure, np.ndarray]:
    """
    Plot a grid of DistSpec panels.

    share_xlim/share_ylim helps comparability when panels are on the same scale/units.
    """
    if ncols <= 0:
        raise ValueError("ncols must be >= 1")

    nrows = math.ceil(len(specs) / ncols) if specs else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for ax, spec in zip(axes, specs):
        hist_with_curve(
            ax,
            spec.samples,
            spec.curve_fn,
            bins=spec.bins,
            density=spec.density,
            n_grid=spec.n_grid,
            xlim_quantiles=spec.xlim_quantiles,
            title=spec.title or spec.name,
            xlabel=spec.xlabel,
            ylabel=spec.ylabel,
            xscale=spec.xscale,
            yscale=spec.yscale,
            show_mean=spec.show_mean,
            show_median=spec.show_median,
            hist_kwargs=spec.hist_kwargs,
            curve_kwargs=spec.curve_kwargs,
            post_ax=spec.post_ax,
        )

    # Hide unused axes
    for ax in axes[len(specs):]:
        ax.axis("off")

    # Apply shared limits after plotting
    active_axes = axes[:len(specs)]
    if len(active_axes) and share_xlim:
        xmins, xmaxs = zip(*(ax.get_xlim() for ax in active_axes))
        xmin, xmax = min(xmins), max(xmaxs)
        for ax in active_axes:
            ax.set_xlim(xmin, xmax)

    if len(active_axes) and share_ylim:
        ymins, ymaxs = zip(*(ax.get_ylim() for ax in active_axes))
        ymin, ymax = min(ymins), max(ymaxs)
        for ax in active_axes:
            ax.set_ylim(ymin, ymax)

    if suptitle:
        fig.suptitle(suptitle)

    return fig, axes
