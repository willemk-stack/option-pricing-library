from __future__ import annotations

import numpy as np
import pandas as pd

from .._mpl import get_plt, pretty_ax, require_columns


def plot_error_vs_runtime(
    df: pd.DataFrame,
    *,
    err_col: str = "abs_err",
    runtime_col: str = "runtime_ms",
    feasible_col: str | None = "feasible_and_within_budget",
    logx: bool = True,
    logy: bool = True,
    figsize=(7, 4),
):
    require_columns(df, [err_col, runtime_col])

    d = df.copy()
    x = d[runtime_col].astype(float).to_numpy()
    y = d[err_col].astype(float).to_numpy()

    plt = get_plt()
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    if feasible_col is not None and feasible_col in d.columns:
        ok = d[feasible_col].astype(bool).to_numpy()
        ax.plot(x[~ok], y[~ok], "o", label="infeasible")
        ax.plot(x[ok], y[ok], "o", label="feasible")
        ax.legend()
    else:
        ax.plot(x, y, "o", label="configs")
        ax.legend()

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    ax.set_xlabel("Runtime (ms)")
    ax.set_ylabel(err_col)
    ax.set_title("PDE vs BS: error vs runtime")
    pretty_ax(ax)
    return fig, ax


def plot_convergence(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str = "abs_err",
    group_col: str | None = None,
    logx: bool = True,
    logy: bool = True,
    figsize=(7, 4),
):
    require_columns(df, [x_col, y_col])

    d = df.copy()
    plt = get_plt()
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    if group_col is None or group_col not in d.columns:
        dd = d.groupby(x_col, as_index=False)[y_col].mean(numeric_only=True)
        x = dd[x_col].astype(float).to_numpy()
        y = dd[y_col].astype(float).to_numpy()
        order = np.argsort(x)
        ax.plot(x[order], y[order], marker="o", label=f"mean({y_col})")
    else:
        for key, sub in d.groupby(group_col, dropna=False):
            dd = sub.groupby(x_col, as_index=False)[y_col].mean(numeric_only=True)
            x = dd[x_col].astype(float).to_numpy()
            y = dd[y_col].astype(float).to_numpy()
            order = np.argsort(x)
            ax.plot(x[order], y[order], marker="o", label=f"{group_col}={key}")
        ax.legend()

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Convergence: {y_col} vs {x_col}")
    pretty_ax(ax)
    return fig, ax


def plot_price_scatter(
    df: pd.DataFrame,
    *,
    x_col: str = "bs",
    y_col: str = "pde",
    figsize=(5, 5),
):
    require_columns(df, [x_col, y_col])

    d = df.copy()
    x = d[x_col].astype(float).to_numpy()
    y = d[y_col].astype(float).to_numpy()

    plt = get_plt()
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    ax.plot(x, y, "o", label="runs")
    if len(x) > 0:
        lo = float(np.nanmin([np.nanmin(x), np.nanmin(y)]))
        hi = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            ax.plot([lo, hi], [lo, hi], ls="--", label="y=x")
    ax.legend()

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("PDE vs BS: price scatter")
    pretty_ax(ax)
    return fig, ax


__all__ = [
    "plot_error_vs_runtime",
    "plot_convergence",
    "plot_price_scatter",
]
