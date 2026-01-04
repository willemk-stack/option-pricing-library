from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from .._mpl import get_plt, pretty_ax, require_columns

if TYPE_CHECKING:
    pass


def plot_mc_bs_errorbars(
    df: pd.DataFrame,
    *,
    case_col: str = "case",
    err_col: str = "err",
    se_col: str = "se",
    sort: Literal["abs_z", "abs_err", "case"] = "abs_z",
    zcrit: float | None = None,
    figsize=(12, 5),
):
    """Plot MC-vs-BS errors and standardized errors.

    Left panel: error with ±zcrit*SE band.
    Right panel: z-scores with ±zcrit reference lines.
    """
    require_columns(df, [case_col, err_col, se_col])

    d = df.copy()
    d["_err"] = d[err_col].astype(float)
    d["_se"] = d[se_col].astype(float)
    d["_z"] = np.where(d["_se"] > 0, d["_err"] / d["_se"], np.nan)
    d["_abs_z"] = np.abs(d["_z"])
    d["_abs_err"] = np.abs(d["_err"])

    if zcrit is None:
        zcrit = float(d["zcrit"].iloc[0]) if "zcrit" in d.columns and len(d) else 1.96

    if sort == "abs_z":
        d = d.sort_values("_abs_z", ascending=False)
    elif sort == "abs_err":
        d = d.sort_values("_abs_err", ascending=False)
    else:
        d = d.sort_values(case_col, ascending=True)

    labels = d[case_col].astype(str).to_list()
    x = np.arange(len(d))

    plt = get_plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # Error bars
    band = zcrit * d["_se"].to_numpy()
    ax1.errorbar(
        x, d["_err"], yerr=band, fmt="o", capsize=3, label="MC - BS (±zcrit·SE)"
    )
    ax1.axhline(0.0, lw=1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_ylabel("Price error")
    ax1.set_title("MC vs BS errors")
    ax1.legend()
    pretty_ax(ax1)

    # z-scores
    ax2.plot(x, d["_z"], "o", label="z = (MC - BS) / SE")
    ax2.axhline(0.0, lw=1.0)
    ax2.axhline(+zcrit, ls="--", label=f"+{zcrit:g}")
    ax2.axhline(-zcrit, ls="--", label=f"-{zcrit:g}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_ylabel("z-score")
    ax2.set_title("Standardized errors")
    ax2.legend()
    pretty_ax(ax2)

    return fig, (ax1, ax2)


def plot_convergence(
    dfs: Sequence[pd.DataFrame],
    *,
    x_col: str = "n_paths",
    y_col: str = "abs_err",
    label_col: str = "case",
    agg: Literal["mean", "max", "median"] = "mean",
    figsize=(7, 4),
):
    """Plot aggregate error vs number of MC paths from multiple result DataFrames."""
    if not dfs:
        raise ValueError("dfs must be a non-empty sequence of DataFrames")

    for d in dfs:
        require_columns(d, [x_col, y_col])

    plt = get_plt()
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    xs: list[float] = []
    ys: list[float] = []
    for d in dfs:
        x = float(d[x_col].iloc[0])
        xs.append(x)
        vals = d[y_col].astype(float).to_numpy()
        if agg == "mean":
            y = float(np.nanmean(vals))
        elif agg == "median":
            y = float(np.nanmedian(vals))
        else:
            y = float(np.nanmax(vals))
        ys.append(y)

    order = np.argsort(xs)
    xs_arr = np.asarray(xs, dtype=float)[order]
    ys_arr = np.asarray(ys, dtype=float)[order]

    ax.plot(xs_arr, ys_arr, marker="o", label=f"{agg}({y_col})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of paths")
    ax.set_ylabel(f"{agg} absolute error")
    ax.set_title("MC convergence (aggregate)")
    ax.legend()
    pretty_ax(ax)
    return fig, ax


def plot_se_scaling(
    df: pd.DataFrame,
    *,
    x_col: str = "n_paths",
    se_col: str = "se",
    figsize=(7, 4),
):
    """Plot reported MC standard error vs n_paths (expected ~ 1/sqrt(n_paths))."""
    require_columns(df, [x_col, se_col])

    d = df.copy()
    d = d.groupby(x_col, as_index=False)[se_col].mean(numeric_only=True)

    plt = get_plt()
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    x = d[x_col].astype(float).to_numpy()
    se = d[se_col].astype(float).to_numpy()
    order = np.argsort(x)
    x, se = x[order], se[order]

    ax.plot(x, se, marker="o", label="mean SE")
    # reference curve: se0 * sqrt(n0/n)
    if len(x) >= 1 and np.all(x > 0) and np.isfinite(se[0]) and se[0] > 0:
        ref = se[0] * np.sqrt(x[0] / x)
        ax.plot(x, ref, ls="--", label="~ 1/sqrt(n)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of paths")
    ax.set_ylabel("Standard error")
    ax.set_title("SE scaling with paths")
    ax.legend()
    pretty_ax(ax)
    return fig, ax
