from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

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


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}. Present columns: {list(df.columns)}"
        )


def plot_mc_bs_errorbars(
    df: pd.DataFrame,
    *,
    mode: Literal["error", "z"] = "z",
    case_col: str = "case",
    err_col: str = "MC-BS",
    se_col: str = "SE",
    zcrit: float = 1.96,
    sort: Literal["abs_z", "abs_err"] = "abs_z",
    figsize: tuple[float, float] = (11, 5),
) -> tuple[Figure, Axes]:
    """
    Plot MC vs BS discrepancies across cases.

    mode:
      - "error": horizontal error bars for (MC - BS) with ± zcrit*SE
      - "z":     standardized error z = (MC - BS) / SE with reference lines at ±zcrit

    Expected columns in df:
      case_col, err_col, se_col
    """
    _require_columns(df, [case_col, err_col, se_col])

    d = df.copy()
    d["_err"] = d[err_col].astype(float)
    d["_se"] = d[se_col].astype(float)
    d["_z"] = np.where(d["_se"] > 0, d["_err"] / d["_se"], np.nan)
    d["_abs_z"] = np.abs(d["_z"])
    d["_abs_err"] = np.abs(d["_err"])

    if sort == "abs_z":
        d = d.sort_values("_abs_z", ascending=False, na_position="last")
    else:  # "abs_err"
        d = d.sort_values("_abs_err", ascending=False, na_position="last")

    d = d.reset_index(drop=True)
    y = np.arange(len(d), dtype=float)
    labels = d[case_col].astype(str).tolist()

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if mode == "error":
        x = d["_err"].to_numpy(dtype=float)
        xerr = (zcrit * d["_se"]).to_numpy(dtype=float)

        ax.errorbar(x, y, xerr=xerr, fmt="o", capsize=4)

        ax.axvline(0.0, linewidth=1.0)
        ax.set_xlabel(f"{err_col}  (error bars: ±{zcrit}·{se_col})")
        ax.set_title("MC − BS across cases (with Monte Carlo uncertainty)")

        if np.any(d["_se"].to_numpy() == 0):
            ax.text(
                0.99,
                0.01,
                "Note: cases with SE=0 have no error bars (all simulated payoffs identical).",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
            )
    else:  # mode == "z"
        x = d["_z"].to_numpy(dtype=float)
        finite = np.isfinite(x)

        ax.axvspan(-zcrit, +zcrit, alpha=0.12, label=f"|z| ≤ {zcrit} (~95%)")

        ax.plot(x[finite], y[finite], "o", label="cases")
        out = finite & (np.abs(x) > zcrit)
        if np.any(out):
            ax.plot(x[out], y[out], "o", label="outside band")

        ax.axvline(0.0, linewidth=1.0)
        ax.axvline(+zcrit, linestyle="--", linewidth=1.0)
        ax.axvline(-zcrit, linestyle="--", linewidth=1.0)

        ax.set_xlabel(f"z = ({err_col}) / {se_col}")
        ax.set_title("Standardized MC error across cases")

        if np.any(~finite):
            ax.text(
                0.99,
                0.01,
                "Some cases have z undefined (SE=0; e.g., all payoffs identical).",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
            )

    ax.legend()
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    _pretty_ax(ax)
    return fig, ax


def plot_convergence(
    df_conv: pd.DataFrame,
    *,
    n_col: str = "n_paths",
    bs_col: str = "BS",
    mc_col: str = "MC",
    se_col: str = "SE",
    zcrit: float = 1.96,
    connect: bool = False,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (7.5, 4.8),
) -> tuple[Figure, Axes]:
    """
    Convergence plot as error vs paths:
      y = MC - BS, with ± zcrit*SE bars

    Expected columns: n_paths, MC, SE, BS.
    """
    _require_columns(df_conv, [n_col, mc_col, se_col, bs_col])

    d = df_conv.sort_values(n_col).reset_index(drop=True)

    n = d[n_col].to_numpy(dtype=float)
    mc = d[mc_col].to_numpy(dtype=float)
    se = d[se_col].to_numpy(dtype=float)
    bs = float(d[bs_col].iloc[0])

    err = mc - bs
    yerr = zcrit * se

    if ax is None:
        plt = _get_plt()
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    ax.errorbar(n, err, yerr=yerr, fmt="o", capsize=4)
    if connect:
        ax.plot(n, err)

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("n_paths (log scale)")
    ax.set_ylabel("MC − BS")
    ax.set_title(f"MC convergence (error with ±{zcrit}·SE)")

    _pretty_ax(ax)
    return fig, ax


def plot_se_scaling(
    df_conv: pd.DataFrame,
    *,
    n_col: str = "n_paths",
    se_col: str = "SE",
    fit_ref: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (7.5, 4.8),
) -> tuple[Figure, Axes]:
    """
    Plot SE vs n_paths on log-log axes.
    If fit_ref=True, overlays a reference ~ 1/sqrt(N) curve anchored at the first point.

    Expects columns: n_paths, SE.
    """
    _require_columns(df_conv, [n_col, se_col])

    d = df_conv.sort_values(n_col).reset_index(drop=True)
    n = d[n_col].to_numpy(dtype=float)
    se = d[se_col].to_numpy(dtype=float)

    if ax is None:
        plt = _get_plt()
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    ax.plot(n, se, marker="o", label="SE")

    if fit_ref and len(n) >= 1 and se[0] > 0:
        n0 = n[0]
        se0 = se[0]
        se_ref = se0 * np.sqrt(n0 / n)
        ax.plot(n, se_ref, linestyle="--", label=r"reference $\propto 1/\sqrt{N}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_paths (log scale)")
    ax.set_ylabel("SE (log scale)")
    ax.set_title("Monte Carlo standard error scaling")

    ax.legend()
    _pretty_ax(ax)
    return fig, ax
