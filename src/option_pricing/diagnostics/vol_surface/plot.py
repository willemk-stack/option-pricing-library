from __future__ import annotations

from typing import Any

import numpy as np

from .compute import (
    calendar_dW,
    calendar_dW_from_report,
    call_prices_from_smile,
    first_failing_smile,
    surface_slices,
)


def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Plotting requires matplotlib. Install it with: pip install matplotlib"
        ) from e
    return plt


def plot_smile_slices(
    surface: Any,
    *,
    forward,
    title: str = "Surface — smile slices",
    figsize=(9, 4),
):
    """Plot all smiles as strike vs implied vol curves."""
    plt = _get_plt()
    plt.figure(figsize=figsize)

    for sl in surface_slices(surface, forward=forward):
        plt.plot(sl.K, sl.iv, marker="o", linewidth=1, label=f"T={sl.T:g}y")

    plt.xlabel("Strike K")
    plt.ylabel("Implied vol")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_queried_smile(
    *,
    K: np.ndarray,
    iv: np.ndarray,
    T: float,
    title: str | None = None,
    figsize=(9, 4),
):
    """Plot a queried smile curve (K vs iv)."""
    plt = _get_plt()
    plt.figure(figsize=figsize)
    plt.plot(K, iv, linewidth=2)
    plt.xlabel("Strike K")
    plt.ylabel("Implied vol")
    plt.title(title or f"Queried smile at T={float(T):g}y")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_call_monotonicity_diagnostics(
    surface: Any,
    *,
    T: float,
    forward,
    df,
    bad_indices: np.ndarray,
    bs_model: Any | None = None,
    figsize=(9, 4),
):
    """Plot call prices vs strike and highlight violating intervals."""
    plt = _get_plt()
    K, C, _iv = call_prices_from_smile(
        surface, T=T, forward=forward, df=df, bs_model=bs_model
    )

    plt.figure(figsize=figsize)
    plt.plot(K, C, marker="o", linewidth=1.5, label="Call price")

    bad_i = np.asarray(bad_indices, dtype=int)
    if bad_i.size:
        plt.scatter(K[bad_i], C[bad_i], s=60, label="Violation start")
        plt.scatter(K[bad_i + 1], C[bad_i + 1], s=60, label="Violation end")

    plt.xlabel("Strike K")
    plt.ylabel("Call price (discounted)")
    plt.title(
        f"Call monotonicity check at T={float(T):g}y — bad intervals={bad_i.size}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_calendar_heatmap(
    surface: Any,
    *,
    x_grid: np.ndarray,
    title: str = "Calendar check — negative cells indicate violations",
    figsize=(10, 4),
):
    """Plot Δw heatmap over expiry steps and x_grid."""
    plt = _get_plt()
    xg = np.asarray(x_grid, dtype=float)
    dW = calendar_dW(surface, x_grid=xg)

    plt.figure(figsize=figsize)
    plt.imshow(dW, aspect="auto", origin="lower")
    plt.colorbar(label=r"Δw = w(T_{i+1},x) - w(T_i,x)")
    plt.yticks(
        np.arange(len(surface.smiles) - 1),
        [
            f"{float(surface.smiles[i].T):g}→{float(surface.smiles[i + 1].T):g}"
            for i in range(len(surface.smiles) - 1)
        ],
    )
    plt.xticks(
        np.linspace(0, len(xg) - 1, 5),
        np.round(np.linspace(xg[0], xg[-1], 5), 3),
    )
    plt.xlabel("log-moneyness x grid (approx)")
    plt.ylabel("Expiry step")
    plt.title(title)
    plt.show()


def plot_first_strike_monotonicity_violation(
    surface: Any,
    report: Any,
    *,
    forward,
    df,
    bs_model: Any | None = None,
    figsize=(9, 4),
):
    """Find the first strike-monotonicity violation and plot call prices highlighting bad intervals.

    If there are no violations, prints a short message and returns None.
    """
    found = first_failing_smile(report)
    if found is None:
        print("No strike monotonicity violations found.")
        return None

    T_fail, rep_fail = found
    bad_i = np.asarray(rep_fail.bad_indices, dtype=int)
    return plot_call_monotonicity_diagnostics(
        surface,
        T=T_fail,
        forward=forward,
        df=df,
        bad_indices=bad_i,
        bs_model=bs_model,
        figsize=figsize,
    )


def plot_calendar_heatmap_from_report(
    surface: Any,
    report: Any,
    *,
    title: str = "Calendar check — negative cells indicate violations",
    figsize=(10, 4),
):
    """If calendar check failed, plot Δw heatmap. Otherwise prints status and returns None."""
    maybe = calendar_dW_from_report(surface, report)
    if maybe is None:
        cal = getattr(report, "calendar_total_variance", None)
        if cal is None:
            print("Calendar check not available.")
        else:
            print("Calendar performed:", bool(getattr(cal, "performed", False)))
            print("Calendar OK:", bool(getattr(cal, "ok", True)))
            print("Calendar message:", str(getattr(cal, "message", "")))
        return None
    xg, _dW = maybe
    return plot_calendar_heatmap(surface, x_grid=xg, title=title, figsize=figsize)
