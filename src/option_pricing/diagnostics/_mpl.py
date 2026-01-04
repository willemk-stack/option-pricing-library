"""Shared Matplotlib helpers for diagnostics plots.

Centralize small plotting utilities so individual diagnostics modules don't
copy/paste boilerplate. This module intentionally keeps things lightweight.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def get_plt():
    """Import and return matplotlib.pyplot with a helpful error if missing."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Plotting requires matplotlib. Install it with: pip install matplotlib"
        ) from e
    return plt


def pretty_ax(ax: Axes) -> None:
    """Apply a minimal consistent style to an axis."""
    ax.grid(axis="both", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    leg = ax.get_legend()
    if leg is not None:
        leg.set_frame_on(True)
        frame = leg.get_frame()
        if frame is not None:
            frame.set_alpha(0.95)


def require_columns(df, cols: Iterable[str]) -> None:
    """Raise ValueError if DataFrame is missing any required columns."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
