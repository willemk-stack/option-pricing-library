from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path


def load_pyplot():
    try:
        import matplotlib as mpl

        mpl.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Publishing visuals requires matplotlib. Install it with: pip install -e '.[plot]'"
        ) from e
    return plt


@contextmanager
def publishing_style():
    plt = load_pyplot()
    with plt.rc_context(
        {
            "figure.dpi": 120,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "axes.titlepad": 8.0,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "DejaVu Sans",
            "savefig.bbox": "tight",
        }
    ):
        yield plt


def save_figure(fig, out_path: str | Path, *, dpi: int) -> Path:
    plt = load_pyplot()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
