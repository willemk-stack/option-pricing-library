from __future__ import annotations

import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

LIGHT_THEME = "light"
DARK_THEME = "dark"
PUBLISHING_THEMES = (LIGHT_THEME, DARK_THEME)


@dataclass(frozen=True)
class AssetVariants:
    base: Path
    light: Path
    dark: Path

    def path_for(self, theme: str) -> Path:
        normalized = str(theme).strip().lower()
        if normalized == DARK_THEME:
            return self.dark
        if normalized == LIGHT_THEME:
            return self.light
        raise ValueError(f"Unsupported publishing theme: {theme}")


def themed_asset_paths(out_path: str | Path) -> AssetVariants:
    path = Path(out_path)
    suffix = path.suffix
    stem = path.stem
    return AssetVariants(
        base=path,
        light=path.with_name(f"{stem}.light{suffix}"),
        dark=path.with_name(f"{stem}.dark{suffix}"),
    )


def copy_light_variant(variants: AssetVariants) -> Path:
    variants.base.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(variants.light, variants.base)
    return variants.base


def publishing_palette(theme: str = LIGHT_THEME) -> dict[str, str]:
    normalized = str(theme).strip().lower()
    if normalized == DARK_THEME:
        return {
            "figure_face": "#0F172A",
            "axes_face": "#0F172A",
            "text": "#E5EDF7",
            "muted_text": "#A8B5C7",
            "grid": "#7C8BA0",
            "spine": "#94A3B8",
            "legend_face": "#162033",
            "legend_edge": "#314056",
            "reference": "#E5EDF7",
        }
    if normalized == LIGHT_THEME:
        return {
            "figure_face": "#FFFFFF",
            "axes_face": "#FFFFFF",
            "text": "#0B1F33",
            "muted_text": "#5A7185",
            "grid": "#A7B8C8",
            "spine": "#A7B8C8",
            "legend_face": "#FFFFFF",
            "legend_edge": "#D7E0EA",
            "reference": "#243447",
        }
    raise ValueError(f"Unsupported publishing theme: {theme}")


def style_colorbar(colorbar, *, theme: str = LIGHT_THEME) -> None:
    palette = publishing_palette(theme)
    colorbar.outline.set_edgecolor(palette["spine"])
    colorbar.ax.tick_params(colors=palette["text"])
    colorbar.ax.yaxis.label.set_color(palette["text"])


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
def publishing_style(theme: str = LIGHT_THEME):
    plt = load_pyplot()
    palette = publishing_palette(theme)
    with plt.rc_context(
        {
            "figure.dpi": 120,
            "figure.facecolor": palette["figure_face"],
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.facecolor": palette["axes_face"],
            "axes.edgecolor": palette["spine"],
            "axes.labelcolor": palette["text"],
            "axes.titlecolor": palette["text"],
            "grid.alpha": 0.18,
            "grid.color": palette["grid"],
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "axes.titlepad": 8.0,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "xtick.color": palette["text"],
            "xtick.labelsize": 9,
            "ytick.color": palette["text"],
            "ytick.labelsize": 9,
            "text.color": palette["text"],
            "legend.facecolor": palette["legend_face"],
            "legend.edgecolor": palette["legend_edge"],
            "font.family": "DejaVu Sans",
            "savefig.bbox": "tight",
            "savefig.facecolor": palette["figure_face"],
            "savefig.edgecolor": palette["figure_face"],
        }
    ):
        yield plt


def save_figure(fig, out_path: str | Path, *, dpi: int) -> Path:
    plt = load_pyplot()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return path
