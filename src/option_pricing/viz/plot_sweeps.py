from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from ..diagnostics.greeks.sweep import SweepResult
from ..types import PricingInputs

Style = Literal["pretty", "minimal"]


def _mpl_context(style: Style):
    import matplotlib as mpl

    if style == "minimal":
        return mpl.rc_context({})

    return mpl.rc_context(
        {
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 2.25,
            "figure.dpi": 120,
        }
    )


def plot_sweep(
    res: SweepResult,
    *,
    base: PricingInputs | None = None,
    method: str | None = None,
    title_prefix: str | None = None,
    style: Style = "pretty",
    show_strike: bool = True,
    vega_per_1pct: bool = True,
    figsize: tuple[float, float] = (11, 8),  # better for 3x2
    show: bool = True,
    savepath: str | Path | None = None,
    dpi: int = 150,
):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, ScalarFormatter
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "plot_sweep requires matplotlib. Install it with: pip install matplotlib"
        ) from e

    x = np.asarray(res.x, dtype=float)
    vega = np.asarray(res.vega, dtype=float) * (0.01 if vega_per_1pct else 1.0)

    panels = [
        ("Price", np.asarray(res.price, dtype=float), None),
        ("Delta", np.asarray(res.delta, dtype=float), 0.0),
        ("Gamma", np.asarray(res.gamma, dtype=float), 0.0),
        ("Vega" + (" (per 1%)" if vega_per_1pct else ""), vega, 0.0),
        ("Theta", np.asarray(res.theta, dtype=float), 0.0),
    ]

    with _mpl_context(style):
        fig, axs = plt.subplots(
            3, 2, sharex=True, figsize=figsize, constrained_layout=True
        )
        axs_flat = axs.ravel()

        # ----- compact header -----
        title_bits: list[str] = [b for b in [title_prefix, method] if b]
        title = " — ".join(title_bits) if title_bits else None

        subtitle = None
        K_line = None
        if base is not None:
            kind = getattr(base.spec.kind, "name", str(base.spec.kind))
            K = float(base.spec.strike)
            T = float(base.spec.expiry)
            t = float(base.t)
            tau = max(0.0, T - t)
            r = float(base.market.rate)
            q = float(getattr(base.market, "dividend_yield", 0.0))
            sig = float(base.sigma)

            subtitle = f"{kind} | K={K:g}, τ={tau:g}, r={r:g}, q={q:g}, σ={sig:g}"
            K_line = K if show_strike else None

        if title:
            fig.suptitle(title, fontsize=13, y=1.02)
        if subtitle:
            fig.text(
                0.5, 0.99, subtitle, ha="center", va="top", fontsize=10, alpha=0.85
            )

        # ----- plotting -----
        for ax, (label, y, y0) in zip(axs_flat[:5], panels, strict=False):
            ax.plot(x, y)

            # Put label as a small subplot title (cleaner than big y-label)
            ax.set_title(label, loc="left")
            ax.set_ylabel("")  # reduce clutter

            # Major grid only (minor ticks ok, but no minor grid)
            ax.minorticks_off()

            # Reference lines
            if y0 is not None:
                ax.axhline(y0, linewidth=1.0, alpha=0.35)
            if K_line is not None:
                ax.axvline(K_line, linestyle=(0, (3, 3)), linewidth=1.2, alpha=0.45)
                ax.annotate(
                    "K",
                    xy=(K_line, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(4, -4),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    fontsize=9,
                    alpha=0.7,
                )

            # Nice tick formatting
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((-3, 4))
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        # Hide unused 6th panel
        axs_flat[5].axis("off")

        # Shared x label on bottom-left
        axs[2, 0].set_xlabel("Spot (S)")

        # Delta looks better constrained
        axs_flat[1].set_ylim(-0.05, 1.05)

        if savepath is not None:
            savepath = Path(savepath)
            savepath.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()

        return fig, axs
