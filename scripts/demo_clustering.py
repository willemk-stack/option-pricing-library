from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Resolve project root
ROOT = Path(__file__).resolve().parents[1]  # Adjust if needed


@dataclass(frozen=True, slots=True)
class GridPlotConfig:
    """Configuration for grid visualization figures."""

    out_dir: Path
    Nx: int = 40
    x_lb: float = 0.0
    x_ub: float = 10.0
    x_center: float = 3.0
    strengths: tuple[float, ...] = (0.1, 2.5, 5.0)
    dpi: int = 200


def generate_sinh_grid(cfg: GridPlotConfig, b: float) -> np.ndarray:
    """Generates a grid clustered around x_center using the sinh mapping."""
    u = np.linspace(-1.0, 1.0, cfg.Nx)

    # Apply sinh clustering and normalize
    raw = np.sinh(b * u)
    raw = raw / np.max(np.abs(raw))

    x = np.empty_like(raw, dtype=float)
    left_scale = cfg.x_center - cfg.x_lb
    right_scale = cfg.x_ub - cfg.x_center

    neg = raw <= 0
    pos = ~neg
    x[neg] = cfg.x_center + raw[neg] * left_scale
    x[pos] = cfg.x_center + raw[pos] * right_scale

    x[0], x[-1] = cfg.x_lb, cfg.x_ub
    return x


def main() -> None:
    # Lazy import for plotting
    try:
        import matplotlib as mpl

        mpl.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Visualizing the grid requires matplotlib. Install with: pip install matplotlib"
        ) from e

    # Setup output directory
    out_dir = ROOT / "docs" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = GridPlotConfig(out_dir=out_dir)

    # --- Plotting Logic ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 2]}
    )

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(cfg.strengths)))

    for i, b in enumerate(cfg.strengths):
        grid = generate_sinh_grid(cfg, b)
        color = colors[i]

        # Plot 1: Rug plot (Density)
        ax1.vlines(grid, i, i + 0.6, colors=color, label=f"b={b}")

        # Plot 2: Mapping (Index vs Position)
        ax2.plot(
            grid,
            np.arange(cfg.Nx),
            "o-",
            markersize=3,
            color=color,
            label=f"Strength b={b}",
        )

    # Formatting
    ax1.axvline(cfg.x_center, color="red", linestyle="--", alpha=0.4, label="Center")
    ax1.set_title(f"Grid Clustering Analysis (Nx={cfg.Nx}, Center={cfg.x_center})")
    ax1.set_yticks([])
    ax1.set_xlabel("Physical Coordinate (x)")
    ax1.legend(loc="upper right", fontsize="small", ncol=len(cfg.strengths) + 1)

    ax2.axvline(cfg.x_center, color="red", linestyle="--", alpha=0.4)
    ax2.set_ylabel("Grid Index")
    ax2.set_xlabel("Physical Coordinate (x)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()

    # Save output
    save_path = cfg.out_dir / "grid_clustering.png"
    plt.savefig(save_path, dpi=cfg.dpi)
    plt.close(fig)

    print(f"Wrote: {save_path}")


if __name__ == "__main__":
    main()
