from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Allow running from a fresh checkout without installing the package first.
try:
    from option_pricing import MarketData, RandomConfig, VolSurface
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(ROOT / "src"))
    from option_pricing import MarketData, RandomConfig, VolSurface


@dataclass(frozen=True, slots=True)
class DocsFiguresConfig:
    """Configuration for synthetic figures in the docs.

    This keeps the docs build reproducible and makes it easy to tweak the
    synthetic market setup without touching plotting logic.
    """

    out_dir: Path
    market: MarketData
    expiries: np.ndarray
    strikes: np.ndarray

    # Dense plotting grid resolution
    n_T: int = 60
    n_K: int = 80

    # Figure output
    dpi: int = 200

    # Optional: tiny noise to make quotes look more "market-like".
    # Set to 0.0 to keep the surface perfectly deterministic.
    iv_noise_std: float = 0.0
    random: RandomConfig = field(default_factory=RandomConfig)


def _make_rng(cfg: RandomConfig) -> np.random.Generator:
    """Create a NumPy RNG consistent with RandomConfig."""
    if cfg.rng_type == "pcg64":
        return np.random.Generator(np.random.PCG64(cfg.seed))
    if cfg.rng_type == "mt19937":
        return np.random.Generator(np.random.MT19937(cfg.seed))
    if cfg.rng_type == "sobol":
        raise ValueError(
            "sobol rng_type is not supported here (requires scipy). "
            "Use rng_type='pcg64' or 'mt19937'."
        )
    raise ValueError(f"Unknown rng_type: {cfg.rng_type!r}")


def _build_synthetic_surface(cfg: DocsFiguresConfig) -> VolSurface:
    rng = _make_rng(cfg.random)

    rows: list[tuple[float, float, float]] = []
    for T in cfg.expiries.astype(float):
        F = float(cfg.market.forward(float(T)))
        for K in cfg.strikes.astype(float):
            x = float(np.log(float(K) / F))  # log-moneyness

            # A simple, plausible equity-like surface: term structure + skew + smile.
            base = 0.18 + 0.04 * float(np.exp(-float(T)))
            skew = -0.20
            convex = 0.50
            iv = base + skew * x + convex * (x**2)

            if cfg.iv_noise_std > 0.0:
                iv += float(rng.normal(0.0, cfg.iv_noise_std))

            iv = float(np.clip(iv, 0.05, 1.50))
            rows.append((float(T), float(K), iv))

    # VolSurface expects a callable forward(T)->float; MarketData.forward matches.
    return VolSurface.from_grid(rows, forward=cfg.market.forward)


def main() -> None:
    # Import matplotlib lazily so the package can be used without plot deps,
    # while still allowing the docs workflow to render figures.
    try:
        import matplotlib as mpl

        mpl.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "make_docs_figures.py requires matplotlib. Install it with: pip install -e '.[plot]'"
        ) from e

    out_dir = ROOT / "docs" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Config-driven synthetic market setup ---
    cfg = DocsFiguresConfig(
        out_dir=out_dir,
        market=MarketData(spot=100.0, rate=0.03, dividend_yield=0.01),
        expiries=np.array([0.25, 0.5, 1.0, 2.0], dtype=float),
        strikes=np.linspace(70.0, 130.0, 13, dtype=float),
        # If you ever want a slightly more organic-looking surface:
        # iv_noise_std=0.002,
        random=RandomConfig(seed=0),
    )

    surf = _build_synthetic_surface(cfg)

    # --- Create a dense grid for plotting ---
    T_grid = np.linspace(float(cfg.expiries.min()), float(cfg.expiries.max()), cfg.n_T)
    K_grid = np.linspace(float(cfg.strikes.min()), float(cfg.strikes.max()), cfg.n_K)

    IV = np.empty((len(T_grid), len(K_grid)), dtype=float)
    for i, T in enumerate(T_grid):
        IV[i, :] = surf.iv(K_grid, float(T))

    # --- Plot 1: heatmap surface ---
    fig1 = plt.figure()
    plt.imshow(
        IV,
        aspect="auto",
        origin="lower",
        extent=[K_grid.min(), K_grid.max(), T_grid.min(), T_grid.max()],
    )
    plt.xlabel("Strike K")
    plt.ylabel("Expiry T (years)")
    plt.title("Synthetic implied volatility surface (interpolated)")
    plt.colorbar(label="Implied vol")
    plt.tight_layout()
    plt.savefig(cfg.out_dir / "iv_surface.png", dpi=cfg.dpi)
    plt.close(fig1)

    # --- Plot 2: smile slices (multiple expiries) ---
    fig2 = plt.figure()
    for T in cfg.expiries.astype(float):
        plt.plot(
            cfg.strikes,
            surf.iv(cfg.strikes, float(T)),
            marker="o",
            label=f"T={T:g}",
        )
    plt.xlabel("Strike K")
    plt.ylabel("Implied vol")
    plt.title("Smile slices from VolSurface")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.out_dir / "iv_smiles.png", dpi=cfg.dpi)
    plt.close(fig2)

    print(f"Wrote: {cfg.out_dir / 'iv_surface.png'}")
    print(f"Wrote: {cfg.out_dir / 'iv_smiles.png'}")


if __name__ == "__main__":
    main()
