from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from option_pricing import VolSurface


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "docs" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Synthetic market setup ---
    spot = 100.0
    r = 0.03
    q = 0.01

    def forward(T: float) -> float:
        return float(spot * np.exp((r - q) * T))

    # --- Build sparse "market" IV quotes on a grid (T, K, iv) ---
    expiries = np.array([0.25, 0.5, 1.0, 2.0], dtype=float)
    strikes = np.linspace(70.0, 130.0, 13)

    rows: list[tuple[float, float, float]] = []
    for T in expiries:
        F = forward(float(T))
        for K in strikes:
            x = np.log(K / F)  # log-moneyness
            base = 0.18 + 0.04 * np.exp(-T)  # term structure
            skew = -0.20  # equity-like skew
            convex = 0.50  # smile curvature
            iv = base + skew * x + convex * (x**2)
            iv = float(np.clip(iv, 0.05, 1.50))
            rows.append((float(T), float(K), iv))

    surf = VolSurface.from_grid(rows, forward=forward)

    # --- Create a dense grid for plotting ---
    T_grid = np.linspace(expiries.min(), expiries.max(), 60)
    K_grid = np.linspace(strikes.min(), strikes.max(), 80)

    IV = np.empty((len(T_grid), len(K_grid)), dtype=float)
    for i, T in enumerate(T_grid):
        IV[i, :] = surf.iv(K_grid, float(T))

    # --- Plot 1: heatmap surface ---
    plt.figure()
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
    plt.savefig(out_dir / "iv_surface.png", dpi=200)

    # --- Plot 2: smile slices (multiple expiries) ---
    plt.figure()
    for T in expiries:
        plt.plot(strikes, surf.iv(strikes, float(T)), marker="o", label=f"T={T:g}")
    plt.xlabel("Strike K")
    plt.ylabel("Implied vol")
    plt.title("Smile slices from VolSurface")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "iv_smiles.png", dpi=200)

    print(f"Wrote: {out_dir / 'iv_surface.png'}")
    print(f"Wrote: {out_dir / 'iv_smiles.png'}")


if __name__ == "__main__":
    main()
