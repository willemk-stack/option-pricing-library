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

    # --- Create dense plotting grids in (T, x) where x = log(K/F(T)) ---
    T_grid = np.linspace(expiries.min(), expiries.max(), 60)

    # Choose x-range based on the original strike grid across expiries (so coverage matches your quotes)
    x_nodes = []
    for T in expiries:
        F = forward(float(T))
        x_nodes.extend(np.log(strikes / F))
    x_min = float(np.min(x_nodes))
    x_max = float(np.max(x_nodes))

    # Optional padding so the edges don’t look cramped
    pad = 0.10 * (x_max - x_min)
    x_grid = np.linspace(x_min - pad, x_max + pad, 80)

    IV = np.empty((len(T_grid), len(x_grid)), dtype=float)
    for i, T in enumerate(T_grid):
        F = forward(float(T))
        K_for_x = F * np.exp(x_grid)  # K = F(T) * exp(x)
        IV[i, :] = surf.iv(K_for_x, float(T))  # evaluate surface in strike space

    # --- Plot 1: heatmap surface in (x, T) ---
    plt.figure()
    plt.imshow(
        IV,
        aspect="auto",
        origin="lower",
        extent=[x_grid.min(), x_grid.max(), T_grid.min(), T_grid.max()],
    )
    plt.xlabel(r"log-moneyness  $x=\log(K/F(T))$")
    plt.ylabel("Expiry T (years)")
    plt.title("Synthetic implied volatility surface (interpolated)")
    plt.colorbar(label="Implied vol")
    plt.tight_layout()
    plt.savefig(out_dir / "iv_surface_logmoneyness.png", dpi=200)

    # --- Plot 2: smile slices vs log-moneyness ---
    plt.figure()
    for T in expiries:
        F = forward(float(T))
        K_for_x = F * np.exp(x_grid)
        plt.plot(x_grid, surf.iv(K_for_x, float(T)), marker="o", label=f"T={T:g}")

    plt.xlabel(r"log-moneyness  $x=\log(K/F(T))$")
    plt.ylabel("Implied vol")
    plt.title("Smile slices (vs log-moneyness)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "iv_smiles_logmoneyness.png", dpi=200)

    print(f"Wrote: {out_dir / 'iv_surface_logmoneyness.png'}")
    print(f"Wrote: {out_dir / 'iv_smiles_logmoneyness.png'}")


if __name__ == "__main__":
    main()
