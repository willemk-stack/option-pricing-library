# User guides

This section shows how to use **option_pricing** in practical workflows, with runnable examples and explanations of the key inputs (spot, rates, dividend yield, volatility, and time to expiry).

## Start here

- [Installation](../installation.md) — install the package and verify everything works.
- [Quickstart](quickstart.md) — pricing your first option and understanding the basic inputs.

## Guides

- [Black-Scholes](black_scholes.md) — vanilla European option pricing and Greeks.
- [Monte Carlo](monte_carlo.md) — simulation-based pricing and estimators.
- [Binomial CRR](binomial_crr.md) — lattice pricing and intuition around convergence.
- [Diagnostics](diagnostics.md) — inspecting numerical behavior and debugging inputs.
- [Implied volatility](implied_vol.md) — invert prices to implied vol and interpret results.

## Conventions used in the guides

- Rates and yields are treated as **continuously compounded** where applicable.
- Times are in **years** (make sure inputs are consistent).
- Strikes and spot/forward levels must be **positive**.