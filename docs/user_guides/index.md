# User guides

These guides focus on *how to use the library in practice*.
They are organized from the simplest workflows to the more advanced volatility and PDE tooling.

## Start here

- [Installation](../installation.md) - install the package and verify the environment.
- [Instruments](instruments.md) - recommended public entry point for most users.
- [Quickstart](quickstart.md) - compact walkthrough using the convenience `PricingInputs` API.
- [Market APIs](market_api.md) - understand flat `MarketData` versus curves-first `PricingContext`.

## Vanilla pricing

- [Black-Scholes](black_scholes.md) - closed-form pricing and analytic Greeks.
- [Monte Carlo](monte_carlo.md) - GBM pricing, standard errors, and reproducibility.
- [Binomial CRR](binomial_crr.md) - lattice pricing, convergence, and American exercise.
- [PDE pricing](pde_pricing.md) - finite-difference pricing under Black-Scholes and local vol.
- [Diagnostics](diagnostics.md) - notebook-friendly helpers for comparisons, sweeps, and surface reports.

## Volatility workflows

- [Implied volatility](implied_vol.md) - invert Black-Scholes prices to implied vol.
- [Volatility surface](vol_surface.md) - build, query, and sanity-check grid-based or SVI-based surfaces.
- [eSSVI](essvi.md) - build analytic eSSVI surfaces, calibrate nodes, and project a smooth Dupire-ready surface.
- [SVI](svi.md) - calibrate analytic SVI slices and inspect fit diagnostics.
- [SVI repair](svi_repair.md) - detect and repair butterfly-arbitrage issues in a slice.
- [Local volatility](local_vol.md) - derive a local-vol surface from a differentiable implied surface and use it in PDE pricing.

## Flagship demos

- [Decision guide](flagship_capstone2_page.md) - choose the right flagship notebook for surface engineering, the eSSVI bridge, or local-vol/PDE numerics.
- [Surface flagship](flagship_surface.md) - the SVI/no-arbitrage/repair story for static-surface engineering.
- [eSSVI bridge](flagship_essvi_bridge.md) - the smooth term-structure handoff into Dupire-ready local vol.
- [Local vol + PDE flagship](flagship_localvol_pde.md) - the numerics-heavy local-vol diagnostics, PDE repricing, and convergence story.

## Conventions used throughout

- Rates and dividend yields are treated as continuously compounded.
- Times are in years.
- `PricingInputs` uses absolute expiry `T` together with valuation time `t`, so `tau = T - t`.
- When `t=0`, `OptionSpec.expiry` and `tau` have the same numeric value in the flat-input workflow.
- Instrument objects such as `VanillaOption` use `expiry` to mean time-to-expiry directly.
- Strike, spot, and forward inputs must be positive where log-moneyness is used.
