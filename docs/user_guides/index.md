# Guides

Use this section either to evaluate the repo quickly or to start from the public API and work outward.

## Start with the strongest proof

- [Decision guide](decision_guide.md) - quickest route to the surface, eSSVI, local-vol/PDE, performance, and architecture pages.
- [Surface repair workflow](surface_workflow.md) - quoted surface diagnostics, SVI fit quality, and repair evidence.
- [eSSVI smooth handoff](essvi_smooth_handoff.md) - why slice-wise repair is not the final Dupire handoff, and what the smoothed surface fixes.
- [Local-vol and PDE validation](localvol_pde_validation.md) - repricing accuracy, error structure, and convergence evidence.
- [Performance evidence](../performance.md) - committed scaling plots, runtime/error tradeoffs, and reproducibility notes.
- [Architecture](../architecture.md) - typed package structure, dependency direction, and system-design intent.

## Start with the public API

- [Installation](../installation.md) - install the package and choose the right extras.
- [Instruments](instruments.md) - recommended public entry point for most users.
- [Quickstart](quickstart.md) - compact walkthrough using the convenience `PricingInputs` API.
- [Market APIs](market_api.md) - flat `MarketData` versus curves-first `PricingContext`.

## Pricing engines

- [Black-Scholes](black_scholes.md) - closed-form pricing and analytic Greeks.
- [Monte Carlo](monte_carlo.md) - GBM pricing, standard errors, and reproducibility.
- [Binomial CRR](binomial_crr.md) - lattice pricing, convergence, and American exercise.
- [PDE pricing](pde_pricing.md) - finite-difference pricing under Black-Scholes and local vol.
- [Diagnostics](diagnostics.md) - notebook-friendly helpers for comparisons, sweeps, and reports.
- [Heston diagnostics](heston_diagnostics.md) - one-call slice review for convergence, smoothness, continuity, and visible failure modes.

## Volatility workflows

- [Implied volatility](implied_vol.md) - invert Black-Scholes prices to implied vol.
- [Volatility surface](vol_surface.md) - build, query, and sanity-check grid-based or SVI-based surfaces.
- [eSSVI](essvi.md) - build analytic eSSVI surfaces, calibrate nodes, and project a smooth Dupire-ready surface.
- [SVI](svi.md) - calibrate analytic SVI slices and inspect fit diagnostics.
- [SVI repair](svi_repair.md) - detect and repair butterfly-arbitrage issues in a slice.
- [Local volatility](local_vol.md) - derive a local-vol surface from a differentiable implied surface and use it in PDE pricing.

## Conventions used throughout

- Rates and dividend yields are treated as continuously compounded.
- Times are in years.
- `PricingInputs` uses absolute expiry `T` together with valuation time `t`, so `tau = T - t`.
- When `t=0`, `OptionSpec.expiry` and `tau` have the same numeric value in the flat-input workflow.
- Instrument objects such as `VanillaOption` use `expiry` to mean time-to-expiry directly.
- Strike, spot, and forward inputs must be positive where log-moneyness is used.
