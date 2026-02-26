# Option Pricing Library

A typed Python library for **vanilla option pricing**, **implied-volatility workflows**, **volatility surfaces**, and **finite-difference PDE pricing**.

The project started as a compact pricing library for textbook models, and now also includes a stronger numerics stack for:

- **Black–Scholes(-Merton)** analytic pricing and Greeks
- **CRR binomial tree** pricing
- **Monte Carlo under GBM**
- **Implied volatility inversion**
- **Smile and surface construction**
- **No-arbitrage diagnostics**
- **Local-volatility extraction and diagnostics**
- **Finite-difference PDE pricing and convergence checks**

## Best places to start

### New to the library
- [Installation](installation.md)
- [Quickstart](user_guides/quickstart.md)
- [API reference](api/index.md)

### Flagship workflows
- [Local volatility](user_guides/local_vol.md)
- [PDE pricing](user_guides/pde_pricing.md)
- [Roadmap](roadmap.md)

### Theory and background
- [Notes index](notes/index.md)

## Recommended workflow

The library supports a few ways to work, but the intended learning path is:

### 1. Start with the convenient pricing workflow
Use `PricingInputs` and the top-level pricers when you want a compact, direct interface for pricing and experiments.

This is the easiest way to:
- price vanilla options quickly
- compare Black–Scholes, tree, and Monte Carlo methods
- test implied-vol and surface workflows

### 2. Move to instruments when you want cleaner contracts
Use the instruments layer when you want a clearer separation between:
- **what** is being priced
- **how** it is being priced

This is a better fit for reusable workflows and a more explicit contract model.

### 3. Use curves / surfaces / PDE modules for advanced work
Use the lower-level market, vol, diagnostics, and PDE modules when you want:
- surface construction
- no-arbitrage checks
- local-vol extraction
- finite-difference pricing and diagnostics

## Flagship capstones

### Capstone 1 — Volatility surfaces, no-arbitrage, and SVI

This capstone focuses on building and validating implied-volatility structures:

- implied-volatility inversion
- smile construction
- surface interpolation
- no-arbitrage diagnostics
- SVI fitting and repair workflows

A good place to begin is the vol-surface demo and the user guides around surfaces and diagnostics.

### Capstone 2 — Local Vol + PDE Pricing + Diagnostics

This is the main numerics-focused capstone in the repo:

**surface quotes → implied surface → local-vol diagnostics → PDE pricing → convergence / repricing checks**

This part of the project is designed to show that local volatility is not treated as “just a formula,” but as a **numerical engineering workflow** with validation and diagnostics.

It includes:
- local-vol extraction from surfaces
- instability and invalid-region diagnostics
- PDE pricing under local volatility
- convergence checks
- discontinuous-payoff remedies
- validation against analytic baselines and reference implementations

Start here:
- [Local volatility guide](user_guides/local_vol.md)
- [PDE pricing guide](user_guides/pde_pricing.md)

## Validation philosophy

This library is meant to be **test-backed** and **numerically transparent**.

That means the project emphasizes:
- analytic cross-checks where available
- convergence and stability checks
- explicit diagnostics for surfaces and local volatility
- notebook demos that illustrate the workflows end to end

## Documentation map

- **Getting started**
  - [Installation](installation.md)
  - [Quickstart](user_guides/quickstart.md)

- **User guides**
  - [Local volatility](user_guides/local_vol.md)
  - [PDE pricing](user_guides/pde_pricing.md)

- **Reference**
  - [API reference](api/index.md)

- **Background**
  - [Notes index](notes/index.md)

- **Project planning**
  - [Roadmap](roadmap.md)

## Scope

The library is strongest today in:
- vanilla pricing workflows
- implied-volatility tooling
- surface diagnostics
- local-vol / PDE capstone work

Some modules are still better understood as **advanced research / portfolio components** rather than fully productized end-user workflows. The roadmap calls out what is complete, what is experimental, and what comes next.