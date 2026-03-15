# Local vol + PDE flagship

This is the repo's main **numerics flagship**.

Primary notebook:

- `demos/08_localvol_pde_repricing.ipynb`

Supporting notebook:

- `demos/05_pde_pricing_and_diagnostics.ipynb`

## Thesis

The key claim is:

> Given a Dupire-ready implied surface, the library can build local vol carefully, expose instability explicitly, and price with a validated finite-difference PDE workflow.

## What it shows

- `LocalVolSurface.from_implied(...)` driven by `ESSVISmoothedSurface`
- invalid masks, denominator diagnostics, and worst-point reporting
- one PDE anchor to keep solver credibility visible
- repricing grids against the originating implied surface
- a compact convergence sweep

## Module signals

- `LocalVolSurface`
- `option_pricing.pricers.pde.*`
- repricing and convergence diagnostics

## Positioning

- This notebook intentionally does **not** spend most of its time re-teaching SVI repair.
- The surface-building story lives upstream in the Surface flagship and the eSSVI bridge.
- The PDE-only notebook remains useful when you want to defend the solver before talking about surfaces at all.
