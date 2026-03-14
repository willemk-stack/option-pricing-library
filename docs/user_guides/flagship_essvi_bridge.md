# eSSVI bridge

This page covers the repo's **smooth implied-surface bridge** between static-surface engineering and Dupire-oriented numerics.

Primary notebook:

- `demos/07_essvi_smooth_surface_for_dupire.ipynb`

## Thesis

The key claim is:

> SVI is excellent for per-slice diagnostics and repair, but eSSVI is the cleaner object when the next step needs a smooth term structure and analytic `w_T`.

## What it shows

- `calibrate_essvi(...)` on a reusable market scenario
- `ESSVINodalSurface` as the exact calibrated node-level object
- `project_essvi_nodes(...)` as the explicit smoothing step
- nodal-versus-smoothed comparison tables
- a compact seam / `w_T` comparison against the slice-stack SVI route
- the final handoff into `LocalVolSurface.from_implied(...)`

## Module signals

- `option_pricing.vol.ssvi.*`
- `project_essvi_nodes`
- eSSVI validation and projection objects

## Why it matters

This page is where the repo's messaging becomes consistent:

- **SVI** owns static-surface engineering.
- **eSSVI** owns the preferred Dupire-ready handoff.

The next step after this page is [Local vol + PDE flagship](flagship_localvol_pde.md).
