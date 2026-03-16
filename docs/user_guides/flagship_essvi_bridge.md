---
hide:
  - navigation
  - toc
---

# eSSVI bridge

This page covers the repo's **smooth implied-surface bridge** between static-surface engineering and Dupire-oriented numerics.

<div class="cta-row cta-row--duo">
  <a class="md-button md-button--primary" href="https://github.com/willemk-stack/option-pricing-library/blob/main/demos/07_essvi_smooth_surface_for_dupire.ipynb">Open the notebook</a>
  <a class="md-button" href="../flagship_localvol_pde/">Continue to local vol + PDE</a>
</div>

<figure class="figure-frame">
  <img src="../../assets/generated/docs/docs_essvi_smoothness.png" alt="eSSVI smoothness figure" />
  <figcaption>The bridge story is about smoothness: explicit projection gives a Dupire-ready term structure with controlled seam behavior.</figcaption>
</figure>

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

## Snapshot results

| Knot / metric | Repaired SVI seam jump | Smoothed eSSVI seam jump | Note |
| --- | --- | --- | --- |
| `T = 0.15` | `0.080696` | `0.000082` | largest early-maturity seam improvement |
| `T = 0.50` | `0.043331` | `0.000012` | smooth projection stays stable mid-curve |
| `T = 1.50` | `0.013674` | `0.000010` | improvement persists at longer maturities |
| Projection summary | `price_rmse = 0.02494` | `max_abs_price_error = 0.11453` | `projection_dupire_invalid_count = 0` |

The point of this page is that eSSVI is the cleaner Dupire handoff: it gives up a little exact nodal fidelity in exchange for the smooth time derivatives the next stage actually needs.

## Why it matters

This page is where the repo's messaging becomes consistent:

- **SVI** owns static-surface engineering.
- **eSSVI** owns the preferred Dupire-ready handoff.

The next step after this page is [Local vol + PDE flagship](flagship_localvol_pde.md).
