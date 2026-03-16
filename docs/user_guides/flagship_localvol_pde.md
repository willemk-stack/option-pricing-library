---
hide:
  - navigation
  - toc
---

# Local vol + PDE flagship

This is the repo's main **numerics flagship**.

<div class="cta-row cta-row--trio">
  <a class="md-button md-button--primary" href="https://github.com/willemk-stack/option-pricing-library/blob/main/demos/08_localvol_pde_repricing.ipynb">Open the flagship notebook</a>
  <a class="md-button" href="https://github.com/willemk-stack/option-pricing-library/blob/main/demos/05_pde_pricing_and_diagnostics.ipynb">Open the PDE appendix</a>
  <a class="md-button" href="https://github.com/willemk-stack/option-pricing-library/blob/main/demos/09_surface_to_localvol_pde_integration.ipynb">Open the integration proof</a>
</div>

<figure class="figure-frame">
  <img src="../../assets/generated/numerics/pde_roundtrip_scatter.png" alt="PDE repricing scatter" />
  <figcaption>The local-vol/PDE proof is about repricing accuracy and visible diagnostics, not just producing a local-vol surface.</figcaption>
</figure>

<figure class="figure-frame figure-frame--compact">
  <img src="../../assets/generated/numerics/pde_convergence.png" alt="PDE convergence chart" />
  <figcaption>Representative local-vol PDE convergence sweep for the published bundle.</figcaption>
</figure>

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

## Snapshot results

| Metric | Published bundle value |
| --- | --- |
| Repriced options | `154` |
| Mean abs price error | `0.0008067` |
| Max abs price error | `0.0044506` |
| Mean abs IV error | `1.0171 bp` |
| Max abs IV error | `18.7059 bp` |

The numerics story is that the repo makes the repricing error, runtime, and convergence behavior visible enough to defend in an interview instead of asking readers to trust a PDE black box.

## Positioning

- This notebook intentionally does **not** spend most of its time re-teaching SVI repair.
- The surface-building story lives upstream in the Surface flagship and the eSSVI bridge.
- The PDE-only notebook remains useful when you want to defend the solver before talking about surfaces at all.
