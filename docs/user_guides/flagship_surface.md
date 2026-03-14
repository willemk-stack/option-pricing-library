# Surface flagship

This is the notebook and doc path for the repo's **static-surface engineering** story.

Primary notebook:

- `demos/06_surface_noarb_svi_repair.ipynb`

## Thesis

The key claim is:

> Given a noisy quote set, the library can build an implied surface, diagnose static arbitrage issues, fit analytic SVI smiles, repair bad slices, and compare interpolation choices in a disciplined way.

## What it shows

- quote ingestion and `VolSurface.from_grid(...)`
- per-expiry SVI fitting through `VolSurface.from_svi(...)`
- no-arbitrage diagnostics via `check_surface_noarb(...)`
- repair-aware workflows built around the SVI toolbox
- interpolation judgment for slice-stack surfaces

## Module signals

- `option_pricing.vol`
- `option_pricing.vol.svi`
- `option_pricing.diagnostics.vol_surface.*`

## What it does not try to prove

- It is **not** the main Dupire/local-vol story anymore.
- The generic slice-stack path remains useful for demos and audits, but the repo now positions the smooth eSSVI projection as the stronger Dupire handoff.

If the audience asks, "what should feed local vol?", move directly to [eSSVI bridge](flagship_essvi_bridge.md).
