---
hide:
  - navigation
  - toc
---

# Surface flagship

This page is the fastest way to inspect the repo's **static-surface engineering** proof.

<div class="cta-row cta-row--duo">
  <a class="md-button md-button--primary" href="https://github.com/willemk-stack/option-pricing-library/blob/main/demos/06_surface_noarb_svi_repair.ipynb">Open the notebook</a>
  <a class="md-button" href="../flagship_essvi_bridge/">Continue to the eSSVI bridge</a>
</div>

<figure class="figure-frame">
  <img src="../../assets/generated/static/quote_surface_compare.png" alt="Comparison of quoted implied-vol data and repaired SVI surface values across strikes and expiries" />
  <figcaption>Primary proof for the surface workflow: quoted structure stays visible alongside the repaired SVI fit, so a reviewer can see where the library preserves market shape versus where it regularizes noisy slices.</figcaption>
</figure>

<div class="snapshot-grid">
  <figure class="figure-frame figure-frame--compact">
    <img src="../../assets/generated/static/svi_repaired_surface_heatmap.png" alt="Heatmap of the repaired SVI implied-vol surface over log-moneyness and maturity" />
    <figcaption>The repaired surface heatmap makes maturity-to-maturity continuity visible before any downstream local-vol handoff is considered.</figcaption>
  </figure>
  <figure class="figure-frame figure-frame--compact">
    <img src="../../assets/generated/static/svi_smile_slices.png" alt="Repaired SVI smile slices for multiple expiries plotted against log-moneyness" />
    <figcaption>Per-expiry smile slices keep the no-arbitrage and repair story inspectable slice by slice instead of hiding it inside one composite surface view.</figcaption>
  </figure>
</div>

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

## Snapshot results

| Expiry `T` | Mean abs SVI IV residual (bp) | Max abs SVI IV residual (bp) | Slice diagnostics after repair |
| --- | --- | --- | --- |
| `0.10` | `15.55` | `56.81` | `pass` |
| `0.25` | `9.13` | `34.09` | `flagged` |
| `1.00` | `6.98` | `21.96` | `flagged` |
| `2.00` | `6.80` | `20.45` | `pass` |

The page-level takeaway is not that every expiry becomes trivial, but that the repo makes the fit quality and no-arbitrage failures explicit instead of hiding them behind a single polished surface image.

## What it does not try to prove

- It is **not** the main Dupire/local-vol story anymore.
- The generic slice-stack path remains useful for demos and audits, but the repo now positions the smooth eSSVI projection as the stronger Dupire handoff.

If the audience asks, "what should feed local vol?", move directly to [eSSVI bridge](flagship_essvi_bridge.md).
