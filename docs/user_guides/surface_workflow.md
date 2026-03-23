hide:
    - toc

# Surface repair workflow

Start the proof sequence with the failure mode itself: noisy option quotes rarely arrive in a form that is smooth, interpretable, or ready for downstream numerics.

<div class="cta-row cta-row--duo" markdown="1">
[Open the notebook](https://github.com/willemk-stack/option-pricing-library/blob/main/demos/06_surface_noarb_svi_repair.ipynb){ .md-button .md-button--primary }
[Next: eSSVI smooth handoff](essvi_smooth_handoff.md){ .md-button }
</div>

<figure markdown class="diagram">
  ![Comparison of quoted implied-vol data and repaired SVI surface values across strikes and expiries](../assets/generated/static/quote_surface_compare.light.png){ .diagram-img .diagram-light }
  ![Comparison of quoted implied-vol data and repaired SVI surface values across strikes and expiries](../assets/generated/static/quote_surface_compare.dark.png){ .diagram-img .diagram-dark }
  <figcaption>Quoted structure remains visible next to the repaired SVI fit, so the surface regularization is inspectable rather than hidden behind one polished plot.</figcaption>
</figure>

<div class="snapshot-grid" markdown="1">

<figure class="diagram" style="--diagram-max-width: 720px" markdown="1">
![Heatmap of the repaired SVI implied-vol surface over log-moneyness and maturity](../assets/generated/static/svi_repaired_surface_heatmap.light.png){ .diagram-img .diagram-light }
![Heatmap of the repaired SVI implied-vol surface over log-moneyness and maturity](../assets/generated/static/svi_repaired_surface_heatmap.dark.png){ .diagram-img .diagram-dark }
<figcaption>The repaired surface heatmap makes maturity continuity and remaining stress regions visible before any local-vol step is considered.</figcaption>
</figure>

<figure class="diagram" style="--diagram-max-width: 720px" markdown="1">
![Repaired SVI smile slices for multiple expiries plotted against log-moneyness](../assets/generated/static/svi_smile_slices.light.png){ .diagram-img .diagram-light }
![Repaired SVI smile slices for multiple expiries plotted against log-moneyness](../assets/generated/static/svi_smile_slices.dark.png){ .diagram-img .diagram-dark }
<figcaption>Per-expiry slices keep fit quality and repair behavior visible slice by slice, which is what a reviewer needs to inspect.</figcaption>
</figure>

</div>

## Raw failure modes

Static implied-vol surfaces have two failure modes that matter in practice:

- the quotes are noisy and inconsistent across maturities
- a fitted surface can look smooth while still hiding slice-level fit stress or static-arbitrage problems

## What to inspect in the repair

The workflow keeps the surface engineering explicit:

- ingest quoted implied vols into `VolSurface.from_grid(...)`
- fit analytic SVI slices with `VolSurface.from_svi(...)`
- run static no-arbitrage diagnostics before and after repair
- compare the repaired fit to the original quote structure instead of replacing the quote view with one summary surface

## Slice-level evidence

| Expiry `T` | Mean abs SVI IV residual (bp) | Max abs SVI IV residual (bp) | Slice diagnostics after repair |
| --- | --- | --- | --- |
| `0.10` | `15.55` | `56.81` | `pass` |
| `0.25` | `9.13` | `34.09` | `flagged` |
| `1.00` | `6.98` | `21.96` | `flagged` |
| `2.00` | `6.80` | `20.45` | `pass` |

The main result is not that every slice becomes trivial. It is that the repo makes fit quality, flagged slices, and repair tradeoffs visible enough to defend.
If the next question is what should feed local vol, continue to [eSSVI smooth handoff](essvi_smooth_handoff.md), where the proof path shifts from static repair quality to the Dupire-oriented handoff.
