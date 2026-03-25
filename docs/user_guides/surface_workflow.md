# Surface repair workflow

<div class="doc-intro" markdown="1">
<p class="doc-intro__kicker">Proof path step 1</p>
<p class="doc-intro__lead">Start the proof sequence with the failure mode itself: noisy option quotes rarely arrive in a form that is smooth, interpretable, or ready for downstream numerics.</p>
<p class="doc-intro__support">This page keeps the quoted structure, the repaired SVI fit, and the slice-level stress visible at the same time so the repair step can be defended rather than implied.</p>
<div class="doc-pill-row">
  <span class="doc-pill">Quoted vs repaired</span>
  <span class="doc-pill">Static no-arb checks</span>
  <span class="doc-pill">Slice-level evidence</span>
</div>
</div>

<div class="cta-row cta-row--duo" markdown="1">
[Open the notebook](https://github.com/willemk-stack/option-pricing-library/blob/main/demos/06_surface_noarb_svi_repair.ipynb){ .md-button .md-button--primary }
[Next: eSSVI smooth handoff](essvi_smooth_handoff.md){ .md-button }
</div>

<div class="doc-card-grid" markdown="1">
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Failure mode</p>
<p class="doc-card__title">What can go wrong</p>
- Quotes can be noisy across strikes and maturities.
- A fitted surface can look smooth while still hiding slice-level residual stress.
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">What to inspect</p>
<p class="doc-card__title">Reviewer checklist</p>
- Compare the repaired fit with the quoted surface, not just one polished summary view.
- Check whether flagged slices remain visible after repair instead of being averaged away.
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Next transition</p>
<p class="doc-card__title">Why this page is not enough</p>
- Static repair is necessary but not sufficient.
- The next question is whether the surface is smooth enough in time for the Dupire handoff.
</div>
</div>

<figure markdown class="diagram">
  ![Comparison of quoted implied-vol data and repaired SVI surface values across strikes and expiries](../assets/generated/static/quote_surface_compare.light.png){ .diagram-img .diagram-light }
  ![Comparison of quoted implied-vol data and repaired SVI surface values across strikes and expiries](../assets/generated/static/quote_surface_compare.dark.png){ .diagram-img .diagram-dark }
  <figcaption>Quoted structure remains visible next to the repaired SVI fit, so the surface regularization is inspectable rather than hidden behind one summary plot.</figcaption>
</figure>

<div class="snapshot-grid" markdown="1">

<figure class="diagram" style="--diagram-max-width: 720px" markdown="1">
![Heatmap of the repaired SVI implied-vol surface over log-moneyness and maturity](../assets/generated/static/svi_repaired_surface_heatmap.light.png){ .diagram-img .diagram-light }
![Heatmap of the repaired SVI implied-vol surface over log-moneyness and maturity](../assets/generated/static/svi_repaired_surface_heatmap.dark.png){ .diagram-img .diagram-dark }
<figcaption>The repaired heatmap shows continuity and remaining stress regions before any local-vol step is considered.</figcaption>
</figure>

<figure class="diagram" style="--diagram-max-width: 720px" markdown="1">
![Repaired SVI smile slices for multiple expiries plotted against log-moneyness](../assets/generated/static/svi_smile_slices.light.png){ .diagram-img .diagram-light }
![Repaired SVI smile slices for multiple expiries plotted against log-moneyness](../assets/generated/static/svi_smile_slices.dark.png){ .diagram-img .diagram-dark }
<figcaption>Per-expiry slices keep fit quality and repair behavior visible one maturity at a time, which is what a reviewer needs to inspect.</figcaption>
</figure>

</div>

## Raw failure modes

<p class="doc-section-lead">Noise and hidden slice stress are the two reasons this page exists.</p>

- Static implied-vol surfaces often arrive with inconsistent strike and maturity structure.
- Even when a repaired surface looks smooth globally, it can still hide slice-level fit stress or static-arbitrage problems.

## What to inspect in the repair

<p class="doc-section-lead">The workflow keeps the surface engineering explicit so the repair step can be reviewed rather than trusted on presentation alone.</p>

- Ingest quoted implied vols into `VolSurface.from_grid(...)`.
- Fit analytic SVI slices with `VolSurface.from_svi(...)`.
- Run static no-arbitrage diagnostics before and after repair.
- Compare the repaired fit with the original quote structure instead of replacing the quote view with one summary surface.

## Slice-level evidence

| Expiry `T` | Mean abs SVI IV residual (bp) | Max abs SVI IV residual (bp) | Slice diagnostics after repair |
| --- | --- | --- | --- |
| `0.10` | `15.55` | `56.81` | `pass` |
| `0.25` | `9.13` | `34.09` | `flagged` |
| `1.00` | `6.98` | `21.96` | `flagged` |
| `2.00` | `6.80` | `20.45` | `pass` |

<div class="doc-panel" markdown="1">
<p class="doc-panel__label">Main takeaway</p>
The result is not that every slice becomes trivial. The result is that fit quality, flagged slices, and repair tradeoffs remain visible enough to defend. The next proof step is the smooth Dupire handoff on the <a href="essvi_smooth_handoff.md">eSSVI smooth handoff</a> page.
</div>
