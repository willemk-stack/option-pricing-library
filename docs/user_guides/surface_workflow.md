# Surface repair workflow

<div class="doc-intro" markdown="1">
<p class="doc-intro__kicker">Proof path step 1</p>
<p class="doc-intro__lead">Start the proof sequence with the failure mode itself: noisy option quotes rarely arrive in a form that is smooth, interpretable, or ready for downstream numerics.</p>
<p class="doc-intro__support">This page treats surface repair as an engineering judgment problem, not a cosmetic preprocessing step. The quoted structure, the repaired SVI fit, and the slice-level stress stay visible together so the repair can be defended rather than assumed.</p>
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

## Problem

<p class="doc-section-lead">The workflow starts here because downstream numerics inherit the quality of the implied-vol surface they receive.</p>

- Quoted surfaces are often noisy across strike and maturity, even before any model choice is made.
- A repaired surface can look globally smooth while still hiding stressed expiries or poor local fit.
- If the repair step is hidden, the later proof pages inherit a cleaner-looking surface without showing what was actually regularized.

## Why Naive Repair Stories Fail

<p class="doc-section-lead">A single polished surface view is not enough to defend the repair. The review has to keep the raw structure and the remaining stress in view.</p>

<div class="doc-panel doc-panel--quiet" markdown="1">
<p class="doc-panel__label">Naive path</p>
Showing only one repaired heatmap or one pass/fail no-arbitrage summary can make the surface look finished too early. That hides whether the quoted structure was respected and whether difficult slices remained visible after repair.
</div>

- Global smoothness can hide slice-level residual stress.
- A repaired surface can pass broad checks while still carrying flagged expiries.
- Static repair is necessary, but it does not answer the separate time-continuity question needed for the Dupire handoff.

## Chosen Method

<p class="doc-section-lead">The page keeps the repair workflow explicit: preserve the quoted surface as the reference object, fit analytic SVI slices, and keep diagnostics attached to the repaired result.</p>

| Layer | Object or check | Design choice and reason |
| --- | --- | --- |
| Quoted input | `VolSurface.from_grid(...)` | Preserve the observed strike/maturity structure as the reference rather than overwriting it with a polished fitted view |
| Repair step | `calibrate_svi(...)` and `VolSurface.from_svi(...)` | Repair each expiry with an inspectable analytic surface instead of an opaque smoothing pass |
| Static diagnostics | no-arbitrage and slice-fit summaries | Let repaired slices be judged instead of treating the repair as automatically trustworthy |
| Evidence pairing | quoted-versus-repaired view plus per-expiry slices | Keep both global structure and expiry-level behavior visible at the same time |
| Next handoff | smooth eSSVI projection is handled separately | Avoid implying that static repair alone is sufficient for Dupire-oriented work |

## Evidence

<p class="doc-section-lead">The evidence should answer three questions quickly: what changed relative to the quotes, where the repaired surface is still delicate, and whether the remaining stress stays visible enough to inspect.</p>

<figure markdown class="diagram diagram--hero">
  ![Comparison of quoted implied-vol data and repaired SVI surface values across strikes and expiries](../assets/generated/static/quote_surface_compare.light.png){ .diagram-img .diagram-light }
  ![Comparison of quoted implied-vol data and repaired SVI surface values across strikes and expiries](../assets/generated/static/quote_surface_compare.dark.png){ .diagram-img .diagram-dark }
  <figcaption>Quoted structure remains visible next to the repaired SVI fit, so the surface regularization is inspectable rather than hidden behind one summary plot.</figcaption>
</figure>

<div class="snapshot-grid" markdown="1">

<figure class="diagram diagram--quiet" style="--diagram-max-width: 720px" markdown="1">
![Heatmap of the repaired SVI implied-vol surface over log-moneyness and maturity](../assets/generated/static/svi_repaired_surface_heatmap.light.png){ .diagram-img .diagram-light }
![Heatmap of the repaired SVI implied-vol surface over log-moneyness and maturity](../assets/generated/static/svi_repaired_surface_heatmap.dark.png){ .diagram-img .diagram-dark }
<figcaption>The repaired heatmap shows continuity and remaining stress regions before any local-vol step is considered.</figcaption>
</figure>

<figure class="diagram diagram--quiet" style="--diagram-max-width: 720px" markdown="1">
![Repaired SVI smile slices for multiple expiries plotted against log-moneyness](../assets/generated/static/svi_smile_slices.light.png){ .diagram-img .diagram-light }
![Repaired SVI smile slices for multiple expiries plotted against log-moneyness](../assets/generated/static/svi_smile_slices.dark.png){ .diagram-img .diagram-dark }
<figcaption>Per-expiry slices keep fit quality and repair behavior visible one maturity at a time, which is what a reviewer needs to inspect.</figcaption>
</figure>

</div>

### Slice-level evidence

| Expiry `T` | Mean abs SVI IV residual (bp) | Max abs SVI IV residual (bp) | Slice diagnostics after repair |
| --- | --- | --- | --- |
| `0.10` | `15.55` | `56.81` | `pass` |
| `0.25` | `9.13` | `34.09` | `flagged` |
| `1.00` | `6.98` | `21.96` | `flagged` |
| `2.00` | `6.80` | `20.45` | `pass` |

<div class="doc-panel doc-panel--quiet" markdown="1">
<p class="doc-panel__label">What to notice</p>
The repair does not pretend every expiry becomes trivial. The flagged slices remain visible, which is exactly what a reviewer should want from a production-minded repair step.
</div>

## Tradeoffs

<p class="doc-section-lead">The repair step is deliberately conservative about what it claims. It regularizes the surface enough to make the workflow usable, but it does not erase the evidence of delicacy.</p>

<div class="doc-panel doc-panel--strong" markdown="1">
<p class="doc-panel__label">Main takeaway</p>
The result is not that every slice becomes easy. The result is that the noisy quoted surface is replaced with a defensible analytic repair while fit quality, flagged slices, and remaining stress stay visible enough to inspect. That is why the next proof step is the <a href="essvi_smooth_handoff.md">eSSVI smooth handoff</a> page rather than an immediate jump to local vol.
</div>

- Keep quoted-versus-repaired comparisons visible in review-facing work instead of showing only the fitted surface.
- Treat flagged slices as useful information about surface delicacy, not as clutter to suppress.
- Use [eSSVI smooth handoff](essvi_smooth_handoff.md) to answer the separate time-continuity question before the local-vol step begins.
