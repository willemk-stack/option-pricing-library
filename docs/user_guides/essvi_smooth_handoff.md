# eSSVI smooth handoff

<div class="doc-intro" markdown="1">
<p class="doc-intro__kicker">Proof path step 2</p>
<p class="doc-intro__lead">The repaired surface is still a stack of slices. This step focuses on what changes once those nodes are projected into a smooth eSSVI surface, because Dupire depends on that change more than on the static repair alone.</p>
<p class="doc-intro__support">The point of this page is not to celebrate smoothing in the abstract. It is to show that the time-continuity problem is understood, measured, and improved before local-vol extraction begins.</p>
<div class="doc-pill-row">
  <span class="doc-pill">`w_T` seam control</span>
  <span class="doc-pill">Smooth projection</span>
  <span class="doc-pill">Dupire-ready surface</span>
</div>
</div>

<div class="cta-row cta-row--duo" markdown="1">
[Open the notebook](https://github.com/willemk-stack/option-pricing-library/blob/main/demos/07_essvi_smooth_surface_for_dupire.ipynb){ .md-button .md-button--primary }
[Next: local-vol and PDE validation](localvol_pde_validation.md){ .md-button }
</div>

<div class="doc-card-grid" markdown="1">
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Problem</p>
<p class="doc-card__title">Slice-wise repair is not enough</p>
- Slice-level SVI repair can still leave time-direction seams.
- Dupire depends on a surface whose term structure is smooth enough to differentiate.
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">What changes</p>
<p class="doc-card__title">eSSVI projection</p>
- Fit exact nodes, then project them into a smooth surface with explicit validation.
- Keep the nodal and smoothed objects visible instead of hiding the transition.
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Success signal</p>
<p class="doc-card__title">Reviewer-facing outcome</p>
- Seam stress drops materially across maturities.
- The final projection keeps `projection_dupire_invalid_count = 0`.
</div>
</div>

<figure markdown class="diagram">
  ![Heatmap of the smoothed eSSVI implied-vol surface across log-moneyness and maturity](../assets/generated/dupire/essvi_smoothed_surface_heatmap.light.png){ .diagram-img .diagram-light }
  ![Heatmap of the smoothed eSSVI implied-vol surface across log-moneyness and maturity](../assets/generated/dupire/essvi_smoothed_surface_heatmap.dark.png){ .diagram-img .diagram-dark }
  <figcaption>The smoothed eSSVI surface is the preferred Dupire handoff because it gives the local-vol step a time-continuous surface instead of a stack of repaired slices.</figcaption>
</figure>

<div class="snapshot-grid" markdown="1">

<figure class="diagram" style="--diagram-max-width: 720px" markdown="1">
![Heatmap of Gatheral local volatility extracted from the smoothed eSSVI surface](../assets/generated/dupire/localvol_gatheral_heatmap.light.png){ .diagram-img .diagram-light }
![Heatmap of Gatheral local volatility extracted from the smoothed eSSVI surface](../assets/generated/dupire/localvol_gatheral_heatmap.dark.png){ .diagram-img .diagram-dark }
<figcaption>Once the handoff is smoothed, the local-vol field becomes an inspectable object rather than a hidden intermediate.</figcaption>
</figure>

<figure class="diagram" style="--diagram-max-width: 720px" markdown="1">
![Heatmap of differences between Gatheral and call-grid Dupire local-vol estimates](../assets/generated/dupire/gatheral_vs_dupire_diff_heatmap.light.png){ .diagram-img .diagram-light }
![Heatmap of differences between Gatheral and call-grid Dupire local-vol estimates](../assets/generated/dupire/gatheral_vs_dupire_diff_heatmap.dark.png){ .diagram-img .diagram-dark }
<figcaption>The difference view shows where the two extraction routes agree and where the handoff still deserves scrutiny.</figcaption>
</figure>

</div>

## What changes after smoothing

<p class="doc-section-lead">Slice-wise SVI is useful for static repair, but Dupire depends on time continuity more than on slice polish alone.</p>

The key issue is `w_T` behavior. Downstream local-vol extraction needs a surface whose term structure is smooth enough to trust, not just one whose individual slices look reasonable in isolation.

## Why that matters for the handoff

<p class="doc-section-lead">The eSSVI workflow addresses the time-direction problem explicitly instead of treating smoothing as a cosmetic post-process.</p>

- Calibrate an exact nodal eSSVI surface.
- Project those nodes into a smooth surface with explicit validation.
- Compare the nodal and smoothed surfaces instead of hiding the smoothing step.
- Hand the smoothed surface into `LocalVolSurface.from_implied(...)`.

## Seam and projection evidence

| Knot / metric | Repaired SVI seam jump | Smoothed eSSVI seam jump | Note |
| --- | --- | --- | --- |
| `T = 0.15` | `0.080696` | `0.000082` | largest early-maturity seam improvement |
| `T = 0.50` | `0.043331` | `0.000012` | smooth projection stays stable mid-curve |
| `T = 1.50` | `0.013674` | `0.000010` | improvement persists at longer maturities |
| Projection summary | `price_rmse = 0.02494` | `max_abs_price_error = 0.11453` | `projection_dupire_invalid_count = 0` |

<div class="doc-panel" markdown="1">
<p class="doc-panel__label">Main takeaway</p>
The result is not perfect nodal fidelity. The result is that seam stress drops by orders of magnitude while the projected surface remains valid for the Dupire handoff. That sets up the final proof page: <a href="localvol_pde_validation.md">Local-vol and PDE validation</a>.
</div>
