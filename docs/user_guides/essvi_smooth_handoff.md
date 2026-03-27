# eSSVI smooth handoff

<div class="doc-intro" markdown="1">
<p class="doc-intro__kicker">Proof path step 2</p>
<p class="doc-intro__lead">Static repair is necessary, but it is not enough for Dupire. This page focuses on the transition from repaired slices to a smooth eSSVI surface, because the handoff succeeds or fails on time-direction behavior more than on slice polish alone.</p>
<p class="doc-intro__support">The question is not whether smoothing makes the surface look nicer. The question is whether the workflow reduces seam stress, preserves an admissible surface, and gives local-vol extraction a defensible time-continuous object before the PDE stage begins.</p>
<div class="doc-pill-row">
  <span class="doc-pill">`w_T` seam control</span>
  <span class="doc-pill">Explicit projection</span>
  <span class="doc-pill">Dupire-oriented surface</span>
</div>
</div>

<div class="cta-row cta-row--duo" markdown="1">
[Open the notebook](https://github.com/willemk-stack/option-pricing-library/blob/main/demos/07_essvi_smooth_surface_for_dupire.ipynb){ .md-button .md-button--primary }
[Next: local-vol and PDE validation](localvol_pde_validation.md){ .md-button }
</div>

## Problem

<p class="doc-section-lead">Slice-wise SVI repair can produce good-looking individual maturities while still leaving the time direction too rough for a trustworthy Dupire handoff.</p>

- The repaired surface is still a stack of slices, not automatically a smooth time-differentiable object.
- Dupire depends on how total variance moves across expiry, not just on whether each isolated smile is repaired cleanly.
- If the handoff stays implicit, the numerical problems only become visible later in the local-vol or PDE steps.

## Why Naive Approaches Fail

<p class="doc-section-lead">The generic slice-stack route is useful for demos and static inspection, but it is not the strongest Dupire-oriented path because its time derivative behavior is only piecewise smooth.</p>

<div class="doc-panel doc-panel--quiet" markdown="1">
<p class="doc-panel__label">Naive path</p>
For a generic expiry stack such as <code>VolSurface.from_svi(...)</code>, the local-vol bridge still relies on piecewise-linear interpolation in total variance across expiry. That keeps <code>w_T</code> only piecewise constant. The page therefore treats slice repair and smooth handoff as two separate engineering questions.
</div>

- Exact slice fit does not guarantee smooth maturity-to-maturity behavior.
- A local-vol workflow can look fine in one view while still carrying seam stress in the time direction.
- The handoff needs its own diagnostics instead of borrowing confidence from the repaired slices.

## Chosen Method

<p class="doc-section-lead">The workflow makes the transition explicit: keep the exact calibrated nodes visible, then decide separately whether a smooth continuous eSSVI surface is admissible for Dupire-oriented work.</p>

| Layer | Object or check | Design choice and reason |
| --- | --- | --- |
| Exact calibration output | `ESSVINodalSurface(fit.nodes)` | Preserve the calibrated nodes exactly so the fitted surface remains inspectable before any smoothing tradeoff is accepted |
| Smooth projection | `project_essvi_nodes(fit.nodes)` -> `ESSVISmoothedSurface` | Produce a continuous surface only through an explicit projection step, rather than hiding smoothing inside the local-vol code |
| Parametric continuity | projected `theta`, `psi`, and `eta` term structures | Use a continuous eSSVI parameter surface when analytic `w`, `w_y`, `w_yy`, and `w_T` matter for Dupire-oriented work |
| Validation layer | `evaluate_essvi_constraints(...)`, `validate_essvi_nodes(...)`, `validate_essvi_continuous(...)` | Separate nodal admissibility, continuous-surface constraints, and sampled static no-arbitrage checks instead of collapsing them into one pass/fail |
| Local-vol input | `LocalVolSurface.from_implied(...)` | Hand the next stage a surface whose time direction is an explicit design choice rather than an interpolation accident |

## Evidence

<p class="doc-section-lead">The evidence on this page should answer three questions quickly: did seam stress drop, did the projection stay admissible, and did the resulting handoff produce an inspectable local-vol object?</p>

<div class="doc-card-grid doc-card-grid--quiet" markdown="1">
<div class="doc-card doc-card--quiet" markdown="1">
<p class="doc-card__eyebrow">Worst seam jump</p>
<p class="doc-card__title">`0.080696 -> 0.000082`</p>
Largest early-maturity improvement at `T = 0.15`.
</div>
<div class="doc-card doc-card--quiet" markdown="1">
<p class="doc-card__eyebrow">Mid-curve seam jump</p>
<p class="doc-card__title">`0.043331 -> 0.000012`</p>
The improvement persists away from the shortest maturities.
</div>
<div class="doc-card doc-card--quiet" markdown="1">
<p class="doc-card__eyebrow">Projection admissibility</p>
<p class="doc-card__title">`projection_dupire_invalid_count = 0`</p>
The projected surface stays usable for the next Dupire-oriented step.
</div>
</div>

<figure markdown class="diagram diagram--hero">
  ![Heatmap of the smoothed eSSVI implied-vol surface across log-moneyness and maturity](../assets/generated/dupire/essvi_smoothed_surface_heatmap.light.png){ .diagram-img .diagram-light }
  ![Heatmap of the smoothed eSSVI implied-vol surface across log-moneyness and maturity](../assets/generated/dupire/essvi_smoothed_surface_heatmap.dark.png){ .diagram-img .diagram-dark }
  <figcaption>The smoothed eSSVI surface is the preferred Dupire handoff because it makes the time-direction choice explicit before local-vol extraction begins.</figcaption>
</figure>

<div class="snapshot-grid" markdown="1">

<figure class="diagram diagram--quiet" style="--diagram-max-width: 720px" markdown="1">
![Heatmap of Gatheral local volatility extracted from the smoothed eSSVI surface](../assets/generated/dupire/localvol_gatheral_heatmap.light.png){ .diagram-img .diagram-light }
![Heatmap of Gatheral local volatility extracted from the smoothed eSSVI surface](../assets/generated/dupire/localvol_gatheral_heatmap.dark.png){ .diagram-img .diagram-dark }
<figcaption>The local-vol field becomes an inspectable intermediate once the handoff surface is smooth enough to differentiate sensibly.</figcaption>
</figure>

<figure class="diagram diagram--quiet" style="--diagram-max-width: 720px" markdown="1">
![Heatmap of differences between Gatheral and call-grid Dupire local-vol estimates](../assets/generated/dupire/gatheral_vs_dupire_diff_heatmap.light.png){ .diagram-img .diagram-light }
![Heatmap of differences between Gatheral and call-grid Dupire local-vol estimates](../assets/generated/dupire/gatheral_vs_dupire_diff_heatmap.dark.png){ .diagram-img .diagram-dark }
<figcaption>The difference view shows where the two extraction routes agree and where the handoff still deserves scrutiny.</figcaption>
</figure>

</div>

### Seam and projection checks

| Knot / metric | Repaired SVI seam jump | Smoothed eSSVI seam jump | Why it matters |
| --- | --- | --- | --- |
| `T = 0.15` | `0.080696` | `0.000082` | largest early-maturity seam improvement |
| `T = 0.50` | `0.043331` | `0.000012` | smooth projection stays stable mid-curve |
| `T = 1.50` | `0.013674` | `0.000010` | improvement persists at longer maturities |
| Projection summary | `price_rmse = 0.02494` | `max_abs_price_error = 0.11453` | `projection_dupire_invalid_count = 0` |

## Tradeoffs

<p class="doc-section-lead">The smoothed handoff is a deliberate tradeoff, not an attempt to maximize every metric at once.</p>

<div class="doc-panel doc-panel--strong" markdown="1">
<p class="doc-panel__label">Main takeaway</p>
The payoff is a time-consistent Dupire-oriented surface and a zero-invalid projection summary. The cost is that the projection is not trying to preserve perfect nodal fidelity at any price; it accepts a controlled projection error so the next local-vol step receives a smoother and more defensible input surface. That tradeoff is preferable here because the next page is about repricing and convergence, not about exact node interpolation.
</div>

- Keep `ESSVINodalSurface` visible when the exact calibrated nodes are the point of the analysis.
- Prefer `ESSVISmoothedSurface` when the handoff into local vol depends on analytic `w_T` and continuous term-structure behavior.
- Use [Local-vol and PDE validation](localvol_pde_validation.md) as the final check on whether this tradeoff pays off numerically.
