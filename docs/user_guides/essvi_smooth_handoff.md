---
hide:
  - navigation
  - toc
---

# eSSVI smooth handoff

The repaired surface is still a stack of slices. This step focuses on what changes once those nodes are projected into a smooth eSSVI surface, because Dupire depends on that change more than on the static repair alone.

<div class="cta-row cta-row--duo" markdown="1">
[Open the notebook](https://github.com/willemk-stack/option-pricing-library/blob/main/demos/07_essvi_smooth_surface_for_dupire.ipynb){ .md-button .md-button--primary }
[Next: local-vol and PDE validation](localvol_pde_validation.md){ .md-button }
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
<figcaption>Once the handoff is smoothed, the local-vol field becomes an object that can be inspected for shape and stability instead of treated as a hidden intermediate.</figcaption>
</figure>

<figure class="diagram" style="--diagram-max-width: 720px" markdown="1">
![Heatmap of differences between Gatheral and call-grid Dupire local-vol estimates](../assets/generated/dupire/gatheral_vs_dupire_diff_heatmap.light.png){ .diagram-img .diagram-light }
![Heatmap of differences between Gatheral and call-grid Dupire local-vol estimates](../assets/generated/dupire/gatheral_vs_dupire_diff_heatmap.dark.png){ .diagram-img .diagram-dark }
<figcaption>The difference view shows where the two extraction routes agree and where the handoff still deserves scrutiny.</figcaption>
</figure>

</div>

## What changes after smoothing

Slice-wise SVI is useful for static-surface repair, but it is not the cleanest object to hand to Dupire. The issue is time continuity: downstream local-vol extraction needs a surface whose term structure and `w_T` behavior are smooth enough to trust.

## Why that matters for the handoff

The eSSVI workflow addresses that directly:

- calibrate an exact nodal eSSVI surface
- project those nodes into a smooth surface with explicit validation
- compare the nodal and smoothed surfaces instead of hiding the smoothing step
- hand the smoothed surface into `LocalVolSurface.from_implied(...)`

## Seam and projection evidence

| Knot / metric | Repaired SVI seam jump | Smoothed eSSVI seam jump | Note |
| --- | --- | --- | --- |
| `T = 0.15` | `0.080696` | `0.000082` | largest early-maturity seam improvement |
| `T = 0.50` | `0.043331` | `0.000012` | smooth projection stays stable mid-curve |
| `T = 1.50` | `0.013674` | `0.000010` | improvement persists at longer maturities |
| Projection summary | `price_rmse = 0.02494` | `max_abs_price_error = 0.11453` | `projection_dupire_invalid_count = 0` |

The important result is not perfect nodal fidelity. It is that the projected surface materially reduces seam stress while keeping the final Dupire-invalid projection count at zero.
That smoother handoff sets up [Local-vol and PDE validation](localvol_pde_validation.md), where the next question is what the reduced seam stress buys in repricing accuracy, error structure, and convergence evidence.
