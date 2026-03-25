# Local-vol and PDE validation

<div class="doc-intro" markdown="1">
<p class="doc-intro__kicker">Proof path step 3</p>
<p class="doc-intro__lead">With the handoff smoothed, the remaining question is numerical: does the local-vol and PDE workflow reprice cleanly, show where error lives, and behave sensibly as the grid is refined?</p>
<p class="doc-intro__support">This page is where the repo proves the workflow is validated rather than merely implemented. The figures show structure, the table shows aggregate error, and the convergence view makes the mesh tradeoff explicit.</p>
<div class="doc-pill-row">
  <span class="doc-pill">Repricing scatter</span>
  <span class="doc-pill">Error localization</span>
  <span class="doc-pill">Convergence evidence</span>
</div>
</div>

<div class="cta-row cta-row--trio" markdown="1">
[Open the notebook](https://github.com/willemk-stack/option-pricing-library/blob/main/demos/08_localvol_pde_repricing.ipynb){ .md-button .md-button--primary }
[Review performance evidence](../performance.md){ .md-button }
[See architecture](../architecture.md){ .md-button }
</div>

<div class="doc-card-grid" markdown="1">
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Reviewer question</p>
<p class="doc-card__title">Does it reprice cleanly?</p>
- The repricing cloud should stay close to the identity line across the grid, not just on average.
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Reviewer question</p>
<p class="doc-card__title">Where does error live?</p>
- Price and implied-vol error should be localized by strike and maturity rather than hidden inside one aggregate number.
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Reviewer question</p>
<p class="doc-card__title">Is the mesh defensible?</p>
- The chosen PDE grid should look like a visible tradeoff, not an arbitrary hard-coded default.
</div>
</div>

<figure markdown class="diagram">
  ![Scatter plot comparing local-vol PDE repriced values with target Black-76 prices across the repricing grid](../assets/generated/numerics/pde_roundtrip_scatter.light.png){ .diagram-img .diagram-light }
  ![Scatter plot comparing local-vol PDE repriced values with target Black-76 prices across the repricing grid](../assets/generated/numerics/pde_roundtrip_scatter.dark.png){ .diagram-img .diagram-dark }
  <figcaption>The repricing cloud stays close to the identity line, so accuracy can be judged from visible structure rather than from one summary claim.</figcaption>
</figure>

<div class="snapshot-grid" markdown="1">

<figure class="diagram" style="--diagram-max-width: 720px" markdown="1">
![Heatmap of local-vol PDE pricing error across strike and maturity on the repricing grid](../assets/generated/numerics/pde_price_error_heatmap.light.png){ .diagram-img .diagram-light }
![Heatmap of local-vol PDE pricing error across strike and maturity on the repricing grid](../assets/generated/numerics/pde_price_error_heatmap.dark.png){ .diagram-img .diagram-dark }
<figcaption>The price-error heatmap shows where the workflow is most stressed across strike and maturity.</figcaption>
</figure>

<figure class="diagram" style="--diagram-max-width: 720px" markdown="1">
![Convergence plot for a representative local-vol PDE solve as the numerical grid is refined](../assets/generated/numerics/pde_convergence.light.png){ .diagram-img .diagram-light }
![Convergence plot for a representative local-vol PDE solve as the numerical grid is refined](../assets/generated/numerics/pde_convergence.dark.png){ .diagram-img .diagram-dark }
<figcaption>The convergence sweep makes the mesh tradeoff explicit instead of asking the reader to trust one chosen grid.</figcaption>
</figure>

</div>

## What the figures answer

<p class="doc-section-lead">The local-vol/PDE page should answer three questions quickly: repricing quality, error structure, and mesh behavior.</p>

- Start from the smoothed eSSVI handoff rather than a rough slice stack.
- Expose invalid masks, denominator diagnostics, and worst-point behavior.
- Reprice a representative option grid against the originating implied surface.
- Run at least one convergence sweep so the PDE mesh choice is inspectable.

## Aggregate repricing result

| Metric | Published bundle value |
| --- | --- |
| Repriced options | `154` |
| Mean abs price error | `0.0008067` |
| Max abs price error | `0.0044506` |
| Mean abs IV error | `1.0171 bp` |
| Max abs IV error | `18.7059 bp` |

<div class="doc-panel" markdown="1">
<p class="doc-panel__label">Main takeaway</p>
The evidence here is deliberately two-layered: the figures show where the workflow is stressed, and the table shows that the aggregate repricing error stays small. Continue to <a href="../performance.md">Performance evidence</a> for the runtime story, or return to <a href="../architecture.md">Architecture</a> for the system-level dependency map behind the workflow.
</div>
