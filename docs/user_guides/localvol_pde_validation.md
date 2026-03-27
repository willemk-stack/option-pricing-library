# Local-vol and PDE validation

<div class="doc-intro" markdown="1">
<p class="doc-intro__kicker">Proof path step 3</p>
<p class="doc-intro__lead">With the handoff smoothed, the remaining question is numerical: does the local-vol and PDE workflow reprice cleanly, show where error lives, and behave sensibly as the grid is refined?</p>
<p class="doc-intro__support">This page is where the repo proves the workflow is validated rather than merely implemented. The figures show structure, the aggregate metrics stay honest about scale, and the convergence view makes the mesh tradeoff explicit instead of hidden in one hard-coded grid.</p>
<div class="doc-pill-row">
  <span class="doc-pill">Repricing scatter</span>
  <span class="doc-pill">Error localization</span>
  <span class="doc-pill">Convergence evidence</span>
</div>
</div>

<div class="cta-row cta-row--duo" markdown="1">
[Open the notebook](https://github.com/willemk-stack/option-pricing-library/blob/main/demos/08_localvol_pde_repricing.ipynb){ .md-button .md-button--primary }
[Review performance evidence](../performance.md){ .md-button }
</div>

## Problem

<p class="doc-section-lead">A smooth handoff is still not enough. The last question is whether the numerical workflow is accurate, inspectable, and honest about its mesh tradeoffs.</p>

- Local-vol extraction and PDE repricing need to show more than successful execution.
- Aggregate error alone is not enough if the worst stress is localized in one region of the grid.
- The chosen mesh should be justified by visible convergence behavior rather than implied as a magic default.

## Why Naive Validation Fails

<p class="doc-section-lead">A small average error or one clean scatter plot can still hide where the workflow is strained. Validation has to show structure as well as summary.</p>

<div class="doc-panel doc-panel--quiet" markdown="1">
<p class="doc-panel__label">Naive validation</p>
One good-looking scatter plot or one mean error number can make the workflow seem finished too early. That hides where the worst error lives, whether invalid regions are controlled, and what extra grid density actually buys.
</div>

- Aggregate means hide localization.
- Repricing without a convergence view does not justify the mesh choice.
- Local-vol validation needs a representative numerical reference when no closed-form target exists for the full workflow.

## Chosen Validation Method

<p class="doc-section-lead">The page validates the workflow in layers: start from the smooth eSSVI handoff, expose extraction diagnostics, reprice a representative grid, and inspect convergence separately.</p>

| Layer | Object or check | Design choice and reason |
| --- | --- | --- |
| Input surface | `ESSVISmoothedSurface` from the previous step | Start from an explicit time-continuous handoff rather than a rough slice stack |
| Local-vol diagnostics | invalid masks, denominator checks, worst-point inspection | Keep extraction failure modes visible before making pricing claims |
| Repricing bundle | representative `154`-option grid against the originating implied surface | Show the numerical workflow where it is actually used |
| Error localization | repricing scatter plus strike/maturity heatmap | Let reviewers see structure instead of trusting one mean |
| Mesh check | convergence sweep against a finer reference solve | Make the PDE grid a defended tradeoff rather than a hard-coded habit |

## Evidence

<p class="doc-section-lead">The evidence should answer three questions quickly: does the repricing cloud stay controlled, where does the error live, and does the convergence view support the chosen grid?</p>

<div class="doc-card-grid doc-card-grid--quiet" markdown="1">
<div class="doc-card doc-card--quiet" markdown="1">
<p class="doc-card__eyebrow">Repriced options</p>
<p class="doc-card__title">`154`</p>
Representative local-vol/PDE grid used for the published check.
</div>
<div class="doc-card doc-card--quiet" markdown="1">
<p class="doc-card__eyebrow">Mean abs price error</p>
<p class="doc-card__title">`8.07e-4`</p>
Aggregate price error stays small while the figures keep the structure visible.
</div>
<div class="doc-card doc-card--quiet" markdown="1">
<p class="doc-card__eyebrow">Max abs IV error</p>
<p class="doc-card__title">`18.7 bp`</p>
Worst-point error remains inspectable instead of being buried inside one average.
</div>
</div>

<figure markdown class="diagram diagram--hero">
  ![Scatter plot comparing local-vol PDE repriced values with target Black-76 prices across the repricing grid](../assets/generated/numerics/pde_roundtrip_scatter.light.png){ .diagram-img .diagram-light }
  ![Scatter plot comparing local-vol PDE repriced values with target Black-76 prices across the repricing grid](../assets/generated/numerics/pde_roundtrip_scatter.dark.png){ .diagram-img .diagram-dark }
  <figcaption>The repricing cloud stays close to the identity line, so accuracy can be judged from visible structure rather than from one summary claim.</figcaption>
</figure>

<div class="snapshot-grid" markdown="1">

<figure class="diagram diagram--quiet" style="--diagram-max-width: 720px" markdown="1">
![Heatmap of local-vol PDE pricing error across strike and maturity on the repricing grid](../assets/generated/numerics/pde_price_error_heatmap.light.png){ .diagram-img .diagram-light }
![Heatmap of local-vol PDE pricing error across strike and maturity on the repricing grid](../assets/generated/numerics/pde_price_error_heatmap.dark.png){ .diagram-img .diagram-dark }
<figcaption>The price-error heatmap shows where the workflow is most stressed across strike and maturity.</figcaption>
</figure>

<figure class="diagram diagram--quiet" style="--diagram-max-width: 720px" markdown="1">
![Convergence plot for a representative local-vol PDE solve as the numerical grid is refined](../assets/generated/numerics/pde_convergence.light.png){ .diagram-img .diagram-light }
![Convergence plot for a representative local-vol PDE solve as the numerical grid is refined](../assets/generated/numerics/pde_convergence.dark.png){ .diagram-img .diagram-dark }
<figcaption>The convergence sweep makes the mesh tradeoff explicit instead of asking the reader to trust one chosen grid.</figcaption>
</figure>

</div>

### Aggregate repricing result

| Metric | Published bundle value |
| --- | --- |
| Repriced options | `154` |
| Mean abs price error | `0.0008067` |
| Max abs price error | `0.0044506` |
| Mean abs IV error | `1.0171 bp` |
| Max abs IV error | `18.7059 bp` |

## Tradeoffs

<p class="doc-section-lead">The page is validating representative numerical behavior, not claiming universal perfection. The practical question is how much runtime to spend for how much additional error reduction.</p>

<div class="doc-panel doc-panel--strong" markdown="1">
<p class="doc-panel__label">Main takeaway</p>
The evidence here is deliberately two-layered: the scatter and heatmap show where the workflow is stressed, while the table and convergence view show that the aggregate repricing error stays controlled on a defensible grid. The remaining judgment is practical rather than theatrical: when extra grid density is worth the runtime cost, and when the published default is already enough.
</div>

- Treat aggregate error as necessary but not sufficient; the figures are what keep worst-point behavior visible.
- Use the convergence plot to justify the mesh, and [Performance evidence](../performance.md) to decide whether the runtime cost of finer grids is worthwhile.
- Return to [Architecture](../architecture.md) if you want the system-level safeguard story behind the workflow.
