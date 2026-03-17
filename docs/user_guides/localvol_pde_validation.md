---
hide:
  - navigation
  - toc
---

# Local-vol and PDE validation

This page is the numerics proof page: it shows a local-vol workflow that is validated with repricing error, error structure, and convergence evidence instead of a single headline price.

<div class="cta-row cta-row--trio">
  <a class="md-button md-button--primary" href="https://github.com/willemk-stack/option-pricing-library/blob/main/demos/08_localvol_pde_repricing.ipynb">Open the notebook</a>
  <a class="md-button" href="../../performance/">Review performance evidence</a>
  <a class="md-button" href="../../architecture/">See architecture</a>
</div>

<figure markdown class="diagram">
  ![Scatter plot comparing local-vol PDE repriced values with target Black-76 prices across the repricing grid](../assets/generated/numerics/pde_roundtrip_scatter.light.png){ .diagram-img .diagram-light }
  ![Scatter plot comparing local-vol PDE repriced values with target Black-76 prices across the repricing grid](../assets/generated/numerics/pde_roundtrip_scatter.dark.png){ .diagram-img .diagram-dark }
  <figcaption>The repricing cloud stays close to the identity line, so accuracy can be judged from visible structure rather than a single summary claim.</figcaption>
</figure>

<div class="snapshot-grid">
  <figure class="diagram" style="--diagram-max-width: 720px">
    <img alt="Heatmap of local-vol PDE pricing error across strike and maturity on the repricing grid" class="diagram-img diagram-light" src="../../assets/generated/numerics/pde_price_error_heatmap.light.png" />
    <img alt="Heatmap of local-vol PDE pricing error across strike and maturity on the repricing grid" class="diagram-img diagram-dark" src="../../assets/generated/numerics/pde_price_error_heatmap.dark.png" />
    <figcaption>The price-error heatmap shows where the workflow is most stressed across strike and maturity.</figcaption>
  </figure>
  <figure class="diagram" style="--diagram-max-width: 720px">
    <img alt="Convergence plot for a representative local-vol PDE solve as the numerical grid is refined" class="diagram-img diagram-light" src="../../assets/generated/numerics/pde_convergence.light.png" />
    <img alt="Convergence plot for a representative local-vol PDE solve as the numerical grid is refined" class="diagram-img diagram-dark" src="../../assets/generated/numerics/pde_convergence.dark.png" />
    <figcaption>The convergence sweep makes the mesh tradeoff explicit instead of asking the reader to trust a single chosen grid.</figcaption>
  </figure>
</div>

## Hard problem

Local-vol and PDE workflows are easy to overstate. A plausible repricing answer can still hide unstable local-vol extraction, poor mesh choices, or unexplained error concentration.

## Method

The library treats this as a diagnostics-first numerical workflow:

- start from the smoothed eSSVI handoff rather than a rough slice stack
- expose invalid masks, denominator diagnostics, and worst-point behavior
- reprice a representative option grid against the originating implied surface
- run at least one convergence sweep so the PDE mesh choice is inspectable

## Evidence

| Metric | Published bundle value |
| --- | --- |
| Repriced options | `154` |
| Mean abs price error | `0.0008067` |
| Max abs price error | `0.0044506` |
| Mean abs IV error | `1.0171 bp` |
| Max abs IV error | `18.7059 bp` |

This is the page where the repo proves the workflow is validated rather than merely implemented. The figures show where error lives, and the table gives the aggregate repricing result.

## Best next click

- Open [Performance evidence](../performance.md) for the measured scaling, digital-remedy tradeoffs, and end-to-end stage-budget benchmarks.
- Open [Architecture](../architecture.md) if the next question is how the surface, local-vol, PDE, and diagnostics layers fit together.
