hide:
    - toc

# Decision guide

Use this page to route into the strongest engineering proof in the repo. The proof sequence now runs from surface repair and smooth local-vol handoff through PDE validation, then into Heston stochastic-volatility calibration and model-comparison diagnostics.

<div class="cta-row cta-row--trio" markdown="1">
[Start with surface repair](surface_workflow.md){ .md-button .md-button--primary }
[Review Heston model comparison](heston.md){ .md-button }
[Review local-vol and PDE validation](localvol_pde_validation.md){ .md-button }
</div>

## Choose by question

| If you want to show... | Open this page | Supporting notebook | Why it matters |
| --- | --- | --- | --- |
| The library can repair and diagnose a noisy quote surface | [Surface repair workflow](surface_workflow.md) | [06_surface_noarb_svi_repair.ipynb](https://github.com/willemk-stack/option-pricing-library/blob/main/demos/06_surface_noarb_svi_repair.ipynb) | This is the clearest static-surface engineering proof: quotes, diagnostics, SVI fit quality, and repair are all visible. |
| The local-vol handoff is smoother than a slice stack | [eSSVI smooth handoff](essvi_smooth_handoff.md) | [07_essvi_smooth_surface_for_dupire.ipynb](https://github.com/willemk-stack/option-pricing-library/blob/main/demos/07_essvi_smooth_surface_for_dupire.ipynb) | This is where the repo proves the time-derivative problem is understood and explicitly improved. |
| The numerics are validated rather than just implemented | [Local-vol and PDE validation](localvol_pde_validation.md) | [08_localvol_pde_repricing.ipynb](https://github.com/willemk-stack/option-pricing-library/blob/main/demos/08_localvol_pde_repricing.ipynb) | This page shows repricing accuracy, error structure, and convergence rather than a black-box PDE result. |
| The repo can compare volatility models rather than just fit one | [Heston](heston.md) | [13_heston_calibration_vs_localvol.ipynb](https://github.com/willemk-stack/option-pricing-library/blob/main/demos/13_heston_calibration_vs_localvol.ipynb) | Shows model-choice judgment: Heston interpretability versus eSSVI/local-vol vanilla-fit flexibility, with calibration diagnostics and validation evidence. |
| The benchmark story is real and reproducible | [Performance evidence](../performance.md) | Benchmark artifacts under `benchmarks/artifacts/` | This is where scaling, remedy tradeoffs, and end-to-end stage budgets are measured. |
| The system design is deliberate and typed | [Architecture](../architecture.md) | [09_surface_to_localvol_pde_integration.ipynb](https://github.com/willemk-stack/option-pricing-library/blob/main/demos/09_surface_to_localvol_pde_integration.ipynb) | The architecture page explains how the pricing, volatility, numerics, and validation layers fit together. |

## Proof sequence

1. Read [Surface repair workflow](surface_workflow.md).
2. Continue to [eSSVI smooth handoff](essvi_smooth_handoff.md).
3. Review [Local-vol and PDE validation](localvol_pde_validation.md).
4. Finish with [Heston](heston.md) for stochastic-volatility pricing, calibration diagnostics, Monte Carlo validation, and Heston-vs-local-vol/eSSVI comparison.

Use [Performance evidence](../performance.md) and [Architecture](../architecture.md) after that sequence when the follow-up questions turn to scaling, cost, or system design.
