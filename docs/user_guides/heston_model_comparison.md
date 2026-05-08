# Heston model comparison

This page is the reviewer-facing entry point for the Capstone 3 model-choice
story. It compares a fitted Heston stochastic-volatility model against the
repo's eSSVI/local-vol workflow on a common vanilla target without claiming
that one model is universally superior.

<div class="cta-row cta-row--trio" markdown="1">
[Open Heston guide](heston.md){ .md-button .md-button--primary }
[Open Heston diagnostics](heston_diagnostics.md){ .md-button }
[Open comparison notebook](https://github.com/willemk-stack/option-pricing-library/blob/main/demos/13_heston_calibration_vs_localvol.ipynb){ .md-button }
</div>

## What this page proves

The point is not that Heston is automatically better. The point is that the
library can compare volatility models with explicit diagnostics: fit quality,
calibration stability, Monte Carlo validation, local-vol/PDE evidence, and
model-purpose tradeoffs.

Heston is not presented as automatically superior. The comparison asks what
each model is good for: Heston gives interpretable stochastic variance
dynamics; eSSVI/local vol gives flexible vanilla surface fit and direct
Dupire/PDE validation evidence.

## What is being compared

- Heston Fourier repricing after calibration.
- Heston Monte Carlo cross-checks where appropriate.
- eSSVI implied-surface repricing as the flexible vanilla-surface layer.
- Direct local-vol/PDE rows as a small Dupire/PDE validation audit.
- Train/held-out and ATM/wing error buckets where available in the
  comparison outputs.

The table-level details for this workflow live in the
[Heston versus local volatility note](../notes/heston_vs_local_vol.md).

## Evidence produced by the comparison report

| Evidence | What it shows | What it does not prove |
| --- | --- | --- |
| Quote-level Heston repricing errors | Whether the fitted Heston model stays close to the common quote target at the individual-row level | It does not prove parameter uniqueness or robustness outside the reviewed target |
| eSSVI repricing errors | How well the surface-first path fits the same vanilla target without imposing Heston dynamics | It does not prove dynamic realism or path-dependent suitability |
| Direct local-vol/PDE rows | Whether selected Dupire/PDE repricing rows remain aligned with the common target | It does not prove full-surface PDE accuracy across all grids or scenarios |
| ATM, downside, and upside buckets | Where model fit is concentrated or weak instead of hiding everything inside one average error | It does not prove behavior outside the sampled bucket definitions |
| Held-out comparison, when available | Whether the ranking changes once some rows are kept out of calibration | It does not prove broad out-of-sample generalization across dates or markets |
| Tradeoff summary | A reviewer-facing summary of fit, interpretability, diagnostics, and validation tradeoffs | It does not declare a universal model winner |

The report is bounded evidence on one shared vanilla target. It is meant to
show what the repository can measure and compare, not to make a universal claim
about stochastic-volatility superiority.

## Comparison report architecture

```text
Common quote target
      |
      +--> Heston calibration -> Fourier repricing -> fit diagnostics
      +--> Heston MC cross-check
      +--> eSSVI surface repricing
      +--> local-vol/PDE audit rows
```

## Why Heston is not just another surface fitter

Heston gives a stochastic variance story: mean reversion, long-run variance,
vol-of-vol, spot/variance correlation, and initial variance all have model
meaning. That matters when the review question includes dynamics, simulation,
or path-dependent interpretation rather than only a static vanilla fit.

## Why eSSVI/local vol still matters

The eSSVI/local-vol side is often the better vanilla-surface fit tool. It is
flexible, smooths the surface handoff, and supports direct Dupire/PDE
validation. It does not tell the same stochastic-variance story as Heston, and
that is exactly why the comparison is useful rather than redundant.

For the smoother-surface and PDE side of the proof path, use
[Local-vol and PDE validation](localvol_pde_validation.md).

## Calibration diagnostics

A fitted Heston parameter vector is not enough. Review multistart behavior,
bounds, residual structure, objective sensitivity, train/held-out errors, and
whether the fit is weakly identified. The broader API and workflow context is
in the [Heston guide](heston.md), while the notebook-facing review layer is in
[Heston diagnostics](heston_diagnostics.md).

## Monte Carlo validation

Monte Carlo results should be read with standard errors, confidence intervals,
path counts, time steps, and scheme labels. A tight confidence interval does
not remove discretization bias, and a matching point estimate does not make the
stochastic-volatility fit economically unique.

## When not to trust the result

- One optimizer start converged but multistart disagrees.
- Fitted parameters sit on or near bounds.
- IV residuals are structured by maturity or wing.
- Backend disagreement persists after robust quadrature reruns.
- Monte Carlo validation misses Fourier outside the expected error budget.
- Held-out errors are materially worse than fit errors.

## Minimal reviewer path

1. Read this page for the model-choice framing.
2. Open the [Heston guide](heston.md) for the namespaced pricing and
   calibration workflow.
3. Open [Heston diagnostics](heston_diagnostics.md) for report interpretation.
4. Review the [Heston versus local volatility note](../notes/heston_vs_local_vol.md)
   for the comparison outputs.
5. Run or inspect the
   [comparison notebook on `main`](https://github.com/willemk-stack/option-pricing-library/blob/main/demos/13_heston_calibration_vs_localvol.ipynb).

## What not to conclude

This comparison does not prove that either Heston or local volatility is
universally superior, and it is not a production-trading claim. It shows a
bounded model-comparison workflow inside this repository, with diagnostics that
make the tradeoffs visible enough for review on the configured comparison
target.