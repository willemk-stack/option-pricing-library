# SVI calibration design

!!! note "Status"
  This note now reflects the current raw-SVI slice calibration path in the
  repository. It stays scoped to the slice-level fitter, its repair hooks,
  and the workflow helpers that assemble surfaces one expiry at a time.

## Role in the library

This note explains why the repository does not treat raw SVI calibration as a
single unconstrained least-squares fit and call the job finished. The goal is a
reviewable vanilla-surface workflow that can feed repair, smoothing, and
Dupire-oriented downstream steps without hiding shape risk.

See also the [SVI guide](../user_guides/svi.md), the
[eSSVI guide](../user_guides/essvi.md), and the
[volatility API overview](../api/vol.md).

## Why raw SVI is not calibrated directly

Raw SVI parameters are awkward optimizer coordinates. They are differently
scaled, can wander into poorly conditioned regions, and can produce a low local
residual without producing a surface that is stable enough for downstream use.

The design target is therefore not just a small market-fit residual. The design
target is a fit that is inspectable, regularized, and either already
arbitrage-aware or repairable in a documented second step.

## Constrained parameter transforms

The optimizer-facing parameter space should encode admissibility and practical
search ranges rather than leaving every bound and sign rule to ad hoc clipping.
Transforms also make multistart calibration easier to reason about because the
seed grid lives in a domain with explicit interpretation.

On this branch the core fitter uses `SVITransformLeeCap`, with optimizer
coordinates `u = [u_a, u_R, u_L, u_m, u_sigma]`. `u_a` and `u_sigma` go through
`softplus` to produce strictly positive `alpha` and `sigma`; `u_R` and `u_L`
go through a sigmoid into Lee-capped right and left slope magnitudes in
`(1e-4, slope_cap)`, with the default `slope_cap = 1.999`; and `u_m` maps
directly to `m`.

The raw-SVI parameters are then reconstructed as

$$
b = \tfrac12 (s_R + s_L^{mag}) + \varepsilon,
\qquad
\rho = \frac{s_R - s_L^{mag}}{s_R + s_L^{mag} + \varepsilon},
\qquad
a = \alpha - b\sigma\sqrt{1 - \rho^2}.
$$

`calibrate_svi(...)` uses that transform throughout the slice fit. The repair
entrypoints operate on raw/JW parameter objects after calibration rather than
on a second optimizer transform, and the eSSVI workflow uses a separate
parameterization documented elsewhere.

## Residual vector: market fit plus regularization

The residual vector should combine market-fit error with explicit penalty terms
for surface quality. That keeps the optimization contract honest: a fit that is
numerically cheap but structurally fragile should not look identical to a fit
that is slightly worse on raw residuals but materially better for downstream
surface use.

The primary calibration path here is total variance, not price or implied
volatility. `VolSurface.from_svi(...)` first converts each quote bucket to
`y = log(K / F(T))` and `w = T * iv^2`, then `SVIObjective.residual(...)`
builds the data block as `sqrt_w * (w_model - w_obs)` before appending the
regularization residuals. `robust_data_only=True` changes only the data weights;
the regularization terms remain explicit residual components rather than being
recast into prices or Black implied vols.

## Data-scaled regularization

Regularization should scale with the data being fit. If penalty strength is not
normalized against quote count and residual magnitude, the same nominal weight
can behave very differently across sparse and dense slices.

The practical intent is to keep regularization visible but not dominant: strong
enough to discourage pathological fits, weak enough that market mismatch is not
silently relabeled as a penalty choice.

This branch only partially data-scales regularization. `default_reg_from_data`
sets `m_prior` from the minimum-variance location, `m_scale` from the
interquantile span of `y`, `sigma_floor` from the median spacing of the sorted
quote grid, and rescales `lambda_m`, `lambda_inv_sigma`, and
`lambda_slope_L/R` using quote-count factors `clip(10 / n, 0.25, 4.0)` and
`clip(30 / n, 1.0, 10.0)`. If a smile is both very dense and clearly winged,
those rail penalties are turned off.

By contrast, `lambda_g`, `g_scale`, `g_floor`, and `g_n_grid` stay at
configuration defaults unless the caller overrides them. There is also no
separate surface-level normalization in the core fitter on this branch:
`VolSurface.from_svi(...)` calibrates each expiry independently and surface
fallback lives in workflow helpers rather than in a second SVI objective.

## Analytical Jacobian blocks

Calibration code benefits when derivative information is explicit about which
parts are analytic and which remain numerical. That matters for speed,
conditioning review, and regression diagnostics.

In the core slice fitter, SciPy is not asked to finite-difference the
objective. Every `least_squares(...)` call passes `jac=obj.jac`, and
`SVIObjective.jac(...)` assembles analytic blocks for the data residual
(`svi_jac_wrt_params(...)` chained through `SVITransformLeeCap.dp_du(...)`),
the `m` prior, the inverse-sigma hinge, the wing-slope target penalties, and
the `g`-penalty via `gatheral_g_jac_params(...)`.

That statement is intentionally scoped to the core fitter. The transform
Jacobian is regression-tested against finite differences, but this branch does
not currently ship a separate full-objective Jacobian-versus-finite-difference
test for the entire `SVIObjective` stack.

## Robust loss and IRLS behavior

Outlier resistance should be described as part of the optimization contract, not
as a hidden implementation detail. If the calibration path uses a robust loss,
iteratively reweighted least squares, or an equivalent reweighting loop, the
final weights and any convergence caveats should be surfaced in diagnostics.

The default loss is `soft_l1` with `f_scale = 1.0`. If
`robust_data_only=False`, the fitter hands the full residual vector to SciPy's
robust-loss interface directly. If `robust_data_only=True` and
`loss != "linear"`, the code runs an explicit IRLS outer loop: it builds data
weights from `_robust_rhoprime(...)`, floors them at `irls_w_floor` (default
`1e-4`), optionally damps them, and resolves a linear-loss least-squares
subproblem up to `irls_max_outer = 8` times.

The resulting diagnostics expose both the solver-side and weighting-side state:
`solver.irls_outer_iters`, `robust_weights_min`, `robust_weights_median`,
`robust_weights_max`, `robust_weights_frac_floored`, and
`robust_weights_entropy`.

## Post-fit butterfly-arbitrage repair

The repo's broader surface workflow already distinguishes between fitting and
repair. This note keeps the same distinction: a raw best-fit SVI parameter set
can be useful evidence, but the downstream surface contract may still require a
repair step before the result is treated as an acceptable vanilla input.

That separation matters because a diagnostic should be able to say both of the
following at once:

- the optimizer found the smallest residual in one region;
- the arbitrage-feasible surface used downstream was produced after a repair or
  feasibility projection step.

## Diagnostics exposing best fit vs arbitrage-feasible fit

The review surface should keep the best-fit and arbitrage-feasible outputs
separate whenever they differ materially. Otherwise the docs risk hiding the
core engineering tradeoff: fit quality versus surface usability.

Useful diagnostics include residual summaries, no-arbitrage checks, penalty
contributions, fit-versus-repair diffs, and downstream smoothness signals for
the Dupire-oriented handoff.

Current branch limitation: `SVIFitResult` stores only the final chosen
`params` plus one `SVIFitDiagnostics` bundle. If repair changes the slice,
diagnostics are rebuilt around the repaired parameters, so callers that need a
side-by-side pre-repair versus post-repair record must keep that comparison
explicitly in notebook or workflow code.

## Validation and diagnostics

- [Surface repair workflow](../user_guides/surface_workflow.md) is the primary
  public proof page for quote repair, SVI fit quality, and no-arbitrage review.
- [eSSVI smooth handoff](../user_guides/essvi_smooth_handoff.md) shows the next
  stage where surface smoothness matters for Dupire-oriented use.
- [Local-vol and PDE validation](../user_guides/localvol_pde_validation.md)
  shows why a surface design note cannot stop at calibration residuals.
- [SVI repair guide](../user_guides/svi_repair.md) documents the public repair
  entrypoints and the calibration-time repair hook.
- [test_svi.py](https://github.com/willemk-stack/option-pricing-library/blob/main/tests/vol/svi/test_svi.py)
  covers transform round-trips, the transform Jacobian against finite
  differences, and a low-noise synthetic fit.
- [test_svi_calibrate_smoke_paths.py](https://github.com/willemk-stack/option-pricing-library/blob/main/tests/vol/svi/test_svi_calibrate_smoke_paths.py)
  and
  [test_svi_calibrate_additional_paths.py](https://github.com/willemk-stack/option-pricing-library/blob/main/tests/vol/svi/test_svi_calibrate_additional_paths.py)
  cover linear, robust-all, and repair-mode calibration branches.
- [test_svi_repair.py](https://github.com/willemk-stack/option-pricing-library/blob/main/tests/vol/svi/test_svi_repair.py)
  and
  [test_svi_repair_robustness.py](https://github.com/willemk-stack/option-pricing-library/blob/main/tests/vol/svi/test_svi_repair_robustness.py)
  cover repair feasibility, repair-trigger behavior, and multi-seed stress
  workflows.

## References

- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*.
  Wiley.
- Gatheral, J., & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
  *Quantitative Finance*, 14(1), 59-71.

## Known limitations

This note does not claim production readiness, and it does not prove that the
current calibration architecture is globally optimal. It is a reviewer-facing
map of why the repo treats SVI calibration as a constrained and diagnosable
workflow rather than a raw parameter fit.

The default regularization schedule is also implementation-defined. It is
designed to be inspectable and overrideable, but it is not presented here as a
faithful reproduction of a single published penalty recipe.