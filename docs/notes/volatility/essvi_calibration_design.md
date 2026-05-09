# eSSVI calibration design

!!! note "Status: implementation policy"
    This note explains the repository's current eSSVI calibration and
    smooth-handoff policy. It distinguishes mathematical constraints,
    literature-backed conventions, repository defaults, diagnostics, and known
    limitations.

## Role in the library

eSSVI is the library's cross-maturity surface layer between repaired SVI slices
and Dupire/local-vol workflows. The slice-level SVI path remains the first
repairable object to inspect, while eSSVI provides a nodal and then smoothed
surface with explicit total-variance derivatives.

The public path is:

1. repair or fit inspectable per-expiry SVI slices;
2. calibrate eSSVI nodes from vanilla prices;
3. preserve `ESSVINodalSurface` as the exact calibrated-node object;
4. project those nodes into `ESSVISmoothedSurface` only when a
   time-differentiable Dupire input is needed.

The user-facing entrypoints are documented in the
[eSSVI guide](../../user_guides/essvi.md), the
[eSSVI smooth handoff page](../../user_guides/essvi_smooth_handoff.md), and
the [volatility API overview](../../api/vol.md).

## Relationship to raw SVI and repaired slices

Raw SVI and eSSVI serve different review roles. The
[SVI calibration design note](svi_calibration_design.md) explains how a single
expiry is fit, regularized, and repair-checked. eSSVI does not replace that
slice evidence. It organizes a cross-maturity handoff after the slice evidence
has been inspected.

The repository policy is to keep the raw or repaired slice stack visible for
review, then treat eSSVI as a separate modeling and numerical decision. This is
important because a stack of acceptable slices can still have maturity-direction
kinks that are hostile to Dupire differentiation.

## Mathematical setup and parameterization

For log-moneyness \(y = \log(K/F(T))\) and maturity \(T\), the implemented
continuous eSSVI representation is parameterized by three maturity term
structures:

- \(\theta(T)\), the at-the-money total variance level;
- \(\psi(T)\), the scaled smile-slope magnitude;
- \(\eta(T)\), the signed skew term.

The classic SSVI quantities are recovered as

\[
\phi(T) = \frac{\psi(T)}{\theta(T)},
\qquad
\rho(T) = \frac{\eta(T)}{\psi(T)}.
\]

The total variance is evaluated as

\[
w(y,T)
=
\frac{1}{2}
\left(
\theta(T) + \eta(T)y
+ \sqrt{\theta(T)^2 + 2\theta(T)\eta(T)y + \psi(T)^2 y^2}
\right).
\]

Mathematical admissibility is enforced through positivity, correlation, Lee
slope, and Gatheral-Jacquier butterfly constraints. In code these are exposed
by `ESSVITermStructures.validate(...)`,
`evaluate_essvi_constraints(...)`, `validate_essvi_nodes(...)`, and
`validate_essvi_continuous(...)`.

## Calibration objective and constraints

`calibrate_essvi(...)` is the primary calibration entrypoint. It calibrates
global eSSVI nodes from arrays of log-moneyness, maturity, market option price,
and optional option side. The objective is price-space least squares:

- input prices are converted to Black implied vols where needed to extract
  total-variance diagnostics;
- the calibrated object is an `ESSVINodeSet`;
- residuals are weighted by `sqrt_weights` when provided;
- invalid candidates receive a large finite residual rather than being silently
  accepted;
- node validation is recorded in `ESSVIFitDiagnostics.node_validation`.

The mathematical requirement is that the fitted nodes satisfy the eSSVI
no-arbitrage constraints. The repository policy is narrower: record the
calibration diagnostics, keep the exact nodal surface inspectable, and require
a separate projection diagnostic before using the result as a smooth Dupire
input.

## Smooth cross-maturity handoff

`project_essvi_nodes(...)` is the explicit handoff step. It starts from an
`ESSVINodeSet`, checks monotonicity of \(\theta\), \(g_+ = \psi(1+\rho)\), and
\(g_- = \psi(1-\rho)\), then builds maturity term structures with origin
anchoring and shape-preserving interpolation. The result carries both:

- `fallback_surface`, an `ESSVINodalSurface` that preserves exact calibrated
  nodes; and
- `surface`, an `ESSVISmoothedSurface` when projection and validation succeed.

This projection is repository policy for Dupire-oriented workflows because the
local-vol formulas consume \(w\), \(w_y\), \(w_{yy}\), and especially \(w_T\).
Exact nodal agreement is useful evidence, but it is not the same thing as a
stable maturity derivative.

## Numerical and modeling risks

The main risks are deliberately exposed rather than hidden inside one pass/fail
label:

- **Calendar and slice arbitrage**: node and continuous-surface validation can
  fail separately.
- **Seam stress**: exact per-expiry fits can leave maturity derivative jumps
  that affect local-vol extraction.
- **Unstable derivatives**: \(w_T\), \(w_y\), and \(w_{yy}\) are more sensitive
  than plotted implied vol.
- **Invalid Dupire denominator regions**: the projection diagnostics count
  invalid Gatheral local-variance points.
- **Extrapolation risk**: short-end anchoring and long-end continuation are
  modeling choices, not market data.
- **Over-smoothing**: a smooth projected surface can reduce visible local-vol
  stress while moving away from exact calibrated prices.

The current implementation therefore treats smooth projection as an accepted
tradeoff only after diagnostics are recorded. It is not a universal statement
that the smooth surface is always more correct than the nodal surface.

## Diagnostics and validation evidence

The public evidence is intentionally split by workflow stage:

- [Surface repair workflow](../../user_guides/surface_workflow.md) shows the
  repaired SVI slice evidence that precedes the handoff.
- [eSSVI guide](../../user_guides/essvi.md) documents the calibration,
  validation, nodal-surface, and projection APIs.
- [eSSVI smooth handoff](../../user_guides/essvi_smooth_handoff.md) is the
  public proof page for the smooth projection decision and its generated
  diagnostics.
- [Dupire local vol](../local-vol-pde/dupire_local_vol.md) explains why
  maturity and strike derivatives are fragile.
- [Local-vol and PDE validation](../../user_guides/localvol_pde_validation.md)
  tests whether the smooth handoff remains numerically useful downstream.
- [Validation matrix](../../validation_matrix.md) maps the eSSVI handoff claim
  to its bounded evidence.
- [test_essvi_surface.py](https://github.com/willemk-stack/option-pricing-library/blob/main/tests/vol/essvi/test_essvi_surface.py)
  covers the total-variance formula, derived ratios, analytic derivatives,
  price delegation, guards, and local-vol integration.
- [test_essvi_calibrate.py](https://github.com/willemk-stack/option-pricing-library/blob/main/tests/vol/essvi/test_essvi_calibrate.py)
  covers theta extraction, constraint reports, synthetic calibration recovery,
  node interpolation, and projection success.
- [test_essvi_mingone_projection.py](https://github.com/willemk-stack/option-pricing-library/blob/main/tests/vol/essvi/test_essvi_mingone_projection.py)
  covers the projected surface as a Dupire local-vol input and the nodal
  fallback surface.
- [test_essvi_objective.py](https://github.com/willemk-stack/option-pricing-library/blob/main/tests/vol/essvi/test_essvi_objective.py)
  covers option-side handling, weighted price residuals, and input validation.

## References

The note relies on the local `Finance-books` source library:

- *GJ - No-Arbitrage SVI (SSVI).pdf*
    in `03_Volatility_Surface/02_SVI_SSVI_eSSVI`.
- *Hendriks - Extended SSVI.pdf*
    in `03_Volatility_Surface/02_SVI_SSVI_eSSVI`.
- *Mingone - No-Arbitrage Global eSSVI.pdf*
    in `03_Volatility_Surface/02_SVI_SSVI_eSSVI`.
- *Pasquazzi - eSSVI Surface Calibration.pdf*
    in `03_Volatility_Surface/02_SVI_SSVI_eSSVI`.

## Known limitations

This note describes the current repository policy; it does not claim that the
calibration objective is globally optimal, that the projection is the only
reasonable smoother, or that synthetic validation is live-market evidence.

The smooth handoff is designed for Dupire/local-vol diagnostics. It should not
be read as a guarantee for exotic pricing, hedging performance, or all
extrapolated maturity/wing regimes.
