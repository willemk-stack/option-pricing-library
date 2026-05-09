# Heston Fourier diagnostics

!!! note "Status"
    This note documents the repository's Fourier-diagnostics contract and review policy.
    Warning flags, severity language, and tuning guidance should be read as implementation
    diagnostics unless explicitly cited or benchmark-backed.

## Why this exists

The Heston Fourier pricer can produce a finite number that is still numerically
fragile. This note focuses on the practical stability issues behind that
behavior: branch handling, complex-log stability, oscillatory integrands,
exposed integration settings, and diagnostics for truncation, convergence,
discontinuities, and non-finite values.

This note documents the diagnostics contract added to
`src/option_pricing/models/heston/fourier.py`:

- panel-local failure and warning bitmaps
- whole-integral warning bitmaps
- summary metrics that explain why a warning fired
- the main tuning knobs behind those warnings

The current Heston Fourier stack separates:

- the characteristic-function implementation in `charfunc.py`
- the probability integrals and diagnostics entry points in `fourier.py`
- the reusable composite quadrature machinery in `numerics/quadrature.py`

## Characteristic-function stability

Literature-backed convention plus repository policy: the implementation follows
a Gatheral-style affine form in `charfunc.py`, with branch-cut concerns framed
by the Heston trap discussion in the source library. The discriminant handling
in `_stable_discriminant(...)` flips the sign of the square-root branch whenever
the implied `g` ratio would be non-finite or unstable. The branch-stability
tests cover this repository implementation choice; they do not prove all
algebraically equivalent forms are equally safe.

That branch choice does not eliminate every numerical problem in the Fourier
inversion. It reduces one major source of instability, but the inversion still
needs diagnostics around:

- truncation
- local resolution near the origin
- oscillation
- cancellation
- non-finite panel values

## Diagnostic layers

There are two layers of diagnostics.

### 1. Panel-local diagnostics

These exist only for the fixed-rule `gauss_legendre` backend, because only that
backend has an explicit panel structure.

Each panel gets a bitmap in `panel_reason`, with convenience mask
`panel_invalid = panel_reason != 0`.

The current flags are:

- `NONFINITE_PANEL_CONTRIB`
- `NONFINITE_INTEGRAND`
- `UNDERRESOLVED_NEAR_ORIGIN`
- `UNDERRESOLVED_TAIL`
- `TAIL_TOO_LARGE`
- `OSCILLATION_SPIKE`

Interpretation:

- `NONFINITE_PANEL_CONTRIB` and `NONFINITE_INTEGRAND` are effectively hard
  failures for that panel.
- `UNDERRESOLVED_NEAR_ORIGIN`, `UNDERRESOLVED_TAIL`, and `OSCILLATION_SPIKE`
  are caution flags that tell you where to look next.
- `TAIL_TOO_LARGE` is stronger: the result can still be finite, but truncation
  is actively suspect.

### 2. Whole-integral diagnostics

Each scalar integral gets one summary bitmap in `warning_flags`.

The current flags are:

- `NONFINITE_TOTAL`
- `NONFINITE_PROBABILITY`
- `PROBABILITY_OUT_OF_RANGE`
- `LARGE_TAIL_FRACTION`
- `EXCESSIVE_CANCELLATION`
- `TOO_MANY_BAD_PANELS`
- `QUAD_ERROR_LARGE`

Interpretation:

Repository review policy:

- `NONFINITE_TOTAL`, `NONFINITE_PROBABILITY`, and
  `PROBABILITY_OUT_OF_RANGE` are hard trust failures unless a rerun identifies
  a benign data or configuration problem.
- `LARGE_TAIL_FRACTION` and `QUAD_ERROR_LARGE` are soft cautions that should be
  checked with a denser rerun before rejecting the point outright.
- `EXCESSIVE_CANCELLATION` and `TOO_MANY_BAD_PANELS` mean the finite number may
  still be fragile and should be reviewed together with backend agreement and
  config-sweep behavior. These severity labels are implementation policy
  covered by the
  [Heston diagnostics tests](https://github.com/willemk-stack/option-pricing-library/blob/main/tests/diagnostics/heston/test_heston_report.py),
  not literature theorem statements.

## Operational flag guide

| Flag | What it usually means | First knob to try | Usual severity |
| --- | --- | --- | --- |
| `NONFINITE_PANEL_CONTRIB` | A panel contribution itself blew up or became undefined | Re-run and inspect the regime or integrand values before tuning thresholds | Hard |
| `NONFINITE_INTEGRAND` | The integrand sampled on that panel contains non-finite values | Re-run and inspect the integrand or characteristic-function stability first | Hard |
| `UNDERRESOLVED_NEAR_ORIGIN` | Too much mass is concentrated in the first few panels | More local resolution near the origin: more panels, then clustered spacing or more nodes | Soft |
| `UNDERRESOLVED_TAIL` | The tail is not decaying convincingly by the end of the grid | Raise `u_max` first | Soft |
| `TAIL_TOO_LARGE` | Truncation is too aggressive and tail mass is still material | Raise `u_max` first | Usually hard for trust, even if finite |
| `OSCILLATION_SPIKE` | Likely coarse local resolution or oscillatory panel structure | More panels first, then more nodes per panel | Soft |
| `NONFINITE_TOTAL` | The integrated result is non-finite | Re-run and inspect the failure source before trusting any output | Hard |
| `NONFINITE_PROBABILITY` | The probability derived from the integral is non-finite | Re-run and inspect the failure source before trusting any output | Hard |
| `PROBABILITY_OUT_OF_RANGE` | The final probability is outside `[0, 1]` beyond tolerance | Raise resolution and compare backends before trusting | Hard |
| `LARGE_TAIL_FRACTION` | Too much absolute mass remains in the last panels | Raise `u_max` first | Soft |
| `EXCESSIVE_CANCELLATION` | The result is finite but fragile because large terms nearly cancel | Compare denser configs and backends before trusting | Soft to severe |
| `TOO_MANY_BAD_PANELS` | Multiple local warnings are piling up, not just one isolated panel | Densify the grid and inspect the worst-panel table | Severe review signal |
| `QUAD_ERROR_LARGE` | SciPy `quad` is reporting a relatively large estimated integration error | Re-run with a denser fixed-rule Gauss-Legendre config for comparison | Soft |

## Why both levels are needed

A panel flag answers:

> where is the numerical problem?

A whole-integral flag answers:

> should I trust the final result?

Those are not the same question.

Example:

- every panel may be finite
- but the final probability may still be outside `[0, 1]`
- or the total may only be small because large positive and negative panel
  contributions nearly cancel

That is why the code stores both local and global diagnostics.

A finite flagged result is not automatically invalid. A point can deserve review
because one heuristic fired while still remaining stable across backends and
denser quadrature settings. That is why the notebook-facing diagnostics keep the
raw warning facts separate from the higher-level strike suspiciousness policy.

For calibration input, the warning-to-action policy is:

- `block`: non-finite total integral or non-finite probability;
- `quarantine`: probability materially outside `[0, 1]` or persistent backend
  disagreement after robust/diagnostics rerun;
- `review`: high tail fraction, cancellation warnings, isolated oscillation
  spikes, near-origin underresolution, too many bad panels, or large quadrature
  error estimates;
- `ok`: no warning flags.

Calibration summaries should report blocked, quarantined, and reviewed quote
counts so filtering decisions are visible.

## Summary metrics

Two summary metrics are stored alongside `warning_flags` for fixed-rule
diagnostics.

### `tail_abs_fraction`

Definition:

\[
\frac{\sum |\text{last tail panels}|}{\sum |\text{all panel contributions}|}
\]

Interpretation:

- high value means the integral may not have decayed enough by `u_max`
- suggests truncation may be too aggressive
- usually addressed by larger `u_max` and sometimes more panels

### `cancellation_ratio`

Definition:

\[
\frac{\sum |\text{all panel contributions}|}{|\text{total integral}|}
\]

Interpretation:

- high value means the final answer is the delicate difference of large terms
- finite result, but potentially very sensitive to small discretization changes
- shows up as smile noise, parameter perturbation sensitivity, or weak
  reproducibility across quadrature settings

## Current heuristic knobs

The diagnostics logic is intentionally controlled by module-level constants so
the thresholds can be tuned without changing the public API.

### Probability and range

- `HESTON_DIAG_PROBABILITY_TOL`
  Small tolerance used before flagging `PROBABILITY_OUT_OF_RANGE`.

Typical reason to tweak:

- you see harmless floating-point drifts around 0 or 1 and want fewer false
  positives

### Tail checks

- `HESTON_DIAG_TAIL_PANEL_COUNT`
  Number of last panels treated as the tail.
- `HESTON_DIAG_TAIL_ABS_FRACTION_WARN`
  Tail-mass threshold for `LARGE_TAIL_FRACTION` and `TAIL_TOO_LARGE`.
- `HESTON_DIAG_TAIL_LAST_OVER_PREV_WARN`
  Ratio used to decide whether the tail is not decaying quickly enough.

Typical reason to tweak:

- wing strikes or short maturities are producing too many tail warnings
- or you want more aggressive truncation diagnostics

### Near-origin checks

- `HESTON_DIAG_NEAR_ORIGIN_PANEL_COUNT`
  Number of early panels treated as "near origin."
- `HESTON_DIAG_NEAR_ORIGIN_ABS_FRACTION_WARN`
  Threshold for `UNDERRESOLVED_NEAR_ORIGIN`.

Typical reason to tweak:

- near-Black or low-vol-of-vol regimes concentrate too much mass near the
  origin

### Oscillation checks

- `HESTON_DIAG_OSCILLATION_SPIKE_FACTOR`
  Flags a panel when its absolute contribution is much larger than the average
  of its immediate neighbors.
- `HESTON_DIAG_OSCILLATION_ABS_MASS_FLOOR`
  Absolute floor below which oscillation spikes are ignored as too small to
  matter for pricing.
- `HESTON_DIAG_OSCILLATION_REL_MASS_FLOOR`
  Relative floor, scaled by total absolute panel mass, that prevents tiny wing
  contributions from being escalated into meaningful oscillation warnings.

Typical reason to tweak:

- dense-strike smile calculations at large `|x|` are showing suspicious local
  spikes

### Whole-result checks

- `HESTON_DIAG_CANCELLATION_RATIO_WARN`
  Threshold for `EXCESSIVE_CANCELLATION`.
- `HESTON_DIAG_BAD_PANEL_COUNT_WARN`
  Absolute bad-panel count needed before the whole result gets
  `TOO_MANY_BAD_PANELS`.
- `HESTON_DIAG_BAD_PANEL_FRACTION_WARN`
  Fractional bad-panel threshold used together with the absolute count so the
  escalation scales with panel count instead of overreacting to a few flagged
  panels on denser grids.
- `HESTON_DIAG_QUAD_REL_ERROR_WARN`
  Relative error threshold for `quad`.

Typical reason to tweak:

- you want a stricter or looser "trust" threshold for production diagnostics

Two current guardrails are worth calling out because they reduce false alarms
without masking genuine instability:

- `TOO_MANY_BAD_PANELS` is driven by `max(BAD_PANEL_COUNT_WARN,
  BAD_PANEL_FRACTION_WARN * n_panels)`, so a denser panelization is not
  penalized just for having more panels available to be flagged.
- Panels whose only reason is `UNDERRESOLVED_NEAR_ORIGIN` do not by themselves
  escalate to `TOO_MANY_BAD_PANELS`; they stay visible as local caution flags
  unless other problems pile on.

## Interactions with quadrature settings

The diagnostics are only useful if interpreted together with the quadrature
configuration.

The current fixed-rule settings that matter most are:

- `u_max`
- `n_panels`
- `nodes_per_panel`
- `panel_spacing`
- `cluster_strength`

These are part of `QuadratureConfig`, and the implementation already exposes a
heuristic recommender via `recommend_heston_quadrature_config(...)`. The current
recommender adjusts truncation, panel count, local resolution, and clustering
based on maturity, `|x|`, vol-of-vol, correlation, and the requested quality
preset.

## Practical tuning guidance

### Increase `u_max` first when

- `LARGE_TAIL_FRACTION` fires
- `UNDERRESOLVED_TAIL` fires
- the result keeps moving in the wings

### Increase `n_panels` when

- `OSCILLATION_SPIKE` fires
- panel-level structure looks too coarse
- large `|x|` cases are noisy

### Increase `nodes_per_panel` when

- oscillation is local within panels rather than across many panels
- near-origin structure looks under-resolved despite enough panels

### Use clustered spacing when

- most absolute mass lives near the first few panels
- low `eta` or near-deterministic regimes need more local origin resolution

### Switch from `balanced` to `robust` or `diagnostics` when

- you are building reference values
- debugging a regression grid
- validating continuity under small parameter perturbations

### Treat the first rerun as a classification check, not a panic reaction

- If a flagged point becomes stable under a denser balanced or robust config
  while backend agreement and config-sweep span remain tight, the original flag
  was likely conservative review noise rather than a pricing failure.
- If the point keeps moving materially as you densify `u_max`, `n_panels`, or
  `nodes_per_panel`, treat the warning as evidence of real numerical fragility.

## Intended downstream use

The low-level diagnostics in `fourier.py` should feed a higher-level
diagnostics and report module under `option_pricing.diagnostics.heston`, where
they can be turned into:

- worst-panel tables
- per-flag counts
- strike-slice heatmaps
- truncation and tolerance convergence reports
- notebook-ready summary tables

That keeps raw numerical facts close to the pricer and presentation and
reporting logic in the diagnostics package.

## References

- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. *The Review of Financial Studies*, 6(2), 327-343.
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley.
- Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2006). The Little Heston Trap.
- Davis, P. J., & Rabinowitz, P. (1984). *Methods of Numerical Integration* (2nd ed.). Academic Press.
- Hale, N., & Townsend, A. Fast and accurate computation of Gauss-Legendre and Gauss-Jacobi quadrature nodes and weights.
