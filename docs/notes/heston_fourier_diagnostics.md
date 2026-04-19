# Heston Fourier diagnostics

## Why this exists

The Heston Fourier pricer can produce a finite number that is still numerically fragile. This note focuses on the practical stability issues behind that behavior: branch handling, complex-log stability, oscillatory integrands, exposed integration settings, and diagnostics for truncation, convergence, discontinuities, and non-finite values.

This note documents the diagnostics contract added to `src/option_pricing/models/heston/fourier.py`:

- panel-local failure / warning bitmaps
- whole-integral warning bitmaps
- summary metrics that explain why a warning fired
- the main tuning knobs behind those warnings

The current Heston Fourier stack separates:
- the characteristic-function implementation in `charfunc.py`
- the probability integrals and diagnostics entry points in `fourier.py`
- the reusable composite quadrature machinery in `numerics/quadrature.py`

## Characteristic-function stability

The implementation uses a stable Gatheral-style affine form in `charfunc.py`. The discriminant handling in `_stable_discriminant(...)` flips the sign of the square-root branch whenever the implied `g` ratio would be non-finite or unstable. This is the main safeguard against the classic Heston branch-cut / complex-log instability.

That branch choice does not eliminate every numerical problem in the Fourier inversion. It reduces one major source of instability, but the inversion still needs diagnostics around:
- truncation
- local resolution near the origin
- oscillation
- cancellation
- non-finite panel values

## Diagnostic layers

There are two layers of diagnostics.

### 1. Panel-local diagnostics

These exist only for the fixed-rule `gauss_legendre` backend, because only that backend has an explicit panel structure.

Each panel gets a bitmap in `panel_reason`, with convenience mask `panel_invalid = panel_reason != 0`.

The current flags are:

- `NONFINITE_PANEL_CONTRIB`
- `NONFINITE_INTEGRAND`
- `UNDERRESOLVED_NEAR_ORIGIN`
- `UNDERRESOLVED_TAIL`
- `TAIL_TOO_LARGE`
- `OSCILLATION_SPIKE`

Interpretation:
- the first two are effectively hard failures for a panel
- the rest are soft warnings / heuristics

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
- the first three are usually hard failures
- the rest indicate a numerically fragile but finite result

## Why both levels are needed

A panel flag answers:
> where is the numerical problem?

A whole-integral flag answers:
> should I trust the final result?

Those are not the same question.

Example:
- every panel may be finite,
- but the final probability may still be outside `[0, 1]`,
- or the total may only be small because large positive and negative panel contributions nearly cancel.

That is why the code stores both local and global diagnostics.

## Summary metrics

Two summary metrics are stored alongside `warning_flags` for fixed-rule diagnostics.

### `tail_abs_fraction`

Definition:

```text
sum(abs(last tail panels)) / sum(abs(all panel contributions))
```

Interpretation:
- high value means the integral may not have decayed enough by `u_max`
- suggests truncation may be too aggressive
- usually addressed by larger `u_max` and sometimes more panels

### `cancellation_ratio`

Definition:

```text
sum(abs(all panel contributions)) / abs(total_integral)
```

Interpretation:
- high value means the final answer is the delicate difference of large terms
- finite result, but potentially very sensitive to small discretization changes
- shows up as smile noise, parameter perturbation sensitivity, or weak reproducibility across quadrature settings

## Current heuristic knobs

The diagnostics logic is intentionally controlled by module-level constants so the thresholds can be tuned without changing the public API.

### Probability / range

- `HESTON_DIAG_PROBABILITY_TOL`
  Small tolerance used before flagging `PROBABILITY_OUT_OF_RANGE`.

Typical reason to tweak:
- you see harmless floating-point drifts around 0 or 1 and want fewer false positives.

### Tail checks

- `HESTON_DIAG_TAIL_PANEL_COUNT`
  Number of last panels treated as the tail.
- `HESTON_DIAG_TAIL_ABS_FRACTION_WARN`
  Tail-mass threshold for `LARGE_TAIL_FRACTION` and `TAIL_TOO_LARGE`.
- `HESTON_DIAG_TAIL_LAST_OVER_PREV_WARN`
  Ratio used to decide whether the tail is not decaying quickly enough.

Typical reason to tweak:
- wing strikes or short maturities are producing too many tail warnings,
- or you want more aggressive truncation diagnostics.

### Near-origin checks

- `HESTON_DIAG_NEAR_ORIGIN_PANEL_COUNT`
  Number of early panels treated as “near origin”.
- `HESTON_DIAG_NEAR_ORIGIN_ABS_FRACTION_WARN`
  Threshold for `UNDERRESOLVED_NEAR_ORIGIN`.

Typical reason to tweak:
- near-Black or low-vol-of-vol regimes concentrate too much mass near the origin.

### Oscillation checks

- `HESTON_DIAG_OSCILLATION_SPIKE_FACTOR`
  Flags a panel when its absolute contribution is much larger than the average of its immediate neighbors.

Typical reason to tweak:
- dense-strike smile calculations at large `|x|` are showing suspicious local spikes.

### Whole-result checks

- `HESTON_DIAG_CANCELLATION_RATIO_WARN`
  Threshold for `EXCESSIVE_CANCELLATION`.
- `HESTON_DIAG_BAD_PANEL_COUNT_WARN`
  Number of flagged panels needed before the whole result gets `TOO_MANY_BAD_PANELS`.
- `HESTON_DIAG_QUAD_REL_ERROR_WARN`
  Relative error threshold for `quad`.

Typical reason to tweak:
- you want a stricter or looser “trust” threshold for production diagnostics.

## Interactions with quadrature settings

The diagnostics are only useful if interpreted together with the quadrature configuration.

The current fixed-rule settings that matter most are:

- `u_max`
- `n_panels`
- `nodes_per_panel`
- `panel_spacing`
- `cluster_strength`

These are part of `QuadratureConfig`, and the implementation already exposes a heuristic recommender via `recommend_heston_quadrature_config(...)`. The current recommender adjusts truncation, panel count, local resolution, and clustering based on maturity, `|x|`, vol-of-vol, correlation, and the requested quality preset.

## Practical tuning guidance

### Increase `u_max` first when:
- `LARGE_TAIL_FRACTION` fires,
- `UNDERRESOLVED_TAIL` fires,
- the result keeps moving in the wings.

### Increase `n_panels` when:
- `OSCILLATION_SPIKE` fires,
- panel-level structure looks too coarse,
- large `|x|` cases are noisy.

### Increase `nodes_per_panel` when:
- oscillation is local within panels rather than across many panels,
- near-origin structure looks under-resolved despite enough panels.

### Use clustered spacing when:
- most absolute mass lives near the first few panels,
- low `eta` / near-deterministic regimes need more local origin resolution.

### Switch from `balanced` to `robust` or `diagnostics` when:
- you are building reference values,
- debugging a regression grid,
- validating continuity under small parameter perturbations.

## Intended downstream use

The low-level diagnostics in `fourier.py` should feed a higher-level diagnostics/report module under `option_pricing.diagnostics.heston`, where they can be turned into:

- worst-panel tables
- per-flag counts
- strike-slice heatmaps
- truncation / tolerance convergence reports
- notebook-ready summary tables

That keeps raw numerical facts close to the pricer and presentation/reporting logic in the diagnostics package.
