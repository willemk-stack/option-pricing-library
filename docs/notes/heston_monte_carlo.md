# Heston Monte Carlo diagnostics

!!! note "Status"
	This note documents the repository's Monte Carlo validation workflow and interpretation
	guidance. Scheme labels, review thresholds, and recommendation text should be read as
	diagnostics policy rather than a universal ranking of simulation methods.

## Purpose

The Heston Monte Carlo diagnostics utilities package three review questions into notebook-friendly pandas tables:

- how the Monte Carlo estimate moves as the timestep changes
- how runtime trades off against error for a fixed path count
- how full-truncation Euler and Andersen QE compare on the same case

The public entrypoints are:

- `run_heston_mc_comparison_sweep(...)`
- `summarize_bias_vs_timestep(...)`
- `summarize_runtime_vs_error(...)`
- `compare_heston_mc_schemes(...)`

The plotting helpers consume the prepared tables directly:

- `plot_heston_mc_bias_vs_timestep(...)`
- `plot_heston_mc_runtime_vs_error(...)`
- `plot_heston_mc_scheme_comparison(...)`

All of the public helpers return plain pandas DataFrames or matplotlib figures so they can be used in notebooks, reports, and lightweight regression checks without hiding the underlying Monte Carlo behavior.

The public Heston simulation and Monte Carlo pricing entrypoints default to `quadratic_exponential`. Pass `scheme="euler_full_truncation"` explicitly when you want Euler as the baseline scheme for side-by-side diagnostics or educational examples.

When you inspect the low-level Euler path generator directly, `simulate_heston_euler_paths(...)` now records `result.metadata["negative_variance_proposal_rate"]`: the fraction of raw variance proposals that would have gone negative before the full-truncation floor was applied. That complements the nonnegative stored `var_paths` by showing how stressed the naive Euler proposal was.

## Bias vs timestep

Use `run_heston_mc_comparison_sweep(...)` to generate one row per `(scheme, n_steps, repeat)` run and then pass the result into `summarize_bias_vs_timestep(...)`.

The bias summary reports:

- `mean_signed_error` as the estimated bias relative to the semi-analytic Heston reference
- `mean_abs_error` and `rmse` as accuracy summaries
- `mean_stderr` and `mean_ci_half_width` as Monte Carlo noise summaries
- `coverage_rate` as a quick check of how often the confidence interval still contains the reference price

Typical notebook flow:

```python
from option_pricing.diagnostics.heston import (
	HestonMCComparisonCase,
	HestonMCSweepConfig,
	run_heston_mc_comparison_sweep,
	summarize_bias_vs_timestep,
)

case = HestonMCComparisonCase(
	ctx=market.to_context(),
	params=params,
	kind=OptionType.CALL,
	strike=100.0,
	tau=1.0,
)
cfg = HestonMCSweepConfig(n_steps_grid=(8, 16, 32, 64), n_paths=20_000)

sweep = run_heston_mc_comparison_sweep(case, cfg)
bias_summary = summarize_bias_vs_timestep(sweep)
```

## Runtime vs error

`summarize_runtime_vs_error(...)` groups the same sweep by `(scheme, n_steps, dt)` and reports mean and median runtime together with error metrics.

This is useful when a smaller discretization error comes with a materially larger runtime cost. The `error_per_second` field is deliberately simple:

```python
mean_abs_error / mean_runtime_seconds
```

Treat it as a review aid, not as a single scalar that proves one scheme is always better.

## Scheme comparison: full-truncation Euler vs Andersen QE

`compare_heston_mc_schemes(...)` reduces the sweep to one row per scheme and selects the lowest-`mean_abs_error` configuration observed in the provided table.

The recommendation field is intentionally conservative:

- `quadratic_exponential` -> `production candidate`
- `euler_full_truncation` -> `baseline / educational`
- anything else -> `review evidence before using`

This summary is only as broad as the sweep you ran. A tiny notebook sweep is a good cross-check, not proof of global superiority across maturities, strikes, or stressed parameter regimes.

## Why the vanilla control variate is disabled by default for validation

The exact same vanilla option can be used as a control variate under Heston Monte Carlo, but it is disabled by default in these diagnostics.

For validation work, enabling that control by default can make the Monte Carlo estimator look artificially close to the semi-analytic price you are comparing against. That hides the raw Monte Carlo behavior you usually want to inspect when you are studying timestep bias, runtime, or scheme differences.

The control variate remains available through `HestonMCSweepConfig(use_control_variate=True)` when variance reduction is the actual subject of the experiment.

## Interpreting confidence intervals and discretization bias

Each raw sweep row stores:

- `stderr`
- `ci_low`
- `ci_high`
- `ci_half_width`
- `covered_reference`

These columns answer a different question from timestep bias.

- The confidence interval is about Monte Carlo sampling noise at a fixed discretization.
- The bias summaries are about how the discretized scheme deviates from the semi-analytic reference as `dt` changes.

If the confidence interval is narrow but the signed error stays materially away from zero as `dt` shrinks slowly, you are looking at a discretization issue rather than a pure Monte Carlo variance issue. If intervals are wide and coverage is erratic, increase the path count before over-interpreting small differences between schemes.
