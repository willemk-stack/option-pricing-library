# Heston calibration evidence

The final calibration diagnostics use
`run_heston_calibration_fit_diagnostics(...)` to turn a fitted Heston parameter
set or `HestonMultistartResult` into plain evidence tables.

The report has the standard diagnostics shape:

- `meta`: objective, backend, quadrature, quote counts, and review notes
- `tables`: residuals, smile fit, IV residual grid, parameter recovery,
  multistart runs, held-out errors, and objective slices
- `arrays`: aligned quote vectors, model prices, model IVs, and parameter arrays

## What is reported

`residuals` is one row per quote. It includes the market price, Heston model
price, price residual, market IV when available, model IV from Black inversion,
IV residual in volatility points and basis points, weights, and the train or
held-out flag.

`smile_fit` is the plotting-ready subset for smile overlays by maturity. It is
data only; plotting helpers consume it later.

`iv_residual_grid` is returned in long form. No interpolation is applied, so
irregular quote grids stay explicit instead of being silently reshaped.

`parameter_recovery` includes truth columns when synthetic truth is supplied.
Without truth, it becomes a fitted-parameter summary and omits unsupported truth
columns.

`multistart_runs` records every optimizer run when a `HestonMultistartResult` is
supplied, including failed runs. Failed seeds are evidence about initialization
sensitivity, not noise to be hidden.

`held_out_errors` is populated only when a held-out mask is supplied. Fit errors
and held-out errors should be reported separately.

`objective_slices` evaluates small constrained-coordinate slices around the
fitted point, currently including `kappa` versus `vbar`, `eta` versus `rho`, and
`v` versus `vbar`.

REVIEW: Objective slices are local diagnostic approximations. They can be
runtime-dependent and coordinate-convention-dependent, and they do not prove
global identifiability.

## Interpreting the evidence

Heston calibration can be weakly identifiable from vanilla options. Several
parameter combinations may produce similar vanilla prices, especially along
mean-reversion, long-run variance, vol-of-vol, and correlation tradeoffs.

Multistart is diagnostic, not magic. It can show whether different starts
converge to the same basin, whether some starts fail, and whether near-best
solutions have similar costs. It cannot make a weakly identified calibration
unique.

Vega-scaled price residuals may be useful as an IV-like proxy during
optimization, but they are not direct IV RMSE unless model prices are explicitly
inverted back to implied volatilities and those IV residuals are reported. The
fit diagnostics do perform that inversion where possible and report IV residuals
separately from price residuals.

Held-out errors should not be mixed with fit errors. If a mask is supplied, the
diagnostics summarize train and held-out rows separately. If no mask is
supplied, the report leaves held-out evaluation empty instead of pretending it
happened.

## Minimal usage

```python
from option_pricing.diagnostics.heston import (
    run_heston_calibration_fit_diagnostics,
)

report = run_heston_calibration_fit_diagnostics(
    quotes=quotes,
    fit=multistart_result,
    true_params=true_params,        # optional
    held_out_mask=held_out_mask,    # optional
    quad_cfg=quad_cfg,
)

report.tables["residuals"].head()
report.tables["held_out_errors"]
report.tables["objective_slices"].head()
```
