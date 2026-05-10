# Heston calibration evidence

!!! note "Status: validation evidence"
    This note documents the repository's calibration-fit diagnostics. It is
    bounded validation evidence for configured quote sets, objective choices,
    and quadrature settings; it is not a guarantee of parameter uniqueness.

The final calibration diagnostics use
`run_heston_calibration_fit_diagnostics(...)` to turn a fitted Heston parameter
set or `HestonMultistartResult` into plain evidence tables.

The report has the standard diagnostics shape:

- `meta`: objective, backend, quadrature, quote counts, Feller status, quote
  policy counts, and notes
- `tables`: residuals, smile fit, IV residual grid, parameter recovery,
  constraint diagnostics, quote policy, multistart runs, held-out errors, and
  objective slices
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

`constraint_diagnostics` reports the Feller ratio, margin, and status. The
policy is reported-not-hard-enforced by default; optional soft regularization
can penalize violations without blocking them.

`quote_policy` and `quote_policy_summary` map integration diagnostics to
calibration actions: `block`, `quarantine`, `review`, or `ok`. Hard failures
such as non-finite integrals or non-finite probabilities should not silently
enter calibration. Review warnings such as high tail fraction or cancellation
should be documented when retained.

LIMITATION: Objective slices are local diagnostic approximations. They can be
runtime-dependent and coordinate-convention-dependent, and they do not prove
global identifiability.

## Interpreting the evidence

Repository review policy: treat vanilla-only Heston calibration as potentially
weakly identified unless multistart runs, residual tables, and held-out
diagnostics say otherwise. Several parameter combinations may produce similar
vanilla prices, especially along mean-reversion, long-run variance,
vol-of-vol, and correlation tradeoffs. The committed evidence surface is the
`multistart_runs`, `held_out_errors`, and residual tables emitted by
`run_heston_calibration_fit_diagnostics(...)`, plus the generated
[multistart panel](../../assets/generated/heston/heston_multistart_stability_panel.svg)
and focused
[calibration-fit tests](https://github.com/willemk-stack/option-pricing-library/blob/main/tests/diagnostics/heston/test_heston_calibration_fit.py).

Multistart is diagnostic, not magic. It can show whether different starts
converge to the same basin, whether some starts fail, and whether near-best
solutions have similar costs. It cannot make a weakly identified calibration
unique.

Calibration optimizes vega-scaled price residuals as a robust proxy for IV
error; it does not optimize direct IV RMSE on this branch. Those residuals are
not direct IV RMSE unless model prices are explicitly inverted back to implied
volatilities and the IV residuals are then reported. The fit diagnostics do
perform that inversion where possible and report IV residuals separately from
price residuals.

### Objective versus reported metrics

| Quantity | Status | Interpretation |
|---|---|---|
| `price_rmse` | available calibration objective | Direct price residual objective |
| `relative_price_rmse` | available calibration objective | Price residual scaled by the target price |
| `vega_scaled_price` | available calibration objective and the default on this branch | Price residual scaled by Black-Scholes vega to approximate IV-error behavior without repeated IV inversion |
| `bid_ask_normalized` | available calibration objective | Residual scaled by bid-ask width |
| direct `iv_rmse` optimization | not implemented on this branch | Do not describe calibration as directly minimizing IV RMSE |
| reported IV residuals / IV RMSE | reporting metric | Computed after repricing and Black inversion when diagnostics are available |

When describing the project, say that calibration uses vega-scaled price
residuals and reports implied-volatility residual diagnostics; do not describe
this as direct IV-RMSE optimization unless such an objective is explicitly
implemented.

Loss-function choice is therefore part of the repository calibration contract,
not a universal Heston fact. Christoffersen and Jacobs motivate treating the
estimation loss and evaluation loss consistently; this branch records both the
optimized residual type and the reported IV diagnostics so the distinction is
visible.

Analytic Jacobians are default-on only on the guarded fixed Gauss-Legendre
calibration path with `eta >= HESTON_ANALYTIC_JAC_ETA_MIN` (`1e-6`) and bounded
parameter ranges. Price-only deterministic-limit checks near `eta=0` remain
valid pricing diagnostics, but they do not validate the Cui analytic-gradient
formula near zero vol-of-vol.

Held-out errors should not be mixed with fit errors. If a mask is supplied, the
diagnostics summarize train and held-out rows separately. If no mask is
supplied, the report leaves held-out evaluation empty instead of pretending it
happened.

## Known limitations

Heston calibration can be weakly identified from vanilla prices alone. Similar
residuals may come from different combinations of mean reversion, long-run
variance, vol-of-vol, correlation, and initial variance. The diagnostic tables
make that risk visible; they do not remove it.

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
report.tables["constraint_diagnostics"]
report.tables["quote_policy_summary"]
report.tables["held_out_errors"]
report.tables["objective_slices"].head()
```

## References

- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. *The Review of Financial Studies*, 6(2), 327-343.
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley.
- Cui, Y., del Baño Rollin, S., & Germano, G. (2016). Full and fast calibration of the Heston stochastic volatility model.
- Christoffersen, P., & Jacobs, K. (2003). The importance of the loss function in option valuation. CIRANO Scientific Series 2003s-52.
