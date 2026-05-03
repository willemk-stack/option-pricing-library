# Heston versus local volatility

`run_heston_vs_local_vol_comparison(...)` is the Capstone 3 comparison layer
for a fitted Heston model and the repo-native eSSVI/local-vol-facing vanilla
surface path. It uses one common `HestonQuoteSet` target, reprices that target
with both models, and packages the evidence as tables plus metadata notes.

REVIEW: This comparison uses the repo-native eSSVI nodal implied surface as a
local-vol-facing proxy. It does not run direct Dupire/PDE local-vol repricing.
Direct PDE repricing should be audited separately if the capstone conclusion
depends on pathwise local-vol pricing.

## What is compared

The diagnostic compares:

- the common target quote set or surface nodes
- fitted Heston repricing on those quotes
- eSSVI nodal implied-surface repricing on the same quotes
- optional train versus held-out partitions
- quote-level price and implied-vol residuals
- ATM, downside-wing, and upside-wing error buckets
- qualitative model tradeoffs

The final comparison notebook is
`demos/13_heston_calibration_vs_localvol.ipynb`. It calibrates Heston once,
calls `run_heston_vs_local_vol_comparison(...)`, and then displays only tables
plots, and notes already produced by the diagnostic and plot helpers.

## Diagnostic outputs

Use these report fields rather than rebuilding comparison tables in a notebook:

- `comparison.tables["fit_errors"]`: quote-level market/model prices, market
  and model implied vols, residuals, moneyness bucket, and sample label
- `comparison.tables["error_summary"]`: RMSE, MAE, and max absolute error by
  model and by `all`, `atm`, `downside_wing`, and `upside_wing` buckets
- `comparison.tables["held_out_comparison"]`: train versus held-out errors
  when a held-out mask is supplied
- `comparison.tables["tradeoff_summary"]`: concise fit-quality,
  interpretability, extrapolation, and dynamics notes
- `comparison.meta["notes"]`: `REVIEW:` limitations that should travel with
  any capstone artifact

The moneyness buckets are intentionally simple:

- `atm`: `abs(log_moneyness) <= 0.03`
- `downside_wing`: `log_moneyness < -0.03`
- `upside_wing`: `log_moneyness > 0.03`

REVIEW: These bucket thresholds are capstone diagnostics, not universal smile
region definitions.

## Reading the result

Heston is preferable when interpretable stochastic-volatility dynamics matter:
mean reversion, long-run variance, vol-of-vol, spot/variance correlation, and
initial variance all map to model behavior. That makes it a better candidate
when the analysis needs dynamics, path-dependent intuition, or forward-looking
variance scenarios.

Local-vol/eSSVI is preferable when the main target is vanilla surface fit. The
eSSVI side is a flexible implied-surface representation and can often match
vanilla smiles more tightly at calibrated nodes than a single five-parameter
Heston fit.

The comparison does not prove that the local-vol PDE repricer reproduces every
vanilla price on the fitted surface. Smoothness, extrapolation, Dupire
stability, and PDE repricing accuracy remain separate evidence items covered by
the existing local-vol and eSSVI workflows.

## Minimal usage

```python
from option_pricing.diagnostics.heston import (
    run_heston_vs_local_vol_comparison,
)
from option_pricing.vol.ssvi import ESSVICalibrationConfig

comparison = run_heston_vs_local_vol_comparison(
    quotes=quotes,
    heston_fit=multistart_result,
    held_out_mask=held_out_mask,
    essvi_cfg=ESSVICalibrationConfig(max_nfev=1000),
)

comparison.tables["fit_errors"]
comparison.tables["error_summary"]
comparison.tables["held_out_comparison"]
comparison.tables["tradeoff_summary"]
comparison.meta["notes"]
```
