# Heston versus local volatility

`run_heston_vs_local_vol_comparison(...)` compares a fitted Heston model with
the repo-native eSSVI path on the same vanilla quote target.

The local-vol side currently uses:

```text
calibrate_essvi(...) -> ESSVINodalSurface -> implied-price repricing
```

REVIEW: This is a local-vol-facing implied-surface proxy. It does not run a
Dupire local-vol PDE repricer. Direct local-vol repricing is heavier and should
be audited separately when the question is pathwise local-vol pricing rather
than vanilla-surface fit.

## What is compared

The comparison table reports, for both models:

- market IV and model IV
- IV residuals in basis points
- market price and model price when available
- price residuals
- expiry, strike, log-moneyness, call/put flag, and train or held-out label

The summary table reports RMSE, MAE, and max absolute error for prices and IV
residual basis points. It also breaks the fit into simple moneyness buckets:

- `atm`: `abs(log_moneyness) <= 0.03`
- `downside_wing`: `log_moneyness < -0.03`
- `upside_wing`: `log_moneyness > 0.03`

REVIEW: These bucket thresholds are capstone diagnostics, not universal smile
region definitions.

If a held-out mask is supplied, the comparison also reports train versus
held-out errors by model. Conclusions should use those held-out rows separately
from in-sample fit rows.

## Reading the tradeoff

Heston can fit smooth skewed smiles when the target is compatible with a
five-parameter stochastic volatility structure. It also provides interpretable
dynamics: mean reversion, long-run variance, vol-of-vol, spot/variance
correlation, and initial variance.

Heston can be too rigid for vanilla-only smile fitting. A market smile with
localized curvature or maturity-specific shape can require more flexibility
than a single Heston parameter set provides.

The eSSVI/local-vol proxy can fit vanilla implied-vol targets more tightly at
the calibrated nodes because it is a surface representation rather than a
single stochastic variance process. That extra vanilla flexibility does not by
itself give Heston-style stochastic dynamics.

Local volatility can match vanilla marginals under its surface assumptions, but
smoothness, extrapolation, Dupire stability, and PDE repricing quality are
separate evidence items. The existing local-vol and eSSVI tests cover those
layers; this comparison focuses on the shared vanilla target.

Model comparison conclusions depend on the chosen data target, fit partition,
weights, and local-vol proxy. A synthetic Heston surface should favor Heston on
interpretability and may favor Heston on exact repricing; a flexible market
surface may favor eSSVI/local-vol on vanilla fit quality.

## Minimal usage

```python
from option_pricing.diagnostics.heston import (
    run_heston_vs_local_vol_comparison,
)
from option_pricing.vol.ssvi import ESSVICalibrationConfig

comparison = run_heston_vs_local_vol_comparison(
    quotes=quotes,
    heston_fit=multistart_result,
    held_out_mask=held_out_mask,  # optional
    essvi_cfg=ESSVICalibrationConfig(max_nfev=1000),
)

comparison.tables["error_summary"]
comparison.tables["tradeoff_summary"]
```
