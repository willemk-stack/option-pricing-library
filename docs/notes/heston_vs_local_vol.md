# Heston versus local volatility

!!! note "Status"
    This note documents a capstone model-comparison diagnostic. It is designed to compare
    Heston, eSSVI, and local-vol/PDE behavior on a common quote target; it is not a claim
    that one model is universally superior.

`run_heston_vs_local_vol_comparison(...)` is the Capstone 3 comparison layer for
a fitted Heston model, the repo-native eSSVI implied-surface path, and a small
direct Dupire/PDE local-vol repricing audit. It uses one common
`HestonQuoteSet` target so Heston, eSSVI, and local-vol PDE residuals are
computed against the same quotes.

The direct local-vol PDE audit is selected-subset evidence. Heston and eSSVI
proxy errors are available over the full comparison fixture, but direct PDE
rows are emitted only for the successful deterministic PDE audit quote indices.
The mixed full summary is retained for continuity; the matched direct-PDE subset
summary is the apples-to-apples table for comparing all three models on the
same selected quotes.

The final target is a deterministic market-like synthetic fixture. It is not
market data and is not generated from the fitted Heston model. This avoids
advantaging Heston with Heston-generated recovery data while keeping the
comparison deterministic and version-controlled.

## What Is Compared

The diagnostic compares:

- fitted Heston repricing on the common quote target;
- eSSVI nodal implied-surface repricing on the same quotes as a supporting
  proxy;
- direct local-vol PDE repricing on a small deterministic validation grid;
- optional train versus held-out partitions;
- quote-level price and implied-vol residuals;
- ATM, downside-wing, and upside-wing error buckets;
- qualitative tradeoffs around fit quality, interpretability, smoothness,
  extrapolation, and dynamics.

The final comparison notebook is
`demos/13_heston_calibration_vs_localvol.ipynb`. It should call library
diagnostics and plotting helpers rather than rebuilding Heston, eSSVI, Dupire,
or PDE logic in notebook cells.

## Diagnostic Outputs

Use these report fields rather than rebuilding comparison tables in a notebook:

- `comparison.tables["fit_errors"]`: quote-level market/model prices, market
  and model implied vols, residuals, moneyness bucket, and sample label;
- `comparison.tables["direct_local_vol_pde"]`: selected validation-grid rows
  with target IV, Heston price/IV, local-vol PDE price/IV, residuals, status,
  runtime, and PDE grid metadata;
- `comparison.tables["direct_local_vol_pde_summary"]`: direct PDE quote count,
  success count, grid size, surface source, and average residuals;
- `comparison.tables["error_summary"]`: RMSE, MAE, and max absolute error by
  model and by `all`, `atm`, `downside_wing`, and `upside_wing` buckets;
- `comparison.tables["direct_local_vol_pde_matched_error_summary"]`: the same
  bucket summary, filtered to successful direct-PDE quote indices so Heston,
  eSSVI proxy, and direct PDE use identical per-bucket quote counts;
- `comparison.tables["held_out_comparison"]`: train versus held-out errors when
  a held-out mask is supplied;
- `comparison.tables["tradeoff_summary"]`: concise fit-quality,
  interpretability, extrapolation, and dynamics notes;
- `comparison.meta["notes"]`: `NOTE:` and `LIMITATION:` statements that should
  travel with any capstone artifact.

The moneyness buckets are intentionally simple:

- `atm`: `abs(log_moneyness) <= 0.03`;
- `downside_wing`: `log_moneyness < -0.03`;
- `upside_wing`: `log_moneyness > 0.03`.

These thresholds are capstone diagnostics, not universal smile-region
definitions.

## Reading The Result

Heston is preferable when interpretable stochastic-volatility dynamics matter:
mean reversion, long-run variance, vol-of-vol, spot/variance correlation, and
initial variance all map to model behavior. That makes it a better candidate
when the analysis needs dynamics, path-dependent intuition, or forward-looking
variance scenarios.

The eSSVI surface side is preferable when the main target is vanilla surface
fit. It is a flexible implied-surface representation and can often match
vanilla smiles more tightly at calibrated nodes than a single five-parameter
Heston fit. The direct local-vol PDE rows are a separate audit of whether the
Dupire handoff and PDE grid reprice selected quotes consistently; they do not
globally prove local-vol accuracy across all strikes, maturities, boundaries,
or extrapolation regimes. Do not compare a full-set eSSVI proxy bucket directly
with a selected-subset direct PDE bucket; use
`direct_local_vol_pde_matched_error_summary` for that question.

## Minimal Usage

```python
from option_pricing.diagnostics.heston import (
    build_market_like_heston_quote_set,
    run_heston_vs_local_vol_comparison,
)
from option_pricing.vol.ssvi import ESSVICalibrationConfig

quotes = build_market_like_heston_quote_set()

comparison = run_heston_vs_local_vol_comparison(
    quotes=quotes,
    heston_fit=multistart_result,
    held_out_mask=held_out_mask,
    essvi_cfg=ESSVICalibrationConfig(max_nfev=1000),
    local_vol_pde_max_quotes=9,
    local_vol_pde_Nx=81,
    local_vol_pde_Nt=121,
)

comparison.tables["fit_errors"]
comparison.tables["direct_local_vol_pde"]
comparison.tables["error_summary"]
comparison.tables["direct_local_vol_pde_matched_error_summary"]
comparison.tables["tradeoff_summary"]
comparison.meta["notes"]
```

## References

- Heston, S. L. (1993). A Closed-Form Solution for Options with Stochastic
    Volatility with Applications to Bond and Currency Options. *Review of
    Financial Studies*, 6(2), 327-343.
- Dupire, B. (1994). Pricing with a smile. *Risk*, 7(1), 18-20.
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*.
    Wiley.
- Gatheral, J., & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
    *Quantitative Finance*, 14(1), 59-71.
