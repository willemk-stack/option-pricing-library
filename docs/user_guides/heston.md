# Heston

The Heston workflow in this library is split into pricing, calibration, and
diagnostics:

- Fourier pricing and backend diagnostics live under
  `option_pricing.models.heston` and `option_pricing.diagnostics.heston`.
- Calibration uses `HestonQuoteSet`, `default_heston_seed`,
  `heston_seed_grid`, `calibrate_heston`, and `calibrate_heston_multistart`.
- Final evidence uses `run_heston_calibration_fit_diagnostics(...)`.
- Vanilla model comparison uses `run_heston_vs_local_vol_comparison(...)`.

## Calibration workflow

Use a `HestonQuoteSet` with mid prices, implied vols, and Black vegas when the
objective is vega-scaled. The default seed is market-aware, but it is only a
starting point.

```python
from option_pricing.models.heston.calibration import calibrate_heston_multistart

result = calibrate_heston_multistart(
    quotes=quotes,
    objective_type="vega_scaled_price",
    bounds=bounds,
    quad_cfg=quad_cfg,
    max_seeds=8,
    parameter_transform="bounded",
)
```

Multistart is diagnostic, not magic. It helps expose initialization sensitivity,
failed seeds, and near-best basins, but it does not make weakly identified
parameters unique.

## Final evidence

```python
from option_pricing.diagnostics.heston import (
    run_heston_calibration_fit_diagnostics,
)

fit_report = run_heston_calibration_fit_diagnostics(
    quotes=quotes,
    fit=result,
    held_out_mask=held_out_mask,
    quad_cfg=quad_cfg,
)
```

Report held-out errors separately from fit errors. If no held-out mask is
supplied, the diagnostics leave that table empty.

Vega-scaled price residuals may be used as an IV-like proxy during calibration,
but they are not direct IV RMSE unless model prices are inverted back to implied
volatility and IV residuals are explicitly reported. The final fit report
includes those direct IV residuals when inversion succeeds.

## Heston versus local volatility

```python
from option_pricing.diagnostics.heston import (
    run_heston_vs_local_vol_comparison,
)

comparison = run_heston_vs_local_vol_comparison(
    quotes=quotes,
    heston_fit=result,
    held_out_mask=held_out_mask,
)
```

The comparison currently uses the repo-native eSSVI implied surface as a
local-vol-facing proxy. REVIEW: It does not run a Dupire PDE repricer. Model
comparison conclusions depend on the target, weighting, train/held-out split,
and local-vol proxy.
