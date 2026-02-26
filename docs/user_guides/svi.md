# SVI

This guide shows how to calibrate and use analytic SVI smiles.

In this library, SVI is used in **raw parameterization**:

- `a`
- `b`
- `rho`
- `m`
- `sigma`

and models **total variance** as a function of log-moneyness.

## The coordinate system

SVI works with:

- log-moneyness `y = ln(K / F(T))`
- total variance `w = T * iv^2`

If your market data starts as `(T, K, iv)`, you usually convert each expiry slice into `y` and `w` first.

## Evaluate the raw SVI formula directly

```python
import numpy as np
from option_pricing.vol.svi import SVIParams, svi_total_variance

params = SVIParams(a=0.02, b=0.50, rho=-0.20, m=0.10, sigma=0.30)
y = np.linspace(-1.0, 1.0, 51)
w = svi_total_variance(y, params)
```

## Calibrate a single SVI slice

```python
import numpy as np
from option_pricing.vol.svi import DomainCheckConfig, calibrate_svi

y = np.linspace(-0.8, 0.8, 41)
w_obs = 0.04 + 0.18 * np.sqrt((y + 0.05) ** 2 + 0.08**2) - 0.02 * (y + 0.05)

fit = calibrate_svi(
    y=y,
    w_obs=w_obs,
    loss="linear",
    domain_check=DomainCheckConfig(
        mode="fixed_logmoneyness",
        y_min=-1.0,
        y_max=1.0,
        n_grid=101,
    ),
)

print(fit.params)
print(fit.diag.ok)
print(fit.diag.summary)
```

The return value is `SVIFitResult` with:

- `params`: calibrated `SVIParams`
- `diag`: rich fit diagnostics

## Wrap calibrated parameters as a smile object

```python
from option_pricing.vol.svi import SVISmile

smile = SVISmile(T=1.0, params=fit.params)
print(smile.iv_at(0.0))
print(smile.dw_dy(0.0))
print(smile.d2w_dy2(0.0))
```

Those derivative methods are one reason SVI is useful as an intermediate representation for local-vol work.

## Diagnostics you will usually inspect

```python
checks = fit.diag.checks
print(checks.rmse_unw)
print(checks.butterfly_ok)
print(checks.min_g)
print(checks.sL, checks.sR)
```

Common signals:

- `fit.diag.ok` is the top-level pass/fail summary
- `checks.butterfly_ok` tells you whether the slice passed the butterfly test
- `checks.rmse_unw` gives you an unweighted fit error summary
- `checks.sL` and `checks.sR` are the fitted wing slopes

## Build a whole SVI surface from quotes

If you already have `(T, K, iv)` market quotes, the easiest route is usually:

```python
surface_svi = VolSurface.from_svi(rows, forward=forward)
```

That calibrates a slice per expiry and stores the result as analytic SVI smiles.

## Repair during calibration

If you want the calibration routine to repair butterfly-arbitrage violations automatically, pass repair options directly into `calibrate_svi(...)` or `VolSurface.from_svi(...)`:

```python
fit = calibrate_svi(
    y=y,
    w_obs=w_obs,
    repair_butterfly=True,
    repair_method="line_search",
)
```

The repair workflow is described in [SVI repair](svi_repair.md).
