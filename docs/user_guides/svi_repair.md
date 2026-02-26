# SVI repair

This guide focuses on detecting and repairing butterfly-arbitrage problems in a single SVI slice.

The main tools are:

- `check_butterfly_arbitrage(...)`
- `repair_butterfly_raw(...)`
- `repair_butterfly_jw_optimal(...)`
- `calibrate_svi(..., repair_butterfly=True)`

## Check whether a slice is butterfly-safe

```python
from option_pricing.vol.svi import SVIParams, check_butterfly_arbitrage

p_bad = SVIParams(a=0.10, b=1.30, rho=0.70, m=0.10, sigma=0.20)

check = check_butterfly_arbitrage(
    p_bad,
    y_domain_hint=(-1.25, 1.25),
    w_floor=0.0,
    g_floor=0.0,
    tol=1e-10,
)

print(check.ok)
print(check.failure_reason)
print(check.g_left_inf, check.g_right_inf)
```

The returned `ButterflyCheck` object includes both analytic wing-limit information and scanned-domain diagnostics.

## Repair a bad raw SVI slice

```python
from option_pricing.vol.svi import repair_butterfly_raw

p_fix = repair_butterfly_raw(
    p_bad,
    T=1.0,
    y_domain_hint=(-1.25, 1.25),
    w_floor=0.0,
    method="line_search",
)
```

Then re-check it:

```python
check_fix = check_butterfly_arbitrage(
    p_fix,
    y_domain_hint=(-1.25, 1.25),
    w_floor=0.0,
    g_floor=0.0,
    tol=1e-10,
)
print(check_fix.ok)
```

## Repair methods

`repair_butterfly_raw` currently supports two high-level repair paths:

- `method="project"`
- `method="line_search"`

In practice, `line_search` is often a good default because it tends to stay closer to the original slice while still producing a feasible result.

## Advanced repair refinement

For a paper-style JW-space refinement after obtaining a feasible slice, use:

```python
import numpy as np
from option_pricing.vol.svi import repair_butterfly_jw_optimal

y_obj = np.linspace(-1.25, 1.25, 121)
y_penalty = np.linspace(-2.0, 2.0, 201)

p_refined = repair_butterfly_jw_optimal(
    p_bad,
    T=1.0,
    y_obj=y_obj,
    y_penalty=y_penalty,
    y_domain_hint=(-1.25, 1.25),
    w_floor=0.0,
)
```

That is more specialized than the basic `repair_butterfly_raw(...)` workflow, but it is available when you want tighter control over the repair objective.

## Repair automatically during calibration

You can ask the calibration routine to repair the slice before returning it:

```python
fit = calibrate_svi(
    y=y,
    w_obs=w_obs,
    repair_butterfly=True,
    repair_method="line_search",
    refit_after_repair=True,
)
```

That same pattern can be passed through `VolSurface.from_svi(..., calibrate_kwargs=...)`.

## Notes

- These repair routines work in **total variance** space, so the `T` argument matters.
- Repair is about making the slice usable and butterfly-safe, not about preserving the original parameters exactly.
- After repair, it is still worth inspecting the fit error and wing diagnostics from `SVIFitDiagnostics`.
