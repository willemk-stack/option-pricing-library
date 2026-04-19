# Heston diagnostics user guide

This page is a practical guide to the Heston Fourier diagnostics returned by:

- `P_j_with_diagnostics(...)`
- `P_j_batch_with_diagnostics(...)`

These diagnostics exist to expose integration stability signals alongside the Heston Fourier probabilities and to make quadrature settings explicit when results look fragile.

## Quick start

```python
import numpy as np

from option_pricing.models.heston.fourier import (
    P_j_with_diagnostics,
    P_j_batch_with_diagnostics,
    recommend_heston_quadrature_config,
)
from option_pricing.models.heston.params import HestonParams

params = HestonParams(
    kappa=2.0,
    vbar=0.04,
    eta=0.6,
    rho=-0.7,
    v=0.04,
)

cfg = recommend_heston_quadrature_config(
    x=0.75,
    tau=0.5,
    params=params,
    quality="balanced",
)

diag = P_j_with_diagnostics(
    x=0.25,
    tau=0.5,
    params=params,
    j=0,
    backend="gauss_legendre",
    quad_cfg=cfg,
)

print(diag.probability)
print(diag.warning_flags)
print(diag.tail_abs_fraction)
print(diag.cancellation_ratio)
print(diag.panel_reason)
```

## Scalar diagnostics object

For `backend="gauss_legendre"` the scalar diagnostics object contains:

- `backend`
- `quad_cfg`
- `j`, `x`, `tau`
- `total_integral`
- `probability`
- `panel_edges`
- `panel_contribs`
- `panel_invalid`
- `panel_reason`
- `warning_flags`
- `tail_abs_fraction`
- `cancellation_ratio`

For `backend="quad"` the object contains:

- whole-integral diagnostics
- `quad_error_estimate`

and panel-local fields are `None`.

## Batch diagnostics object

For a batch `x` input, the leading shape of every array field matches `x.shape`.

For example, if `x.shape == (n_strikes,)` and the rule has `p` panels:

- `probability.shape == (n_strikes,)`
- `warning_flags.shape == (n_strikes,)`
- `panel_contribs.shape == (n_strikes, p)`
- `panel_reason.shape == (n_strikes, p)`

If `x.shape == (n_expiries, n_strikes)`:

- `warning_flags.shape == (n_expiries, n_strikes)`
- `panel_reason.shape == (n_expiries, n_strikes, p)`

## How to interpret warning flags

### Hard failures
Treat these as “do not trust this result”:

- `NONFINITE_TOTAL`
- `NONFINITE_PROBABILITY`

Usually also treat this as a failure:

- `PROBABILITY_OUT_OF_RANGE`

### Soft warnings
Treat these as “finite but numerically suspicious”:

- `LARGE_TAIL_FRACTION`
- `EXCESSIVE_CANCELLATION`
- `TOO_MANY_BAD_PANELS`
- `QUAD_ERROR_LARGE`

These often suggest that you should rerun with stronger settings before trusting the number for calibration or regression baselines.

## Typical recovery actions

### Tail warnings
Symptoms:
- `LARGE_TAIL_FRACTION`
- tail panel flags
- wing instability

Actions:
- increase `u_max`
- often also increase `n_panels`
- consider `quality="robust"` or `quality="diagnostics"`

### Origin-resolution warnings
Symptoms:
- `UNDERRESOLVED_NEAR_ORIGIN`
- most mass concentrated in first few panels

Actions:
- use clustered spacing
- increase `nodes_per_panel`
- increase `n_panels`

### Oscillation warnings
Symptoms:
- `OSCILLATION_SPIKE`
- noisy smile points at large `|x|`

Actions:
- increase `n_panels`
- possibly increase `nodes_per_panel`
- use a more conservative quality preset

### Cancellation warnings
Symptoms:
- very large `cancellation_ratio`
- unstable results under tiny setting changes

Actions:
- compare with a more robust fixed rule
- compare against `quad`
- avoid treating the current run as a reference value

## Recommended workflow

### For fast exploratory runs
Use:
- `quality="fast"` or `quality="balanced"`
- inspect only whole-integral warnings

### For production-style slice pricing
Use:
- `quality="balanced"` or `quality="robust"`
- review both whole-integral flags and worst panels on suspicious strikes

### For regression baselines and notebook diagnostics
Use:
- `quality="diagnostics"`
- run `P_j_batch_with_diagnostics(...)`
- store `warning_flags`, `tail_abs_fraction`, and `cancellation_ratio`
- inspect where warnings cluster over strike / maturity grids

## What should be documented elsewhere

Once the docs structure is cleaned up, this page should likely be split into:
- one implementation note for the characteristic function and branch handling
- one user-facing diagnostics guide
- one report/notebook guide under `option_pricing.diagnostics.heston`
