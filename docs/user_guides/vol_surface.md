# Volatility surface

This guide covers the library's volatility-surface containers and the most common ways to build them.

The two central ideas are:

- a **smile** at one expiry, represented in total variance
- a **surface** across expiries, interpolated in total variance

## Core objects

```python
from option_pricing.vol import Smile, VolSurface
```

A `Smile` stores:

- `T`: expiry in years
- `y = ln(K / F(T))`: log-moneyness grid
- `w = T * iv^2`: total variance on that grid

A `VolSurface` stores multiple smile slices plus a `forward(T)` callable.

## Build a simple grid-based surface from `(T, K, iv)` points

```python
import numpy as np
from option_pricing.vol import VolSurface

S, r, q = 100.0, 0.02, 0.00

def forward(T: float) -> float:
    return float(S * np.exp((r - q) * float(T)))

rows = [
    (0.5, 90.0, 0.24),
    (0.5, 100.0, 0.20),
    (0.5, 110.0, 0.22),
    (1.0, 90.0, 0.25),
    (1.0, 100.0, 0.21),
    (1.0, 110.0, 0.23),
]

surface = VolSurface.from_grid(rows, forward=forward)
```

## Query implied vol at any strike and expiry

```python
iv_scalar = surface.iv(K=100.0, T=0.75)
iv_vec = surface.iv(K=np.array([90.0, 100.0, 110.0]), T=0.75)
```

Within a slice, the object interpolates in total variance.
Across expiries, the default public query path uses no-arbitrage-aware interpolation at constant log-moneyness.

## Work directly with slices

```python
slice_noarb = surface.slice(0.75, method="no_arb")
```

If your endpoint slices are differentiable analytic smiles such as SVI, you can also request linear-in-total-variance interpolation:

```python
slice_linear = surface.slice(0.75, method="linear_w")
```

That path is especially useful for local-vol construction because it preserves `dw/dy` and `d2w/dy2` when the endpoint slices provide them.

## Build an SVI-based surface directly

If your inputs are market quotes `(T, K, iv)`, you can calibrate one SVI smile per expiry and wrap the result in a `VolSurface` in one step:

```python
surface_svi = VolSurface.from_svi(
    rows,
    forward=forward,
    calibrate_kwargs={
        "repair_butterfly": True,
        "repair_method": "line_search",
    },
)
```

That gives you analytic per-expiry slices instead of grid-interpolated smiles.

## Surface no-arbitrage sanity checks

```python
from option_pricing.vol import check_surface_noarb

def df(T: float) -> float:
    return float(np.exp(-r * float(T)))

rep = check_surface_noarb(surface, df=df)
print(rep.ok)
print(rep.message)
```

These checks are lightweight sanity checks for:

- strike monotonicity / convexity proxies within each expiry
- calendar monotonicity in total variance across expiries

## Notes

- `VolSurface.from_grid(...)` is lightweight and dependency-light.
- `VolSurface.from_svi(...)` is the right starting point if you plan to build a [Local volatility](local_vol.md) surface later.
- For full SVI calibration details, see [SVI](svi.md).
