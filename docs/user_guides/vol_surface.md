# Volatility surface user guide (`VolSurface`)

This guide shows how to build and query a simple volatility surface using the library’s two container types:

- `Smile`: a single-expiry **total-variance** curve on a log-moneyness grid
- `VolSurface`: a piecewise-linear surface in **total variance**, interpolated in both strike and expiry

It also introduces the basic **no-arbitrage sanity checks** in `option_pricing.vol.arbitrage`.

---

## Data model: log-moneyness and total variance

For an expiry `T` (in years), the surface works in:

- **log-moneyness**:  

\[
x = \ln\left(\frac{K}{F(T)}\right)
\]

where `F(T)` is the forward at expiry.
  

- **total variance**:  

\[
w(T,x) = T \cdot \sigma_{imp}(T,x)^2
\]

A `Smile` stores `(T, x_grid, w_grid)` and provides interpolation in **total variance** space.

---

## Key types

```python
from option_pricing import Smile, VolSurface
```

### `Smile(T, x, w)`

- `T`: expiry in years
- `x`: strictly increasing log-moneyness grid
- `w`: total variance values on that grid (`w = T * iv^2`)

It supports:

- `smile.w_at(xq)` → total variance (linear interpolation in `x`, flat extrapolation)
- `smile.iv_at(xq)` → implied vol via `sqrt(max(w/T, 0))`

### `VolSurface(expiries, smiles, forward)`

- `expiries`: sorted expiries (years)
- `smiles`: `Smile` objects (same length as `expiries`)
- `forward`: callable `forward(T) -> float` (must be strictly positive)

The main method is:

- `surface.iv(K, T)` → implied vol at strike(s) `K` and expiry `T`

---

## Build a surface from `(T, K, iv)` points

The easiest way to create a surface is:

```python
from option_pricing import VolSurface

surface = VolSurface.from_grid(
    rows=[(T1, K1, iv1), (T1, K2, iv2), (T2, K1, iv3), ...],
    forward=forward,   # your forward curve function
)
```

### Forward curve example

If you have spot `S`, continuous rate `r`, and dividend yield `q`:

```python
import numpy as np

S, r, q = 100.0, 0.02, 0.00

def forward(T: float) -> float:
    return float(S * np.exp((r - q) * float(T)))
```

### Input requirements

- `T > 0`
- `K > 0`
- `iv >= 0` (recommended; small negatives will lead to odd results)
- For each expiry, strikes must map to a **strictly increasing** `x = ln(K / F(T))` grid.
  Duplicate strikes at the same expiry will typically fail.

If your expiries come from floating data sources, `from_grid` can bucket `T` with rounding:

```python
surface = VolSurface.from_grid(rows, forward=forward, expiry_round_decimals=10)
```

---

## Query the surface

```python
iv = surface.iv(K=100.0, T=1.0)                 # scalar strike
iv_vec = surface.iv(K=np.array([...]), T=0.75)  # vector strikes
```

### Interpolation / extrapolation behavior

- **Within an expiry** (`x` direction): linear interpolation in `w`, flat extrapolation at endpoints
- **Across expiries** (`T` direction): linear interpolation in `w` between bracketing expiries
- If `T` is outside the known range, expiry is **clamped** to the nearest smile

This makes the object very lightweight and predictable, but it is not a smooth fitter.

---

## No-arbitrage sanity checks

The module `option_pricing.vol.arbitrage` provides quick sanity checks for common surface problems.

```python
from option_pricing.vol.arbitrage import check_surface_noarb
```

### Surface-level check

```python
import numpy as np

def df(T: float) -> float:
    return float(np.exp(-r * float(T)))

rep = check_surface_noarb(surface, df=df)

print(rep.ok)
print(rep.message)
```

This runs two checks:

1) **Strike check per expiry (proxy)**: call price should be non-increasing in strike  
2) **Calendar check** in total variance: `w(T_{i+1}, x) >= w(T_i, x)` on a common `x` grid

### Interpreting the report

`check_surface_noarb` returns a `SurfaceNoArbReport`:

- `rep.ok`: overall boolean
- `rep.smile_monotonicity`: per-expiry `(T, MonotonicityReport)`
- `rep.calendar_total_variance`: a `CalendarVarianceReport`

Each `MonotonicityReport` includes:

- `ok`: pass/fail
- `bad_indices`: indices `i` where the call price increases from strike `i` to `i+1`
- `max_violation`: the largest observed increase (in price units)

The calendar report includes:

- `performed`: `False` if there is only one expiry or no overlapping `x` range
- `bad_pairs`: grid locations where calendar variance is violated
- `max_violation`: largest negative `Δw` (reported as a positive number)

---

## Practical tips

- These are **sanity checks**, not a full arbitrage-free surface construction.
  Real market surfaces typically require a parameterization (SVI, SABR, splines with constraints, …).
- If the calendar check is skipped, it usually means your smiles do not overlap in `x`.
  Try using a consistent strike grid across maturities.
- If strike monotonicity fails, the smile is often too steep/noisy in one wing.
  Smoothing the smile or removing bad quotes is a common first fix.

---

## Demo notebook

See: `demos/05_vol_surface_and_noarb.ipynb` for a runnable example with plots and diagnostics.
