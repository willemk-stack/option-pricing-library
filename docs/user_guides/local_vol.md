# Local volatility

This guide shows how to derive a local-vol surface from a differentiable implied-vol surface and then use it in PDE pricing.

## Important limitation

`LocalVolSurface` supports two different implied-surface paths.

For a generic expiry stack such as `VolSurface.from_svi(...)`, it is still a **demo-grade** bridge from implied vol to local vol.
In that path, the time derivative `w_T` comes from piecewise-linear interpolation in total variance across expiry, so `w_T` is only piecewise constant.

For a continuous time-differentiable implied surface such as `ESSVISmoothedSurface`, the object can consume analytic `w`, `w_y`, `w_yy`, and `w_T` directly. That is the preferred route for Dupire-oriented work.

## Why SVI is a common starting point

`LocalVolSurface` needs each expiry slice to provide:

- `w_at(y)`
- `dw_dy(y)`
- `d2w_dy2(y)`

A plain grid surface built with `VolSurface.from_grid(...)` does **not** provide those derivatives.
An SVI-based surface does.

That said, SVI is only one route. If you need a smoother time-consistent surface, the eSSVI workflow is the stronger choice and is now the preferred Dupire-oriented path in the docs and flagship demos.

## Build an implied SVI surface first

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
    (1.5, 90.0, 0.26),
    (1.5, 100.0, 0.22),
    (1.5, 110.0, 0.24),
]

surface_svi = VolSurface.from_svi(
    rows,
    forward=forward,
    calibrate_kwargs={
        "repair_butterfly": True,
        "repair_method": "line_search",
        "butterfly_min_g_tol": None,
        "butterfly_min_g_tol_scale": 1.0,
    },
)
```

## Derive a local-vol surface

```python
import warnings
from option_pricing.vol import LocalVolSurface

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    localvol = LocalVolSurface.from_implied(surface_svi)
```

If you already have discount and forward callables from a market context, pass them explicitly:

```python
ctx = market.to_context()

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    localvol = LocalVolSurface.from_implied(
        surface_svi,
        forward=ctx.fwd,
        discount=ctx.df,
    )
```

## Preferred Dupire path: smooth eSSVI projection

If you have calibrated eSSVI nodes, project them first and pass the resulting `ESSVISmoothedSurface` into `LocalVolSurface`:

```python
from option_pricing.vol import LocalVolSurface, project_essvi_nodes

projection = project_essvi_nodes(fit.nodes)
if projection.surface is None:
    raise ValueError(projection.diag.message)

ctx = market.to_context()
localvol = LocalVolSurface.from_implied(
    projection.surface,
    forward=ctx.fwd,
    discount=ctx.df,
)
```

That path avoids the piecewise-constant-in-time `w_T` approximation used by the generic slice-stack interpolation route.

## Query local volatility

```python
import numpy as np

K = np.array([90.0, 100.0, 110.0])
sigma_loc = localvol.local_vol(K, 1.0)
sigma2_loc = localvol.local_var(K, 1.0)
```

## Inspect diagnostics instead of only the final number

```python
report = localvol.local_var_diagnostics(K=np.array([100.0]), T=1.0)
print(report)
```

This is useful when you want to understand *why* a point is unstable or invalid rather than silently taking square roots of whatever came out.

## Use the local-vol surface in PDE pricing

```python
from option_pricing.pricers.pde_pricer import local_vol_price_pde_european
from option_pricing.pricers.pde.domain import BSDomainConfig, BSDomainPolicy
from option_pricing.numerics.grids import SpacingPolicy
from option_pricing.numerics.pde import AdvectionScheme
from option_pricing.numerics.pde.domain import Coord

domain_cfg = BSDomainConfig(
    policy=BSDomainPolicy.LOG_NSIGMA,
    n_sigma=6.0,
    center="strike",
    spacing=SpacingPolicy.CLUSTERED,
    cluster_strength=2.0,
)

price = local_vol_price_pde_european(
    p,
    lv=localvol,
    coord=Coord.LOG_S,
    domain_cfg=domain_cfg,
    Nx=201,
    Nt=201,
    method="cn",
    advection=AdvectionScheme.CENTRAL,
)
```

## What fails with a grid-only surface?

```python
from option_pricing.vol import LocalVolSurface, VolSurface

surface_grid = VolSurface.from_grid(rows, forward=forward)
lv_bad = LocalVolSurface.from_implied(surface_grid)

# This raises TypeError because grid slices do not provide y-derivatives.
# lv_bad.local_vol(np.array([100.0]), 1.0)
```

## Notes

- The fastest demo-grade workflow is usually: market quotes -> `VolSurface.from_svi(...)` -> `LocalVolSurface.from_implied(...)` -> `local_vol_price_pde_european(...)`.
- The preferred Dupire-oriented workflow is: market prices -> `calibrate_essvi(...)` -> `project_essvi_nodes(...)` -> `ESSVISmoothedSurface` -> `LocalVolSurface.from_implied(...)` -> `local_vol_price_pde_european(...)`.
- The PDE solver advances in time-to-expiry `tau`, and `LocalVolSurface.local_var(K, T)` uses the same expiry variable in this codebase. The PDE wiring therefore passes solver time through directly instead of reversing it as `T_total - tau`.
- For a more time-consistent implied surface with explicit `w_T`, use the eSSVI workflow in [eSSVI](essvi.md) and feed `ESSVISmoothedSurface` into `LocalVolSurface.from_implied(...)`.
- For the implied-surface step, see [Volatility surface](vol_surface.md) and [SVI](svi.md).
- For PDE controls, see [PDE pricing](pde_pricing.md).
