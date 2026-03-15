# eSSVI

This guide covers the library's extended SSVI implementation and the three main ways it appears in the public volatility API.

In this codebase, eSSVI is parameterized by maturity term structures for:

- `theta(T)`
- `psi(T)`
- `eta(T)`

with the classic SSVI quantities recovered as:

- `phi(T) = psi(T) / theta(T)`
- `rho(T) = eta(T) / psi(T)`

That representation is exposed through `ESSVITermStructures` and the related surface classes in `option_pricing.vol`.

## Surface types

The implementation exposes three related surface objects:

- `ESSVIImpliedSurface`: continuous analytic surface driven directly by `ESSVITermStructures`
- `ESSVINodalSurface`: exact arbitrage-safe nodal surface built from calibrated expiry nodes and Mingone interpolation
- `ESSVISmoothedSurface`: smooth projection of nodal parameters for time-differentiable workflows such as Dupire local vol

In practice:

- use `ESSVIImpliedSurface` when you already know the term structures
- use `ESSVINodalSurface` as the direct output of calibration
- use `ESSVISmoothedSurface` when you need `w_T` and a Dupire-oriented continuous surface

## Build a continuous eSSVI surface directly

```python
import numpy as np

from option_pricing.types import MarketData, OptionType
from option_pricing.vol import (
    ESSVIImpliedSurface,
    ESSVITermStructures,
    EtaTermStructure,
    PsiTermStructure,
    ThetaTermStructure,
    validate_essvi_continuous,
)

params = ESSVITermStructures(
    theta_term=ThetaTermStructure(
        value=lambda T: 0.04 + 0.03 * np.asarray(T, dtype=np.float64),
        first_derivative=lambda T: np.full_like(np.asarray(T, dtype=np.float64), 0.03),
    ),
    psi_term=PsiTermStructure(
        value=lambda T: 0.18 + 0.02 * np.asarray(T, dtype=np.float64),
        first_derivative=lambda T: np.full_like(np.asarray(T, dtype=np.float64), 0.02),
    ),
    eta_term=EtaTermStructure(
        value=lambda T: -0.03 + 0.005 * np.asarray(T, dtype=np.float64),
        first_derivative=lambda T: np.full_like(np.asarray(T, dtype=np.float64), 0.005),
    ),
)

surface = ESSVIImpliedSurface(params=params)

y = np.linspace(-0.5, 0.5, 5)
T = 1.0

w = surface.w(y, T)
iv = surface.iv(y, T)
dw_dy = surface.dw_dy(y, T)
d2w_dy2 = surface.d2w_dy2(y, T)
dw_dT = surface.dw_dT(y, T)

market = MarketData(spot=100.0, rate=0.01, dividend_yield=0.0)
report = validate_essvi_continuous(
    params,
    market,
    expiries=np.array([0.25, 0.5, 1.0, 1.5]),
)
print(report.ok)

strike = np.array([80.0, 90.0, 100.0, 110.0, 120.0], dtype=np.float64)
call_price = surface.price(
    kind=OptionType.CALL,
    strike=strike,
    T=T,
    market=market,
)
```

`ESSVIImpliedSurface` is the right object when you need a continuous parameter surface with analytic smile derivatives and an explicit `w_T`.

## Calibrate eSSVI from market option prices

Calibration works on arrays of log-moneyness, expiry, and market option prices.

- `y = ln(K / F(T))`
- `T` is time to expiry in years
- `price_mkt` is the market option price for that quote
- `is_call` is optional; if you omit it, the code uses calls for `y >= 0` and puts for `y < 0`

```python
import numpy as np

from option_pricing.types import MarketData
from option_pricing.vol import (
    ESSVICalibrationConfig,
    ESSVINodalSurface,
    calibrate_essvi,
    project_essvi_nodes,
)

market = MarketData(spot=100.0, rate=0.01, dividend_yield=0.0)

expiries = np.array([0.25, 0.5, 1.0, 1.5], dtype=np.float64)
y_grid = np.array([-0.4, -0.2, 0.0, 0.2, 0.4], dtype=np.float64)

y = np.tile(y_grid, expiries.size)
T = np.repeat(expiries, y_grid.size)

# Fill this from your market quotes after converting strikes to y = ln(K / F(T)).
price_mkt = np.asarray([...], dtype=np.float64)
is_call = y >= 0.0

fit = calibrate_essvi(
    y=y,
    T=T,
    price_mkt=price_mkt,
    market=market,
    is_call=is_call,
    cfg=ESSVICalibrationConfig(strict_validation=True),
)

print(fit.diag.node_validation.ok)
print(fit.diag.price_rmse)
print(fit.diag.max_abs_price_error)

nodal_surface = ESSVINodalSurface(fit.nodes)
projection = project_essvi_nodes(fit.nodes)
smoothed_surface = projection.surface
```

The calibration result is nodal by design. That gives you an exact arbitrage-aware `ESSVINodalSurface`, and then an explicit projection step decides whether a smooth continuous `ESSVISmoothedSurface` is admissible.

## Validate continuous or nodal constraints

Use the validation helpers at the level you are working on:

```python
from option_pricing.vol import (
    evaluate_essvi_constraints,
    validate_essvi_continuous,
    validate_essvi_nodes,
)

constraint_report = evaluate_essvi_constraints(params, expiries)
node_report = validate_essvi_nodes(fit.nodes)
surface_report = validate_essvi_continuous(params, market, expiries=expiries)
```

These reports separate:

- parametric admissibility checks such as Lee and Gatheral-Jacquier margins
- nodal monotonicity and Mingone interpolation conditions
- sampled static no-arbitrage checks on the resulting implied surface

Use `validate_essvi_continuous(...)` for new code. `validate_essvi_surface(...)` remains as a deprecated compatibility alias.

## Use eSSVI as a local-vol input surface

The main Dupire-oriented path is:

```python
from option_pricing.vol import LocalVolSurface

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

This is the preferred route when you want a time-differentiable implied surface for local-vol diagnostics or PDE workflows. The older generic slice-stack route remains available, but it still relies on piecewise time interpolation.

## Notes

- `calibrate_essvi(...)` is the primary calibration entrypoint.
- `calibrate_essvi_global(...)` is the same global node calibration under the more explicit name.
- `calibrate_essvi_smooth(...)` is a deprecated compatibility helper; use `calibrate_essvi(...)` followed by `project_essvi_nodes(...)` instead.
- The eSSVI objects live under `option_pricing.vol` and `option_pricing.vol.ssvi`; they are not re-exported from the package root.