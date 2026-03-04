# PDE pricing

The library includes finite-difference PDE pricers for:

- European vanilla options under Black-Scholes
- European vanilla options under a local-vol surface
- digital options through lower-level PDE entry points

This guide focuses on the two most common vanilla workflows.

## Black-Scholes PDE pricing

The main entry point is `bs_price_pde_european`.

```python
from option_pricing.pricers.pde_pricer import bs_price_pde_european
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

price = bs_price_pde_european(
    p,
    coord=Coord.LOG_S,
    domain_cfg=domain_cfg,
    Nx=201,
    Nt=201,
    method="cn",
    advection=AdvectionScheme.CENTRAL,
)
```

## Compare PDE to the analytic Black-Scholes value

```python
from option_pricing import bs_price

pde = bs_price_pde_european(
    p,
    coord=Coord.LOG_S,
    domain_cfg=domain_cfg,
    Nx=201,
    Nt=201,
    method="cn",
)
analytic = bs_price(p)
print(pde, analytic, pde - analytic)
```

## Return the final numerical solution as well

```python
price, sol = bs_price_pde_european(
    p,
    coord=Coord.LOG_S,
    domain_cfg=domain_cfg,
    Nx=201,
    Nt=201,
    method="cn",
    return_solution=True,
)

print(sol.grid.x.shape)
print(sol.u_final.shape)
```

## Domain choices

The PDE wrapper uses `BSDomainConfig` to build a numerical domain.
A good default is:

- `policy=BSDomainPolicy.LOG_NSIGMA`
- `coord=Coord.LOG_S`
- clustered spacing around the strike

Manual domains are also possible if you know exactly what you want.

## Local-vol PDE pricing

If you already have a `LocalVolSurface`, use `local_vol_price_pde_european`.
The interface is intentionally very similar.

```python
from option_pricing.pricers.pde_pricer import local_vol_price_pde_european

price_lv = local_vol_price_pde_european(
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

That workflow is covered in more detail in [Local volatility](local_vol.md).

## Notes

- The PDE wrappers in this guide are for European vanilla pricing.
- `Nx` and `Nt` are the main resolution controls.
- Clustered grids are often a good idea when the payoff kink sits near the strike.
