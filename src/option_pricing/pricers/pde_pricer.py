import numpy as np

from option_pricing import PricingInputs
from option_pricing.numerics.grids import GridConfig
from option_pricing.numerics.pde import AdvectionScheme, PDESolution1D, solve_pde_1d
from option_pricing.pricers.pde.black_scholes import bs_pde_wiring
from option_pricing.pricers.pde.domain import Coord, DomainConfig, compute_bounds


def bs_price_pde(
    p: PricingInputs,
    *,
    coord: Coord | str = Coord.LOG_S,
    domain_cfg: DomainConfig,
    Nx: int = 400,
    Nt: int = 400,
    method: str = "cn",
    advection: AdvectionScheme = AdvectionScheme.CENTRAL,
    return_solution: bool = False,
) -> float | tuple[float, PDESolution1D]:
    coord = Coord(coord)

    bounds = compute_bounds(p, coord=coord, cfg=domain_cfg)
    wiring = bs_pde_wiring(p, coord, x_lb=bounds.x_lb, x_ub=bounds.x_ub)

    if not (bounds.x_lb < wiring.x_0 < bounds.x_ub):
        raise ValueError("Computed bounds do not contain x0 (spot).")

    grid_cfg = GridConfig(
        Nx=Nx,
        Nt=Nt,
        x_lb=bounds.x_lb,
        x_ub=bounds.x_ub,
        T=p.tau,
        spacing=domain_cfg.spacing,
        x_center=bounds.x_center,
        cluster_strength=domain_cfg.cluster_strength,
    )

    sol = solve_pde_1d(
        wiring.problem,
        grid_cfg=grid_cfg,
        method=method,
        advection=advection,
        store="final",
    )

    x = np.asarray(sol.grid.x, dtype=float)
    u = np.asarray(sol.u_final, dtype=float)

    # (Optional safety) ensure monotonic x for interp
    if x.size >= 2 and not np.all(np.diff(x) > 0):
        raise ValueError("Grid x must be strictly increasing for interpolation.")

    price = float(np.interp(wiring.x_0, x, u))
    return (price, sol) if return_solution else price
