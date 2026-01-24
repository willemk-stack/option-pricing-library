from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from ..instruments.base import ExerciseStyle
from ..instruments.digital import DigitalOption
from ..market.curves import PricingContext
from ..numerics.grids import Grid, GridConfig
from ..numerics.pde import AdvectionScheme, PDESolution1D, solve_pde_1d
from ..numerics.pde.domain import Coord
from ..numerics.pde.ic_remedies import ICRemedy, ic_cell_average, ic_l2_projection
from ..types import DigitalSpec, MarketData, OptionSpec, PricingInputs
from .pde.digital_black_scholes import bs_pde_wiring as digital_bs_pde_wiring

# BS-specific domain selection (Option A)
from .pde.domain import BSDomainConfig, bs_compute_bounds

# IMPORTANT: use the correct wiring modules
from .pde.european_black_scholes import bs_pde_wiring as european_bs_pde_wiring

# Python 3.12 type aliases (optional)
type VanillaInputs = PricingInputs[OptionSpec]
type DigitalInputs = PricingInputs[DigitalSpec]
type ICTransform = Callable[[Grid, Callable[[float], float]], np.ndarray]


def bs_price_pde_european(
    p: VanillaInputs,
    *,
    coord: Coord | str = Coord.LOG_S,
    domain_cfg: BSDomainConfig,
    Nx: int = 400,
    Nt: int = 400,
    method: str = "cn",
    advection: AdvectionScheme = AdvectionScheme.CENTRAL,
    return_solution: bool = False,
) -> float | tuple[float, PDESolution1D]:
    coord = Coord(coord)

    # Option A: BS-specific bounds provider
    bounds = bs_compute_bounds(p, coord=coord, cfg=domain_cfg)

    wiring = european_bs_pde_wiring(p, coord, x_lb=bounds.x_lb, x_ub=bounds.x_ub)

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

    if x.size >= 2 and not np.all(np.diff(x) > 0):
        raise ValueError("Grid x must be strictly increasing for interpolation.")

    price = float(np.interp(wiring.x_0, x, u))
    return (price, sol) if return_solution else price


def bs_digital_price_pde_from_ctx(
    inst: DigitalOption,
    *,
    ctx: PricingContext,
    sigma: float,
    domain_cfg: BSDomainConfig,
    coord: Coord | str = Coord.LOG_S,
    Nx: int = 400,
    Nt: int = 400,
    method: str = "rannacher",
    advection: AdvectionScheme = AdvectionScheme.CENTRAL,
    return_solution: bool = False,
    ic_remedy: ICRemedy | str = ICRemedy.CELL_AVG,
    ic_transformFn: ICTransform | None = None,
) -> float | tuple[float, PDESolution1D]:
    """
    Core digital PDE pricer using curves-first inputs.

    Domain selection uses an effective flat (r,q) inferred from df and forward at tau.
    Pricing still uses the BS PDE wiring based on those inputs.
    """
    coord = Coord(coord)

    if inst.exercise != ExerciseStyle.EUROPEAN:
        raise ValueError("Digital PDE pricer currently supports European exercise only")

    tau = float(inst.expiry)
    if tau <= 0.0:
        raise ValueError("Need expiry > 0")

    df = float(ctx.df(tau))
    F = float(ctx.fwd(tau))
    S0 = float(ctx.spot)

    # effective flat rates for *bounds + wiring parameters*
    r_eff = -math.log(df) / tau
    carry = math.log(F / S0) / tau
    q_eff = r_eff - carry

    p_eff: DigitalInputs = PricingInputs(
        spec=DigitalSpec(
            kind=inst.kind,
            strike=float(inst.strike),
            expiry=tau,
            payout=float(inst.payout),
        ),
        market=MarketData(spot=S0, rate=r_eff, dividend_yield=q_eff),
        sigma=float(sigma),
        t=0.0,
    )

    bounds = bs_compute_bounds(p_eff, coord=coord, cfg=domain_cfg)
    wiring = digital_bs_pde_wiring(p_eff, coord, x_lb=bounds.x_lb, x_ub=bounds.x_ub)

    # --- IC remedy selection (override wins) ---
    ic_transform = ic_transformFn

    if ic_transform is None:
        remedy = ICRemedy(ic_remedy)  # allows passing "cell_avg" as a string

        if remedy == ICRemedy.NONE:
            ic_transform = None

        elif remedy == ICRemedy.CELL_AVG:
            # Discontinuity at strike in solver coordinates
            xK = float(np.asarray(wiring.to_x(p_eff.spec.strike)).reshape(()))

            def ic_transform(grid: Grid, ic: Callable[[float], float]) -> np.ndarray:
                return ic_cell_average(grid, ic, breakpoints=(xK,))

        elif remedy == ICRemedy.L2_PROJ:
            xK = float(np.asarray(wiring.to_x(p_eff.spec.strike)).reshape(()))

            def ic_transform(grid: Grid, ic: Callable[[float], float]) -> np.ndarray:
                return ic_l2_projection(grid=grid, ic=ic, breakpoints=(xK,))

        else:
            raise ValueError(f"Unsupported IC remedy: {remedy}")

    if not (bounds.x_lb < wiring.x_0 < bounds.x_ub):
        raise ValueError("Computed bounds do not contain x0 (spot).")

    grid_cfg = GridConfig(
        Nx=Nx,
        Nt=Nt,
        x_lb=bounds.x_lb,
        x_ub=bounds.x_ub,
        T=tau,
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
        ic_transform=ic_transform,
    )

    x = np.asarray(sol.grid.x, dtype=float)
    u = np.asarray(sol.u_final, dtype=float)

    if x.size >= 2 and not np.all(np.diff(x) > 0):
        raise ValueError("Grid x must be strictly increasing for interpolation.")

    price = float(np.interp(wiring.x_0, x, u))
    return (price, sol) if return_solution else price


def bs_digital_price_pde(
    p: DigitalInputs,
    *,
    coord: Coord | str = Coord.LOG_S,
    domain_cfg: BSDomainConfig,
    Nx: int = 400,
    Nt: int = 400,
    method: str = "rannacher",
    advection: AdvectionScheme = AdvectionScheme.CENTRAL,
    return_solution: bool = False,
    ic_remedy: ICRemedy | str = ICRemedy.CELL_AVG,
    ic_transformFn: ICTransform | None = None,
) -> float | tuple[float, PDESolution1D]:
    """
    PricingInputs wrapper (Option 1): DigitalSpec drives payout.
    """
    inst = DigitalOption(
        kind=p.spec.kind,
        strike=float(p.spec.strike),
        expiry=float(p.tau),  # instruments use tau
        payout=float(p.spec.payout),
        exercise=ExerciseStyle.EUROPEAN,
    )
    return bs_digital_price_pde_from_ctx(
        inst,
        ctx=p.ctx,
        sigma=p.sigma,
        domain_cfg=domain_cfg,
        coord=coord,
        Nx=Nx,
        Nt=Nt,
        method=method,
        advection=advection,
        return_solution=return_solution,
        ic_remedy=ic_remedy,
        ic_transformFn=ic_transformFn,
    )
