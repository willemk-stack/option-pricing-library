from __future__ import annotations

import numpy as np

from ...instruments.digital import DigitalOption
from ...models.black_scholes.pde import bs_coord_maps
from ...models.local_vol.pde import local_vol_pde_coeffs
from ...numerics.pde import LinearParabolicPDE1D
from ...numerics.pde.boundary import RobinBC, RobinBCSide
from ...numerics.pde.domain import Coord
from ...types import DigitalSpec, OptionType, PricingInputs
from ...vol.surface import LocalVolSurface
from .pde_wiring import PDEWiring1D


def _bc_constr_digital(*, kind: OptionType, r: float, Q: float) -> RobinBC:
    if kind == OptionType.CALL:

        def left_gamma(tau: float) -> float:
            return 0.0

        def right_gamma(tau: float) -> float:
            return float(Q * np.exp(-r * tau))

    elif kind == OptionType.PUT:

        def left_gamma(tau: float) -> float:
            return float(Q * np.exp(-r * tau))

        def right_gamma(tau: float) -> float:
            return 0.0

    else:
        raise ValueError(f"Unsupported option type: {kind}")

    left = RobinBCSide(alpha=lambda tau: 1.0, beta=lambda tau: 0.0, gamma=left_gamma)
    right = RobinBCSide(alpha=lambda tau: 1.0, beta=lambda tau: 0.0, gamma=right_gamma)
    return RobinBC(left=left, right=right)


def local_vol_pde_wiring(
    p: PricingInputs[DigitalSpec],
    lv: LocalVolSurface,
    coord: Coord | str,
    *,
    x_lb: float,
    x_ub: float,
    tau_floor: float = 1e-8,
    sigma2_floor: float = 1e-14,
    sigma2_cap: float | None = None,
) -> PDEWiring1D:
    coord = Coord(coord)

    r = float(p.market.rate)
    q = float(p.market.dividend_yield)
    to_x, to_S = bs_coord_maps(coord)

    x0 = float(np.asarray(to_x(p.S)).reshape(()))
    if not (x_lb < x0 < x_ub):
        raise ValueError(f"Need x_lb < x0 < x_ub, got {x_lb=}, {x0=}, {x_ub=}")

    kind = p.spec.kind
    K = float(p.spec.strike)
    Q = float(p.spec.payout)

    bc = _bc_constr_digital(kind=kind, r=r, Q=Q)

    coeffs = local_vol_pde_coeffs(
        coord=coord,
        local_var=lv.local_var,
        r=r,
        q=q,
        tau_floor=tau_floor,
        sigma2_floor=sigma2_floor,
        sigma2_cap=sigma2_cap,
    )

    opt = DigitalOption(expiry=float(p.tau), strike=K, payout=Q, kind=kind)
    payoff = opt.payoff

    def ic(x: float) -> float:
        return float(payoff(to_S(x)))

    problem = LinearParabolicPDE1D(a=coeffs.a, b=coeffs.b, c=coeffs.c, bc=bc, ic=ic)
    return PDEWiring1D(coord=coord, to_x=to_x, to_S=to_S, x_0=x0, problem=problem)
