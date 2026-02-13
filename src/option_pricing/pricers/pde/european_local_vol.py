from __future__ import annotations

import numpy as np

from ...instruments.vanilla import VanillaOption
from ...models.black_scholes.pde import bs_coord_maps
from ...models.local_vol.pde import local_vol_pde_coeffs
from ...numerics.pde import LinearParabolicPDE1D
from ...numerics.pde.boundary import RobinBC, RobinBCSide
from ...numerics.pde.domain import Coord
from ...types import OptionSpec, OptionType, PricingInputs
from ...typing import ArrayLike, FloatArray
from ...vol.surface import LocalVolSurface
from .pde_wiring import PDEWiring1D


def _bc_constr(
    *, kind: OptionType, K: float, r: float, q: float, S_min: float, S_max: float
) -> RobinBC:
    """Vanilla far-field Dirichlet BCs expressed as Robin with beta=0."""
    if kind == OptionType.CALL:

        def left_gamma(tau: float) -> float:
            return 0.0

        def right_gamma(tau: float) -> float:
            return float(S_max * np.exp(-q * tau) - K * np.exp(-r * tau))

    elif kind == OptionType.PUT:

        def left_gamma(tau: float) -> float:
            return float(K * np.exp(-r * tau) - S_min * np.exp(-q * tau))

        def right_gamma(tau: float) -> float:
            return 0.0

    else:
        raise ValueError(f"Unsupported option type: {kind}")

    left = RobinBCSide(alpha=lambda tau: 1.0, beta=lambda tau: 0.0, gamma=left_gamma)
    right = RobinBCSide(alpha=lambda tau: 1.0, beta=lambda tau: 0.0, gamma=right_gamma)
    return RobinBC(left=left, right=right)


def local_vol_pde_wiring(
    p: PricingInputs[OptionSpec],
    lv: LocalVolSurface,
    coord: Coord | str,
    *,
    x_lb: float,
    x_ub: float,
    # guardrails forwarded to coeff builder
    tau_floor: float = 1e-8,
    sigma2_floor: float = 1e-14,
    sigma2_cap: float | None = None,
) -> PDEWiring1D:
    """Build a fully specified 1D local-vol PDE (coeffs + BC + terminal)."""

    coord = Coord(coord)

    r = float(p.market.rate)
    q = float(p.market.dividend_yield)

    to_x, to_S = bs_coord_maps(coord)
    x0 = float(np.asarray(to_x(p.S)).reshape(()))

    if not (x_lb < x0 < x_ub):
        raise ValueError(
            f"Domain bounds must contain x0 strictly inside (solver coords): "
            f"x_lb={x_lb}, x0={x0}, x_ub={x_ub}"
        )

    S_min = float(to_S(x_lb))
    S_max = float(to_S(x_ub))

    # LocalVolSurface is defined as sigma_loc(K,T). Under Dupire this is used as
    # sigma_loc(S,T) by plugging S into the K slot.
    def local_var(S: ArrayLike, tau: float) -> FloatArray:
        return lv.local_var(S, float(tau))

    coeffs = local_vol_pde_coeffs(
        coord=coord,
        local_var=local_var,
        r=r,
        q=q,
        tau_floor=tau_floor,
        sigma2_floor=sigma2_floor,
        sigma2_cap=sigma2_cap,
    )

    kind = p.spec.kind
    K = float(p.spec.strike)
    bc = _bc_constr(kind=kind, K=K, r=r, q=q, S_min=S_min, S_max=S_max)

    opt = VanillaOption(expiry=float(p.tau), strike=K, kind=kind)
    payoff = opt.payoff

    def ic(x: float) -> float:
        return float(payoff(float(to_S(x))))

    problem = LinearParabolicPDE1D(a=coeffs.a, b=coeffs.b, c=coeffs.c, bc=bc, ic=ic)

    return PDEWiring1D(coord=coord, to_x=to_x, to_S=to_S, x_0=x0, problem=problem)
