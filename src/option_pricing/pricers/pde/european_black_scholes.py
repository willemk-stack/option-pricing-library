from __future__ import annotations

import numpy as np

from ...instruments.vanilla import VanillaOption

# NOTE: update this import to your final location/name for BS PDE coeff helpers
# e.g. option_pricing/models/black_scholes/pde_coeffs.py
from ...models.black_scholes.pde import bs_coord_maps, bs_pde_coeffs
from ...numerics.pde import LinearParabolicPDE1D
from ...numerics.pde.boundary import RobinBC, RobinBCSide
from ...numerics.pde.domain import Coord
from ...types import OptionSpec, OptionType, PricingInputs
from .pde_wiring import BSPDEWiring1D


def _bc_constr(
    *, kind: OptionType, K: float, r: float, q: float, S_min: float, S_max: float
) -> RobinBC:
    """
    Dirichlet boundary values expressed as Robin with beta=0:
        u = gamma(tau)
    i.e. alpha=1, beta=0, gamma=g(tau)
    """
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


def bs_pde_wiring(
    p: PricingInputs[OptionSpec],  # works for vanilla-style inputs
    coord: Coord | str,
    *,
    x_lb: float,
    x_ub: float,
) -> BSPDEWiring1D:
    """
    Build the fully specified PDE problem (coeffs + BC + terminal condition),
    in solver coordinates (x).
    """
    coord = Coord(coord)

    sigma = float(p.sigma)
    r = float(p.market.rate)
    q = float(p.market.dividend_yield)

    to_x, to_S = bs_coord_maps(coord)
    x0 = float(np.asarray(to_x(p.S)).reshape(()))

    # BS PDE coefficients (model-specific, product-agnostic)
    coeffs = bs_pde_coeffs(coord=coord, sigma=sigma, r=r, q=q)
    a, b, c = coeffs.a, coeffs.b, coeffs.c

    # Validate bounds in solver coordinates
    if not (x_lb < x0 < x_ub):
        raise ValueError(
            f"Domain bounds must contain x0 strictly inside (solver coords): "
            f"x_lb={x_lb}, x0={x0}, x_ub={x_ub}"
        )

    S_min = float(to_S(x_lb))
    S_max = float(to_S(x_ub))

    kind = p.spec.kind
    K = float(p.spec.strike)

    # Product-specific BCs
    bc = _bc_constr(kind=kind, K=K, r=r, q=q, S_min=S_min, S_max=S_max)

    # Terminal condition from instrument payoff (tau-based expiry convention)
    opt = VanillaOption(expiry=float(p.tau), strike=K, kind=kind)
    payoff = opt.payoff

    def ic(x: float) -> float:
        return float(payoff(to_S(x)))

    problem = LinearParabolicPDE1D(a=a, b=b, c=c, bc=bc, ic=ic)

    return BSPDEWiring1D(coord=coord, to_x=to_x, to_S=to_S, x_0=x0, problem=problem)
