from __future__ import annotations

import numpy as np

from ...instruments.digital import DigitalOption

# update to your final location/name for BS PDE helpers
from ...models.black_scholes.pde import bs_coord_maps, bs_pde_coeffs
from ...numerics.pde import LinearParabolicPDE1D
from ...numerics.pde.boundary import RobinBC, RobinBCSide
from ...numerics.pde.domain import Coord
from ...types import DigitalSpec, OptionType, PricingInputs
from .pde_wiring import BSPDEWiring1D


def _bc_constr_digital(*, kind: OptionType, r: float, Q: float) -> RobinBC:
    """
    Digital option far-field Dirichlet boundaries (expressed as Robin with beta=0).

    For large S:
      - digital call tends to payout * df = Q * exp(-r*tau)
      - digital put tends to 0
    For small S:
      - digital put tends to Q * exp(-r*tau)
      - digital call tends to 0
    """
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


def bs_pde_wiring(
    p: PricingInputs[DigitalSpec],
    coord: Coord | str,
    *,
    x_lb: float,
    x_ub: float,
) -> BSPDEWiring1D:
    coord = Coord(coord)

    sigma = float(p.sigma)
    r = float(p.market.rate)
    q = float(p.market.dividend_yield)

    to_x, to_S = bs_coord_maps(coord)
    x0 = float(np.asarray(to_x(p.S)).reshape(()))

    coeffs = bs_pde_coeffs(coord=coord, sigma=sigma, r=r, q=q)
    a, b, c = coeffs.a, coeffs.b, coeffs.c

    if not (x_lb < x0 < x_ub):
        raise ValueError(
            f"Domain bounds must contain x0 strictly inside (solver coords): "
            f"x_lb={x_lb}, x0={x0}, x_ub={x_ub}"
        )

    kind = p.spec.kind
    K = float(p.spec.strike)
    Q = float(p.spec.payout)

    # Digital far-field boundaries only need r and payout
    bc = _bc_constr_digital(kind=kind, r=r, Q=Q)

    opt = DigitalOption(expiry=float(p.tau), strike=K, payout=Q, kind=kind)
    payoff = opt.payoff

    def ic(x: float) -> float:
        return float(payoff(to_S(x)))

    problem = LinearParabolicPDE1D(a=a, b=b, c=c, bc=bc, ic=ic)

    return BSPDEWiring1D(coord=coord, to_x=to_x, to_S=to_S, x_0=x0, problem=problem)
