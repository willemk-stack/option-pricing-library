from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np

from ...instruments.vanilla import VanillaOption
from ...numerics.pde import LinearParabolicPDE1D
from ...numerics.pde.boundary import RobinBC, RobinBCSide
from ...numerics.pde.domain import Coord
from ...types import OptionType, PricingInputs
from ...typing import ArrayLike

type CoordFn = Callable[[ArrayLike], ArrayLike]


@dataclass(frozen=True, slots=True)
class BSPDEWiring:
    coord: Coord
    to_x: CoordFn
    to_S: CoordFn
    x_0: float
    problem: LinearParabolicPDE1D


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
    p: PricingInputs,
    coord: Coord | str,
    *,
    x_lb: float,
    x_ub: float,
) -> BSPDEWiring:
    coord = Coord(coord)

    sigma = float(p.sigma)
    r = float(p.r)
    q = float(p.q)

    if coord == Coord.LOG_S:
        to_x = cast(CoordFn, np.log)
        to_S = cast(CoordFn, np.exp)
        x0 = float(np.log(float(p.S)))

        if not np.isfinite(x_lb) or not np.isfinite(x_ub):
            raise ValueError("LOG_S bounds must be finite.")

        def a(x, tau):
            x = np.asarray(x)
            return 0.5 * sigma**2 + 0.0 * x

        def b(x, tau):
            x = np.asarray(x)
            return (r - q - 0.5 * sigma**2) + 0.0 * x

        def c(x, tau):
            x = np.asarray(x)
            return (-r) + 0.0 * x

    elif coord == Coord.S:
        to_x = cast(CoordFn, lambda z: z)
        to_S = cast(CoordFn, lambda z: z)
        x0 = float(p.S)

        def a(S, tau):
            return 0.5 * sigma**2 * (np.asarray(S) ** 2)

        def b(S, tau):
            return (r - q) * np.asarray(S)

        def c(S, tau):
            S = np.asarray(S)
            return (-r) + 0.0 * S

    else:
        raise ValueError(f"Unsupported coord type: {coord}")

    # Check in solver coordinates (x-space), not S-space.
    if not (x_lb < x0 < x_ub):
        raise ValueError(
            f"Domain bounds must contain x0 strictly inside (solver coords): "
            f"x_lb={x_lb}, x0={x0}, x_ub={x_ub}"
        )

    S_min = float(to_S(x_lb))
    S_max = float(to_S(x_ub))

    kind = p.spec.kind
    K = float(p.K)

    bc = _bc_constr(kind=kind, K=K, r=r, q=q, S_min=S_min, S_max=S_max)

    opt = VanillaOption(expiry=p.tau, strike=p.K, kind=kind)
    payoff = opt.payoff

    def ic(x):
        return payoff(to_S(x))

    problem = LinearParabolicPDE1D(a=a, b=b, c=c, bc=bc, ic=ic)

    return BSPDEWiring(coord=coord, to_x=to_x, to_S=to_S, x_0=x0, problem=problem)
