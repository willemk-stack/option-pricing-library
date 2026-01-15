from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import cast

import numpy as np

from option_pricing import OptionType, PricingInputs, VanillaOption
from option_pricing.numerics.pde import LinearParabolicPDE1D
from option_pricing.numerics.pde.boundary import DirichletBC
from option_pricing.typing import ArrayLike

type CoordFn = Callable[[ArrayLike], ArrayLike]


class Coord(str, Enum):
    LOG_S = "logS"
    S = auto()


@dataclass(frozen=True, slots=True)
class BSPDEWiring:
    coord: Coord
    to_x: CoordFn
    to_S: CoordFn
    x_0: float
    problem: LinearParabolicPDE1D


def _bc_constr(
    *, kind: OptionType, K: float, r: float, q: float, S_min: float, S_max: float
) -> DirichletBC:
    if kind == OptionType.CALL:

        def left(tau: float) -> float:
            return 0.0

        def right(tau: float) -> float:
            return float(S_max * np.exp(-q * tau) - K * np.exp(-r * tau))

    elif kind == OptionType.PUT:

        def left(tau: float) -> float:
            return float(K * np.exp(-r * tau) - S_min * np.exp(-q * tau))

        def right(tau: float) -> float:
            return 0.0

    else:
        raise ValueError(f"Unsupported option type: {kind}")

    return DirichletBC(left=left, right=right)


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

    S_min = float(to_S(x_lb))
    S_max = float(to_S(x_ub))

    kind = p.spec.kind
    K = float(p.K)

    bc = _bc_constr(kind=kind, K=K, r=r, q=q, S_min=S_min, S_max=S_max)

    opt = VanillaOption(expiry=p.tau, strike=p.K, kind=kind)
    payoff = opt.payoff

    def ic(x):
        return payoff(to_S(x))  # works for scalars/arrays if payoff does

    problem = LinearParabolicPDE1D(a=a, b=b, c=c, bc=bc, ic=ic)

    return BSPDEWiring(coord=coord, to_x=to_x, to_S=to_S, x_0=x0, problem=problem)
