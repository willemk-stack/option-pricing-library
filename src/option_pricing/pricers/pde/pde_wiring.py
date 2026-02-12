from collections.abc import Callable
from dataclasses import dataclass

from ...numerics.pde import LinearParabolicPDE1D
from ...numerics.pde.domain import Coord
from ...typing import ArrayLike

type CoordFn = Callable[[ArrayLike], ArrayLike]


@dataclass(frozen=True, slots=True)
class PDEWiring1D:
    coord: Coord
    to_x: CoordFn
    to_S: CoordFn
    x_0: float
    problem: LinearParabolicPDE1D
