from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from option_pricing.typing import ScalarFn


class BoundaryCondition(Protocol):
    def left(self, tau: float) -> float: ...
    def right(self, tau: float) -> float: ...


@dataclass(frozen=True)
class DirichletBC:
    left: ScalarFn  # boundary value at x_min for time tau
    right: ScalarFn  # boundary value at x_max for time tau
