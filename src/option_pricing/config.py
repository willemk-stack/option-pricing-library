from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from option_pricing.numerics.root_finding import RootMethod


class SeedStrategy(str, Enum):
    USE_GUESS = "use_guess"  # use provided sigma0
    HEURISTIC = "heuristic"  # compute initial guess
    LAST_SOLUTION = "last_solution"  # warm start from last solved point


RngType = Literal["pcg64", "mt19937", "sobol"]


@dataclass(frozen=True, slots=True)
class NumericsConfig:
    abs_tol: float = 1e-10
    rel_tol: float = 1e-8
    max_iter: int = 100
    eps: float = 1e-12
    min_vega: float = 1e-10
    diagnostics: bool = False

    def __post_init__(self) -> None:
        if self.abs_tol <= 0 or self.rel_tol <= 0:
            raise ValueError("abs_tol and rel_tol must be > 0")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if self.eps <= 0:
            raise ValueError("eps must be > 0")
        if self.min_vega <= 0:
            raise ValueError("min_vega must be > 0")


@dataclass(frozen=True, slots=True)
class RandomConfig:
    seed: int = 0
    rng_type: RngType = "pcg64"


@dataclass(frozen=True, slots=True)
class ImpliedVolConfig:
    root_method: RootMethod = RootMethod.BRACKETED_NEWTON
    sigma_lo: float = 1e-8
    sigma_hi: float = 5.0
    bounds_eps: float = 1e-12
    seed_strategy: SeedStrategy = SeedStrategy.HEURISTIC
    numerics: NumericsConfig = field(default_factory=NumericsConfig)

    def __post_init__(self) -> None:
        if self.sigma_lo <= 0 or self.sigma_hi <= 0:
            raise ValueError("sigma bounds must be > 0")
        if self.sigma_lo >= self.sigma_hi:
            raise ValueError("sigma_lo must be < sigma_hi")
        if self.bounds_eps < 0:
            raise ValueError("bounds_eps must be >= 0")


# Future: FiniteDiffGreeksConfig, MCConfig
