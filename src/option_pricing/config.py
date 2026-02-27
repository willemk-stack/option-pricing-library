from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

import numpy as np

from option_pricing.numerics.root_finding import RootMethod


class SeedStrategy(StrEnum):
    """Strategy for initializing root-finding seeds in implied volatility calculation.

    Attributes
    ----------
    USE_GUESS
        Use the provided initial guess (sigma0).
    HEURISTIC
        Compute initial guess using a heuristic method.
    LAST_SOLUTION
        Warm-start from the last solved volatility (if available).
    """

    USE_GUESS = "use_guess"
    HEURISTIC = "heuristic"
    LAST_SOLUTION = "last_solution"


RngType = Literal["pcg64", "mt19937", "sobol"]


@dataclass(frozen=True, slots=True)
class NumericsConfig:
    """Numerical solver configuration.

    Parameters controlling root-finding and other numerical algorithms.

    Attributes
    ----------
    abs_tol
        Absolute tolerance (default: 1e-10).
    rel_tol
        Relative tolerance (default: 1e-8).
    max_iter
        Maximum number of iterations (default: 100).
    eps
        Perturbation size for finite differences (default: 1e-12).
    min_vega
        Minimum vega threshold to avoid division issues (default: 1e-10).
    diagnostics
        Whether to collect solver diagnostics (default: False).
    """

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
    """Random number generator configuration.

    Attributes
    ----------
    seed
        Seed value for reproducibility (default: 0).
    rng_type
        RNG algorithm: 'pcg64', 'mt19937', or 'sobol' (default: 'pcg64').
    """

    seed: int = 0
    rng_type: RngType = "pcg64"


@dataclass(frozen=True, slots=True)
class ImpliedVolConfig:
    """Configuration for implied volatility (IV) solvers.

    Controls root-finding algorithm, bounds, seeding strategy, and numerical tolerances.

    Attributes
    ----------
    root_method
        Root-finding method from :class:`~option_pricing.numerics.root_finding.RootMethod`
        (default: BRACKETED_NEWTON).
    sigma_lo
        Lower volatility bound (default: 1e-8).
    sigma_hi
        Upper volatility bound (default: 5.0).
    bounds_eps
        Slack for bracket bounds (default: 1e-12).
    seed_strategy
        Initial guess strategy from :class:`SeedStrategy` (default: HEURISTIC).
    numerics
        Numerical solver configuration (:class:`NumericsConfig`).
    """

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


# Future: FiniteDiffGreeksConfig,
# MCConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class MCConfig:
    """Monte Carlo pricing configuration.

    Attributes
    ----------
    n_paths
        Number of Monte Carlo paths (default: 100,000).
    antithetic
        Use antithetic variates for variance reduction (default: False).
    random
        Random number generator configuration (:class:`RandomConfig`).
    rng
        Optional pre-constructed numpy RandomGenerator. If None, one is
        created from the random config.
    """

    n_paths: int = 100_000
    antithetic: bool = False
    random: RandomConfig = field(default_factory=RandomConfig)
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        if self.n_paths <= 0:
            raise ValueError("n_paths must be > 0")
