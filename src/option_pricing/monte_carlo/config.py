"""Monte Carlo run configuration."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

RngType = Literal["pcg64", "mt19937", "sobol"]


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
        Random number generator configuration (`RandomConfig`).
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
