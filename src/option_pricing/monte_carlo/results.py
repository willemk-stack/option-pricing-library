"""Result containers and diagnostics for Monte Carlo runs."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

from ..typing import FloatArray


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    level: float
    low: float
    high: float
    half_width: float


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    price: float
    stderr: float
    n_paths: int
    effective_n: int
    discount: float
    sample_mean: float
    sample_std: float | None = None
    seed: int | None = None
    antithetic: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_tuple(self) -> tuple[float, float]:
        return self.price, self.stderr

    def confidence_interval(self, level: float = 0.95) -> ConfidenceInterval:
        """Return a normal-approximation confidence interval for the MC price."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must be between 0 and 1")

        alpha = 1.0 - float(level)
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        half_width = float(z * self.stderr)

        return ConfidenceInterval(
            level=float(level),
            low=float(self.price - half_width),
            high=float(self.price + half_width),
            half_width=half_width,
        )


def monte_carlo_result_from_samples(
    samples: FloatArray,
    *,
    discount: float,
    n_paths: int,
    antithetic: bool,
    seed: int | None = None,
    metadata: Mapping[str, object] | None = None,
) -> MonteCarloResult:
    x = np.asarray(samples, dtype=float)

    if x.ndim != 1:
        raise ValueError("samples must be one-dimensional")
    if len(x) == 0:
        raise ValueError("at least one sample is required")

    sample_mean = float(np.mean(x))
    sample_std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    effective_n = len(x)

    return MonteCarloResult(
        price=float(discount) * sample_mean,
        stderr=float(discount) * sample_std / float(np.sqrt(effective_n)),
        n_paths=int(n_paths),
        effective_n=effective_n,
        discount=float(discount),
        sample_mean=sample_mean,
        sample_std=sample_std,
        seed=seed,
        antithetic=bool(antithetic),
        metadata={} if metadata is None else dict(metadata),
    )
