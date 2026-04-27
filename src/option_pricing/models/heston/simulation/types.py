"""Typed configs and results for Heston simulation."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from ....typing import FloatArray

HestonScheme = Literal["euler_full_truncation", "quadratic_exponential"]


@dataclass(frozen=True, slots=True)
class HestonSimulationResult:
    spot_paths: FloatArray
    var_paths: FloatArray
    dt: float
    metadata: Mapping[str, float] = field(default_factory=dict)
