"""Typed configs and results for Heston simulation."""

from dataclasses import dataclass
from typing import Literal

from ....typing import FloatArray

HestonScheme = Literal["euler_full_truncation"]


@dataclass(frozen=True, slots=True)
class HestonSimulationResult:
    spot_paths: FloatArray
    var_paths: FloatArray
    dt: float
