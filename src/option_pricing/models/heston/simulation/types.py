"""Typed configs and results for Heston simulation."""

from dataclasses import dataclass


@dataclass
class HestonMCConfig:
    n_paths: int
    n_steps: int
    seed: int | None = None
    antithetic: bool = False


@dataclass
class SimulationResult:
    spot_paths: object
    var_paths: object
    dt: float
