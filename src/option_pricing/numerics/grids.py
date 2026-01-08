# src/option_pricing/numerics/grids.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "SpacingPolicy",
    "GridConfig",
    "Grid",
    "build_x_grid",
    "build_time_grid",
    "build_grid",
    "const_bc",
]


class SpacingPolicy(str, Enum):
    UNIFORM = "uniform"
    CLUSTERED = "clustered"


@dataclass(frozen=True, slots=True)
class GridConfig:
    Nx: int
    Nt: int
    x_lb: float
    x_ub: float
    T: float
    spacing: SpacingPolicy = SpacingPolicy.UNIFORM
    x_center: float | None = None
    cluster_strength: float = 2.0

    def validate(self) -> None:
        if self.Nx < 3:
            raise ValueError("Nx must be >= 3")
        if self.Nt < 2:
            raise ValueError("Nt must be >= 2")
        if not (self.x_lb < self.x_ub):
            raise ValueError("Need x_lb < x_ub")
        if self.T <= 0:
            raise ValueError("T must be > 0")

        if self.spacing == SpacingPolicy.CLUSTERED:
            if self.x_center is None:
                raise ValueError("x_center required for clustered spacing")
            if not (self.x_lb <= self.x_center <= self.x_ub):
                raise ValueError("x_center must be within [x_lb, x_ub]")
            if self.cluster_strength <= 0:
                raise ValueError("cluster_strength must be > 0")


@dataclass(frozen=True, slots=True)
class Grid:
    t: NDArray[np.floating]
    x: NDArray[np.floating]


def _build_x_grid_validated(cfg: GridConfig) -> NDArray[np.floating]:
    if cfg.spacing == SpacingPolicy.UNIFORM:
        x = np.linspace(cfg.x_lb, cfg.x_ub, cfg.Nx, dtype=float)
    else:
        u = np.linspace(-1.0, 1.0, cfg.Nx, dtype=float)
        b = float(cfg.cluster_strength)

        raw = np.sinh(b * u)
        raw = raw / np.max(np.abs(raw))

        x = np.empty_like(raw, dtype=float)

        xc_opt = cfg.x_center
        assert xc_opt is not None
        xc = float(xc_opt)

        left_scale = xc - cfg.x_lb
        right_scale = cfg.x_ub - xc

        neg = raw <= 0.0
        pos = ~neg
        x[neg] = xc + raw[neg] * left_scale
        x[pos] = xc + raw[pos] * right_scale

        x[0] = cfg.x_lb
        x[-1] = cfg.x_ub

    if not np.all(np.diff(x) > 0):
        raise ValueError("x grid must be strictly increasing")
    return x


def _build_time_grid_validated(cfg: GridConfig) -> NDArray[np.floating]:
    return np.linspace(0.0, cfg.T, cfg.Nt, dtype=float)


def build_x_grid(cfg: GridConfig) -> NDArray[np.floating]:
    cfg.validate()
    return _build_x_grid_validated(cfg)


def build_time_grid(cfg: GridConfig) -> NDArray[np.floating]:
    cfg.validate()
    return _build_time_grid_validated(cfg)


def build_grid(cfg: GridConfig) -> Grid:
    cfg.validate()
    return Grid(t=_build_time_grid_validated(cfg), x=_build_x_grid_validated(cfg))


def const_bc(val: float) -> Callable[[float], float]:
    v = float(val)
    return lambda _t: v
