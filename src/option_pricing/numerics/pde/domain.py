from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol

import numpy as np

from ..grids import GridConfig, SpacingPolicy


class Coord(str, Enum):
    LOG_S = "logS"
    S = "S"


class DomainPolicy(str, Enum):
    MANUAL = "MANUAL"
    STRIKE_MULTIPLE = "STRIKE_MULTIPLE"


@dataclass(frozen=True, slots=True)
class DomainConfig:
    policy: DomainPolicy

    # MANUAL
    x_lb: float | None = None
    x_ub: float | None = None

    # STRIKE_MULTIPLE
    multiple: float = 6.0

    # If using LOG_S, we must keep S_min > 0
    s_min_floor: float | None = None

    # Optional: where to center clustered grids (geometry-only)
    center: Literal["strike", "spot"] = "strike"

    # Grid preferences
    spacing: SpacingPolicy = SpacingPolicy.UNIFORM
    cluster_strength: float = 2.0


@dataclass(frozen=True, slots=True)
class DomainBounds:
    x_lb: float
    x_ub: float
    x_center: float | None
    S_min: float
    S_max: float


class DomainInputs(Protocol):
    # minimal surface needed by this module
    @property
    def S(self) -> float: ...
    @property
    def K(self) -> float: ...
    @property
    def tau(self) -> float: ...


def compute_bounds(
    p: DomainInputs,
    *,
    coord: Coord,
    cfg: DomainConfig,
) -> DomainBounds:
    # --- basic validation
    S0 = float(p.S)
    K = float(p.K)
    tau = float(p.tau)

    if S0 <= 0.0:
        raise ValueError("Spot must be > 0")
    if K <= 0.0:
        raise ValueError("Strike must be > 0")
    if tau <= 0.0:
        raise ValueError("tau must be > 0")

    # --- coordinate transforms
    def _to_x(S: float) -> float:
        return float(np.log(S)) if coord == Coord.LOG_S else float(S)

    def _to_S(x: float) -> float:
        return float(np.exp(x)) if coord == Coord.LOG_S else float(x)

    xS = _to_x(S0)
    xK = _to_x(K)

    # --- pick x_center (optional)
    def _x_center_raw() -> float:
        return xS if cfg.center == "spot" else xK  # default: strike

    # --- ensure positive S_min for LOG_S
    def _apply_floor(S_min: float) -> float:
        if coord != Coord.LOG_S:
            return float(S_min)

        floor = cfg.s_min_floor
        if floor is None:
            floor = 1e-12 * max(S0, K)

        if floor <= 0.0:
            raise ValueError("s_min_floor must be > 0 when using LOG_S")

        return float(max(S_min, floor))

    # --- helper: ensure domain contains spot and strike in solver coordinates
    def _ensure_contains_spot_and_strike(
        x_lb: float, x_ub: float
    ) -> tuple[float, float]:
        lo = float(min(x_lb, xS, xK))
        hi = float(max(x_ub, xS, xK))

        width = hi - lo
        eps = 1e-9 * max(1.0, width)

        return lo - eps, hi + eps

    # --- helper: finalize DomainBounds consistently (apply floor, clamp x_center)
    def _finalize(x_lb: float, x_ub: float, x_center: float | None) -> DomainBounds:
        if not (x_lb < x_ub):
            raise ValueError("Computed invalid bounds (x_lb >= x_ub)")

        if coord == Coord.LOG_S:
            S_min = _apply_floor(_to_S(x_lb))
            x_lb = _to_x(S_min)  # floor adjustment can move x_lb upward
            S_max = _to_S(x_ub)
        else:
            S_min = float(x_lb)
            S_max = float(x_ub)

        if x_center is not None:
            x_center = float(np.clip(float(x_center), x_lb, x_ub))

        return DomainBounds(
            x_lb=float(x_lb),
            x_ub=float(x_ub),
            x_center=x_center,
            S_min=float(S_min),
            S_max=float(S_max),
        )

    # --- MANUAL
    if cfg.policy == DomainPolicy.MANUAL:
        if cfg.x_lb is None or cfg.x_ub is None:
            raise ValueError("MANUAL policy requires cfg.x_lb and cfg.x_ub")

        x_lb = float(cfg.x_lb)
        x_ub = float(cfg.x_ub)

        # Make manual "safe" by ensuring spot/strike inclusion
        x_lb, x_ub = _ensure_contains_spot_and_strike(x_lb, x_ub)

        return _finalize(x_lb, x_ub, _x_center_raw())

    # --- STRIKE_MULTIPLE (geometry-only)
    if cfg.policy == DomainPolicy.STRIKE_MULTIPLE:
        m = float(cfg.multiple)
        if m <= 1.0:
            raise ValueError("STRIKE_MULTIPLE requires cfg.multiple > 1")

        S_ref_hi = max(S0, K)
        S_ref_lo = min(S0, K)

        S_max = m * S_ref_hi
        S_min = S_ref_lo / m

        if coord == Coord.LOG_S:
            S_min = _apply_floor(S_min)
            x_lb = _to_x(S_min)
            x_ub = _to_x(S_max)
        else:
            x_lb, x_ub = float(S_min), float(S_max)

        # Enforce inclusion defensively
        x_lb, x_ub = _ensure_contains_spot_and_strike(x_lb, x_ub)

        return _finalize(x_lb, x_ub, _x_center_raw())

    raise ValueError(f"Unsupported DomainPolicy: {cfg.policy}")


def make_grid_config(
    p: DomainInputs,
    *,
    coord: Coord | str,
    dom: DomainConfig,
    Nx: int,
    Nt: int,
) -> GridConfig:
    coord_ = Coord(coord)
    b = compute_bounds(p, coord=coord_, cfg=dom)

    return GridConfig(
        Nx=int(Nx),
        Nt=int(Nt),
        x_lb=float(b.x_lb),
        x_ub=float(b.x_ub),
        T=float(p.tau),
        spacing=dom.spacing,
        x_center=b.x_center if dom.spacing == SpacingPolicy.CLUSTERED else None,
        cluster_strength=float(dom.cluster_strength),
    )
