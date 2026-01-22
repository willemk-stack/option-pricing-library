from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np

from ...types import PricingInputs
from ..grids import GridConfig, SpacingPolicy


class Coord(str, Enum):
    LOG_S = "logS"
    S = "S"


class DomainPolicy(str, Enum):
    MANUAL = "MANUAL"
    STRIKE_MULTIPLE = "STRIKE_MULTIPLE"
    LOG_NSIGMA = "LOG_NSIGMA"


@dataclass(frozen=True, slots=True)
class DomainConfig:
    policy: DomainPolicy

    # MANUAL
    x_lb: float | None = None
    x_ub: float | None = None

    # STRIKE_MULTIPLE
    multiple: float = 6.0

    # LOG_NSIGMA
    n_sigma: float = 6.0

    # If using LOG_S, we must keep S_min > 0
    s_min_floor: float | None = None

    # Optional: where to center clustered grids
    center: Literal["strike", "spot", "drifted_spot"] = "strike"

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


def compute_bounds(
    p: PricingInputs,
    *,
    coord: Coord,
    cfg: DomainConfig,
) -> DomainBounds:
    # --- basic validation
    S0 = float(p.S)
    K = float(p.K)
    sigma = float(p.sigma)
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
        if cfg.center == "spot":
            return xS
        if cfg.center == "drifted_spot":
            mu = float(np.log(S0) + (p.r - p.q - 0.5 * sigma**2) * tau)
            return float(mu) if coord == Coord.LOG_S else float(np.exp(mu))
        # default: "strike"
        return xK

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

        # Pad so strict inequalities (lo < x0 < hi) hold.
        width = hi - lo
        eps = 1e-9 * max(1.0, width)

        return lo - eps, hi + eps

    # --- helper: finalize DomainBounds consistently (apply floor, clamp x_center)
    def _finalize(x_lb: float, x_ub: float, x_center: float | None) -> DomainBounds:
        if not (x_lb < x_ub):
            raise ValueError("Computed invalid bounds (x_lb >= x_ub)")

        if coord == Coord.LOG_S:
            S_min = _apply_floor(_to_S(x_lb))
            # floor adjustment can move x_lb slightly upward
            x_lb = _to_x(S_min)
            S_max = _to_S(x_ub)
        else:
            S_min = float(x_lb)
            S_max = float(x_ub)

        # If clustered spacing, keep x_center inside [x_lb, x_ub] to avoid odd clustering behavior
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

        # Optional: for diagnostics you may want this ON; for now we keep manual strict.
        # Uncomment if you want manual to be "always safe":
        x_lb, x_ub = _ensure_contains_spot_and_strike(x_lb, x_ub)

        return _finalize(x_lb, x_ub, _x_center_raw())

    # --- STRIKE_MULTIPLE (already includes spot+strike in S-space by construction)
    if cfg.policy == DomainPolicy.STRIKE_MULTIPLE:
        m = float(cfg.multiple)
        if m <= 1.0:
            raise ValueError("STRIKE_MULTIPLE requires cfg.multiple > 1")

        S_ref_hi = max(S0, K)
        S_ref_lo = min(S0, K)

        S_max = m * S_ref_hi
        S_min = S_ref_lo / m

        # apply floor if log-space
        if coord == Coord.LOG_S:
            S_min = _apply_floor(S_min)
            x_lb = _to_x(S_min)
            x_ub = _to_x(S_max)
        else:
            x_lb, x_ub = float(S_min), float(S_max)

        # still enforce inclusion defensively (harmless here)
        x_lb, x_ub = _ensure_contains_spot_and_strike(x_lb, x_ub)

        return _finalize(x_lb, x_ub, _x_center_raw())

    # --- LOG_NSIGMA (robust: drift band + ensure contains spot+strike)
    if cfg.policy == DomainPolicy.LOG_NSIGMA:
        n = float(cfg.n_sigma)
        if n <= 0.0:
            raise ValueError("LOG_NSIGMA requires cfg.n_sigma > 0")
        if sigma <= 0.0:
            raise ValueError("LOG_NSIGMA requires sigma > 0")

        # drifted log-spot mean (risk-neutral)
        mu_log = float(np.log(S0) + (p.r - p.q - 0.5 * sigma**2) * tau)
        width = float(n * sigma * np.sqrt(tau))

        # band in log-space
        x_lb_log = mu_log - width
        x_ub_log = mu_log + width

        # map to requested coordinate
        if coord == Coord.LOG_S:
            # apply floor to S_min, then re-log
            S_min = _apply_floor(float(np.exp(x_lb_log)))
            x_lb = _to_x(S_min)
            x_ub = float(x_ub_log)  # log-space already
        else:
            # convert band to S, then use S as coordinate
            S_min = float(np.exp(x_lb_log))
            S_max = float(np.exp(x_ub_log))
            x_lb, x_ub = float(S_min), float(S_max)

        # crucial: ensure inclusion of spot and strike (prevents your x0-outside-domain error)
        x_lb, x_ub = _ensure_contains_spot_and_strike(x_lb, x_ub)

        return _finalize(x_lb, x_ub, _x_center_raw())

    raise ValueError(f"Unsupported DomainPolicy: {cfg.policy}")


def make_grid_config(
    p: PricingInputs,
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
