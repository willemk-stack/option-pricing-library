from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol

import numpy as np

from ...numerics.grids import GridConfig, SpacingPolicy
from ...numerics.pde.domain import (
    Coord,
    DomainBounds,
)
from ...numerics.pde.domain import (
    DomainConfig as GeomDomainConfig,
)
from ...numerics.pde.domain import (
    DomainPolicy as GeomDomainPolicy,
)
from ...numerics.pde.domain import (
    compute_bounds as geom_compute_bounds,
)


# ---- Protocols: keep numerics independent; BS domain needs drift inputs
class _HasMarket(Protocol):
    @property
    def rate(self) -> float: ...
    @property
    def dividend_yield(self) -> float: ...


class BSInputs(Protocol):
    @property
    def S(self) -> float: ...
    @property
    def K(self) -> float: ...
    @property
    def sigma(self) -> float: ...
    @property
    def tau(self) -> float: ...
    @property
    def market(self) -> _HasMarket: ...


class BSDomainPolicy(str, Enum):
    MANUAL = "MANUAL"
    STRIKE_MULTIPLE = "STRIKE_MULTIPLE"
    LOG_NSIGMA = "LOG_NSIGMA"


@dataclass(frozen=True, slots=True)
class BSDomainConfig:
    policy: BSDomainPolicy

    # MANUAL
    x_lb: float | None = None
    x_ub: float | None = None

    # STRIKE_MULTIPLE
    multiple: float = 6.0

    # LOG_NSIGMA
    n_sigma: float = 6.0

    # If using LOG_S, we must keep S_min > 0
    s_min_floor: float | None = None

    # Optional: where to center clustered grids (BS adds drifted_spot)
    center: Literal["strike", "spot", "drifted_spot"] = "strike"

    # Grid preferences
    spacing: SpacingPolicy = SpacingPolicy.UNIFORM
    cluster_strength: float = 2.0


def _to_geom_cfg(cfg: BSDomainConfig) -> GeomDomainConfig:
    """
    Map BS domain config to the geometry-only numerics DomainConfig.
    Only valid for MANUAL and STRIKE_MULTIPLE policies.
    """
    if cfg.policy == BSDomainPolicy.MANUAL:
        pol = GeomDomainPolicy.MANUAL
    elif cfg.policy == BSDomainPolicy.STRIKE_MULTIPLE:
        pol = GeomDomainPolicy.STRIKE_MULTIPLE
    else:
        raise ValueError("LOG_NSIGMA is BS-specific and must not be delegated.")

    if cfg.center == "drifted_spot":
        # drifted_spot only makes sense with LOG_NSIGMA; keep behavior explicit
        raise ValueError("center='drifted_spot' is only supported for LOG_NSIGMA.")

    return GeomDomainConfig(
        policy=pol,
        x_lb=cfg.x_lb,
        x_ub=cfg.x_ub,
        multiple=cfg.multiple,
        s_min_floor=cfg.s_min_floor,
        center=cfg.center,  # 'spot' or 'strike'
        spacing=cfg.spacing,
        cluster_strength=cfg.cluster_strength,
    )


def bs_compute_bounds(
    p: BSInputs,
    *,
    coord: Coord | str,
    cfg: BSDomainConfig,
) -> DomainBounds:
    """
    Black-Scholes domain bounds provider.

    - MANUAL, STRIKE_MULTIPLE: delegate to geometry-only numerics.compute_bounds
    - LOG_NSIGMA: BS/lognormal band computed here
    """
    coord = Coord(coord)

    # Delegate geometry-only policies
    if cfg.policy in (BSDomainPolicy.MANUAL, BSDomainPolicy.STRIKE_MULTIPLE):
        geom_cfg = _to_geom_cfg(cfg)
        return geom_compute_bounds(p, coord=coord, cfg=geom_cfg)

    # BS-only policy
    if cfg.policy != BSDomainPolicy.LOG_NSIGMA:
        raise ValueError(f"Unsupported BSDomainPolicy: {cfg.policy}")

    # --- inputs
    S0 = float(p.S)
    K = float(p.K)
    sigma = float(p.sigma)
    tau = float(p.tau)

    if S0 <= 0.0:
        raise ValueError("Spot must be > 0")
    if K <= 0.0:
        raise ValueError("Strike must be > 0")
    if sigma <= 0.0:
        raise ValueError("LOG_NSIGMA requires sigma > 0")
    if tau <= 0.0:
        raise ValueError("tau must be > 0")

    r = float(p.market.rate)
    q = float(p.market.dividend_yield)

    # --- coordinate transforms
    def _to_x(S: float) -> float:
        return float(np.log(S)) if coord == Coord.LOG_S else float(S)

    def _to_S(x: float) -> float:
        return float(np.exp(x)) if coord == Coord.LOG_S else float(x)

    xS = _to_x(S0)
    xK = _to_x(K)

    # --- pick x_center (BS adds drifted_spot)
    def _x_center_raw() -> float:
        if cfg.center == "spot":
            return xS
        if cfg.center == "drifted_spot":
            mu_log = float(np.log(S0) + (r - q - 0.5 * sigma**2) * tau)
            return float(mu_log) if coord == Coord.LOG_S else float(np.exp(mu_log))
        return xK  # "strike"

    # --- floor for LOG_S
    def _apply_floor(S_min: float) -> float:
        if coord != Coord.LOG_S:
            return float(S_min)

        floor = cfg.s_min_floor
        if floor is None:
            floor = 1e-12 * max(S0, K)

        if floor <= 0.0:
            raise ValueError("s_min_floor must be > 0 when using LOG_S")

        return float(max(S_min, floor))

    # --- ensure domain contains spot and strike
    def _ensure_contains_spot_and_strike(
        x_lb: float, x_ub: float
    ) -> tuple[float, float]:
        lo = float(min(x_lb, xS, xK))
        hi = float(max(x_ub, xS, xK))
        width = hi - lo
        eps = 1e-9 * max(1.0, width)
        return lo - eps, hi + eps

    # --- finalize bounds
    def _finalize(x_lb: float, x_ub: float, x_center: float | None) -> DomainBounds:
        if not (x_lb < x_ub):
            raise ValueError("Computed invalid bounds (x_lb >= x_ub)")

        if coord == Coord.LOG_S:
            S_min = _apply_floor(_to_S(x_lb))
            x_lb = _to_x(S_min)
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

    # --- BS LOG_NSIGMA band (defined in log-space)
    n = float(cfg.n_sigma)
    if n <= 0.0:
        raise ValueError("LOG_NSIGMA requires cfg.n_sigma > 0")

    mu_log = float(np.log(S0) + (r - q - 0.5 * sigma**2) * tau)
    width = float(n * sigma * np.sqrt(tau))

    x_lb_log = mu_log - width
    x_ub_log = mu_log + width

    # map band to chosen coordinate
    if coord == Coord.LOG_S:
        S_min = _apply_floor(float(np.exp(x_lb_log)))
        x_lb = _to_x(S_min)
        x_ub = float(x_ub_log)
    else:
        S_min = float(np.exp(x_lb_log))
        S_max = float(np.exp(x_ub_log))
        x_lb, x_ub = float(S_min), float(S_max)

    x_lb, x_ub = _ensure_contains_spot_and_strike(x_lb, x_ub)
    return _finalize(x_lb, x_ub, _x_center_raw())


def bs_make_grid_config(
    p: BSInputs,
    *,
    coord: Coord | str,
    dom: BSDomainConfig,
    Nx: int,
    Nt: int,
) -> GridConfig:
    b = bs_compute_bounds(p, coord=coord, cfg=dom)

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
