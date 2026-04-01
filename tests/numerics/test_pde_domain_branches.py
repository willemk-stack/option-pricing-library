from dataclasses import dataclass

import numpy as np
import pytest

from option_pricing.numerics.grids import SpacingPolicy
from option_pricing.numerics.pde.domain import (
    Coord,
    DomainConfig,
    DomainPolicy,
    compute_bounds,
    make_grid_config,
)


@dataclass(frozen=True)
class _Inputs:
    S: float
    K: float
    tau: float


@pytest.mark.parametrize(
    "S,K,tau",
    [
        (0.0, 100.0, 1.0),
        (100.0, 0.0, 1.0),
        (100.0, 100.0, 0.0),
    ],
)
def test_compute_bounds_basic_validation(S: float, K: float, tau: float):
    p = _Inputs(S=S, K=K, tau=tau)
    cfg = DomainConfig(policy=DomainPolicy.MANUAL, x_lb=80.0, x_ub=120.0)
    with pytest.raises(ValueError):
        compute_bounds(p, coord=Coord.S, cfg=cfg)


def test_manual_bounds_include_spot_and_strike():
    p = _Inputs(S=100.0, K=110.0, tau=1.0)
    cfg = DomainConfig(policy=DomainPolicy.MANUAL, x_lb=95.0, x_ub=105.0)

    bounds = compute_bounds(p, coord=Coord.S, cfg=cfg)

    assert bounds.S_min <= min(p.S, p.K)
    assert bounds.S_max >= max(p.S, p.K)
    assert bounds.x_center is not None
    assert bounds.x_lb < bounds.x_ub
    assert bounds.x_lb <= bounds.x_center <= bounds.x_ub


def test_log_coord_floor_and_s_min_floor_validation():
    p = _Inputs(S=100.0, K=100.0, tau=1.0)
    cfg_bad = DomainConfig(
        policy=DomainPolicy.MANUAL,
        x_lb=np.log(1e-8),
        x_ub=np.log(200.0),
        s_min_floor=0.0,
    )
    with pytest.raises(ValueError):
        compute_bounds(p, coord=Coord.LOG_S, cfg=cfg_bad)

    cfg_floor = DomainConfig(
        policy=DomainPolicy.MANUAL,
        x_lb=np.log(1e-8),
        x_ub=np.log(200.0),
        s_min_floor=90.0,
    )
    bounds = compute_bounds(p, coord=Coord.LOG_S, cfg=cfg_floor)
    assert bounds.S_min >= 90.0


def test_strike_multiple_policy_and_invalid_multiple():
    p = _Inputs(S=100.0, K=90.0, tau=1.0)

    cfg_bad = DomainConfig(policy=DomainPolicy.STRIKE_MULTIPLE, multiple=1.0)
    with pytest.raises(ValueError):
        compute_bounds(p, coord=Coord.S, cfg=cfg_bad)

    cfg = DomainConfig(policy=DomainPolicy.STRIKE_MULTIPLE, multiple=2.0)
    bounds = compute_bounds(p, coord=Coord.S, cfg=cfg)
    assert bounds.S_min > 0.0
    assert bounds.S_max > bounds.S_min


def test_make_grid_config_clustered_vs_uniform():
    p = _Inputs(S=100.0, K=100.0, tau=1.0)

    dom_clustered = DomainConfig(
        policy=DomainPolicy.STRIKE_MULTIPLE,
        multiple=2.0,
        spacing=SpacingPolicy.CLUSTERED,
        center="spot",
    )
    cfg_clustered = make_grid_config(
        p, coord=Coord.LOG_S, dom=dom_clustered, Nx=21, Nt=11
    )
    assert cfg_clustered.spacing == SpacingPolicy.CLUSTERED
    assert cfg_clustered.x_center is not None

    dom_uniform = DomainConfig(
        policy=DomainPolicy.STRIKE_MULTIPLE,
        multiple=2.0,
        spacing=SpacingPolicy.UNIFORM,
    )
    cfg_uniform = make_grid_config(p, coord=Coord.S, dom=dom_uniform, Nx=21, Nt=11)
    assert cfg_uniform.spacing == SpacingPolicy.UNIFORM
    assert cfg_uniform.x_center is None
