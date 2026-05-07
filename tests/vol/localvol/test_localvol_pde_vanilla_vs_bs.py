from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pytest

from option_pricing.numerics.grids import GridConfig, SpacingPolicy
from option_pricing.numerics.pde import AdvectionScheme, solve_pde_1d
from option_pricing.numerics.pde.domain import Coord
from option_pricing.pricers.black_scholes import bs_price
from option_pricing.pricers.pde.domain import (
    BSDomainConfig,
    BSDomainPolicy,
    bs_compute_bounds,
)
from option_pricing.pricers.pde.european_local_vol import local_vol_pde_wiring
from option_pricing.pricers.pde_pricer import local_vol_price_pde_european
from option_pricing.types import OptionType
from option_pricing.vol import LocalVolSurface, VolSurface


@dataclass(frozen=True)
class _ConstSmile:
    """Constant-IV smile slice with analytic derivatives in y."""

    T: float
    sigma: float
    y_min: float = -2.0
    y_max: float = 2.0

    def w_at(self, xq):
        xq_arr = np.asarray(xq, dtype=np.float64)
        out = np.full_like(
            xq_arr, float(self.T) * float(self.sigma) ** 2, dtype=np.float64
        )
        return out

    def iv_at(self, xq):
        xq_arr = np.asarray(xq, dtype=np.float64)
        return np.full_like(xq_arr, float(self.sigma), dtype=np.float64)

    def dw_dy(self, xq):
        xq_arr = np.asarray(xq, dtype=np.float64)
        return np.zeros_like(xq_arr, dtype=np.float64)

    def d2w_dy2(self, xq):
        xq_arr = np.asarray(xq, dtype=np.float64)
        return np.zeros_like(xq_arr, dtype=np.float64)


@dataclass(slots=True)
class _RecordingLocalVol:
    sigma2: float
    tau_calls: list[float] = field(default_factory=list)

    def local_var(self, S, T):
        self.tau_calls.append(float(T))
        S_arr = np.asarray(S, dtype=np.float64)
        return np.full_like(S_arr, float(self.sigma2), dtype=np.float64)


def _default_domain_cfg(
    *, spacing: SpacingPolicy = SpacingPolicy.CLUSTERED
) -> BSDomainConfig:
    return BSDomainConfig(
        policy=BSDomainPolicy.LOG_NSIGMA,
        n_sigma=6.0,
        center="strike",
        spacing=spacing,
        cluster_strength=2.0,
    )


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
def test_localvol_pde_vanilla_matches_bs_on_constant_surface(make_inputs, kind) -> None:
    p = make_inputs(
        S=100.0,
        K=105.0,
        r=0.03,
        q=0.01,
        sigma=0.25,
        T=1.0,
        kind=kind,
    )

    ctx = p.market.to_context()

    expiries = np.asarray([0.25, 0.75, 1.25], dtype=float)
    smiles = tuple(_ConstSmile(T=float(Ti), sigma=float(p.sigma)) for Ti in expiries)
    implied = VolSurface(expiries=expiries, smiles=smiles, forward=ctx.fwd)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        lv = LocalVolSurface.from_implied(implied, forward=ctx.fwd, discount=ctx.df)

    dom = _default_domain_cfg(spacing=SpacingPolicy.CLUSTERED)

    pde = float(
        local_vol_price_pde_european(
            p,
            lv=lv,
            coord=Coord.LOG_S,
            domain_cfg=dom,
            Nx=201,
            Nt=201,
            method="cn",
            advection=AdvectionScheme.CENTRAL,
        )
    )
    ref = float(bs_price(p))

    # Local-vol reduces to BS when sigma_loc is constant.
    assert abs(pde - ref) <= max(3.0e-3, 1.0e-2 * abs(ref))


def test_localvol_pde_wiring_maps_solver_tau_to_calendar_surface_time(
    make_inputs,
) -> None:
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.03,
        q=0.01,
        sigma=0.2,
        T=1.0,
        kind=OptionType.CALL,
    )

    lv = _RecordingLocalVol(sigma2=float(p.sigma) ** 2)
    dom = _default_domain_cfg(spacing=SpacingPolicy.CLUSTERED)
    bounds = bs_compute_bounds(p, coord=Coord.LOG_S, cfg=dom)

    wiring = local_vol_pde_wiring(
        p,
        lv,
        Coord.LOG_S,
        x_lb=bounds.x_lb,
        x_ub=bounds.x_ub,
    )

    grid_cfg = GridConfig(
        Nx=61,
        Nt=9,
        x_lb=bounds.x_lb,
        x_ub=bounds.x_ub,
        T=p.tau,
        spacing=dom.spacing,
        x_center=bounds.x_center,
        cluster_strength=dom.cluster_strength,
    )

    solve_pde_1d(
        wiring.problem,
        grid_cfg=grid_cfg,
        method="cn",
        advection=AdvectionScheme.CENTRAL,
        store="final",
    )

    tau_calls = np.asarray(lv.tau_calls, dtype=float)

    assert tau_calls.size > 0
    assert np.all(tau_calls >= 0.0)
    assert float(np.min(tau_calls)) <= 1.0e-6
    assert float(np.max(tau_calls)) >= float(p.tau) - 1.0e-6
    assert np.all(tau_calls <= float(p.tau) + 1.0e-12)
