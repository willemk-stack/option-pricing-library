from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from option_pricing.diagnostics.vol_surface.pde_repricing import (
    localvol_pde_single_option_convergence_sweep,
)
from option_pricing.numerics.grids import SpacingPolicy
from option_pricing.numerics.pde import AdvectionScheme
from option_pricing.numerics.pde.domain import Coord
from option_pricing.pricers.pde.domain import BSDomainConfig, BSDomainPolicy
from option_pricing.types import OptionType
from option_pricing.vol import LocalVolSurface, VolSurface


@dataclass(frozen=True)
class _ConstSmile:
    T: float
    sigma: float
    y_min: float = -2.0
    y_max: float = 2.0

    def w_at(self, xq):
        xq_arr = np.asarray(xq, dtype=np.float64)
        return np.full_like(
            xq_arr, float(self.T) * float(self.sigma) ** 2, dtype=np.float64
        )

    def iv_at(self, xq):
        xq_arr = np.asarray(xq, dtype=np.float64)
        return np.full_like(xq_arr, float(self.sigma), dtype=np.float64)

    def dw_dy(self, xq):
        xq_arr = np.asarray(xq, dtype=np.float64)
        return np.zeros_like(xq_arr, dtype=np.float64)

    def d2w_dy2(self, xq):
        xq_arr = np.asarray(xq, dtype=np.float64)
        return np.zeros_like(xq_arr, dtype=np.float64)


def test_convergence_sweep_uses_fine_grid_reference(make_inputs) -> None:
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.03,
        q=0.01,
        sigma=0.25,
        T=1.0,
        kind=OptionType.CALL,
    )
    ctx = p.market.to_context()
    expiries = np.asarray([0.5, 1.0, 1.5], dtype=float)
    smiles = tuple(_ConstSmile(T=float(Ti), sigma=float(p.sigma)) for Ti in expiries)
    implied = VolSurface(expiries=expiries, smiles=smiles, forward=ctx.fwd)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        lv = LocalVolSurface.from_implied(implied, forward=ctx.fwd, discount=ctx.df)

    solver_cfg = {
        "coord": Coord.LOG_S,
        "domain_cfg": BSDomainConfig(
            policy=BSDomainPolicy.LOG_NSIGMA,
            n_sigma=6.0,
            center="strike",
            spacing=SpacingPolicy.CLUSTERED,
            cluster_strength=2.0,
        ),
        "method": "cn",
        "advection": AdvectionScheme.CENTRAL,
    }

    reference_grid = (81, 161)
    result = localvol_pde_single_option_convergence_sweep(
        lv=lv,
        market=p.market,
        strike=100.0,
        expiry=1.0,
        grids=[(41, 81), reference_grid],
        solver_cfg=solver_cfg,
        kind=OptionType.CALL,
        reference_grid=reference_grid,
    )

    fine_row = result.grid.loc[
        (result.grid["Nx"] == 81) & (result.grid["Nt"] == 161)
    ].iloc[0]
    assert fine_row["reference_price"] == result.meta["reference_price"]
    assert fine_row["abs_error"] <= 1.0e-12
    assert result.meta["reference_Nx"] == 81
    assert result.meta["reference_Nt"] == 161
    assert "implied_target_price" in result.meta
