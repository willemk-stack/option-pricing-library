from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pytest

from option_pricing.numerics.grids import SpacingPolicy
from option_pricing.numerics.pde import AdvectionScheme
from option_pricing.numerics.pde.domain import Coord
from option_pricing.pricers.black_scholes import bs_price
from option_pricing.pricers.pde.domain import BSDomainConfig, BSDomainPolicy
from option_pricing.pricers.pde_pricer import local_vol_price_pde_european
from option_pricing.types import OptionType
from option_pricing.vol.surface import LocalVolSurface, VolSurface


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
