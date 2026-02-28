from __future__ import annotations

import warnings

import numpy as np
import pytest

from option_pricing.numerics.grids import GridConfig
from option_pricing.numerics.pde import solve_pde_1d
from option_pricing.numerics.pde.domain import Coord
from option_pricing.numerics.pde.ic_remedies import ic_cell_average
from option_pricing.pricers.pde.digital_local_vol import local_vol_pde_wiring
from option_pricing.pricers.pde.domain import (
    BSDomainConfig,
    BSDomainPolicy,
    bs_compute_bounds,
)
from option_pricing.types import DigitalSpec, MarketData, OptionType, PricingInputs
from option_pricing.vol.local_vol_surface import LocalVolSurface
from option_pricing.vol.surface_core import VolSurface
from option_pricing.vol.svi import SVIParams, svi_total_variance

warnings.filterwarnings("ignore", category=FutureWarning)


def _build_synthetic_surface(
    *,
    nT: int,
    nK: int,
    market: MarketData,
    params: SVIParams,
) -> VolSurface:
    expiries = np.linspace(0.25, 2.0, nT, dtype=float)
    strikes = np.linspace(60.0, 140.0, nK, dtype=float)
    rows: list[tuple[float, float, float]] = []

    for T in expiries:
        F = market.forward(T)
        y = np.log(strikes / F)
        w = svi_total_variance(y, params)
        iv = np.sqrt(np.maximum(w / T, 1e-12))
        for K, iv_k in zip(strikes, iv, strict=False):
            rows.append((float(T), float(K), float(iv_k)))

    calibrate_kwargs = {
        "loss": "linear",
        "irls_max_outer": 0,
        "repair_butterfly": False,
    }
    return VolSurface.from_svi(
        rows, forward=market.forward, calibrate_kwargs=calibrate_kwargs
    )


def _price_digital_local_vol_pde(
    *,
    lv: LocalVolSurface,
    market: MarketData,
    strike: float,
    tau: float,
    Nx: int,
    Nt: int,
) -> float:
    domain_cfg = BSDomainConfig(policy=BSDomainPolicy.LOG_NSIGMA, n_sigma=6.0)

    sigma_for_bounds = 0.2
    try:
        sigma_for_bounds = float(np.sqrt(lv.local_var(strike, tau)))
    except Exception:
        sigma_for_bounds = 0.2

    if not np.isfinite(sigma_for_bounds) or sigma_for_bounds <= 0.0:
        sigma_for_bounds = 0.2

    p = PricingInputs(
        spec=DigitalSpec(kind=OptionType.CALL, strike=strike, expiry=tau, payout=1.0),
        market=market,
        sigma=sigma_for_bounds,
        t=0.0,
    )

    bounds = bs_compute_bounds(p, coord=Coord.LOG_S, cfg=domain_cfg)
    wiring = local_vol_pde_wiring(
        p, lv, Coord.LOG_S, x_lb=bounds.x_lb, x_ub=bounds.x_ub
    )

    # No public digital-local-vol pricer yet; wire the PDE directly for the macro benchmark.
    xK = float(np.asarray(wiring.to_x(p.spec.strike)).reshape(()))

    def ic_transform(grid, ic):
        return ic_cell_average(grid, ic, breakpoints=(xK,))

    grid_cfg = GridConfig(
        Nx=Nx,
        Nt=Nt,
        x_lb=bounds.x_lb,
        x_ub=bounds.x_ub,
        T=tau,
        spacing=domain_cfg.spacing,
        x_center=bounds.x_center,
        cluster_strength=domain_cfg.cluster_strength,
    )

    sol = solve_pde_1d(
        wiring.problem,
        grid_cfg=grid_cfg,
        method="rannacher",
        store="final",
        ic_transform=ic_transform,
    )

    x = np.asarray(sol.grid.x, dtype=float)
    u = np.asarray(sol.u_final, dtype=float)
    return float(np.interp(wiring.x_0, x, u))


@pytest.mark.parametrize(
    "nT,nK,Nx,Nt",
    [
        (31, 61, 201, 201),
        (61, 121, 401, 401),
    ],
)
def test_bench_macro_pipeline(benchmark, nT: int, nK: int, Nx: int, Nt: int) -> None:
    market = MarketData(spot=100.0, rate=0.01, dividend_yield=0.0)
    params = SVIParams(a=0.02, b=0.3, rho=-0.4, m=0.0, sigma=0.4)

    surface = _build_synthetic_surface(nT=nT, nK=nK, market=market, params=params)
    lv = LocalVolSurface.from_implied(
        surface,
        forward=market.forward,
        discount=market.df,
    )

    def _run() -> float:
        return _price_digital_local_vol_pde(
            lv=lv,
            market=market,
            strike=100.0,
            tau=1.0,
            Nx=Nx,
            Nt=Nt,
        )

    benchmark(_run)
