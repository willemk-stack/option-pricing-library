from __future__ import annotations

import warnings

import numpy as np


def test_digital_from_localvol_pde_matches_bs_on_flat_svi_surface() -> None:
    """Regression / sanity check for the local-vol -> PDE path.

    We build an *implied* surface using per-expiry SVI smiles, but choose SVI params
    that produce a *flat* total-variance smile (b=0), i.e. constant implied vol.

    For a flat implied surface, the Gatheral/Dupire local volatility should collapse
    to the same constant sigma, and therefore a digital priced under the local-vol
    PDE should match the closed-form Black-76 digital price.
    """

    from option_pricing.instruments.digital import DigitalOption
    from option_pricing.market.curves import PricingContext
    from option_pricing.models.black_scholes import bs as bs_model
    from option_pricing.numerics.grids import GridConfig, SpacingPolicy
    from option_pricing.numerics.pde import AdvectionScheme, solve_pde_1d
    from option_pricing.numerics.pde.domain import Coord
    from option_pricing.numerics.pde.ic_remedies import ic_l2_projection
    from option_pricing.pricers.pde.digital_local_vol import local_vol_pde_wiring
    from option_pricing.pricers.pde.domain import (
        BSDomainConfig,
        BSDomainPolicy,
        bs_compute_bounds,
    )
    from option_pricing.types import DigitalSpec, MarketData, OptionType, PricingInputs
    from option_pricing.vol.surface import LocalVolSurface, VolSurface
    from option_pricing.vol.svi import SVIParams, SVISmile

    # ---- market + contract ----
    S0 = 100.0
    K = 100.0
    r = 0.02
    q = 0.00
    sigma = 0.25
    T = 0.75
    payout = 1.0

    ctx: PricingContext = MarketData(spot=S0, rate=r, dividend_yield=q).to_context()
    inst = DigitalOption(expiry=T, strike=K, payout=payout, kind=OptionType.CALL)

    # Closed-form "known good" reference price
    ref = float(bs_model.digital_call_price(instr=inst, market=ctx, sigma=sigma))

    # ---- implied surface (SVI smiles) ----
    # Choose b=0 so w(y,T) = a = sigma^2 * T (flat smile).
    expiries = np.asarray([0.25, 0.75, 1.25], dtype=float)

    smiles: list[SVISmile] = []
    for Ti in expiries:
        params = SVIParams(
            a=(sigma * sigma) * float(Ti),
            b=0.0,
            rho=0.0,
            m=0.0,
            sigma=0.2,  # irrelevant when b=0, but must be finite
        )
        smiles.append(SVISmile(T=float(Ti), params=params))

    implied = VolSurface(expiries=expiries, smiles=tuple(smiles), forward=ctx.fwd)

    # Local-vol surface derived from SVI (this emits a FutureWarning in the library)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        lv = LocalVolSurface.from_implied(implied, forward=ctx.fwd, discount=ctx.df)

    # ---- local-vol PDE in LOG_S coordinates ----
    coord = Coord.LOG_S

    # Use the BS domain helper (lognormal band) for a robust computational domain,
    # and a clustered grid centered at strike (best practice for discontinuous payoffs).
    p = PricingInputs(
        spec=DigitalSpec(kind=OptionType.CALL, strike=K, expiry=T, payout=payout),
        market=MarketData(spot=S0, rate=r, dividend_yield=q),
        sigma=sigma,  # only for bounds selection
        t=0.0,
    )

    dom = BSDomainConfig(
        policy=BSDomainPolicy.LOG_NSIGMA,
        n_sigma=6.0,
        center="strike",
        spacing=SpacingPolicy.CLUSTERED,
        cluster_strength=2.0,
    )
    bounds = bs_compute_bounds(p, coord=coord, cfg=dom)

    wiring = local_vol_pde_wiring(p, lv, coord, x_lb=bounds.x_lb, x_ub=bounds.x_ub)

    # L2 projection (best default) at the strike discontinuity + Rannacher time-stepping
    xK = float(np.asarray(wiring.to_x(K)).reshape(()))

    def ic_transform(grid, ic_fn):
        return ic_l2_projection(grid=grid, ic=ic_fn, breakpoints=(xK,))

    grid_cfg = GridConfig(
        Nx=401,
        Nt=401,
        x_lb=bounds.x_lb,
        x_ub=bounds.x_ub,
        T=T,
        spacing=dom.spacing,
        x_center=bounds.x_center,
        cluster_strength=dom.cluster_strength,
    )

    sol = solve_pde_1d(
        wiring.problem,
        grid_cfg=grid_cfg,
        method="rannacher",
        advection=AdvectionScheme.CENTRAL,
        store="final",
        ic_transform=ic_transform,
    )

    x = np.asarray(sol.grid.x, dtype=float)
    u = np.asarray(sol.u_final, dtype=float)
    pde = float(np.interp(wiring.x_0, x, u))

    # With a discontinuous payoff, tight tolerances need both smoothing + Rannacher.
    # The flat-SVI construction ensures the local-vol PDE should match Black-76.
    assert (
        abs(pde - ref) < 5e-3
    ), f"local-vol PDE={pde:.8f}, ref(BS)={ref:.8f}, abs_err={abs(pde-ref):.3g}"
