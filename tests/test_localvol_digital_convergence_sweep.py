from __future__ import annotations

import warnings

import numpy as np


def _build_skewed_svi_localvol_surface(
    *,
    S0: float,
    r: float,
    q: float,
    sigma_atm: float,
    expiries: np.ndarray,
):
    """Build a LocalVolSurface from moderately skewed SVI smiles.

    This is intentionally *not* flat-in-y: we want to exercise the full
    SVI -> implied surface -> Dupire/Gatheral local-vol pipeline.
    """
    from option_pricing.types import MarketData
    from option_pricing.vol.surface import LocalVolSurface, VolSurface
    from option_pricing.vol.svi import SVIParams, SVISmile

    ctx = MarketData(spot=S0, rate=r, dividend_yield=q).to_context()

    smiles: list[SVISmile] = []
    for Ti in expiries:
        Ti = float(Ti)

        # Moderate equity-like skew. Enforce w(0,T)=sigma_atm^2*T.
        b = 0.10 * Ti
        svi_sig = 0.30
        rho = -0.40
        m = 0.0

        w_atm = (sigma_atm * sigma_atm) * Ti
        # Since w(0)=a + b*hypot(0,svi_sig)=a + b*svi_sig, set a accordingly.
        a = max(1e-10, float(w_atm - b * svi_sig))

        smiles.append(
            SVISmile(T=Ti, params=SVIParams(a=a, b=b, rho=rho, m=m, sigma=svi_sig))
        )

    implied = VolSurface(
        expiries=np.asarray(expiries, dtype=float),
        smiles=tuple(smiles),
        forward=ctx.fwd,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        lv = LocalVolSurface.from_implied(implied, forward=ctx.fwd, discount=ctx.df)

    return ctx, lv


def _price_localvol_digital_pde(
    *,
    S0: float,
    K: float,
    T: float,
    payout: float,
    r: float,
    q: float,
    sigma_for_bounds: float,
    lv,
    Nx: int,
    Nt: int,
) -> float:
    """Price a unit cash-or-nothing digital call with the PDE in LOG_S.

    Uses the numerics we generally want for digitals:
    - clustered grid around strike
    - Rannacher time-stepping
    - L2 projection of the discontinuous terminal payoff

    (Implementation delegates to the library wiring helpers rather than
    re-implementing coefficient/boundary logic in the tests.)
    """
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

    p = PricingInputs(
        spec=DigitalSpec(
            kind=OptionType.CALL, strike=float(K), expiry=float(T), payout=float(payout)
        ),
        market=MarketData(spot=float(S0), rate=float(r), dividend_yield=float(q)),
        sigma=float(sigma_for_bounds),  # bounds selection only
        t=0.0,
    )

    coord = Coord.LOG_S

    dom = BSDomainConfig(
        policy=BSDomainPolicy.LOG_NSIGMA,
        n_sigma=7.0,
        center="strike",
        spacing=SpacingPolicy.CLUSTERED,
        cluster_strength=2.0,
    )
    bounds = bs_compute_bounds(p, coord=coord, cfg=dom)

    wiring = local_vol_pde_wiring(p, lv, coord, x_lb=bounds.x_lb, x_ub=bounds.x_ub)

    xK = float(np.asarray(wiring.to_x(K)).reshape(()))

    def ic_transform(grid, ic_fn):
        return ic_l2_projection(grid=grid, ic=ic_fn, breakpoints=(xK,))

    grid_cfg = GridConfig(
        Nx=int(Nx),
        Nt=int(Nt),
        x_lb=float(bounds.x_lb),
        x_ub=float(bounds.x_ub),
        T=float(T),
        spacing=dom.spacing,
        x_center=float(bounds.x_center),
        cluster_strength=float(dom.cluster_strength),
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
    return float(np.interp(wiring.x_0, x, u))


def test_localvol_digital_convergence_sweep_over_strikes_and_maturities() -> None:
    """A small convergence sweep for LocalVol digitals.

    This is a *numerical stability* test:

    For a handful of (T, K) points under a skewed SVI->LocalVol surface, we
    price the same digital with three increasingly fine grids and assert that:

    - the solution stays within obvious no-arbitrage bounds [0, payout*df(T)]
    - discretization steps shrink as the grid is refined (coarse->mid vs mid->fine)
    - the mid->fine step is acceptably small in absolute terms
    """
    S0 = 100.0
    r = 0.02
    q = 0.00
    payout = 1.0

    # A reasonably smooth (but not flat) local-vol surface derived from SVI
    expiries = np.asarray([0.10, 0.25, 0.50, 1.00, 2.00], dtype=float)
    sigma_atm = 0.25
    ctx, lv = _build_skewed_svi_localvol_surface(
        S0=S0, r=r, q=q, sigma_atm=sigma_atm, expiries=expiries
    )

    # Small sweep: enough to cover wings + different maturities without being slow
    Ts = [0.25, 1.00]
    Ks = [90.0, 100.0, 110.0]

    # 3-level ladder for a basic convergence/plateau check
    levels = [(201, 201), (301, 301), (401, 401)]

    # Absolute tolerance floor for a digital PV (unit payout). This is intentionally
    # conservative across machines; convergence should typically be tighter.
    max_step_fine = 2.0e-2

    for T in Ts:
        df = float(ctx.df(float(T)))
        for K in Ks:
            prices: list[float] = []
            for Nx, Nt in levels:
                px = _price_localvol_digital_pde(
                    S0=S0,
                    K=float(K),
                    T=float(T),
                    payout=payout,
                    r=r,
                    q=q,
                    sigma_for_bounds=sigma_atm,
                    lv=lv,
                    Nx=Nx,
                    Nt=Nt,
                )
                prices.append(float(px))

            p1, p2, p3 = prices
            # No-arbitrage bounds sanity
            assert (
                0.0 <= p3 <= payout * df + 1e-8
            ), f"T={T}, K={K}: digital out of bounds: {p3}"

            step12 = abs(p2 - p1)
            step23 = abs(p3 - p2)

            # Convergence tendency: when the coarse->mid step is meaningful, refinement should not get worse.
            #
            # For digitals (discontinuous payoff), it can happen that coarse and mid are accidentally very close
            # (e.g. grid alignment / projection effects), making step12 tiny; in that case, a ratio-style check
            # becomes meaningless and can false-fail even when the solution is perfectly stable. So we only apply
            # the "step23 <= c * step12" check when step12 is above a small floor.
            if step12 > 1.0e-4:
                assert step23 <= 1.25 * step12 + 1e-12, (
                    f"T={T}, K={K}: not converging: p1={p1:.8f}, p2={p2:.8f}, p3={p3:.8f}, "
                    f"step12={step12:.3g}, step23={step23:.3g}"
                )

            # Fine step should be small in absolute terms.
            assert (
                step23 < max_step_fine
            ), f"T={T}, K={K}: fine-grid step too large: p2={p2:.8f}, p3={p3:.8f}, step23={step23:.3g}"
