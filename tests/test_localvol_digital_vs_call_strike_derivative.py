from __future__ import annotations

import math
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
    """Helper: build a LocalVolSurface from a moderately skewed SVI implied surface."""
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

    # Local-vol surface derived from SVI
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        lv = LocalVolSurface.from_implied(implied, forward=ctx.fwd, discount=ctx.df)

    return ctx, lv


def _price_localvol_pde_1d(
    *,
    payoff_S,  # payoff at expiry as a function of S
    bc_left_gamma,
    bc_right_gamma,
    S0: float,
    r: float,
    q: float,
    T: float,
    lv,
    bounds,
    Nx: int,
    Nt: int,
    x_breakpoints: tuple[float, ...] = (),
) -> float:
    """Solve the local-vol PDE in LOG_S and return u(tau=T, x0)."""
    from option_pricing.models.black_scholes.pde import bs_coord_maps
    from option_pricing.numerics.grids import GridConfig
    from option_pricing.numerics.pde import (
        AdvectionScheme,
        LinearParabolicPDE1D,
        solve_pde_1d,
    )
    from option_pricing.numerics.pde.boundary import RobinBC, RobinBCSide
    from option_pricing.numerics.pde.domain import Coord
    from option_pricing.numerics.pde.ic_remedies import ic_l2_projection

    coord = Coord.LOG_S
    to_x, to_S = bs_coord_maps(coord)

    x0 = float(np.asarray(to_x(S0)).reshape(()))

    mu = float(r - q)

    # Clamp tau away from 0 to avoid LocalVolSurface(T<=0) guardrails during coefficient eval.
    def _sigma2(x: np.ndarray | float, tau: float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        tau_eff = max(float(tau), 1e-8)
        S = np.exp(x_arr)
        return np.asarray(lv.local_var(S, tau_eff), dtype=float)

    def a(x: np.ndarray | float, tau: float) -> np.ndarray:
        return 0.5 * _sigma2(x, tau)

    def b(x: np.ndarray | float, tau: float) -> np.ndarray:
        sig2 = _sigma2(x, tau)
        return (mu - 0.5 * sig2).astype(float, copy=False)

    def c(x: np.ndarray | float, tau: float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        return (-float(r)) + 0.0 * x_arr

    # Dirichlet boundaries expressed as Robin (beta=0)
    bc = RobinBC(
        left=RobinBCSide(
            alpha=lambda tau: 1.0, beta=lambda tau: 0.0, gamma=bc_left_gamma
        ),
        right=RobinBCSide(
            alpha=lambda tau: 1.0, beta=lambda tau: 0.0, gamma=bc_right_gamma
        ),
    )

    def ic(x: float) -> float:
        return float(payoff_S(float(to_S(x))))

    problem = LinearParabolicPDE1D(a=a, b=b, c=c, bc=bc, ic=ic)

    # L2 projection is robust for kinks/discontinuities. If no breakpoints, skip transform.
    def ic_transform(grid, ic_fn):
        return ic_l2_projection(grid=grid, ic=ic_fn, breakpoints=x_breakpoints)

    grid_cfg = GridConfig(
        Nx=int(Nx),
        Nt=int(Nt),
        x_lb=float(bounds.x_lb),
        x_ub=float(bounds.x_ub),
        T=float(T),
        spacing=bounds.spacing,
        x_center=float(bounds.x_center),
        cluster_strength=float(bounds.cluster_strength),
    )

    sol = solve_pde_1d(
        problem,
        grid_cfg=grid_cfg,
        method="rannacher",
        advection=AdvectionScheme.CENTRAL,
        store="final",
        ic_transform=ic_transform if x_breakpoints else None,
    )

    x = np.asarray(sol.grid.x, dtype=float)
    u = np.asarray(sol.u_final, dtype=float)
    return float(np.interp(x0, x, u))


def test_localvol_digital_matches_minus_strike_derivative_of_call() -> None:
    """
    Internal consistency check (no external libs):

      Digital(K,T)  ≈  - d/dK Call(K,T)

    where Digital is the PV of a unit cash-or-nothing digital paying 1{S_T > K}.

    We do this under a *skewed* LocalVol surface derived from SVI smiles.
    We also check basic numerical stability (coarse vs fine grids) and set the
    tolerance relative to the observed discretization step.
    """
    from option_pricing.instruments.digital import DigitalOption
    from option_pricing.instruments.vanilla import VanillaOption
    from option_pricing.models.black_scholes.pde import bs_coord_maps
    from option_pricing.numerics.grids import SpacingPolicy
    from option_pricing.numerics.pde.domain import Coord
    from option_pricing.pricers.pde.domain import (
        BSDomainConfig,
        BSDomainPolicy,
        bs_compute_bounds,
    )
    from option_pricing.types import DigitalSpec, MarketData, OptionType, PricingInputs

    # --- contract / market ---
    S0 = 100.0
    K = 100.0
    T = 0.75
    payout = 1.0

    r = 0.02
    q = 0.00

    # central-difference step in strike
    h = 0.50  # ~0.5% of K; small enough for FD, large enough to beat PDE noise

    ctx = MarketData(spot=S0, rate=r, dividend_yield=q).to_context()

    # --- local-vol surface (SVI) ---
    expiries = np.asarray([0.25, T, 1.25], dtype=float)
    sigma_atm = 0.25
    ctx, lv = _build_skewed_svi_localvol_surface(
        S0=S0, r=r, q=q, sigma_atm=sigma_atm, expiries=expiries
    )

    # --- shared PDE domain/grid (cluster around K) ---
    coord = Coord.LOG_S
    to_x, _ = bs_coord_maps(coord)
    xK = float(np.asarray(to_x(K)).reshape(()))

    p_for_bounds = PricingInputs(
        spec=DigitalSpec(kind=OptionType.CALL, strike=K, expiry=T, payout=payout),
        market=MarketData(spot=S0, rate=r, dividend_yield=q),
        sigma=sigma_atm,  # only for bounds selection
        t=0.0,
    )

    dom = BSDomainConfig(
        policy=BSDomainPolicy.LOG_NSIGMA,
        n_sigma=7.0,
        center="strike",
        spacing=SpacingPolicy.CLUSTERED,
        cluster_strength=2.0,
    )
    bounds = bs_compute_bounds(p_for_bounds, coord=coord, cfg=dom)

    # We'll carry these through to _price_localvol_pde_1d via a small struct-like shim
    class _Bounds:
        x_lb = bounds.x_lb
        x_ub = bounds.x_ub
        x_center = bounds.x_center
        spacing = dom.spacing
        cluster_strength = dom.cluster_strength

    bnd = _Bounds()

    # --- boundary functions ---
    # Digital call: tends to payout*df(tau) as S -> +inf
    def digital_left_gamma(tau: float) -> float:
        return 0.0

    def digital_right_gamma(tau: float) -> float:
        return float(payout * ctx.df(float(tau)))

    # Call: tends to S*df_q(tau) - K*df(tau) as S -> +inf (Dirichlet at S_ub)
    S_ub = float(math.exp(float(bounds.x_ub)))

    def call_left_gamma(tau: float) -> float:
        return 0.0

    def call_right_gamma_factory(Ki: float):
        def _gamma(tau: float) -> float:
            tau = float(tau)
            return float(S_ub * ctx.df_q(tau) - float(Ki) * ctx.df(tau))

        return _gamma

    # --- payoff functions ---
    digital = DigitalOption(expiry=T, strike=K, payout=payout, kind=OptionType.CALL)
    call_m = VanillaOption(expiry=T, strike=K - h, kind=OptionType.CALL)
    call_p = VanillaOption(expiry=T, strike=K + h, kind=OptionType.CALL)

    # Breakpoints for L2 projection (strike kink/discontinuity)
    xKm = float(np.asarray(to_x(K - h)).reshape(()))
    xKp = float(np.asarray(to_x(K + h)).reshape(()))

    def _compute(Nx: int, Nt: int):
        # Digital PV
        d = _price_localvol_pde_1d(
            payoff_S=digital.payoff,
            bc_left_gamma=digital_left_gamma,
            bc_right_gamma=digital_right_gamma,
            S0=S0,
            r=r,
            q=q,
            T=T,
            lv=lv,
            bounds=bnd,
            Nx=Nx,
            Nt=Nt,
            x_breakpoints=(xK,),
        )

        # Calls for strike-derivative
        Cm = _price_localvol_pde_1d(
            payoff_S=call_m.payoff,
            bc_left_gamma=call_left_gamma,
            bc_right_gamma=call_right_gamma_factory(K - h),
            S0=S0,
            r=r,
            q=q,
            T=T,
            lv=lv,
            bounds=bnd,
            Nx=Nx,
            Nt=Nt,
            x_breakpoints=(xKm,),
        )
        Cp = _price_localvol_pde_1d(
            payoff_S=call_p.payoff,
            bc_left_gamma=call_left_gamma,
            bc_right_gamma=call_right_gamma_factory(K + h),
            S0=S0,
            r=r,
            q=q,
            T=T,
            lv=lv,
            bounds=bnd,
            Nx=Nx,
            Nt=Nt,
            x_breakpoints=(xKp,),
        )

        digital_from_calls = -float((Cp - Cm) / (2.0 * h))
        return d, digital_from_calls

    # Coarse vs fine: stability + consistency
    d_coarse, dfd_coarse = _compute(Nx=301, Nt=301)
    d_fine, dfd_fine = _compute(Nx=401, Nt=401)

    # Bounds sanity
    df = float(ctx.df(T))
    for name, px in {"digital_coarse": d_coarse, "digital_fine": d_fine}.items():
        assert 0.0 <= px <= payout * df + 1e-8, f"{name} out of bounds: {px}"

    # Stability (plateau)
    step_d = abs(d_fine - d_coarse)
    step_fd = abs(dfd_fine - dfd_coarse)
    assert (
        step_d < 2.5e-2
    ), f"digital not stable: coarse={d_coarse:.8f}, fine={d_fine:.8f}, step={step_d:.3g}"
    assert (
        step_fd < 2.5e-2
    ), f"-dC/dK not stable: coarse={dfd_coarse:.8f}, fine={dfd_fine:.8f}, step={step_fd:.3g}"

    # Consistency (anchor tolerance to observed discretization error + a floor)
    diff = abs(d_fine - dfd_fine)
    tol = max(2.5e-2, 10.0 * (step_d + step_fd))
    assert diff < tol, (
        f"digital={d_fine:.8f}, -dC/dK={dfd_fine:.8f}, abs_err={diff:.3g}, tol={tol:.3g} "
        f"(step_d={step_d:.3g}, step_fd={step_fd:.3g})"
    )
