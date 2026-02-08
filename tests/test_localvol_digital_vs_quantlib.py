from __future__ import annotations

import math
import warnings

import numpy as np
import pytest


def _our_localvol_pde_digital_price_from_flat_svi(
    *,
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    payout: float = 1.0,
    Nx: int = 401,
    Nt: int = 401,
) -> float:
    """
    Price a cash-or-nothing digital call under a LocalVol surface derived from
    per-expiry SVI smiles. We choose b=0 so the smile is flat and local vol is
    (numerically) constant.

    Uses "best practice" numerics for discontinuous payoffs:
      - LOG_S coordinate
      - clustered grid centered at strike
      - Rannacher timestepping
      - L2 projection of initial condition at the strike kink
    """
    from option_pricing.instruments.digital import DigitalOption
    from option_pricing.models.black_scholes.pde import bs_coord_maps
    from option_pricing.numerics.grids import GridConfig, SpacingPolicy
    from option_pricing.numerics.pde import (
        AdvectionScheme,
        LinearParabolicPDE1D,
        solve_pde_1d,
    )
    from option_pricing.numerics.pde.boundary import RobinBC, RobinBCSide
    from option_pricing.numerics.pde.domain import Coord
    from option_pricing.numerics.pde.ic_remedies import ic_l2_projection
    from option_pricing.pricers.pde.domain import (
        BSDomainConfig,
        BSDomainPolicy,
        bs_compute_bounds,
    )
    from option_pricing.types import DigitalSpec, MarketData, OptionType, PricingInputs
    from option_pricing.vol.surface import LocalVolSurface, VolSurface
    from option_pricing.vol.svi import SVIParams, SVISmile

    ctx = MarketData(spot=S0, rate=r, dividend_yield=q).to_context()
    inst = DigitalOption(expiry=T, strike=K, payout=payout, kind=OptionType.CALL)

    # --- implied surface (SVI smiles), but flat in y via b=0 ---
    expiries = np.asarray([0.25, float(T), 1.25], dtype=float)
    smiles: list[SVISmile] = []
    for Ti in expiries:
        smiles.append(
            SVISmile(
                T=float(Ti),
                params=SVIParams(
                    a=(sigma * sigma) * float(Ti),
                    b=0.0,
                    rho=0.0,
                    m=0.0,
                    sigma=0.2,  # irrelevant when b=0, but must be finite
                ),
            )
        )

    implied = VolSurface(expiries=expiries, smiles=tuple(smiles), forward=ctx.fwd)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        lv = LocalVolSurface.from_implied(implied, forward=ctx.fwd, discount=ctx.df)

    # --- LOG_S PDE coefficients using local vol ---
    coord = Coord.LOG_S
    to_x, to_S = bs_coord_maps(coord)

    x0 = float(np.asarray(to_x(S0)).reshape(()))
    xK = float(np.asarray(to_x(K)).reshape(()))

    p_for_bounds = PricingInputs(
        spec=DigitalSpec(kind=OptionType.CALL, strike=K, expiry=T, payout=payout),
        market=MarketData(spot=S0, rate=r, dividend_yield=q),
        sigma=sigma,  # used only for bounds selection here
        t=0.0,
    )
    dom = BSDomainConfig(
        policy=BSDomainPolicy.LOG_NSIGMA,
        n_sigma=6.0,
        center="strike",
        spacing=SpacingPolicy.CLUSTERED,
        cluster_strength=2.0,
    )
    bounds = bs_compute_bounds(p_for_bounds, coord=coord, cfg=dom)

    # Far-field Dirichlet in PV terms (expressed as Robin beta=0)
    def left_gamma(tau: float) -> float:
        return 0.0

    def right_gamma(tau: float) -> float:
        return float(payout * ctx.df(float(tau)))

    bc = RobinBC(
        left=RobinBCSide(alpha=lambda tau: 1.0, beta=lambda tau: 0.0, gamma=left_gamma),
        right=RobinBCSide(
            alpha=lambda tau: 1.0, beta=lambda tau: 0.0, gamma=right_gamma
        ),
    )

    mu = float(r - q)

    # avoid T=0 guardrails in LV eval inside the PDE time loop
    def _sigma2(x: np.ndarray | float, tau: float) -> np.ndarray:
        tau_eff = max(float(tau), 1e-8)
        x_arr = np.asarray(x, dtype=float)
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

    payoff = inst.payoff

    def ic(x: float) -> float:
        return float(payoff(float(to_S(x))))

    problem = LinearParabolicPDE1D(a=a, b=b, c=c, bc=bc, ic=ic)

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
        problem,
        grid_cfg=grid_cfg,
        method="rannacher",
        advection=AdvectionScheme.CENTRAL,
        store="final",
        ic_transform=ic_transform,
    )

    x = np.asarray(sol.grid.x, dtype=float)
    u = np.asarray(sol.u_final, dtype=float)
    return float(np.interp(x0, x, u))


def _quantlib_fd_digital_price(
    ql,
    *,
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    payout: float = 1.0,
    tGrid: int = 401,
    xGrid: int = 401,
    dampingSteps: int = 2,
) -> float:
    """
    QuantLib cross-check: price a cash-or-nothing digital using the finite-difference
    Black-Scholes engine in local-vol mode (localVol=True).

    We pass a (flat) implied vol surface; in this case local vol is constant as well.
    """
    # --- dates / curves ---
    calendar = ql.TARGET()
    dc = ql.Actual365Fixed()

    eval_date = ql.Date(2, ql.January, 2020)
    ql.Settings.instance().evaluationDate = eval_date

    maturity = eval_date + int(round(float(T) * 365))

    spot = ql.QuoteHandle(ql.SimpleQuote(float(S0)))
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, float(r), dc))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, float(q), dc))

    # --- build an implied vol surface (here: flat, but still constructed as a surface) ---
    # Note: BlackVarianceSurface expects vols in a matrix [strike x date].
    expiries = [0.25, float(T), 1.25]
    dates = [eval_date + int(round(t * 365)) for t in expiries]

    strikes = np.linspace(0.6 * S0, 1.6 * S0, 41, dtype=float)
    vol_matrix = ql.Matrix(len(strikes), len(dates))
    for i, _k in enumerate(strikes):
        for j, _d in enumerate(dates):
            vol_matrix[i][j] = float(sigma)

    black_surface = ql.BlackVarianceSurface(
        eval_date,
        calendar,
        dates,
        [float(k) for k in strikes],
        vol_matrix,
        dc,
    )
    black_surface.enableExtrapolation()

    vol_ts = ql.BlackVolTermStructureHandle(black_surface)

    process = ql.BlackScholesMertonProcess(spot, q_ts, r_ts, vol_ts)

    # --- option ---
    payoff = ql.CashOrNothingPayoff(ql.Option.Call, float(K), float(payout))
    exercise = ql.EuropeanExercise(maturity)
    opt = ql.VanillaOption(payoff, exercise)

    # --- engine ---
    # Prefer Crank-Nicolson (with damping steps ≈ Rannacher), fallback to Douglas if needed.
    scheme_ctor = getattr(ql.FdmSchemeDesc, "CrankNicolson", None)
    if callable(scheme_ctor):
        scheme = scheme_ctor()
    else:
        scheme = ql.FdmSchemeDesc.Douglas()

    # Signature varies a bit across wheels; this is the common one:
    try:
        engine = ql.FdBlackScholesVanillaEngine(
            process, int(tGrid), int(xGrid), int(dampingSteps), scheme, True
        )
    except TypeError:
        # Some builds expose the illegalLocalVolOverwrite param explicitly
        engine = ql.FdBlackScholesVanillaEngine(
            process, int(tGrid), int(xGrid), int(dampingSteps), scheme, True, None
        )

    opt.setPricingEngine(engine)
    return float(opt.NPV())


def test_localvol_digital_matches_quantlib_fd_on_flat_svi_surface() -> None:
    """
    External cross-check against QuantLib.

    This compares:
      - our LocalVol(SVI)->PDE digital price
      - QuantLib FD digital price with localVol=True

    using a flat-SVI construction (b=0) so both implementations should reduce
    to the same constant-vol digital price.
    """
    ql = pytest.importorskip(
        "QuantLib",
        reason="QuantLib is required for this cross-check (pip install QuantLib).",
    )

    # --- contract / market ---
    S0 = 100.0
    K = 100.0
    r = 0.02
    q = 0.00
    sigma = 0.25
    T = 0.75
    payout = 1.0

    ours = _our_localvol_pde_digital_price_from_flat_svi(
        S0=S0, K=K, r=r, q=q, sigma=sigma, T=T, payout=payout, Nx=401, Nt=401
    )
    ql_price = _quantlib_fd_digital_price(
        ql,
        S0=S0,
        K=K,
        r=r,
        q=q,
        sigma=sigma,
        T=T,
        payout=payout,
        tGrid=401,
        xGrid=401,
        dampingSteps=2,
    )

    # (Optional but useful) also check against closed-form BS digital.
    from option_pricing.instruments.digital import DigitalOption
    from option_pricing.models.black_scholes import bs as bs_model
    from option_pricing.types import MarketData, OptionType

    ctx = MarketData(spot=S0, rate=r, dividend_yield=q).to_context()
    inst = DigitalOption(expiry=T, strike=K, payout=payout, kind=OptionType.CALL)
    ref = float(bs_model.digital_call_price(instr=inst, market=ctx, sigma=sigma))

    # With good damping + clustering, digitals should be accurate to a few bp.
    tol = 5e-3
    assert (
        abs(ours - ql_price) < tol
    ), f"ours={ours:.8f}, QuantLib={ql_price:.8f}, abs_err={abs(ours-ql_price):.3g}"
    assert (
        abs(ours - ref) < tol
    ), f"ours={ours:.8f}, BS(ref)={ref:.8f}, abs_err={abs(ours-ref):.3g}"
    assert (
        abs(ql_price - ref) < tol
    ), f"QuantLib={ql_price:.8f}, BS(ref)={ref:.8f}, abs_err={abs(ql_price-ref):.3g}"


def _our_localvol_pde_digital_price_from_svi_smiles(
    *,
    smiles,
    expiries,
    S0: float,
    K: float,
    r: float,
    q: float,
    T: float,
    payout: float = 1.0,
    Nx: int = 401,
    Nt: int = 401,
    n_sigma_bounds: float = 6.0,
) -> float:
    """Price a digital under LocalVol derived from the supplied SVI smile slices."""
    from option_pricing.instruments.digital import DigitalOption
    from option_pricing.models.black_scholes.pde import bs_coord_maps
    from option_pricing.numerics.grids import GridConfig, SpacingPolicy
    from option_pricing.numerics.pde import (
        AdvectionScheme,
        LinearParabolicPDE1D,
        solve_pde_1d,
    )
    from option_pricing.numerics.pde.boundary import RobinBC, RobinBCSide
    from option_pricing.numerics.pde.domain import Coord
    from option_pricing.numerics.pde.ic_remedies import ic_l2_projection
    from option_pricing.pricers.pde.domain import (
        BSDomainConfig,
        BSDomainPolicy,
        bs_compute_bounds,
    )
    from option_pricing.types import DigitalSpec, MarketData, OptionType, PricingInputs
    from option_pricing.vol.surface import LocalVolSurface, VolSurface

    ctx = MarketData(spot=S0, rate=r, dividend_yield=q).to_context()
    inst = DigitalOption(expiry=T, strike=K, payout=payout, kind=OptionType.CALL)

    implied = VolSurface(
        expiries=np.asarray(expiries, dtype=float),
        smiles=tuple(smiles),
        forward=ctx.fwd,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        lv = LocalVolSurface.from_implied(implied, forward=ctx.fwd, discount=ctx.df)

    coord = Coord.LOG_S
    to_x, to_S = bs_coord_maps(coord)

    x0 = float(np.asarray(to_x(S0)).reshape(()))
    xK = float(np.asarray(to_x(K)).reshape(()))

    p_for_bounds = PricingInputs(
        spec=DigitalSpec(kind=OptionType.CALL, strike=K, expiry=T, payout=payout),
        market=MarketData(spot=S0, rate=r, dividend_yield=q),
        sigma=0.25,  # bounds selection only
        t=0.0,
    )
    dom = BSDomainConfig(
        policy=BSDomainPolicy.LOG_NSIGMA,
        n_sigma=float(n_sigma_bounds),
        center="strike",
        spacing=SpacingPolicy.CLUSTERED,
        cluster_strength=2.0,
    )
    bounds = bs_compute_bounds(p_for_bounds, coord=coord, cfg=dom)

    def left_gamma(tau: float) -> float:
        return 0.0

    def right_gamma(tau: float) -> float:
        return float(payout * ctx.df(float(tau)))

    bc = RobinBC(
        left=RobinBCSide(alpha=lambda tau: 1.0, beta=lambda tau: 0.0, gamma=left_gamma),
        right=RobinBCSide(
            alpha=lambda tau: 1.0, beta=lambda tau: 0.0, gamma=right_gamma
        ),
    )

    mu = float(r - q)

    def _sigma2(x: np.ndarray | float, tau: float) -> np.ndarray:
        tau_eff = max(float(tau), 1e-8)
        x_arr = np.asarray(x, dtype=float)
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

    payoff = inst.payoff

    def ic(x: float) -> float:
        return float(payoff(float(to_S(x))))

    problem = LinearParabolicPDE1D(a=a, b=b, c=c, bc=bc, ic=ic)

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
        problem,
        grid_cfg=grid_cfg,
        method="rannacher",
        advection=AdvectionScheme.CENTRAL,
        store="final",
        ic_transform=ic_transform,
    )

    x = np.asarray(sol.grid.x, dtype=float)
    u = np.asarray(sol.u_final, dtype=float)
    return float(np.interp(x0, x, u))


def _quantlib_fd_digital_price_from_implied_surface(
    ql,
    *,
    implied,
    S0: float,
    K: float,
    r: float,
    q: float,
    T: float,
    payout: float = 1.0,
    tGrid: int = 401,
    xGrid: int = 401,
    dampingSteps: int = 2,
    n_strikes: int = 61,
    strike_min: float | None = None,
    strike_max: float | None = None,
) -> float:
    """QuantLib FD digital with localVol=True from an arbitrary implied surface."""
    calendar = ql.TARGET()
    dc = ql.Actual365Fixed()

    eval_date = ql.Date(2, ql.January, 2020)
    ql.Settings.instance().evaluationDate = eval_date

    maturity = eval_date + int(round(float(T) * 365))

    spot = ql.QuoteHandle(ql.SimpleQuote(float(S0)))
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, float(r), dc))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, float(q), dc))

    expiries = [float(t) for t in np.asarray(implied.expiries, dtype=float).reshape(-1)]
    dates = [eval_date + int(round(t * 365)) for t in expiries]

    k_lo = float(0.6 * S0) if strike_min is None else float(strike_min)
    k_hi = float(1.6 * S0) if strike_max is None else float(strike_max)
    strikes = np.linspace(k_lo, k_hi, int(n_strikes), dtype=float)

    vol_matrix = ql.Matrix(len(strikes), len(dates))
    for j, t in enumerate(expiries):
        vols = np.asarray(implied.iv(strikes, t), dtype=float)
        for i in range(len(strikes)):
            vol_matrix[i][j] = float(vols[i])

    black_surface = ql.BlackVarianceSurface(
        eval_date,
        calendar,
        dates,
        [float(k) for k in strikes],
        vol_matrix,
        dc,
    )
    black_surface.enableExtrapolation()

    vol_ts = ql.BlackVolTermStructureHandle(black_surface)
    process = ql.BlackScholesMertonProcess(spot, q_ts, r_ts, vol_ts)

    payoff = ql.CashOrNothingPayoff(ql.Option.Call, float(K), float(payout))
    exercise = ql.EuropeanExercise(maturity)
    opt = ql.VanillaOption(payoff, exercise)

    scheme_ctor = getattr(ql.FdmSchemeDesc, "CrankNicolson", None)
    scheme = scheme_ctor() if callable(scheme_ctor) else ql.FdmSchemeDesc.Douglas()

    try:
        engine = ql.FdBlackScholesVanillaEngine(
            process, int(tGrid), int(xGrid), int(dampingSteps), scheme, True
        )
    except TypeError:
        engine = ql.FdBlackScholesVanillaEngine(
            process, int(tGrid), int(xGrid), int(dampingSteps), scheme, True, None
        )

    opt.setPricingEngine(engine)
    return float(opt.NPV())


def test_localvol_digital_matches_quantlib_fd_on_skewed_svi_surface() -> None:
    """Cross-check on a *non-flat* (skewed) SVI surface.

    We compare our LocalVol(SVI)->PDE digital price against QuantLib's FD engine
    in local-vol mode. Since the implied->local mapping differs across stacks
    (interpolation and numerical derivatives), we validate:
      - numerical stability (coarse vs fine grid plateau) for both solvers
      - agreement within a tolerance tied to the observed discretization error
    """
    ql = pytest.importorskip(
        "QuantLib",
        reason="QuantLib is required for this cross-check (pip install QuantLib).",
    )

    from option_pricing.types import MarketData
    from option_pricing.vol.surface import VolSurface
    from option_pricing.vol.svi import SVIParams, SVISmile

    S0 = 100.0
    K = 100.0
    r = 0.02
    q = 0.00
    T = 0.75
    payout = 1.0

    # Moderate equity-like skew. We enforce w(0,T)=sigma_atm^2*T by construction.
    sigma_atm = 0.25
    expiries = np.asarray([0.25, 0.75, 1.25], dtype=float)
    smiles = []
    for Ti in expiries:
        b = 0.10 * float(Ti)
        svi_sig = 0.30
        rho = -0.40
        m = 0.0
        w_atm = (sigma_atm * sigma_atm) * float(Ti)
        a = max(1e-10, float(w_atm - b * svi_sig))
        smiles.append(
            SVISmile(
                T=float(Ti),
                params=SVIParams(a=a, b=b, rho=rho, m=m, sigma=svi_sig),
            )
        )

    # Our PDE: plateau check
    ours_coarse = _our_localvol_pde_digital_price_from_svi_smiles(
        smiles=smiles,
        expiries=expiries,
        S0=S0,
        K=K,
        r=r,
        q=q,
        T=T,
        payout=payout,
        Nx=301,
        Nt=301,
    )
    ours_fine = _our_localvol_pde_digital_price_from_svi_smiles(
        smiles=smiles,
        expiries=expiries,
        S0=S0,
        K=K,
        r=r,
        q=q,
        T=T,
        payout=payout,
        Nx=401,
        Nt=401,
    )

    # QuantLib PDE: build implied surface by sampling our SVI surface
    ctx = MarketData(spot=S0, rate=r, dividend_yield=q).to_context()
    implied = VolSurface(expiries=expiries, smiles=tuple(smiles), forward=ctx.fwd)

    ql_coarse = _quantlib_fd_digital_price_from_implied_surface(
        ql,
        implied=implied,
        S0=S0,
        K=K,
        r=r,
        q=q,
        T=T,
        payout=payout,
        tGrid=201,
        xGrid=201,
        dampingSteps=2,
        n_strikes=61,
    )
    ql_fine = _quantlib_fd_digital_price_from_implied_surface(
        ql,
        implied=implied,
        S0=S0,
        K=K,
        r=r,
        q=q,
        T=T,
        payout=payout,
        tGrid=401,
        xGrid=401,
        dampingSteps=2,
        n_strikes=81,
    )

    # Basic bounds
    df = math.exp(-r * T)
    for name, px in {
        "ours_coarse": ours_coarse,
        "ours_fine": ours_fine,
        "ql_coarse": ql_coarse,
        "ql_fine": ql_fine,
    }.items():
        assert 0.0 <= px <= payout * df + 1e-8, f"{name} out of bounds: {px}"

    # Stability / plateau
    ours_step = abs(ours_fine - ours_coarse)
    ql_step = abs(ql_fine - ql_coarse)
    assert (
        ours_step < 2.0e-2
    ), f"ours not stable: coarse={ours_coarse:.8f}, fine={ours_fine:.8f}, step={ours_step:.3g}"
    assert (
        ql_step < 2.0e-2
    ), f"QuantLib not stable: coarse={ql_coarse:.8f}, fine={ql_fine:.8f}, step={ql_step:.3g}"

    # Cross-library agreement.
    # (Tol anchored to observed discretization error + a small floor.)
    tol = max(3.0e-2, 10.0 * (ours_step + ql_step))
    diff = abs(ours_fine - ql_fine)
    assert diff < tol, (
        f"ours_fine={ours_fine:.8f}, QuantLib_fine={ql_fine:.8f}, abs_err={diff:.3g}, "
        f"tol={tol:.3g} (ours_step={ours_step:.3g}, ql_step={ql_step:.3g})"
    )
