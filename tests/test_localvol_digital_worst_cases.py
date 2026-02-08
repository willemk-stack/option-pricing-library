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

    Includes very short expiries so we can stress short-maturity digitals.
    """
    from option_pricing.types import MarketData
    from option_pricing.vol.surface import LocalVolSurface, VolSurface
    from option_pricing.vol.svi import SVIParams, SVISmile

    ctx = MarketData(spot=S0, rate=r, dividend_yield=q).to_context()

    smiles: list[SVISmile] = []
    for Ti in expiries:
        Ti = float(Ti)

        # Moderate equity-like skew. Keep parameters scaling with T so that
        # total variance stays well-behaved as T -> 0.
        b = 0.10 * Ti
        svi_sig = 0.30
        rho = -0.40
        m = 0.0

        w_atm = (sigma_atm * sigma_atm) * Ti
        # w(0)=a + b*sqrt(m^2 + sigma^2) = a + b*svi_sig  (since m=0)
        a = max(1e-12, float(w_atm - b * svi_sig))

        smiles.append(
            SVISmile(T=Ti, params=SVIParams(a=a, b=b, rho=rho, m=m, sigma=svi_sig))
        )

    implied = VolSurface(
        expiries=np.asarray(expiries, dtype=float),
        smiles=tuple(smiles),
        forward=ctx.fwd,
    )

    with warnings.catch_warnings():
        # SciPy warnings can appear due to interpolation internals on some versions
        warnings.simplefilter("ignore")
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
    n_sigma: float = 8.0,
) -> float:
    """Price a unit cash-or-nothing digital call with the PDE in LOG_S.

    Uses numerics that are generally robust for digitals:
    - clustered grid around strike
    - Rannacher time-stepping
    - L2 projection of the discontinuous payoff
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

    ctx = MarketData(spot=S0, rate=r, dividend_yield=q).to_context()

    coord = Coord.LOG_S
    to_x, to_S = bs_coord_maps(coord)
    x0 = float(np.asarray(to_x(S0)).reshape(()))
    xK = float(np.asarray(to_x(K)).reshape(()))

    # PDE bounds (computed from a sigma guess; LV is used inside coefficients)
    p_for_bounds = PricingInputs(
        spec=DigitalSpec(kind=OptionType.CALL, strike=K, expiry=T, payout=payout),
        market=MarketData(spot=S0, rate=r, dividend_yield=q),
        sigma=float(sigma_for_bounds),
        t=0.0,
    )
    dom = BSDomainConfig(
        policy=BSDomainPolicy.LOG_NSIGMA,
        n_sigma=float(n_sigma),
        center="strike",
        spacing=SpacingPolicy.CLUSTERED,
        cluster_strength=2.0,
    )
    bounds = bs_compute_bounds(p_for_bounds, coord=coord, cfg=dom)

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

    digital = DigitalOption(expiry=T, strike=K, payout=payout, kind=OptionType.CALL)

    # Digital call PV tends to payout*df(tau) as S -> +inf, and 0 as S -> 0.
    bc = RobinBC(
        left=RobinBCSide(
            alpha=lambda tau: 1.0, beta=lambda tau: 0.0, gamma=lambda tau: 0.0
        ),
        right=RobinBCSide(
            alpha=lambda tau: 1.0,
            beta=lambda tau: 0.0,
            gamma=lambda tau: float(payout * ctx.df(float(tau))),
        ),
    )

    def ic(x: float) -> float:
        return float(digital.payoff(float(to_S(x))))

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


def test_localvol_digital_worst_cases_short_maturity_and_wings() -> None:
    """Worst-case stress test for LocalVol(PDE) digitals.

    Covers:
    - very short maturities
    - far OTM/ITM strikes (wings)
    - basic monotonicity in strike
    - local-vol sanity (finite and non-negative on the PDE-relevant domain)

    This is a stability/robustness test (not a strict accuracy benchmark).
    """
    S0 = 100.0
    r = 0.02
    q = 0.00
    payout = 1.0
    sigma_atm = 0.25

    # Include short expiries so LV interpolation/differentiation is defined there.
    expiries = np.asarray(
        [1.0 / 365.0, 7.0 / 365.0, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00], dtype=float
    )
    ctx, lv = _build_skewed_svi_localvol_surface(
        S0=S0, r=r, q=q, sigma_atm=sigma_atm, expiries=expiries
    )

    # Short maturities + wings are the numerically nastiest for digitals.
    scenarios: dict[float, list[float]] = {
        float(1.0 / 365.0): [60.0, 100.0, 140.0],
        float(7.0 / 365.0): [60.0, 100.0, 140.0],
        1.0: [50.0, 100.0, 150.0],
    }

    # Two-level ladder: keep runtime reasonable but still catch obvious instability.
    mid = (201, 201)
    fine = (301, 301)

    # Plateau tolerance: digitals are discontinuous, and LV adds coefficient variability.
    # Use a slightly looser floor for very short maturities.
    def _max_step(T: float) -> float:
        return 7.0e-2 if T < 0.05 else 3.5e-2

    # Monotonicity slack (discretization noise allowance)
    mono_eps = 7.5e-3

    for T, Ks in scenarios.items():
        df = float(ctx.df(float(T)))

        # --- local vol diagnostics on a representative domain (ATM-centered bounds)
        # (We only need to check this once per T.)
        # Sample on a wide domain; failures here often indicate Dupire blow-ups.
        from option_pricing.numerics.grids import SpacingPolicy
        from option_pricing.numerics.pde.domain import Coord
        from option_pricing.pricers.pde.domain import (
            BSDomainConfig,
            BSDomainPolicy,
            bs_compute_bounds,
        )
        from option_pricing.types import (
            DigitalSpec,
            MarketData,
            OptionType,
            PricingInputs,
        )

        p_for_bounds = PricingInputs(
            spec=DigitalSpec(kind=OptionType.CALL, strike=S0, expiry=T, payout=payout),
            market=MarketData(spot=S0, rate=r, dividend_yield=q),
            sigma=float(sigma_atm),
            t=0.0,
        )
        dom = BSDomainConfig(
            policy=BSDomainPolicy.LOG_NSIGMA,
            n_sigma=9.0,
            center="spot",
            spacing=SpacingPolicy.UNIFORM,
        )
        bnd = bs_compute_bounds(p_for_bounds, coord=Coord.LOG_S, cfg=dom)
        x_samp = np.linspace(float(bnd.x_lb), float(bnd.x_ub), 256)
        S_samp = np.exp(x_samp)
        lv_var = np.asarray(lv.local_var(S_samp, float(T)), dtype=float)
        assert np.all(
            np.isfinite(lv_var)
        ), f"T={T}: local_var produced non-finite values"
        assert (
            np.min(lv_var) >= -1e-10
        ), f"T={T}: local_var went negative: min={np.min(lv_var)}"

        # --- pricing checks
        fine_prices: list[tuple[float, float]] = []
        for K in Ks:
            p_mid = _price_localvol_digital_pde(
                S0=S0,
                K=float(K),
                T=float(T),
                payout=payout,
                r=r,
                q=q,
                sigma_for_bounds=sigma_atm,
                lv=lv,
                Nx=mid[0],
                Nt=mid[1],
                n_sigma=9.0,
            )
            p_fine = _price_localvol_digital_pde(
                S0=S0,
                K=float(K),
                T=float(T),
                payout=payout,
                r=r,
                q=q,
                sigma_for_bounds=sigma_atm,
                lv=lv,
                Nx=fine[0],
                Nt=fine[1],
                n_sigma=9.0,
            )

            # Basic bounds (PV must be between 0 and payout*df)
            assert (
                0.0 <= p_fine <= payout * df + 1e-8
            ), f"T={T}, K={K}: digital out of bounds: {p_fine}"

            # Stability / plateau under refinement
            step = abs(p_fine - p_mid)
            assert step < _max_step(
                float(T)
            ), f"T={T}, K={K}: refinement step too large: mid={p_mid:.8f}, fine={p_fine:.8f}, step={step:.3g}"

            fine_prices.append((float(K), float(p_fine)))

        # Monotonicity in strike: for a digital call, price should decrease as K increases.
        fine_prices.sort(key=lambda t: t[0])
        for (K1, p1), (K2, p2) in zip(fine_prices[:-1], fine_prices[1:], strict=True):
            assert (
                p1 + mono_eps >= p2
            ), f"T={T}: non-monotone digital: K1={K1}, p1={p1}, K2={K2}, p2={p2}"
