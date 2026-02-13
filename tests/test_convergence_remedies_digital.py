"""Convergence tests for discontinuous payoffs (digital options).

This module is meant to validate the convergence remedies discussed in
"Convergence Remedies For Non-Smooth Payoffs in Option Pricing" (Pooley, Vetzal,
Forsyth, 2002).

Key empirical claim (Table 4 of the paper): for discontinuous payoffs, *second
order* convergence is restored when you combine:

  - a smoothing/projection of the initial condition (payoff), and
  - Rannacher timestepping (a small number of fully-implicit steps before CN).

The tests below are split into:
  1) baseline tests that should pass on the current codebase, and
  2) remedy tests that are automatically skipped until the repo exposes an
     `ic_transform` hook in `solve_pde_1d` and provides the recommended
     `ic_cell_average` / `ic_l2_projection` helpers.

How to run (without installing the package):
  PYTHONPATH=src pytest -q
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest


def bs_digital_call_price(
    *,
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    payout: float = 1.0,
) -> float:
    """Analytic Black–Scholes digital call with payoff `payout * 1{S_T >= K}`.

    This is a thin wrapper around the library implementation (Black-76 via
    forward + discount factor).
    """
    from option_pricing.instruments.digital import DigitalOption
    from option_pricing.models.black_scholes import bs as bs_model
    from option_pricing.types import MarketData, OptionType

    ctx = MarketData(spot=float(S), rate=float(r), dividend_yield=float(q)).to_context()
    inst = DigitalOption(
        expiry=float(T),
        strike=float(K),
        payout=float(payout),
        kind=OptionType.CALL,
    )
    return float(
        bs_model.digital_call_price(instr=inst, market=ctx, sigma=float(sigma))
    )


def _observed_order(err_coarse: float, err_fine: float) -> float:
    """Observed order between two successive refinements (halving dx and dt)."""
    if err_coarse <= 0 or err_fine <= 0:
        return float("nan")
    return math.log(err_coarse / err_fine) / math.log(2.0)


def _can_use_ic_transform() -> bool:
    """Return True iff `solve_pde_1d` exposes an `ic_transform` kwarg."""
    try:
        from option_pricing.numerics.pde import solve_pde_1d
    except Exception:
        return False

    sig = inspect.signature(solve_pde_1d)
    return "ic_transform" in sig.parameters


def _try_import_remedies():
    """Import ic remedies if present; otherwise return (None, None)."""
    try:
        from option_pricing.numerics.pde.ic_remedies import (  # type: ignore
            ic_cell_average,
            ic_l2_projection,
        )
    except Exception:
        return None, None
    return ic_cell_average, ic_l2_projection


def _price_digital_pde(
    *,
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    payout: float,
    Nx: int,
    Nt: int,
    method: str,
    use_clustered: bool = True,
    ic_transform=None,
) -> float:
    """Price a digital via the PDE stack, optionally injecting `ic_transform`."""

    # Local imports keep this file importable even if users run without install.
    from option_pricing.numerics.grids import GridConfig, SpacingPolicy
    from option_pricing.numerics.pde import AdvectionScheme, solve_pde_1d
    from option_pricing.numerics.pde.domain import Coord
    from option_pricing.pricers.pde.digital_black_scholes import bs_pde_wiring
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
        market=MarketData(spot=float(S), rate=float(r), dividend_yield=float(q)),
        sigma=float(sigma),
        t=0.0,
    )

    dom = BSDomainConfig(
        policy=BSDomainPolicy.LOG_NSIGMA,
        n_sigma=6.0,
        center="strike",
        spacing=SpacingPolicy.CLUSTERED if use_clustered else SpacingPolicy.UNIFORM,
        cluster_strength=2.0,
    )

    coord = Coord.LOG_S
    bounds = bs_compute_bounds(p, coord=coord, cfg=dom)
    wiring = bs_pde_wiring(p, coord, x_lb=bounds.x_lb, x_ub=bounds.x_ub)

    grid_cfg = GridConfig(
        Nx=int(Nx),
        Nt=int(Nt),
        x_lb=bounds.x_lb,
        x_ub=bounds.x_ub,
        T=float(T),
        spacing=dom.spacing,
        x_center=bounds.x_center,
        cluster_strength=dom.cluster_strength,
    )

    sol = solve_pde_1d(
        wiring.problem,
        grid_cfg=grid_cfg,
        method=method,
        advection=AdvectionScheme.CENTRAL,
        store="final",
        **({"ic_transform": ic_transform} if ic_transform is not None else {}),
    )

    x = np.asarray(sol.grid.x, dtype=float)
    u = np.asarray(sol.u_final, dtype=float)
    return float(np.interp(wiring.x_0, x, u))


@pytest.mark.parametrize("Nx,Nt", [(161, 101), (201, 151)])
def test_digital_rannacher_smoke_accuracy(Nx: int, Nt: int):
    """Sanity check: digital pricing is in the right ballpark.

    If payoff-conditioning is available (ic_transform + remedies), we assert a tighter
    tolerance using L2 projection + Rannacher. Otherwise, we assert a looser tolerance
    for Rannacher-only (discontinuous payoff).
    """
    S = 40.0
    K = 40.0
    r = 0.05
    q = 0.0
    sigma = 0.3
    T = 0.5
    payout = 1.0

    exact = bs_digital_call_price(S=S, K=K, r=r, q=q, sigma=sigma, T=T, payout=payout)

    ic_cell_average, ic_l2_projection = _try_import_remedies()
    can_hook = _can_use_ic_transform()

    ic_transform = None
    tol = 1e-2  # Rannacher-only tolerance for discontinuous payoffs

    if can_hook and ic_l2_projection is not None:
        # Use L2 projection (preferred) -> tighter tolerance is reasonable.
        from option_pricing.models.black_scholes.pde import bs_coord_maps
        from option_pricing.numerics.pde.domain import Coord

        to_x, _to_S = bs_coord_maps(Coord.LOG_S)
        xK = float(to_x(K))

        def proj_transform(grid, ic_fn):
            return ic_l2_projection(grid, ic_fn, breakpoints=[xK])

        ic_transform = proj_transform
        tol = 5e-3

    pde = _price_digital_pde(
        S=S,
        K=K,
        r=r,
        q=q,
        sigma=sigma,
        T=T,
        payout=payout,
        Nx=Nx,
        Nt=Nt,
        method="rannacher",
        ic_transform=ic_transform,
    )

    assert abs(pde - exact) < tol, (
        f"abs error {abs(pde-exact):.6g} exceeded tol {tol:.6g}. "
        f"(ic_transform={'on' if ic_transform is not None else 'off'})"
    )


@pytest.mark.slow
def test_digital_rannacher_is_not_quadratic_without_smoothing():
    """For a discontinuous payoff, Rannacher alone typically converges ~1st order."""
    S = 40.0
    K = 40.0
    r = 0.05
    q = 0.0
    sigma = 0.3
    T = 0.5
    payout = 1.0

    # Refinement schedule consistent with the paper (dt halves each refinement).
    Nx_list = [41, 81, 161, 321]
    Nt_list = [26, 51, 101, 201]

    exact = bs_digital_call_price(S=S, K=K, r=r, q=q, sigma=sigma, T=T, payout=payout)
    errs = []
    for Nx, Nt in zip(Nx_list, Nt_list, strict=True):
        pde = _price_digital_pde(
            S=S,
            K=K,
            r=r,
            q=q,
            sigma=sigma,
            T=T,
            payout=payout,
            Nx=Nx,
            Nt=Nt,
            method="rannacher",
        )
        errs.append(abs(pde - exact))

    p_last = _observed_order(errs[-2], errs[-1])

    # Weak assertion to avoid flakiness across domain settings.
    assert p_last < 1.8


@pytest.mark.slow
def test_digital_quadratic_convergence_with_smoothing_and_rannacher():
    """Target behavior (paper Table 4): smoothing + Rannacher restores ~2nd order."""
    if not _can_use_ic_transform():
        pytest.skip("solve_pde_1d does not yet expose an ic_transform hook")

    ic_cell_average, ic_l2_projection = _try_import_remedies()
    if ic_cell_average is None or ic_l2_projection is None:
        pytest.skip(
            "ic remedies module not yet implemented: option_pricing.numerics.pde.ic_remedies"
        )

    S = 40.0
    K = 40.0
    r = 0.05
    q = 0.0
    sigma = 0.3
    T = 0.5
    payout = 1.0

    from option_pricing.models.black_scholes.pde import bs_coord_maps
    from option_pricing.numerics.pde.domain import Coord

    to_x, _to_S = bs_coord_maps(Coord.LOG_S)
    xK = float(to_x(K))

    def avg_transform(grid, ic_fn):
        return ic_cell_average(grid, ic_fn, breakpoints=[xK])

    def proj_transform(grid, ic_fn):
        return ic_l2_projection(grid, ic_fn, breakpoints=[xK])

    Nx_list = [41, 81, 161, 321]
    Nt_list = [26, 51, 101, 201]
    exact = bs_digital_call_price(S=S, K=K, r=r, q=q, sigma=sigma, T=T, payout=payout)

    def run(transform):
        errs = []
        for Nx, Nt in zip(Nx_list, Nt_list, strict=True):
            pde = _price_digital_pde(
                S=S,
                K=K,
                r=r,
                q=q,
                sigma=sigma,
                T=T,
                payout=payout,
                Nx=Nx,
                Nt=Nt,
                method="rannacher",
                ic_transform=transform,
            )
            errs.append(abs(pde - exact))
        return errs

    errs_avg = run(avg_transform)
    errs_proj = run(proj_transform)

    p_avg = _observed_order(errs_avg[-2], errs_avg[-1])
    p_proj = _observed_order(errs_proj[-2], errs_proj[-1])

    assert 1.6 <= p_avg <= 2.4
    assert 1.6 <= p_proj <= 2.4
