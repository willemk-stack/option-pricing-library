import math

import numpy as np
import pytest

from option_pricing.numerics.grids import SpacingPolicy
from option_pricing.numerics.pde import AdvectionScheme, PDESolution1D
from option_pricing.numerics.pde.domain import Coord  # FIX: correct Coord import
from option_pricing.pricers.black_scholes import bs_price
from option_pricing.pricers.pde.domain import (  # FIX: BS domain config/policy
    BSDomainConfig as DomainConfig,
)
from option_pricing.pricers.pde.domain import (
    BSDomainPolicy as DomainPolicy,
)
from option_pricing.pricers.pde_pricer import bs_price_pde_european
from option_pricing.types import OptionType


def _default_domain_cfg(
    *, spacing: SpacingPolicy = SpacingPolicy.CLUSTERED
) -> DomainConfig:
    """Conservative BS-specific domain defaults that work well across typical test cases."""
    return DomainConfig(
        policy=DomainPolicy.LOG_NSIGMA,
        n_sigma=6.0,
        center="strike",
        spacing=spacing,
        cluster_strength=2.0,
    )


@pytest.mark.parametrize(
    "coord, rel_tol",
    [
        (Coord.LOG_S, 8e-3),
        (Coord.S, 2.5e-2),
    ],
)
@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
def test_bs_price_pde_matches_black_scholes(make_inputs, coord, rel_tol, kind):
    """Finite-difference PDE price should track the closed-form Black-Scholes price."""
    p = make_inputs(
        S=100.0,
        K=105.0,
        r=0.03,
        q=0.01,
        sigma=0.25,
        T=1.0,
        kind=kind,
    )

    dom = _default_domain_cfg(spacing=SpacingPolicy.CLUSTERED)

    pde = float(
        bs_price_pde_european(
            p,
            coord=coord,
            domain_cfg=dom,
            Nx=201,
            Nt=201,
            method="cn",
            advection=AdvectionScheme.CENTRAL,
        )
    )
    ref = float(bs_price(p))

    abs_err = abs(pde - ref)
    assert abs_err <= max(2.0e-3, rel_tol * abs(ref))


def test_bs_price_pde_return_solution_is_consistent(make_inputs):
    """When return_solution=True, returned price must match interpolation of sol.u_final."""
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.05,
        q=0.02,
        sigma=0.20,
        T=1.0,
        kind=OptionType.CALL,
    )

    dom = _default_domain_cfg(spacing=SpacingPolicy.UNIFORM)

    price, sol = bs_price_pde_european(
        p,
        coord=Coord.LOG_S,
        domain_cfg=dom,
        Nx=151,
        Nt=151,
        method="cn",
        advection=AdvectionScheme.CENTRAL,
        return_solution=True,
    )

    assert isinstance(sol, PDESolution1D)
    assert sol.u.ndim == 2
    assert sol.u.shape[0] == 1
    assert sol.u.shape[1] == sol.grid.x.shape[0]

    x0 = math.log(float(p.S))
    x = np.asarray(sol.grid.x, dtype=float)
    u = np.asarray(sol.u_final, dtype=float)

    interp = float(np.interp(x0, x, u))
    assert abs(float(price) - interp) <= 1e-12


@pytest.mark.parametrize(
    "Nx, Nt, match",
    [
        (3, 50, r"Need Nx>=4"),
        (50, 1, r"Nt must be >= 2"),
    ],
)
def test_bs_price_pde_rejects_too_small_grids(make_inputs, Nx, Nt, match):
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.01,
        q=0.00,
        sigma=0.20,
        T=0.5,
        kind=OptionType.CALL,
    )

    dom = _default_domain_cfg()

    with pytest.raises(ValueError, match=match):
        bs_price_pde_european(
            p,
            coord=Coord.LOG_S,
            domain_cfg=dom,
            Nx=Nx,
            Nt=Nt,
            method="cn",
        )


def test_bs_price_pde_unknown_method_raises(make_inputs):
    p = make_inputs(
        S=100.0,
        K=110.0,
        r=0.03,
        q=0.00,
        sigma=0.25,
        T=1.0,
        kind=OptionType.CALL,
    )

    dom = _default_domain_cfg()

    with pytest.raises(ValueError, match=r"Unknown method"):
        bs_price_pde_european(
            p, coord=Coord.LOG_S, domain_cfg=dom, Nx=101, Nt=101, method="nope"
        )


def test_bs_price_pde_manual_domain_requires_bounds(make_inputs):
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.02,
        q=0.00,
        sigma=0.20,
        T=1.0,
        kind=OptionType.CALL,
    )

    dom = DomainConfig(policy=DomainPolicy.MANUAL)

    with pytest.raises(ValueError, match=r"MANUAL policy requires"):
        bs_price_pde_european(p, coord=Coord.LOG_S, domain_cfg=dom, Nx=101, Nt=101)


def test_bs_price_pde_accepts_coord_string(make_inputs):
    p = make_inputs(
        S=100.0,
        K=95.0,
        r=0.04,
        q=0.01,
        sigma=0.30,
        T=0.75,
        kind=OptionType.PUT,
    )

    dom = _default_domain_cfg()

    price_enum = float(
        bs_price_pde_european(
            p, coord=Coord.LOG_S, domain_cfg=dom, Nx=121, Nt=121, method="cn"
        )
    )
    price_str = float(
        bs_price_pde_european(
            p, coord="logS", domain_cfg=dom, Nx=121, Nt=121, method="cn"
        )
    )

    assert abs(price_enum - price_str) <= 1e-12


@pytest.mark.parametrize("advection", [AdvectionScheme.CENTRAL, AdvectionScheme.UPWIND])
def test_bs_price_pde_runs_for_supported_advection_schemes(make_inputs, advection):
    """Smoke test: both advection schemes should yield finite prices."""
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.02,
        q=0.00,
        sigma=0.35,
        T=1.0,
        kind=OptionType.CALL,
    )

    dom = _default_domain_cfg()

    price = float(
        bs_price_pde_european(
            p,
            coord=Coord.LOG_S,
            domain_cfg=dom,
            Nx=151,
            Nt=151,
            method="cn",
            advection=advection,
        )
    )

    assert np.isfinite(price)
    assert price >= 0.0


@pytest.mark.slow
def test_bs_price_pde_error_decreases_with_grid_refinement(make_inputs):
    """A coarse convergence sanity check (not a strict order test)."""
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.03,
        q=0.01,
        sigma=0.20,
        T=1.0,
        kind=OptionType.CALL,
    )

    dom = _default_domain_cfg(spacing=SpacingPolicy.CLUSTERED)
    ref = float(bs_price(p))

    pde_1 = float(
        bs_price_pde_european(
            p,
            coord=Coord.LOG_S,
            domain_cfg=dom,
            Nx=81,
            Nt=81,
            method="cn",
            advection=AdvectionScheme.CENTRAL,
        )
    )
    pde_2 = float(
        bs_price_pde_european(
            p,
            coord=Coord.LOG_S,
            domain_cfg=dom,
            Nx=161,
            Nt=161,
            method="cn",
            advection=AdvectionScheme.CENTRAL,
        )
    )

    err_1 = abs(pde_1 - ref)
    err_2 = abs(pde_2 - ref)

    assert err_2 <= 0.85 * err_1 + 2e-4
