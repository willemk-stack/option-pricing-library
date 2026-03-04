from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from option_pricing.instruments.digital import DigitalOption
from option_pricing.numerics.pde import AdvectionScheme
from option_pricing.numerics.pde.domain import Coord
from option_pricing.pricers.black_scholes import digital_price
from option_pricing.pricers.pde.domain import BSDomainConfig
from option_pricing.pricers.pde_pricer import bs_digital_price_pde
from option_pricing.types import DigitalSpec, PricingInputs

from .._pde_compare import engine


def _analytic_digital(p: PricingInputs[DigitalSpec]) -> float:
    inst = DigitalOption(
        kind=p.spec.kind,
        strike=float(p.spec.strike),
        expiry=float(p.tau),  # instruments use tau
        payout=float(p.spec.payout),
    )
    return float(digital_price(inst, ctx=p.ctx, sigma=p.sigma))


def run_once(
    p: PricingInputs[DigitalSpec],
    *,
    domain_cfg: BSDomainConfig,
    Nx: int = 401,
    Nt: int = 401,
    coord: Coord | str = Coord.LOG_S,
    method: str = "rannacher",
    advection: str | AdvectionScheme = "central",
    spacing: str = "clustered",
    ic_remedy: str = "cell_avg",
    analytic_price_fn: Callable[[PricingInputs[DigitalSpec]], float] | None = None,
    **pde_kwargs,
) -> dict[str, object]:
    """Run one configuration for a digital option.

    Notes:
      - Digitals are discontinuous, so methods like Rannacher are often preferable.
      - ``ic_remedy`` is passed through to the PDE pricer and also recorded.
    """

    analytic_fn = _analytic_digital if analytic_price_fn is None else analytic_price_fn

    rec = engine.run_once(
        p,
        domain_cfg=domain_cfg,
        analytic_fn=analytic_fn,
        pde_fn=bs_digital_price_pde,
        Nx=Nx,
        Nt=Nt,
        coord=coord,
        method=method,
        advection=advection,
        spacing=spacing,
        contract="digital",
        extra={"ic_remedy": str(ic_remedy)},
        ic_remedy=ic_remedy,
        **pde_kwargs,
    )
    assert isinstance(rec, dict)
    return rec


def run_cases(
    cases: Iterable[tuple[str, PricingInputs[DigitalSpec]]],
    *,
    domain_cfg: BSDomainConfig,
    Nx: int = 401,
    Nt: int = 401,
    coord: Coord | str = Coord.LOG_S,
    method: str = "rannacher",
    advection: str | AdvectionScheme = "central",
    spacing: str = "clustered",
    ic_remedy: str = "cell_avg",
    analytic_price_fn: Callable[[PricingInputs[DigitalSpec]], float] | None = None,
    **pde_kwargs,
) -> list[dict[str, object]]:
    analytic_fn = _analytic_digital if analytic_price_fn is None else analytic_price_fn

    return engine.run_cases(
        cases,
        domain_cfg=domain_cfg,
        analytic_fn=analytic_fn,
        pde_fn=bs_digital_price_pde,
        Nx=Nx,
        Nt=Nt,
        coord=coord,
        method=method,
        advection=advection,
        spacing=spacing,
        contract="digital",
        extra={"ic_remedy": str(ic_remedy)},
        ic_remedy=ic_remedy,
        **pde_kwargs,
    )


def sweep_nx(
    p: PricingInputs[DigitalSpec],
    *,
    domain_cfg: BSDomainConfig,
    Nx_list: Sequence[int] = (101, 201, 401, 801),
    Nt: int = 401,
    coord: Coord | str = Coord.LOG_S,
    method: str = "rannacher",
    advection: str | AdvectionScheme = "central",
    spacing: str = "clustered",
    ic_remedy: str = "cell_avg",
    analytic_price_fn: Callable[[PricingInputs[DigitalSpec]], float] | None = None,
    **pde_kwargs,
) -> list[dict[str, object]]:
    analytic_fn = _analytic_digital if analytic_price_fn is None else analytic_price_fn
    return engine.sweep_nx(
        p,
        domain_cfg=domain_cfg,
        analytic_fn=analytic_fn,
        pde_fn=bs_digital_price_pde,
        Nx_list=Nx_list,
        Nt=Nt,
        coord=coord,
        method=method,
        advection=advection,
        spacing=spacing,
        contract="digital",
        extra={"ic_remedy": str(ic_remedy)},
        ic_remedy=ic_remedy,
        **pde_kwargs,
    )


def sweep_nt(
    p: PricingInputs[DigitalSpec],
    *,
    domain_cfg: BSDomainConfig,
    Nt_list: Sequence[int] = (51, 101, 201, 401),
    Nx: int = 401,
    coord: Coord | str = Coord.LOG_S,
    method: str = "rannacher",
    advection: str | AdvectionScheme = "central",
    spacing: str = "clustered",
    ic_remedy: str = "cell_avg",
    analytic_price_fn: Callable[[PricingInputs[DigitalSpec]], float] | None = None,
    **pde_kwargs,
) -> list[dict[str, object]]:
    analytic_fn = _analytic_digital if analytic_price_fn is None else analytic_price_fn
    return engine.sweep_nt(
        p,
        domain_cfg=domain_cfg,
        analytic_fn=analytic_fn,
        pde_fn=bs_digital_price_pde,
        Nt_list=Nt_list,
        Nx=Nx,
        coord=coord,
        method=method,
        advection=advection,
        spacing=spacing,
        contract="digital",
        extra={"ic_remedy": str(ic_remedy)},
        ic_remedy=ic_remedy,
        **pde_kwargs,
    )


def sweep_nx_nt(
    p: PricingInputs[DigitalSpec],
    *,
    domain_cfg: BSDomainConfig,
    Nx_list: Sequence[int] = (101, 201, 401, 801),
    Nt_list: Sequence[int] = (51, 101, 201, 401),
    coord: Coord | str = Coord.LOG_S,
    method: str = "rannacher",
    advection: str | AdvectionScheme = "central",
    spacing: str = "clustered",
    ic_remedy: str = "cell_avg",
    analytic_price_fn: Callable[[PricingInputs[DigitalSpec]], float] | None = None,
    **pde_kwargs,
) -> list[dict[str, object]]:
    analytic_fn = _analytic_digital if analytic_price_fn is None else analytic_price_fn
    return engine.sweep_nx_nt(
        p,
        domain_cfg=domain_cfg,
        analytic_fn=analytic_fn,
        pde_fn=bs_digital_price_pde,
        Nx_list=Nx_list,
        Nt_list=Nt_list,
        coord=coord,
        method=method,
        advection=advection,
        spacing=spacing,
        contract="digital",
        extra={"ic_remedy": str(ic_remedy)},
        ic_remedy=ic_remedy,
        **pde_kwargs,
    )


__all__ = [
    "run_once",
    "run_cases",
    "sweep_nx",
    "sweep_nt",
    "sweep_nx_nt",
]
