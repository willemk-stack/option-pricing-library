from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, SupportsFloat, SupportsInt, TypedDict, cast, overload

from option_pricing import PricingInputs, bs_price
from option_pricing.numerics.pde import AdvectionScheme, PDESolution1D
from option_pricing.numerics.pde.domain import Coord
from option_pricing.pricers.pde.domain import BSDomainConfig
from option_pricing.pricers.pde_pricer import bs_price_pde_european

from .._pde_compare import engine


@dataclass(frozen=True, slots=True)
class PDEvsBSRun:
    """Single-run result for PDE vs analytic Black-Scholes (vanilla European)."""

    # Identifiers
    method: str
    coord: str
    spacing: str

    # cfg
    advection: AdvectionScheme
    Nx: int
    Nt: int
    x0: float

    # outputs
    bs: float
    pde: float
    abs_err: float
    rel_err: float
    runtime_ms: float


class _EngineRunOnceRec(TypedDict):
    method: str
    coord: str
    spacing: str
    advection: str
    Nx: SupportsInt
    Nt: SupportsInt
    x0: SupportsFloat
    analytic: SupportsFloat
    pde: SupportsFloat
    abs_err: SupportsFloat
    rel_err: SupportsFloat
    runtime_ms: SupportsFloat


@overload
def run_once(
    p: PricingInputs,
    *,
    domain_cfg: BSDomainConfig,
    Nx: int = 401,
    Nt: int = 401,
    coord: Coord | str = Coord.LOG_S,
    method: str = "cn",
    advection: str | AdvectionScheme = "central",
    spacing: str = "clustered",
    bs_price_fn: Callable[[PricingInputs], float] | None = None,
    return_solution: Literal[False] = False,
) -> PDEvsBSRun: ...


@overload
def run_once(
    p: PricingInputs,
    *,
    domain_cfg: BSDomainConfig,
    Nx: int = 401,
    Nt: int = 401,
    coord: Coord | str = Coord.LOG_S,
    method: str = "cn",
    advection: str | AdvectionScheme = "central",
    spacing: str = "clustered",
    bs_price_fn: Callable[[PricingInputs], float] | None = None,
    return_solution: Literal[True],
) -> tuple[PDEvsBSRun, PDESolution1D]: ...


def run_once(
    p: PricingInputs,
    *,
    domain_cfg: BSDomainConfig,
    Nx: int = 401,
    Nt: int = 401,
    coord: Coord | str = Coord.LOG_S,
    method: str = "cn",
    advection: str | AdvectionScheme = "central",
    spacing: str = "clustered",
    bs_price_fn: Callable[[PricingInputs], float] | None = None,
    return_solution: bool = False,
) -> PDEvsBSRun | tuple[PDEvsBSRun, PDESolution1D]:
    """Run one configuration and return a compact dataclass.

    This wrapper delegates timing/error computation to the shared engine and keeps
    the public API stable for notebooks.
    """

    analytic_fn = bs_price if bs_price_fn is None else bs_price_fn

    out = engine.run_once(
        p,
        domain_cfg=domain_cfg,
        analytic_fn=analytic_fn,
        pde_fn=bs_price_pde_european,
        Nx=Nx,
        Nt=Nt,
        coord=coord,
        method=method,
        advection=advection,
        spacing=spacing,
        contract="vanilla",
        return_solution=return_solution,
    )

    rec: dict[str, object]
    sol: PDESolution1D | None
    if return_solution:
        rec, sol = cast(tuple[dict[str, object], PDESolution1D], out)
    else:
        rec = cast(dict[str, object], out)
        sol = None

    r = cast(_EngineRunOnceRec, rec)

    adv = AdvectionScheme(str(r["advection"]))

    res = PDEvsBSRun(
        method=str(r["method"]),
        coord=str(r["coord"]),
        spacing=str(r["spacing"]),
        advection=adv,
        Nx=int(r["Nx"]),
        Nt=int(r["Nt"]),
        x0=float(r["x0"]),
        bs=float(r["analytic"]),
        pde=float(r["pde"]),
        abs_err=float(r["abs_err"]),
        rel_err=float(r["rel_err"]),
        runtime_ms=float(r["runtime_ms"]),
    )

    if return_solution:
        if sol is None:
            raise TypeError(
                "return_solution=True but PDE solver did not return a PDESolution1D."
            )
        return res, sol

    return res


def _with_bs_alias(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Back-compat: add bs=analytic for downstream notebooks."""

    out: list[dict[str, object]] = []
    for r in records:
        rr = dict(r)
        if "analytic" in rr and "bs" not in rr:
            rr["bs"] = rr["analytic"]
        out.append(rr)
    return out


def run_cases(
    cases: Iterable[tuple[str, PricingInputs]],
    *,
    domain_cfg: BSDomainConfig,
    Nx: int = 401,
    Nt: int = 401,
    coord: Coord | str = Coord.LOG_S,
    method: str = "cn",
    advection: str | AdvectionScheme = "central",
    spacing: str = "clustered",
    bs_price_fn: Callable[[PricingInputs], float] | None = None,
) -> list[dict[str, object]]:
    analytic_fn = bs_price if bs_price_fn is None else bs_price_fn

    recs = engine.run_cases(
        cases,
        domain_cfg=domain_cfg,
        analytic_fn=analytic_fn,
        pde_fn=bs_price_pde_european,
        Nx=Nx,
        Nt=Nt,
        coord=coord,
        method=method,
        advection=advection,
        spacing=spacing,
        contract="vanilla",
    )
    return _with_bs_alias(recs)


def sweep_nx(
    p: PricingInputs,
    *,
    domain_cfg: BSDomainConfig,
    Nx_list: Sequence[int] = (101, 201, 401, 801),
    Nt: int = 401,
    coord: Coord | str = Coord.LOG_S,
    method: str = "cn",
    advection: str | AdvectionScheme = "central",
    spacing: str = "clustered",
    bs_price_fn: Callable[[PricingInputs], float] | None = None,
) -> list[dict[str, object]]:
    analytic_fn = bs_price if bs_price_fn is None else bs_price_fn
    recs = engine.sweep_nx(
        p,
        domain_cfg=domain_cfg,
        analytic_fn=analytic_fn,
        pde_fn=bs_price_pde_european,
        Nx_list=Nx_list,
        Nt=Nt,
        coord=coord,
        method=method,
        advection=advection,
        spacing=spacing,
        contract="vanilla",
    )
    return _with_bs_alias(recs)


def sweep_nt(
    p: PricingInputs,
    *,
    domain_cfg: BSDomainConfig,
    Nt_list: Sequence[int] = (51, 101, 201, 401),
    Nx: int = 401,
    coord: Coord | str = Coord.LOG_S,
    method: str = "cn",
    advection: str | AdvectionScheme = "central",
    spacing: str = "clustered",
    bs_price_fn: Callable[[PricingInputs], float] | None = None,
) -> list[dict[str, object]]:
    analytic_fn = bs_price if bs_price_fn is None else bs_price_fn
    recs = engine.sweep_nt(
        p,
        domain_cfg=domain_cfg,
        analytic_fn=analytic_fn,
        pde_fn=bs_price_pde_european,
        Nt_list=Nt_list,
        Nx=Nx,
        coord=coord,
        method=method,
        advection=advection,
        spacing=spacing,
        contract="vanilla",
    )
    return _with_bs_alias(recs)


def sweep_nx_nt(
    p: PricingInputs,
    *,
    domain_cfg: BSDomainConfig,
    Nx_list: Sequence[int] = (101, 201, 401, 801),
    Nt_list: Sequence[int] = (51, 101, 201, 401),
    coord: Coord | str = Coord.LOG_S,
    method: str = "cn",
    advection: str | AdvectionScheme = "central",
    spacing: str = "clustered",
    bs_price_fn: Callable[[PricingInputs], float] | None = None,
) -> list[dict[str, object]]:
    analytic_fn = bs_price if bs_price_fn is None else bs_price_fn
    recs = engine.sweep_nx_nt(
        p,
        domain_cfg=domain_cfg,
        analytic_fn=analytic_fn,
        pde_fn=bs_price_pde_european,
        Nx_list=Nx_list,
        Nt_list=Nt_list,
        coord=coord,
        method=method,
        advection=advection,
        spacing=spacing,
        contract="vanilla",
    )
    return _with_bs_alias(recs)


__all__ = [
    "PDEvsBSRun",
    "run_once",
    "run_cases",
    "sweep_nx",
    "sweep_nt",
    "sweep_nx_nt",
]
