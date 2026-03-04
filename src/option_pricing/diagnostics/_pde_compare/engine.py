"""Shared runner for PDE-vs-analytic diagnostics.

This module is intentionally instrument-agnostic: instrument-specific diagnostics
packages inject the analytic benchmark function and the PDE pricer function.

The runner returns plain dictionaries (records) so downstream tables/plots can be
shared and the output is easy to serialize.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from time import perf_counter
from typing import Any, cast

import numpy as np

from option_pricing import PricingInputs
from option_pricing.numerics.pde import AdvectionScheme, PDESolution1D
from option_pricing.numerics.pde.domain import Coord
from option_pricing.pricers.pde.domain import BSDomainConfig

from .meta import meta_from_inputs

AnalyticFn = Callable[[PricingInputs], float]


def coerce_coord(c: Coord | str) -> Coord:
    return Coord(c)


def coerce_advection(a: str | AdvectionScheme) -> AdvectionScheme:
    if isinstance(a, AdvectionScheme):
        return a
    a_str = str(a).strip().lower()
    try:
        return AdvectionScheme(a_str)
    except ValueError as e:
        raise ValueError(
            f"Unknown advection='{a}'. Expected one of: {[x.value for x in AdvectionScheme]}"
        ) from e


def x0_for_coord(p: PricingInputs, coord: Coord) -> float:
    if coord == Coord.LOG_S:
        return float(np.log(p.S))
    if coord == Coord.S:
        return float(p.S)
    raise ValueError(f"Unhandled coord: {coord}")


def _call_pde(
    pde_fn: Callable[..., Any],
    p: PricingInputs,
    *,
    coord: Coord,
    domain_cfg: BSDomainConfig,
    Nx: int,
    Nt: int,
    method: str,
    advection: AdvectionScheme,
    return_solution: bool,
    pde_kwargs: dict[str, Any],
) -> tuple[float, PDESolution1D | None]:
    """Invoke an injected PDE pricer with common kwargs.

    Expected signature (most pricers in this repo follow this):

        pde_fn(
            p,
            coord=..., domain_cfg=..., Nx=..., Nt=...,
            method=..., advection=..., return_solution=...,
            **pde_kwargs
        )
    """

    out = pde_fn(
        p,
        coord=coord,
        domain_cfg=domain_cfg,
        Nx=int(Nx),
        Nt=int(Nt),
        method=str(method),
        advection=advection,
        return_solution=bool(return_solution),
        **pde_kwargs,
    )

    sol: PDESolution1D | None = None
    if return_solution:
        price, sol = cast(tuple[float, PDESolution1D], out)
        return float(price), sol
    return float(out), None


def run_once(
    p: PricingInputs,
    *,
    domain_cfg: BSDomainConfig,
    analytic_fn: AnalyticFn,
    pde_fn: Callable[..., Any],
    Nx: int = 401,
    Nt: int = 401,
    coord: Coord | str = Coord.LOG_S,
    method: str = "cn",
    advection: str | AdvectionScheme = "central",
    spacing: str = "clustered",
    contract: str | None = None,
    return_solution: bool = False,
    extra: dict[str, Any] | None = None,
    **pde_kwargs: Any,
) -> dict[str, object] | tuple[dict[str, object], PDESolution1D]:
    """Run a single PDE-vs-analytic comparison.

    Returns a record dict, or (record, solution) if *return_solution* is True.
    """

    coord_enum = coerce_coord(coord)
    adv = coerce_advection(advection)

    # Analytic benchmark
    analytic = float(analytic_fn(p))

    # PDE solve (timed)
    t0 = perf_counter()
    pde_price, sol = _call_pde(
        pde_fn,
        p,
        coord=coord_enum,
        domain_cfg=domain_cfg,
        Nx=int(Nx),
        Nt=int(Nt),
        method=str(method),
        advection=adv,
        return_solution=return_solution,
        pde_kwargs=pde_kwargs,
    )
    runtime_ms = (perf_counter() - t0) * 1e3

    abs_err = float(abs(pde_price - analytic))
    rel_err = float(abs_err / abs(analytic)) if analytic != 0.0 else float("nan")

    rec: dict[str, object] = {
        **meta_from_inputs(p),
        "contract": str(contract) if contract is not None else "",
        "coord": str(getattr(coord_enum, "value", coord_enum)),
        "method": str(method),
        "advection": adv.value,
        "spacing": str(spacing),
        "Nx": int(Nx),
        "Nt": int(Nt),
        "x0": float(x0_for_coord(p, coord_enum)),
        "analytic": float(analytic),
        "pde": float(pde_price),
        "err": float(pde_price - analytic),
        "abs_err": float(abs_err),
        "rel_err": float(rel_err),
        "runtime_ms": float(runtime_ms),
    }

    if extra:
        # Extra columns (e.g. ic_remedy for digitals) must be JSON-ish.
        rec.update({k: v for k, v in extra.items()})

    if return_solution:
        if sol is None:
            raise TypeError(
                "return_solution=True but the injected pde_fn did not return (price, PDESolution1D)."
            )
        return rec, sol

    return rec


def run_cases(
    cases: Iterable[tuple[str, PricingInputs]],
    *,
    domain_cfg: BSDomainConfig,
    analytic_fn: AnalyticFn,
    pde_fn: Callable[..., Any],
    Nx: int = 401,
    Nt: int = 401,
    coord: Coord | str = Coord.LOG_S,
    method: str = "cn",
    advection: str | AdvectionScheme = "central",
    spacing: str = "clustered",
    contract: str | None = None,
    extra: dict[str, Any] | None = None,
    **pde_kwargs: Any,
) -> list[dict[str, object]]:
    """Run a named list of cases.

    Always returns a list of records. Failures are captured as ``status='error'``
    records to keep diagnostics pipelines robust.
    """

    coord_enum = coerce_coord(coord)
    adv = coerce_advection(advection)

    records: list[dict[str, object]] = []
    for name, p in cases:
        base: dict[str, object] = {
            "case": str(name),
            **meta_from_inputs(p),
            "contract": str(contract) if contract is not None else "",
            "coord": str(getattr(coord_enum, "value", coord_enum)),
            "method": str(method),
            "advection": adv.value,
            "spacing": str(spacing),
            "Nx": int(Nx),
            "Nt": int(Nt),
            "x0": float(x0_for_coord(p, coord_enum)),
        }

        if extra:
            base.update({k: v for k, v in extra.items()})

        try:
            rec = run_once(
                p,
                domain_cfg=domain_cfg,
                analytic_fn=analytic_fn,
                pde_fn=pde_fn,
                Nx=Nx,
                Nt=Nt,
                coord=coord_enum,
                method=method,
                advection=adv,
                spacing=spacing,
                contract=contract,
                return_solution=False,
                extra=extra,
                **pde_kwargs,
            )
            assert isinstance(rec, dict)
            records.append({"status": "ok", **base, **rec})

        except Exception as e:
            # Keep failure as data.
            try:
                analytic = float(analytic_fn(p))
            except Exception:
                analytic = float("nan")
            records.append(
                {
                    "status": "error",
                    "error": repr(e),
                    **base,
                    "analytic": analytic,
                    "pde": np.nan,
                    "err": np.nan,
                    "abs_err": np.nan,
                    "rel_err": np.nan,
                    "runtime_ms": np.nan,
                }
            )

    return records


def sweep_nx(
    p: PricingInputs,
    *,
    domain_cfg: BSDomainConfig,
    analytic_fn: AnalyticFn,
    pde_fn: Callable[..., Any],
    Nx_list: Sequence[int] = (101, 201, 401, 801),
    Nt: int = 401,
    coord: Coord | str = Coord.LOG_S,
    method: str = "cn",
    advection: str | AdvectionScheme = "central",
    spacing: str = "clustered",
    contract: str | None = None,
    extra: dict[str, Any] | None = None,
    **pde_kwargs: Any,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for Nx in Nx_list:
        rec = run_once(
            p,
            domain_cfg=domain_cfg,
            analytic_fn=analytic_fn,
            pde_fn=pde_fn,
            Nx=int(Nx),
            Nt=int(Nt),
            coord=coord,
            method=method,
            advection=advection,
            spacing=spacing,
            contract=contract,
            return_solution=False,
            extra=extra,
            **pde_kwargs,
        )
        assert isinstance(rec, dict)
        records.append({"sweep": "Nx", **rec})
    return records


def sweep_nt(
    p: PricingInputs,
    *,
    domain_cfg: BSDomainConfig,
    analytic_fn: AnalyticFn,
    pde_fn: Callable[..., Any],
    Nt_list: Sequence[int] = (51, 101, 201, 401),
    Nx: int = 401,
    coord: Coord | str = Coord.LOG_S,
    method: str = "cn",
    advection: str | AdvectionScheme = "central",
    spacing: str = "clustered",
    contract: str | None = None,
    extra: dict[str, Any] | None = None,
    **pde_kwargs: Any,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for Nt in Nt_list:
        rec = run_once(
            p,
            domain_cfg=domain_cfg,
            analytic_fn=analytic_fn,
            pde_fn=pde_fn,
            Nx=int(Nx),
            Nt=int(Nt),
            coord=coord,
            method=method,
            advection=advection,
            spacing=spacing,
            contract=contract,
            return_solution=False,
            extra=extra,
            **pde_kwargs,
        )
        assert isinstance(rec, dict)
        records.append({"sweep": "Nt", **rec})
    return records


def sweep_nx_nt(
    p: PricingInputs,
    *,
    domain_cfg: BSDomainConfig,
    analytic_fn: AnalyticFn,
    pde_fn: Callable[..., Any],
    Nx_list: Sequence[int] = (101, 201, 401, 801),
    Nt_list: Sequence[int] = (51, 101, 201, 401),
    coord: Coord | str = Coord.LOG_S,
    method: str = "cn",
    advection: str | AdvectionScheme = "central",
    spacing: str = "clustered",
    contract: str | None = None,
    extra: dict[str, Any] | None = None,
    **pde_kwargs: Any,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for Nx in Nx_list:
        for Nt in Nt_list:
            rec = run_once(
                p,
                domain_cfg=domain_cfg,
                analytic_fn=analytic_fn,
                pde_fn=pde_fn,
                Nx=int(Nx),
                Nt=int(Nt),
                coord=coord,
                method=method,
                advection=advection,
                spacing=spacing,
                contract=contract,
                return_solution=False,
                extra=extra,
                **pde_kwargs,
            )
            assert isinstance(rec, dict)
            records.append({"sweep": "NxNt", **rec})
    return records


__all__ = [
    "coerce_coord",
    "coerce_advection",
    "x0_for_coord",
    "run_once",
    "run_cases",
    "sweep_nx",
    "sweep_nt",
    "sweep_nx_nt",
]
