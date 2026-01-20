from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from option_pricing import PricingInputs, bs_price
from option_pricing.numerics.pde import AdvectionScheme, PDESolution1D
from option_pricing.pricers.pde.domain import Coord, DomainConfig
from option_pricing.pricers.pde_pricer import bs_price_pde

# ----------------------------
# Results dataclass
# ----------------------------


@dataclass(frozen=True, slots=True)
class PDEvsBSRun:
    # Identifiers
    method: str
    coord: str  # "logS" or "S" (Coord value)
    spacing: str  # "uniform" / "clustered" (reporting)

    # cfg
    advection: AdvectionScheme
    Nx: int
    Nt: int
    x0: float  # spot in x-coordinates

    # outputs
    bs: float
    pde: float
    abs_err: float
    rel_err: float
    runtime_ms: float


# ----------------------------
# Input coercions
# ----------------------------


def _coerce_advection(a: str | AdvectionScheme) -> AdvectionScheme:
    if isinstance(a, AdvectionScheme):
        return a
    a_str = str(a).strip().lower()
    try:
        return AdvectionScheme(a_str)
    except ValueError as e:
        raise ValueError(
            f"Unknown advection='{a}'. Expected one of: "
            f"{[x.value for x in AdvectionScheme]}"
        ) from e


def _coerce_coord(c: Coord | str) -> Coord:
    return Coord(c)


def _x0_for_coord(p: PricingInputs, coord: Coord) -> float:
    if coord == Coord.LOG_S:
        return float(np.log(p.S))
    if coord == Coord.S:
        return float(p.S)
    raise ValueError(f"Unhandled coord: {coord}")


def _kind_str(p: PricingInputs) -> str:
    k = getattr(getattr(p, "spec", None), "kind", None)
    if k is None:
        return ""
    return str(getattr(k, "value", k))


def _meta_from_inputs(p: PricingInputs) -> dict[str, object]:
    # Safe, tolerant extraction for tables/plots
    return {
        "tau": float(getattr(p, "tau", float("nan"))),
        "spot": float(getattr(p, "S", float("nan"))),
        "strike": float(getattr(p, "K", float("nan"))),
        "kind": _kind_str(p),
        "sigma": float(getattr(p, "sigma", float("nan"))),
        "r": float(getattr(p, "r", float("nan"))),
        "q": float(getattr(p, "q", float("nan"))),
    }


# ----------------------------
# Core execution
# ----------------------------


def run_once(
    p: PricingInputs,
    *,
    domain_cfg: DomainConfig,
    Nx: int = 401,
    Nt: int = 401,
    coord: Coord | str = Coord.LOG_S,
    method: str = "cn",
    advection: str | AdvectionScheme = "central",
    spacing: str = "clustered",
    bs_price_fn: Callable[[PricingInputs], float] | None = None,
    return_solution: bool = False,
) -> PDEvsBSRun | tuple[PDEvsBSRun, PDESolution1D]:
    coord_enum = _coerce_coord(coord)
    adv = _coerce_advection(advection)

    # Analytic benchmark
    bs_fn = bs_price if bs_price_fn is None else bs_price_fn
    C_anal = float(bs_fn(p))

    # PDE solve (timed)
    t0 = perf_counter()
    out = bs_price_pde(
        p=p,
        coord=coord_enum,
        domain_cfg=domain_cfg,
        Nx=int(Nx),
        Nt=int(Nt),
        method=str(method),
        advection=adv,
        return_solution=return_solution,
    )
    runtime_ms = (perf_counter() - t0) * 1e3

    sol: PDESolution1D | None = None
    if return_solution:
        C_pde, sol = out  # type: ignore[misc]
    else:
        C_pde = out  # type: ignore[assignment]

    C_pde = float(C_pde)

    abs_err = float(abs(C_pde - C_anal))
    rel_err = float(abs_err / abs(C_anal)) if C_anal != 0.0 else float("nan")

    res = PDEvsBSRun(
        method=str(method),
        coord=str(getattr(coord_enum, "value", coord_enum)),
        spacing=str(spacing),
        advection=adv,
        Nx=int(Nx),
        Nt=int(Nt),
        x0=_x0_for_coord(p, coord_enum),
        bs=C_anal,
        pde=C_pde,
        abs_err=abs_err,
        rel_err=rel_err,
        runtime_ms=float(runtime_ms),
    )

    if return_solution:
        if sol is None:
            raise TypeError(
                "return_solution=True but bs_price_pde did not return a PDESolution1D. "
                "Expected (price, PDESolution1D)."
            )
        return res, sol

    return res


# ----------------------------
# Batch execution (cases/sweeps)
# Returns "records" (dicts) for downstream tables/plots
# ----------------------------


def run_cases(
    cases: Iterable[tuple[str, PricingInputs]],
    *,
    domain_cfg: DomainConfig,
    Nx: int = 401,
    Nt: int = 401,
    coord: Coord | str = Coord.LOG_S,
    method: str = "cn",
    advection: str | AdvectionScheme = "central",
    spacing: str = "clustered",
    bs_price_fn: Callable[[PricingInputs], float] | None = None,
) -> list[dict[str, object]]:
    from dataclasses import asdict

    import numpy as np

    coord_enum = _coerce_coord(coord)
    adv = _coerce_advection(advection)
    bs_fn = bs_price if bs_price_fn is None else bs_price_fn

    records: list[dict[str, object]] = []

    for name, p in cases:
        base = {
            "case": str(name),
            **_meta_from_inputs(p),
            "coord": str(getattr(coord_enum, "value", coord_enum)),
            "method": str(method),
            "advection": adv.value,
            "spacing": str(spacing),
            "Nx": int(Nx),
            "Nt": int(Nt),
            "x0": _x0_for_coord(p, coord_enum),
        }

        try:
            r = run_once(
                p,
                domain_cfg=domain_cfg,
                Nx=Nx,
                Nt=Nt,
                coord=coord_enum,
                method=method,
                advection=adv,
                spacing=spacing,
                bs_price_fn=bs_fn,
                return_solution=False,
            )
            assert isinstance(r, PDEvsBSRun)

            d = asdict(r)
            d["advection"] = r.advection.value
            d["err"] = float(r.pde - r.bs)

            records.append({"status": "ok", **base, **d})

        except Exception as e:
            # Record failure as data (diagnostics-friendly)
            bs_val = float(bs_fn(p))
            records.append(
                {
                    "status": "error",
                    "error": repr(e),
                    **base,
                    "bs": bs_val,
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
    domain_cfg: DomainConfig,
    Nx_list: Sequence[int] = (101, 201, 401, 801),
    Nt: int = 401,
    coord: Coord | str = Coord.LOG_S,
    method: str = "cn",
    advection: str | AdvectionScheme = "central",
    spacing: str = "clustered",
    bs_price_fn: Callable[[PricingInputs], float] | None = None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for Nx in Nx_list:
        r = run_once(
            p,
            domain_cfg=domain_cfg,
            Nx=int(Nx),
            Nt=int(Nt),
            coord=coord,
            method=method,
            advection=advection,
            spacing=spacing,
            bs_price_fn=bs_price_fn,
            return_solution=False,
        )
        assert isinstance(r, PDEvsBSRun)
        d = asdict(r)
        d["advection"] = r.advection.value
        d["err"] = float(r.pde - r.bs)
        records.append({"sweep": "Nx", **_meta_from_inputs(p), **d})
    return records


def sweep_nt(
    p: PricingInputs,
    *,
    domain_cfg: DomainConfig,
    Nt_list: Sequence[int] = (51, 101, 201, 401),
    Nx: int = 401,
    coord: Coord | str = Coord.LOG_S,
    method: str = "cn",
    advection: str | AdvectionScheme = "central",
    spacing: str = "clustered",
    bs_price_fn: Callable[[PricingInputs], float] | None = None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for Nt in Nt_list:
        r = run_once(
            p,
            domain_cfg=domain_cfg,
            Nx=int(Nx),
            Nt=int(Nt),
            coord=coord,
            method=method,
            advection=advection,
            spacing=spacing,
            bs_price_fn=bs_price_fn,
            return_solution=False,
        )
        assert isinstance(r, PDEvsBSRun)
        d = asdict(r)
        d["advection"] = r.advection.value
        d["err"] = float(r.pde - r.bs)
        records.append({"sweep": "Nt", **_meta_from_inputs(p), **d})
    return records


def sweep_nx_nt(
    p: PricingInputs,
    *,
    domain_cfg: DomainConfig,
    Nx_list: Sequence[int] = (101, 201, 401, 801),
    Nt_list: Sequence[int] = (51, 101, 201, 401),
    coord: Coord | str = Coord.LOG_S,
    method: str = "cn",
    advection: str | AdvectionScheme = "central",
    spacing: str = "clustered",
    bs_price_fn: Callable[[PricingInputs], float] | None = None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for Nx in Nx_list:
        for Nt in Nt_list:
            r = run_once(
                p,
                domain_cfg=domain_cfg,
                Nx=int(Nx),
                Nt=int(Nt),
                coord=coord,
                method=method,
                advection=advection,
                spacing=spacing,
                bs_price_fn=bs_price_fn,
                return_solution=False,
            )
            assert isinstance(r, PDEvsBSRun)
            d = asdict(r)
            d["advection"] = r.advection.value
            d["err"] = float(r.pde - r.bs)
            records.append({"sweep": "NxNt", **_meta_from_inputs(p), **d})
    return records


__all__ = [
    "PDEvsBSRun",
    "run_once",
    "run_cases",
    "sweep_nx",
    "sweep_nt",
    "sweep_nx_nt",
]
