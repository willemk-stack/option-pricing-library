"""Reproduce (approximately) the convergence tables from Pooley–Vetzal–Forsyth (2002).

This script is a manual validation harness for the convergence remedies for
non-smooth payoffs.

Run from the repository root:

    PYTHONPATH=src python scripts/convergence_tables_pvf2002.py --case digital
    PYTHONPATH=src python scripts/convergence_tables_pvf2002.py --case supershare
    PYTHONPATH=src python scripts/convergence_tables_pvf2002.py --case all

If you have implemented the recommended convergence remedies, the script will
also produce the "smoothing + Rannacher" tables. If not, it will clearly report
which features are missing.
"""

from __future__ import annotations

import argparse
import inspect
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_d2(*, S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    vol_sqrt = sigma * math.sqrt(T)
    return (math.log(S / K) + (r - q - 0.5 * sigma * sigma) * T) / vol_sqrt


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
    d2 = _bs_d2(S=S, K=K, r=r, q=q, sigma=sigma, T=T)
    return payout * math.exp(-r * T) * _norm_cdf(d2)


def bs_supershare_price(
    *, S: float, K: float, d: float, r: float, q: float, sigma: float, T: float
) -> float:
    p_ge_k = _norm_cdf(_bs_d2(S=S, K=K, r=r, q=q, sigma=sigma, T=T))
    p_ge_kd = _norm_cdf(_bs_d2(S=S, K=K + d, r=r, q=q, sigma=sigma, T=T))
    return math.exp(-r * T) * (p_ge_k - p_ge_kd) / d


def _observed_ratio(d_prev: float, d_next: float) -> float | None:
    if d_prev == 0:
        return None
    return d_prev / d_next


def _can_use_ic_transform() -> bool:
    try:
        from option_pricing.numerics.pde import solve_pde_1d
    except Exception:
        return False
    return "ic_transform" in inspect.signature(solve_pde_1d).parameters


def _try_import_remedies():
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
    ic_transform=None,
) -> float:
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
        spacing=SpacingPolicy.CLUSTERED,
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


@dataclass(frozen=True)
class TableRow:
    Nx: int
    Nt: int
    value: float
    diff: float | None
    ratio: float | None


def _make_table(
    values: Sequence[float], Nx_list: Sequence[int], Nt_list: Sequence[int]
) -> list[TableRow]:
    rows: list[TableRow] = []
    prev = None
    prev_diff = None
    for Nx, Nt, v in zip(Nx_list, Nt_list, values, strict=True):
        diff = None if prev is None else abs(v - prev)
        ratio = (
            None
            if (prev_diff is None or diff in (None, 0.0))
            else _observed_ratio(prev_diff, diff)
        )
        rows.append(
            TableRow(Nx=int(Nx), Nt=int(Nt), value=float(v), diff=diff, ratio=ratio)
        )
        prev = v
        prev_diff = diff
    return rows


def _print_table(
    title: str, rows: Sequence[TableRow], exact: float | None = None
) -> None:
    print("\n" + title)
    if exact is not None:
        print(f"Exact (analytic): {exact:.10f}")
    print(f"{'Nx':>6} {'Nt':>6} {'Value':>14} {'Diff':>14} {'Ratio':>8}")
    for r in rows:
        diff_s = "" if r.diff is None else f"{r.diff:.10f}"
        ratio_s = "" if r.ratio is None else f"{r.ratio:.3f}"
        print(f"{r.Nx:6d} {r.Nt:6d} {r.value:14.10f} {diff_s:>14} {ratio_s:>8}")


def run_digital() -> None:
    # Parameters match Table 1 in the paper.
    S = 40.0
    K = 40.0
    r = 0.05
    q = 0.0
    sigma = 0.3
    T = 0.5
    payout = 1.0

    Nx_list = [41, 81, 161, 321]
    Nt_list = [26, 51, 101, 201]
    exact = bs_digital_call_price(S=S, K=K, r=r, q=q, sigma=sigma, T=T, payout=payout)

    for method in ("implicit", "cn", "rannacher"):
        vals = [
            _price_digital_pde(
                S=S,
                K=K,
                r=r,
                q=q,
                sigma=sigma,
                T=T,
                payout=payout,
                Nx=Nx,
                Nt=Nt,
                method=method,
            )
            for Nx, Nt in zip(Nx_list, Nt_list, strict=True)
        ]
        _print_table(
            f"Digital call (no smoothing), method='{method}'",
            _make_table(vals, Nx_list, Nt_list),
            exact=exact,
        )

    if not _can_use_ic_transform():
        print(
            "\n[INFO] Skipping smoothing tables: solve_pde_1d has no ic_transform hook."
        )
        return

    ic_avg, ic_proj = _try_import_remedies()
    if ic_avg is None or ic_proj is None:
        print(
            "\n[INFO] Skipping smoothing tables: option_pricing.numerics.pde.ic_remedies not found."
        )
        return

    from option_pricing.models.black_scholes.pde import bs_coord_maps
    from option_pricing.numerics.pde.domain import Coord

    to_x, _to_S = bs_coord_maps(Coord.LOG_S)
    xK = float(to_x(K))

    def avg_transform(grid, ic_fn):
        return ic_avg(grid, ic_fn, breakpoints=[xK])

    def proj_transform(grid, ic_fn):
        return ic_proj(grid, ic_fn, breakpoints=[xK])

    for name, transform in ("Averaging", avg_transform), ("Projection", proj_transform):
        vals = [
            _price_digital_pde(
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
            for Nx, Nt in zip(Nx_list, Nt_list, strict=True)
        ]
        _print_table(
            f"Digital call (smoothing={name} + Rannacher)",
            _make_table(vals, Nx_list, Nt_list),
            exact=exact,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["digital", "supershare", "all"], default="all")
    args = ap.parse_args()

    if args.case in ("digital", "all"):
        run_digital()


if __name__ == "__main__":
    main()
