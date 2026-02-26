# ruff: noqa: E402
from __future__ import annotations

"""Local-vol PDE repricing diagnostics.

This module runs an *end-to-end* experiment:

1) take a local-vol surface derived from an implied-vol surface
2) price a grid of vanilla options via the local-vol PDE
3) compare to Black-76 prices implied by the original implied-vol surface

It is intended for demos and diagnostics (not core pricing API).
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from option_pricing.pricers.black_scholes import bs_price_from_ctx
from option_pricing.pricers.pde.domain import BSDomainConfig
from option_pricing.pricers.pde_pricer import local_vol_price_pde_european
from option_pricing.types import MarketData, OptionSpec, OptionType, PricingInputs
from option_pricing.vol.implied_vol import implied_vol_bs
from option_pricing.vol.surface import LocalVolSurface


@dataclass(frozen=True)
class LocalVolRepricingResult:
    grid: pd.DataFrame
    summary: pd.DataFrame
    meta: dict[str, Any]


def _as_1d_float(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Expected a non-empty array")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Expected finite values")
    return arr


def localvol_pde_repricing_grid(
    *,
    lv: LocalVolSurface,
    market: MarketData,
    strikes: np.ndarray,
    expiries: np.ndarray,
    Nx: int,
    Nt: int,
    solver_cfg: dict[str, Any],
    kind: OptionType = OptionType.CALL,
    target: Literal["black76_from_implied"] = "black76_from_implied",
    compute_implied_vol: bool = True,
) -> LocalVolRepricingResult:
    """Run a grid repricing diagnostic for a local-vol PDE pricer."""
    if target != "black76_from_implied":
        raise ValueError(f"Unsupported target={target!r}")

    Ks = _as_1d_float(strikes)
    Ts = _as_1d_float(expiries)
    if np.any(Ks <= 0.0):
        raise ValueError("All strikes must be > 0")
    if np.any(Ts <= 0.0):
        raise ValueError("All expiries must be > 0")

    ctx = market.to_context()

    domain_cfg = solver_cfg.get("domain_cfg", None)
    if domain_cfg is None or not isinstance(domain_cfg, BSDomainConfig):
        raise ValueError(
            "solver_cfg must include a 'domain_cfg' of type BSDomainConfig "
            "(same object you pass to local_vol_price_pde_european)."
        )

    rows: list[dict[str, Any]] = []
    t_total0 = perf_counter()

    for T in Ts:
        target_iv_vec = np.asarray(lv.implied.iv(Ks, float(T)), dtype=float).reshape(-1)
        if target_iv_vec.shape != Ks.shape:
            raise ValueError("lv.implied.iv returned unexpected shape")

        for K, target_iv in zip(Ks, target_iv_vec, strict=True):
            spec = OptionSpec(kind=kind, strike=float(K), expiry=float(T))

            target_price = float(
                bs_price_from_ctx(
                    kind=kind,
                    strike=float(K),
                    sigma=float(target_iv),
                    tau=float(T),
                    ctx=ctx,
                )
            )

            sigma_seed = (
                float(target_iv) if np.isfinite(target_iv) and target_iv > 0 else 0.2
            )
            p = PricingInputs(spec=spec, market=market, sigma=sigma_seed)

            t0 = perf_counter()
            # call PDE pricer without returning full solution so mypy knows
            # the result is a plain float union branch.
            pde_price_raw = local_vol_price_pde_european(
                p,
                lv=lv,
                Nx=int(Nx),
                Nt=int(Nt),
                return_solution=False,
                **solver_cfg,
            )
            # mypy: local_vol_price_pde_european returns float | tuple; we know
            # return_solution=False so it must be float.
            pde_price = float(cast(float, pde_price_raw))
            runtime_ms = 1000.0 * (perf_counter() - t0)

            abs_price_error = abs(pde_price - target_price)

            pde_iv = float("nan")
            abs_iv_error_bp = float("nan")
            iv_ok = False
            iv_err = ""

            if compute_implied_vol:
                try:
                    pde_iv = float(
                        implied_vol_bs(
                            mkt_price=pde_price,
                            spec=spec,
                            market=market,
                            t=0.0,
                        )
                    )
                    iv_ok = bool(np.isfinite(pde_iv))
                    if iv_ok and np.isfinite(target_iv):
                        abs_iv_error_bp = 1e4 * abs(float(pde_iv) - float(target_iv))
                except Exception as e:  # noqa: BLE001
                    iv_err = f"{type(e).__name__}: {e}"

            rows.append(
                {
                    "T": float(T),
                    "K": float(K),
                    "target_iv": float(target_iv),
                    "target_price": float(target_price),
                    "pde_price": float(pde_price),
                    "abs_price_error": float(abs_price_error),
                    "pde_iv": float(pde_iv),
                    "abs_iv_error_bp": float(abs_iv_error_bp),
                    "iv_ok": bool(iv_ok) if compute_implied_vol else True,
                    "iv_error": str(iv_err),
                    "runtime_ms": float(runtime_ms),
                    "Nx": int(Nx),
                    "Nt": int(Nt),
                }
            )

    total_runtime_ms = 1000.0 * (perf_counter() - t_total0)
    grid = pd.DataFrame(rows)

    def _finite_mean(x: pd.Series) -> float:
        x2 = x[np.isfinite(x)]
        return float(x2.mean()) if len(x2) else float("nan")

    def _finite_max(x: pd.Series) -> float:
        x2 = x[np.isfinite(x)]
        return float(x2.max()) if len(x2) else float("nan")

    summary = pd.DataFrame(
        [
            {
                "n_options": int(len(grid)),
                "mean_abs_price_error": float(_finite_mean(grid["abs_price_error"])),
                "max_abs_price_error": float(_finite_max(grid["abs_price_error"])),
                "mean_abs_iv_error_bp": float(
                    _finite_mean(grid["abs_iv_error_bp"])
                    if compute_implied_vol
                    else float("nan")
                ),
                "max_abs_iv_error_bp": float(
                    _finite_max(grid["abs_iv_error_bp"])
                    if compute_implied_vol
                    else float("nan")
                ),
                "mean_runtime_ms": float(_finite_mean(grid["runtime_ms"])),
                "total_runtime_ms": float(total_runtime_ms),
                "Nx": int(Nx),
                "Nt": int(Nt),
            }
        ]
    )

    meta = {
        "solver_cfg": dict(solver_cfg),
        "Nx": int(Nx),
        "Nt": int(Nt),
        "kind": str(kind.value),
        "target": str(target),
        "compute_implied_vol": bool(compute_implied_vol),
    }

    return LocalVolRepricingResult(grid=grid, summary=summary, meta=meta)


__all__ = [
    "LocalVolRepricingResult",
    "localvol_pde_repricing_grid",
]
