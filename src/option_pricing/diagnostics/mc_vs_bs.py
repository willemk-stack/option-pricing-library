from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import numpy as np
import pandas as pd

from option_pricing.models.bs import bs_call_from_inputs
from option_pricing.pricers.mc import mc_call_from_inputs
from option_pricing.types import PricingInputs


def _with_params(
    base: PricingInputs,
    *,
    S: float | None = None,
    K: float | None = None,
    r: float | None = None,
    T: float | None = None,
    sigma: float | None = None,
) -> PricingInputs:
    """
    Create a modified copy of PricingInputs by replacing nested dataclasses.

    Assumes:
      - base.market has fields like spot, rate
      - base.spec has fields like strike, expiry
      - base has fields like market, spec, sigma
    """
    market = base.market
    spec = base.spec

    if S is not None:
        market = replace(market, spot=float(S))
    if r is not None:
        market = replace(market, rate=float(r))

    if K is not None:
        spec = replace(spec, strike=float(K))
    if T is not None:
        spec = replace(spec, expiry=float(T))

    out = replace(base, market=market, spec=spec)

    if sigma is not None:
        out = replace(out, sigma=float(sigma))

    return out


# -----------------------------
# Case generation
# -----------------------------


def default_cases(base: PricingInputs) -> list[tuple[str, PricingInputs]]:
    """
    Small curated set of regimes for demos.

    Note: PricingInputs.T is the (absolute) maturity time, not tau.
    So for short/long maturity tweaks we set T = base.t + tau.
    """
    t0 = base.t
    return [
        ("ATM base", base),
        ("ITM (S=120)", _with_params(base, S=120.0)),
        ("OTM (S=80)", _with_params(base, S=80.0)),
        ("Deep ITM (S=150)", _with_params(base, S=150.0)),
        ("Deep OTM (S=50)", _with_params(base, S=50.0)),
        ("Short tau (1w)", _with_params(base, T=t0 + 1.0 / 52.0)),
        ("Long tau (5y)", _with_params(base, T=t0 + 5.0)),
        ("Low vol (5%)", _with_params(base, sigma=0.05)),
        ("High vol (80%)", _with_params(base, sigma=0.80)),
        ("High rate (10%)", _with_params(base, r=0.10)),
    ]


def grid_cases(
    base: PricingInputs,
    *,
    moneyness: Iterable[float] = (0.7, 0.9, 1.0, 1.1, 1.3),  # S/K
    taus: Iterable[float] = (1.0 / 52.0, 0.25, 1.0, 2.0),  # time-to-maturity
    vols: Iterable[float] = (0.1, 0.2, 0.4),
    rates: Iterable[float] = (0.0, 0.02, 0.05),
) -> list[tuple[str, PricingInputs]]:
    """
    Larger grid for notebook tables.
    Keeps K fixed and sets:
      S = (S/K) * K
      T = t + tau
    """
    out: list[tuple[str, PricingInputs]] = []

    K0 = float(
        base.K
    )  # relies on your property; if you remove it, use base.spec.strike

    for r in rates:
        for sig in vols:
            for tau in taus:
                for m in moneyness:
                    p = _with_params(
                        base,
                        r=float(r),
                        sigma=float(sig),
                        S=float(m * K0),
                        T=float(base.t + tau),
                    )
                    out.append((f"S/K={m:.2f}, tau={tau:g}, sig={sig:g}, r={r:g}", p))

    return out


# -----------------------------
# Notebook-friendly tables
# -----------------------------


def compare_table(
    cases: list[tuple[str, PricingInputs]],
    *,
    n_paths: int = 10_000,
    seed: int | None = 0,
    rng: np.random.Generator | None = None,
    per_case_seed: bool = True,
    zcrit: float = 1.96,
) -> pd.DataFrame:
    """
    One MC run per case + BS benchmark.

    - Uses mc_call_from_inputs(..., seed=..., rng=...)
    - Assumes the MC function returns (price, std_err) where std_err is already SE
    - If seed is provided and per_case_seed=True, uses seed+i for case i (stable & independent-ish)
    - If rng is provided, it is passed through (and will be advanced across cases)
    """
    rows: list[dict] = []

    for i, (name, p) in enumerate(cases):
        tau = p.T - p.t
        if tau < 0:
            raise ValueError(f"{name}: negative time-to-maturity tau = T-t = {tau}")

        seed_i: int | None = None
        rng_i: np.random.Generator | None = None

        if rng is not None:
            rng_i = rng
        elif seed is not None:
            seed_i = int(seed) + i if per_case_seed else int(seed)

        mc, se = mc_call_from_inputs(p, n_paths=int(n_paths), seed=seed_i, rng=rng_i)
        bs = float(bs_call_from_inputs(p))

        err = float(mc - bs)
        rel_err = err / bs if bs != 0.0 else np.nan
        z = err / se if se != 0.0 else np.nan

        ci_low = float(mc - zcrit * se)
        ci_high = float(mc + zcrit * se)
        in_ci = (bs >= ci_low) and (bs <= ci_high)

        rows.append(
            {
                "case": name,
                "t": p.t,
                "S": p.S,
                "K": p.K,
                "r": p.r,
                "sigma": p.sigma,
                "T": p.T,
                "tau": tau,
                "n_paths": int(n_paths),
                "MC": float(mc),
                "SE": float(se),
                "BS": bs,
                "MC-BS": err,
                "rel_err": rel_err,
                "z": z,
                "abs_z": abs(z) if np.isfinite(z) else np.nan,
                "CI_low": ci_low,
                "CI_high": ci_high,
                "BS_in_CI": in_ci,
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values("abs_z", ascending=False, na_position="last").reset_index(
        drop=True
    )


def multi_seed_summary(
    cases: list[tuple[str, PricingInputs]],
    *,
    n_paths: int = 5_000,
    seeds: Iterable[int] = range(20),
) -> pd.DataFrame:
    """
    Repeat MC across multiple seeds per case and summarize:

      - MC_mean vs BS (bias check)
      - MC_std_across_seeds (empirical jitter)
      - avg_reported_SE (mean of your reported SE)
    """
    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds must be non-empty")

    rows: list[dict] = []

    for name, p in cases:
        tau = p.T - p.t
        if tau < 0:
            raise ValueError(f"{name}: negative time-to-maturity tau = T-t = {tau}")

        bs = float(bs_call_from_inputs(p))

        mcs: list[float] = []
        ses: list[float] = []
        for s in seeds:
            mc, se = mc_call_from_inputs(p, n_paths=int(n_paths), seed=int(s), rng=None)
            mcs.append(float(mc))
            ses.append(float(se))

        mcs_arr = np.asarray(mcs, dtype=float)
        ses_arr = np.asarray(ses, dtype=float)

        rows.append(
            {
                "case": name,
                "BS": bs,
                "MC_mean": float(mcs_arr.mean()),
                "mean_error": float(mcs_arr.mean() - bs),
                "MC_std_across_seeds": (
                    float(mcs_arr.std(ddof=1)) if len(mcs_arr) > 1 else 0.0
                ),
                "avg_reported_SE": float(ses_arr.mean()),
                "n_paths": int(n_paths),
                "n_seeds": len(seeds),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("MC_std_across_seeds", ascending=False)
        .reset_index(drop=True)
    )


def convergence_table(
    p: PricingInputs,
    *,
    n_paths_list: Iterable[int] = (1_000, 2_000, 5_000, 10_000, 20_000, 50_000),
    seed: int | None = 0,
) -> pd.DataFrame:
    """
    For one parameter set, show how MC and SE behave as n_paths increases.
    """
    tau = p.T - p.t
    if tau < 0:
        raise ValueError(f"negative time-to-maturity tau = T-t = {tau}")

    bs = float(bs_call_from_inputs(p))
    rows: list[dict] = []

    for n in n_paths_list:
        mc, se = mc_call_from_inputs(p, n_paths=int(n), seed=seed, rng=None)
        err = float(mc - bs)
        rows.append(
            {
                "n_paths": int(n),
                "MC": float(mc),
                "SE": float(se),
                "BS": bs,
                "MC-BS": err,
                "abs_err": abs(err),
            }
        )

    return pd.DataFrame(rows)
