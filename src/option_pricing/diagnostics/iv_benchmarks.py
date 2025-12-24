"""
option_pricing.diagnostics.iv_benchmarks

Diagnostics helpers for:
- Black-Scholes “golden” benchmark pricing and implied-vol recovery
- Synthetic IV smile generation (strike-dependent true vol)
- Single-case IV recovery (handy for notebooks / quick checks)

Option A contract:
- root finders return RootResult
- implied vol inversion returns ImpliedVolResult (diagnostics preserved)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from option_pricing import MarketData, OptionSpec, OptionType, PricingInputs, bs_price
from option_pricing.numerics.root_finding import RootResult, bracketed_newton
from option_pricing.vol.implied_vol import implied_vol_bs_result


# ----------------------------
# Helper function for constant volatility
# ----------------------------
def flat_vol_20(_: float) -> float:
    return 0.20


# ----------------------------
# Single-case helper (notebook-friendly)
# ----------------------------
def single_iv_recovery_from_template(
    p_template: PricingInputs,
    *,
    sigma_true: float,
    sigma_guess: float,
    root_method: Callable[..., RootResult] = bracketed_newton,
    sigma0: float | None = None,
    sigma_lo: float = 1e-8,
    sigma_hi: float = 5.0,
    tol_f: float = 1e-10,
    tol_x: float = 1e-10,
) -> pd.DataFrame:
    """Price with sigma_true, then recover IV from that price."""
    p_true = replace(p_template, sigma=float(sigma_true))
    mkt_price = float(bs_price(p_true))

    ivres = implied_vol_bs_result(
        mkt_price=mkt_price,
        spec=p_template.spec,
        market=p_template.market,
        root_method=root_method,
        t=float(p_template.t),
        sigma0=float(sigma0) if sigma0 is not None else float(sigma_guess),
        sigma_lo=float(sigma_lo),
        sigma_hi=float(sigma_hi),
        tol_f=float(tol_f),
        tol_x=float(tol_x),
    )

    return pd.DataFrame(
        [
            {
                "sigma_true": float(sigma_true),
                "sigma_guess": float(sigma_guess),
                "implied_vol": float(ivres.vol),
                "abs_error": float(abs(ivres.vol - float(sigma_true))),
                "mkt_price": float(mkt_price),
                # diagnostics
                "converged": bool(ivres.root_result.converged),
                "iterations": int(ivres.root_result.iterations),
                "method": str(ivres.root_result.method),
                "f_at_root": float(ivres.root_result.f_at_root),
                "bracket": ivres.root_result.bracket,
            }
        ]
    )


# ----------------------------
# Synthetic smile generator
# ----------------------------
def run_synthetic_iv_smile(
    *,
    S: float,
    r: float,
    q: float,
    tau: float,
    kind: OptionType,
    K_min: float = 60.0,
    K_max: float = 160.0,
    n: int = 51,
    true_vol_fn: Callable[[float], float] | None = None,
    sigma_guess: float = 0.25,
    root_method: Callable[..., RootResult] = bracketed_newton,
    sigma0: float | None = None,
    sigma_lo: float = 1e-8,
    sigma_hi: float = 5.0,
    tol_f: float = 1e-10,
    tol_x: float = 1e-10,
    t: float = 0.0,
    drop_failed: bool = False,
) -> pd.DataFrame:
    """Generate a synthetic IV smile: price using sigma_true(K) then recover IV."""
    if true_vol_fn is None:
        true_vol_fn = flat_vol_20

    market = MarketData(spot=float(S), rate=float(r), dividend_yield=float(q))
    expiry = float(t) + float(tau)

    K_vals = np.linspace(float(K_min), float(K_max), int(n))
    rows: list[dict[str, Any]] = []

    for K in K_vals:
        K = float(K)
        sigma_true = float(true_vol_fn(K))

        spec = OptionSpec(kind=kind, strike=K, expiry=expiry)
        p_true = PricingInputs(spec=spec, market=market, sigma=sigma_true, t=float(t))
        mkt_price = float(bs_price(p_true))

        try:
            ivres = implied_vol_bs_result(
                mkt_price=mkt_price,
                spec=spec,
                market=market,
                root_method=root_method,
                t=float(t),
                sigma0=float(sigma0) if sigma0 is not None else float(sigma_guess),
                sigma_lo=float(sigma_lo),
                sigma_hi=float(sigma_hi),
                tol_f=float(tol_f),
                tol_x=float(tol_x),
            )
            iv = float(ivres.vol)
            ok = True
            abs_error = float(abs(iv - sigma_true))
            err = ""

            rr = ivres.root_result
            converged = bool(rr.converged)
            iterations = int(rr.iterations)
            method = str(rr.method)
            f_at_root = float(rr.f_at_root)
            bracket = rr.bracket

        except Exception as e:
            iv = float("nan")
            ok = False
            abs_error = float("nan")
            err = f"{type(e).__name__}: {e}"
            converged = False
            iterations = 0
            method = ""
            f_at_root = float("nan")
            bracket = None

        rows.append(
            {
                "K": K,
                "log_moneyness": float(np.log(K / float(S))),
                "sigma_true": sigma_true,
                "mkt_price": mkt_price,
                "implied_vol": float(iv),
                "abs_error": abs_error,
                "ok": ok,
                "error": err,
                # diagnostics
                "converged": bool(converged),
                "iterations": int(iterations),
                "method": str(method),
                "f_at_root": float(f_at_root),
                "bracket": bracket,
            }
        )

    df = pd.DataFrame(rows)
    if drop_failed:
        df = df[df["ok"]].reset_index(drop=True)
    return df


def plot_iv_smile(
    df: pd.DataFrame,
    *,
    x: str = "K",
    show_true: bool = True,
    title: str = "IV smile",
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(
        df[x].to_numpy(),
        df["implied_vol"].to_numpy(),
        marker="o",
        linewidth=1,
        label="Recovered IV",
    )
    if show_true:
        plt.plot(
            df[x].to_numpy(),
            df["sigma_true"].to_numpy(),
            linestyle="--",
            linewidth=1,
            label="True σ used to price",
        )
    plt.xlabel(x)
    plt.ylabel("Implied vol")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


# ----------------------------
# Benchmark suite
# ----------------------------
def run_bs_iv_benchmarks(
    *,
    price_tol: float = 5e-4,
    iv_tol: float = 5e-4,
    sigma_lo: float = 1e-8,
    sigma_hi: float = 5.0,
    tol_f: float = 1e-10,
    tol_x: float = 1e-10,
    save_csv_path: str | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Run a small set of Black-Scholes benchmarks.

    Returns a tidy diagnostics table (one row per benchmark case).
    """
    benchmarks: list[dict[str, Any]] = [
        dict(
            name="ATM no-div",
            kind=OptionType.CALL,
            S=100.0,
            K=100.0,
            r=0.05,
            q=0.0,
            tau=1.0,
            sigma_true=0.20,
            mkt_price=10.4506,
        ),
        dict(
            name="ATM no-div",
            kind=OptionType.PUT,
            S=100.0,
            K=100.0,
            r=0.05,
            q=0.0,
            tau=1.0,
            sigma_true=0.20,
            mkt_price=5.57352,
        ),
        dict(
            name="ITM 3M",
            kind=OptionType.CALL,
            S=100.0,
            K=95.0,
            r=0.01,
            q=0.0,
            tau=3 / 12,
            sigma_true=0.50,
            mkt_price=12.5279,
        ),
        dict(
            name="Hull-style",
            kind=OptionType.CALL,
            S=42.0,
            K=40.0,
            r=0.10,
            q=0.0,
            tau=0.5,
            sigma_true=0.20,
            mkt_price=4.7594223929,
        ),
        dict(
            name="Hull-style",
            kind=OptionType.PUT,
            S=42.0,
            K=40.0,
            r=0.10,
            q=0.0,
            tau=0.5,
            sigma_true=0.20,
            mkt_price=0.8085993729,
        ),
    ]

    rows: list[dict[str, Any]] = []

    for b in benchmarks:
        # Here t=0, so expiry == tau. If you later benchmark with nonzero t, set expiry=t+tau.
        spec = OptionSpec(kind=b["kind"], strike=float(b["K"]), expiry=float(b["tau"]))
        market = MarketData(
            spot=float(b["S"]), rate=float(b["r"]), dividend_yield=float(b["q"])
        )

        # Price at sigma_true (sanity check against published mkt_price)
        p_true = PricingInputs(
            spec=spec, market=market, sigma=float(b["sigma_true"]), t=0.0
        )

        try:
            model_price = float(bs_price(p_true))
            price_abs_error = float(abs(model_price - float(b["mkt_price"])))
            price_ok = price_abs_error <= float(price_tol)
            price_error_msg = ""
        except Exception as e:
            model_price = float("nan")
            price_abs_error = float("nan")
            price_ok = False
            price_error_msg = f"{type(e).__name__}: {e}"

        # IV recovery from published mkt_price
        sigma_guess = 0.30
        try:
            ivres = implied_vol_bs_result(
                mkt_price=float(b["mkt_price"]),
                spec=spec,
                market=market,
                root_method=bracketed_newton,
                t=0.0,
                sigma0=float(sigma_guess),
                sigma_lo=float(sigma_lo),
                sigma_hi=float(sigma_hi),
                tol_f=float(tol_f),
                tol_x=float(tol_x),
            )
            iv = float(ivres.vol)
            rr = ivres.root_result

            iv_abs_error = float(abs(iv - float(b["sigma_true"])))
            iv_ok = iv_abs_error <= float(iv_tol)
            iv_error_msg = ""
        except Exception as e:
            iv = float("nan")
            rr = None
            iv_abs_error = float("nan")
            iv_ok = False
            iv_error_msg = f"{type(e).__name__}: {e}"

        rows.append(
            {
                "benchmark": b["name"],
                "kind": b["kind"].value,
                "S": b["S"],
                "K": b["K"],
                "r": b["r"],
                "q": b["q"],
                "tau": b["tau"],
                "sigma_true": b["sigma_true"],
                "sigma_guess": sigma_guess,
                "mkt_price_published": b["mkt_price"],
                "bs_price_at_sigma_true": model_price,
                "price_abs_error": price_abs_error,
                "price_ok": price_ok,
                "price_error": price_error_msg,
                "implied_vol": iv,
                "iv_abs_error": iv_abs_error,
                "iv_ok": iv_ok,
                "iv_error": iv_error_msg,
                # diagnostics
                "converged": (bool(rr.converged) if rr is not None else False),
                "iterations": (int(rr.iterations) if rr is not None else 0),
                "method": (str(rr.method) if rr is not None else ""),
                "f_at_root": (float(rr.f_at_root) if rr is not None else float("nan")),
                "bracket": (rr.bracket if rr is not None else None),
                "price_tol": price_tol,
                "iv_tol": iv_tol,
                "sigma_lo": sigma_lo,
                "sigma_hi": sigma_hi,
                "tol_f": tol_f,
                "tol_x": tol_x,
            }
        )

    df = pd.DataFrame(rows)

    if save_csv_path is not None:
        df.to_csv(save_csv_path, index=False)

    if strict:
        failed = df[~(df["price_ok"] & df["iv_ok"])]
        if not failed.empty:
            raise AssertionError(
                "Some benchmarks failed.\n"
                + failed[
                    [
                        "benchmark",
                        "kind",
                        "price_ok",
                        "iv_ok",
                        "price_error",
                        "iv_error",
                    ]
                ].to_string(index=False)
            )

    return df
