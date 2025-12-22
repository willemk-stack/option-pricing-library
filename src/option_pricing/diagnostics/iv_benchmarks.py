"""
option_pricing.diagnostics.iv_benchmarks

Diagnostics helpers for:
- Black-Scholes “golden” benchmark pricing and implied-vol recovery
- Synthetic IV smile generation (strike-dependent true vol)
- Single-case IV recovery (handy for notebooks / quick checks)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from option_pricing import MarketData, OptionSpec, OptionType, PricingInputs, bs_price
from option_pricing.numerics.root_finding import bracketed_newton
from option_pricing.vol.implied_vol import IV_solver


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
    root_method: Callable[..., float] = bracketed_newton,
    sigma0: float | None = None,
    sigma_lo: float = 1e-8,
    sigma_hi: float = 5.0,
) -> pd.DataFrame:
    """
    Generate a synthetic market price from bs_price at sigma_true, then recover IV from that price.

    Returns a 1-row DataFrame with sigma_true, sigma_guess, implied_vol, abs_error, mkt_price.
    """
    p_true = replace(p_template, sigma=float(sigma_true))
    mkt_price = float(bs_price(p_true))

    p_guess = replace(p_template, sigma=float(sigma_guess))
    iv = float(
        IV_solver(
            p_guess,
            mkt_price,
            root_method=root_method,
            sigma0=(float(sigma0) if sigma0 is not None else float(p_guess.sigma)),
            sigma_lo=float(sigma_lo),
            sigma_hi=float(sigma_hi),
        )
    )

    return pd.DataFrame(
        [
            {
                "sigma_true": float(sigma_true),
                "sigma_guess": float(p_guess.sigma),
                "implied_vol": float(iv),
                "abs_error": float(abs(iv - float(sigma_true))),
                "mkt_price": float(mkt_price),
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
    kind: Any,  # OptionType or "call"/"put" depending on your OptionSpec
    K_min: float = 60.0,
    K_max: float = 160.0,
    n: int = 51,
    true_vol_fn: Callable[[float], float] | None = None,
    sigma_guess: float = 0.25,
    root_method: Callable[..., float] = bracketed_newton,
    sigma0: float | None = None,
    sigma_lo: float = 1e-8,
    sigma_hi: float = 5.0,
    t: float = 0.0,
    drop_failed: bool = False,
) -> pd.DataFrame:
    """
    Create a synthetic implied-vol smile by:
      1) selecting a strike-dependent true vol sigma_true(K)
      2) generating synthetic market prices from bs_price at sigma_true(K)
      3) recovering IV from those prices

    Returns a DataFrame with:
      K, log_moneyness, sigma_true, mkt_price, implied_vol, abs_error, ok
    """
    if true_vol_fn is None:
        true_vol_fn = flat_vol_20  # flat (debug baseline)

    mkt = MarketData(spot=float(S), rate=float(r), dividend_yield=float(q))

    # PricingInputs.tau = expiry - t, so set expiry = t + tau
    expiry = float(t) + float(tau)

    K_vals = np.linspace(float(K_min), float(K_max), int(n))

    rows: list[dict[str, Any]] = []
    for K in K_vals:
        K = float(K)
        sigma_true = float(true_vol_fn(K))

        spec = OptionSpec(kind=kind, strike=K, expiry=expiry)

        # synthetic market price at sigma_true(K)
        p_true = PricingInputs(spec=spec, market=mkt, sigma=sigma_true, t=float(t))
        mkt_price = float(bs_price(p_true))

        # IV recovery
        p_guess = PricingInputs(
            spec=spec, market=mkt, sigma=float(sigma_guess), t=float(t)
        )

        try:
            iv = float(
                IV_solver(
                    p_guess,
                    mkt_price,
                    root_method=root_method,
                    sigma0=(
                        float(sigma0) if sigma0 is not None else float(p_guess.sigma)
                    ),
                    sigma_lo=float(sigma_lo),
                    sigma_hi=float(sigma_hi),
                )
            )
            ok = True
            abs_error = abs(iv - sigma_true)
        except Exception:
            iv = float("nan")
            ok = False
            abs_error = float("nan")

        rows.append(
            {
                "K": K,
                "log_moneyness": float(np.log(K / float(S))),
                "sigma_true": sigma_true,
                "mkt_price": mkt_price,
                "implied_vol": iv,
                "abs_error": abs_error,
                "ok": ok,
            }
        )

    df = pd.DataFrame(rows)
    if drop_failed:
        df = df[df["ok"]].reset_index(drop=True)
    return df


def plot_iv_smile(
    df: pd.DataFrame,
    *,
    x: str = "K",  # or "log_moneyness"
    show_true: bool = True,
    title: str = "IV smile",
) -> None:
    """Convenience plot (kept here so notebooks stay clean)."""
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
# Benchmark suit
# ----------------------------
def run_bs_iv_benchmarks(
    *,
    price_tol: float = 5e-4,
    iv_tol: float = 5e-4,
    sigma_lo: float = 1e-8,
    sigma_hi: float = 5.0,
    save_csv_path: str | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Run a small set of Black-Scholes benchmarks.

    Returns a tidy diagnostics table (one row per benchmark case).
    """

    benchmarks: list[dict[str, Any]] = [
        # Benchmark 1: ATM, no dividends
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
        # Benchmark 2: ITM call, 3 months
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
        # Benchmark 3: Hull-style (use more precise values to keep price_tol tight)
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
        sigma_guess = 0.30 if abs(b["sigma_true"] - 0.30) > 1e-12 else 0.20

        p_guess = PricingInputs(
            spec=OptionSpec(kind=b["kind"], strike=b["K"], expiry=b["tau"]),
            market=MarketData(spot=b["S"], rate=b["r"], dividend_yield=b["q"]),
            sigma=sigma_guess,
            t=0.0,
        )

        # 1) Price check at sigma_true
        p_true = replace(p_guess, sigma=float(b["sigma_true"]))
        try:
            model_price = float(bs_price(p_true))
            price_abs_error = abs(model_price - float(b["mkt_price"]))
            price_ok = price_abs_error <= price_tol
            price_error_msg = ""
        except Exception as e:
            model_price = float("nan")
            price_abs_error = float("nan")
            price_ok = False
            price_error_msg = f"{type(e).__name__}: {e}"

        # 2) IV check (recover sigma_true from published price)
        try:
            iv = float(
                IV_solver(
                    p_guess,
                    float(b["mkt_price"]),
                    root_method=bracketed_newton,
                    sigma0=p_guess.sigma,
                    sigma_lo=sigma_lo,
                    sigma_hi=sigma_hi,
                )
            )
            iv_abs_error = abs(iv - float(b["sigma_true"]))
            iv_ok = iv_abs_error <= iv_tol
            iv_error_msg = ""
        except Exception as e:
            iv = float("nan")
            iv_abs_error = float("nan")
            iv_ok = False
            iv_error_msg = f"{type(e).__name__}: {e}"

        rows.append(
            {
                "benchmark": b["name"],
                "kind": (
                    b["kind"].value if hasattr(b["kind"], "value") else str(b["kind"])
                ),
                "S": b["S"],
                "K": b["K"],
                "r": b["r"],
                "q": b["q"],
                "tau": b["tau"],
                "sigma_true": b["sigma_true"],
                "sigma_guess": p_guess.sigma,
                "mkt_price_published": b["mkt_price"],
                "bs_price_at_sigma_true": model_price,
                "price_abs_error": price_abs_error,
                "price_ok": price_ok,
                "price_error": price_error_msg,
                "implied_vol": iv,
                "iv_abs_error": iv_abs_error,
                "iv_ok": iv_ok,
                "iv_error": iv_error_msg,
                "price_tol": price_tol,
                "iv_tol": iv_tol,
                "sigma_lo": sigma_lo,
                "sigma_hi": sigma_hi,
            }
        )

    df = pd.DataFrame(rows)

    if save_csv_path:
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
