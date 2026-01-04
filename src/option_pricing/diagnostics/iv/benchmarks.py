from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from option_pricing.config import ImpliedVolConfig


def make_synthetic_smile(
    *,
    atm_vol: float = 0.20,
    skew: float = -0.20,
    smile: float = 0.10,
    k0: float = 1.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a simple synthetic "smile" function of strike K.

    We define log-moneyness x = ln(K / k0) and set:

        sigma(K) = atm_vol + skew * x + smile * x^2

    Parameters
    ----------
    atm_vol : float
        Vol at x=0 (K = k0).
    skew : float
        Linear skew in x (negative -> equity-like).
    smile : float
        Quadratic curvature in x.
    k0 : float
        Reference strike level (often spot or forward).

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        A function that maps K (scalar or array) -> sigma (array).
    """
    atm_vol = float(atm_vol)
    skew = float(skew)
    smile = float(smile)
    k0 = float(k0)

    def _sigma(K: np.ndarray) -> np.ndarray:
        K = np.asarray(K, dtype=float)
        x = np.log(K / k0)
        return atm_vol + skew * x + smile * (x**2)

    return _sigma


def iv_recovery_benchmark(
    *,
    Ks: Sequence[float],
    sigma_true_fn: Callable[[float], float],
    price_fn: Callable[[Any], float],
    implied_vol_fn: Callable[[Any, float], float],
    make_pricing_inputs: Callable[[float, float], Any],
) -> pd.DataFrame:
    """
    Generic helper: sweep strikes, generate a "market" price from sigma_true_fn,
    then recover implied vols from those prices.

    This is intentionally generic so it can be reused across pricing engines.
    Your project uses `run_synthetic_iv_smile` below, which wires this up to
    your Black–Scholes price + implied vol routines.
    """
    rows = []
    for K in Ks:
        sigma_true = float(sigma_true_fn(float(K)))
        p_true = make_pricing_inputs(float(K), sigma_true)
        mkt_price = float(price_fn(p_true))

        p_guess = make_pricing_inputs(
            float(K), np.nan
        )  # sigma guess handled in implied_vol_fn
        try:
            iv = float(implied_vol_fn(p_guess, mkt_price))
            ok = True
        except Exception:
            iv = np.nan
            ok = False

        rows.append(
            dict(
                K=float(K),
                sigma_true=sigma_true,
                mkt_price=mkt_price,
                implied_vol=iv,
                abs_error=float(abs(iv - sigma_true)) if np.isfinite(iv) else np.nan,
                converged=bool(ok),
            )
        )
    return pd.DataFrame(rows)


def run_synthetic_iv_smile(
    *,
    S: float,
    r: float,
    q: float,
    tau: float,
    kind: Any,
    K_min: float,
    K_max: float,
    n: int,
    true_vol_fn: Callable[[float], float],
    sigma_guess: float = 0.20,
    cfg: ImpliedVolConfig | None = None,
    drop_failed: bool = True,
    t: float = 0.0,
) -> pd.DataFrame:
    """
    Project-specific benchmark: build a synthetic implied-vol smile by:

    1) generating "market" prices using BS with sigma_true(K)
    2) inverting each price to recover IV(K)

    Returns a DataFrame with columns:
      K, sigma_true, mkt_price, implied_vol, abs_error, converged, iterations
    """
    # Local imports keep diagnostics lightweight until used.
    from option_pricing import MarketData, OptionSpec, PricingInputs, bs_price
    from option_pricing.vol.implied_vol import implied_vol_bs_result

    cfg = ImpliedVolConfig() if cfg is None else cfg

    S = float(S)
    r = float(r)
    q = float(q)
    tau = float(tau)
    K_min = float(K_min)
    K_max = float(K_max)
    n = int(n)

    market = MarketData(spot=S, rate=r, dividend_yield=q)
    Ks = np.linspace(K_min, K_max, n)

    rows = []
    for K in Ks:
        K = float(K)
        sigma_true = float(true_vol_fn(K))

        spec = OptionSpec(kind=kind, strike=K, expiry=tau)
        p_true = PricingInputs(spec=spec, market=market, sigma=sigma_true, t=t)
        mkt_price = float(bs_price(p_true))

        try:
            ivres = implied_vol_bs_result(
                mkt_price=mkt_price,
                spec=spec,
                market=market,
                cfg=cfg,
                t=t,
                sigma0=float(sigma_guess),
            )
            iv = float(ivres.vol)
            converged = bool(getattr(ivres.root_result, "converged", True))
            iterations = int(getattr(ivres.root_result, "iterations", -1))
        except Exception:
            iv = np.nan
            converged = False
            iterations = -1

        rows.append(
            dict(
                K=K,
                sigma_true=sigma_true,
                mkt_price=mkt_price,
                implied_vol=iv,
                abs_error=float(abs(iv - sigma_true)) if np.isfinite(iv) else np.nan,
                converged=converged,
                iterations=iterations,
            )
        )

    df = pd.DataFrame(rows)
    if drop_failed:
        df = df[df["converged"]].reset_index(drop=True)
    return df


def run_bs_iv_benchmarks(
    *,
    price_tol: float = 5e-4,
    iv_tol: float = 5e-4,
    cfg: ImpliedVolConfig | None = None,
    strict: bool = False,
    save_csv_path: str | None = None,
) -> pd.DataFrame:
    """
    Deterministic regression-style benchmark for BS implied vol inversion.

    Builds a grid of (T, K, kind, sigma_true), computes BS prices, then inverts.
    Returns a DataFrame and (optionally) writes it to CSV.

    If strict=True, raises AssertionError if any case exceeds tolerances.
    """
    from option_pricing import (
        MarketData,
        OptionSpec,
        OptionType,
        PricingInputs,
        bs_price,
    )
    from option_pricing.vol.implied_vol import implied_vol_bs_result

    cfg = ImpliedVolConfig() if cfg is None else cfg

    S = 100.0
    r = 0.03
    q = 0.01
    t0 = 0.0
    market = MarketData(spot=S, rate=r, dividend_yield=q)

    taus = [0.25, 0.5, 1.0, 2.0]
    Ks = [60, 80, 90, 100, 110, 120, 140]
    sigmas = [0.05, 0.20, 0.50, 1.00]
    kinds = [OptionType.CALL, OptionType.PUT]

    rows = []
    for tau in taus:
        for K in Ks:
            for kind in kinds:
                for sigma_true in sigmas:
                    spec = OptionSpec(kind=kind, strike=float(K), expiry=float(tau))
                    p_true = PricingInputs(
                        spec=spec, market=market, sigma=float(sigma_true), t=t0
                    )
                    price = float(bs_price(p_true))

                    try:
                        ivres = implied_vol_bs_result(
                            mkt_price=price,
                            spec=spec,
                            market=market,
                            cfg=cfg,
                            t=t0,
                            sigma0=0.20,
                        )
                        iv = float(ivres.vol)
                        converged = bool(getattr(ivres.root_result, "converged", True))
                        iterations = int(getattr(ivres.root_result, "iterations", -1))
                        price_re = float(
                            bs_price(
                                PricingInputs(spec=spec, market=market, sigma=iv, t=t0)
                            )
                        )
                        abs_price_err = abs(price_re - price)
                        abs_iv_err = abs(iv - float(sigma_true))
                    except Exception:
                        iv = np.nan
                        converged = False
                        iterations = -1
                        abs_price_err = np.nan
                        abs_iv_err = np.nan

                    rows.append(
                        dict(
                            tau=float(tau),
                            K=float(K),
                            kind=str(kind),
                            sigma_true=float(sigma_true),
                            price=float(price),
                            implied_vol=iv,
                            abs_price_err=(
                                float(abs_price_err)
                                if np.isfinite(abs_price_err)
                                else np.nan
                            ),
                            abs_iv_err=(
                                float(abs_iv_err) if np.isfinite(abs_iv_err) else np.nan
                            ),
                            converged=converged,
                            iterations=iterations,
                        )
                    )

    df = pd.DataFrame(rows)

    # Optional assertions
    if strict:
        bad_price = df[df["converged"] & (df["abs_price_err"] > float(price_tol))]
        bad_iv = df[df["converged"] & (df["abs_iv_err"] > float(iv_tol))]
        assert (
            bad_price.empty
        ), f"Price error tolerance exceeded in {len(bad_price)} cases."
        assert bad_iv.empty, f"IV error tolerance exceeded in {len(bad_iv)} cases."
        assert df["converged"].all(), "Some benchmark cases did not converge."

    if save_csv_path:
        df.to_csv(save_csv_path, index=False)

    return df
