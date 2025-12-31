from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from option_pricing.types import PricingInputs
else:  # pragma: no cover
    PricingInputs = Any  # type: ignore


def _default_bs_price():
    """Default import path used in this repo for Black-Scholes pricing."""
    try:
        from option_pricing.pricers.black_scholes import bs_price  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import Black-Scholes pricer at option_pricing.pricers.black_scholes.bs_price. "
            "Pass bs_price_fn=... explicitly."
        ) from e
    return bs_price


def _default_mc_price():
    """Default import path used in this repo for Monte Carlo pricing."""
    try:
        from option_pricing.pricers.mc import mc_price  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import MC pricer at option_pricing.pricers.mc.mc_price. "
            "Pass mc_price_fn=... explicitly."
        ) from e
    return mc_price


def _to_float(v: Any, default: float = float("nan")) -> float:
    """Best-effort float conversion used by demo tables."""
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _extract(p: Any, name: str, default: float = float("nan")) -> float:
    return _to_float(getattr(p, name, default), default)


def _get_q_compat(p: Any) -> float:
    """Best-effort dividend yield extraction for legacy demo tables."""
    if hasattr(p, "q"):
        try:
            return _to_float(p.q, float("nan"))
        except Exception:
            pass
    m = getattr(p, "market", None)
    if m is not None and hasattr(m, "dividend_yield"):
        try:
            return _to_float(m.dividend_yield, float("nan"))
        except Exception:
            pass
    return float("nan")


def run_mc_vs_bs_cases(
    cases: Iterable[tuple[str, Any]],
    *,
    n_paths: int,
    bs_price_fn: Callable[[Any], float] | None = None,
    mc_price_fn: Callable[..., Any] | None = None,
    seed: int | None = 123,
    rng: np.random.Generator | None = None,
    per_case_seed: bool = True,
    zcrit: float = 1.96,
) -> pd.DataFrame:
    """Run one MC estimate per case and compare to a BS benchmark.

    Returns a tidy DataFrame with columns:
        case, tau, strike, spot, kind, bs, mc, se, err, abs_err, z, abs_z, zcrit, n_paths
    """
    bs_price_fn = bs_price_fn or _default_bs_price()
    mc_price_fn = mc_price_fn or _default_mc_price()

    rows: list[dict[str, object]] = []
    for i, (name, p) in enumerate(cases):
        seed_i = None
        if seed is not None:
            seed_i = int(seed + i) if per_case_seed else int(seed)

        out = mc_price_fn(p, n_paths=int(n_paths), seed=seed_i, rng=rng)

        # allow different MC return conventions
        if isinstance(out, tuple) and len(out) >= 2:
            mc_val, se_val = _to_float(out[0]), _to_float(out[1])
        elif isinstance(out, dict) and "price" in out:
            mc_val = _to_float(out.get("price"))
            se_raw = out.get("std_err")
            if se_raw is None:
                se_raw = out.get("se")
            if se_raw is None:
                se_raw = out.get("stderr")
            se_val = _to_float(se_raw)
        else:
            mc_val, se_val = _to_float(out), float("nan")

        bs_val = _to_float(bs_price_fn(p))
        err = mc_val - bs_val
        z = err / se_val if np.isfinite(se_val) and se_val > 0 else np.nan

        rows.append(
            {
                "case": name,
                "tau": _extract(p, "tau", np.nan),
                "spot": _extract(p, "S", _extract(p, "spot")),
                "strike": _extract(p, "K", _extract(p, "strike")),
                "kind": getattr(
                    p, "kind", getattr(getattr(p, "spec", None), "kind", "")
                ),
                "bs": bs_val,
                "mc": mc_val,
                "se": se_val,
                "err": err,
                "abs_err": abs(err),
                "z": z,
                "abs_z": abs(z) if np.isfinite(z) else np.nan,
                "zcrit": float(zcrit),
                "n_paths": int(n_paths),
            }
        )

    return pd.DataFrame(rows)


def compare_table(
    cases: list[tuple[str, PricingInputs]],
    *,
    n_paths: int = 10_000,
    seed: int | None = 0,
    rng: np.random.Generator | None = None,
    per_case_seed: bool = True,
    zcrit: float = 1.96,
) -> pd.DataFrame:
    """One MC run per case + BS benchmark (legacy-friendly columns)."""
    df = run_mc_vs_bs_cases(
        cases,
        n_paths=int(n_paths),
        seed=seed,
        rng=rng,
        per_case_seed=per_case_seed,
        zcrit=float(zcrit),
    )

    # Legacy aliases expected by older notebooks
    if "bs" in df.columns:
        df["BS"] = df["bs"]
    if "mc" in df.columns:
        df["MC"] = df["mc"]
    if "se" in df.columns:
        df["SE"] = df["se"]
    if "err" in df.columns:
        df["MC-BS"] = df["err"]

    if "q" not in df.columns:
        df["q"] = [_get_q_compat(p) for _, p in cases]

    return df


def convergence_table(
    p: PricingInputs,
    *,
    n_paths_list: Iterable[int] = (
        1_000,
        2_000,
        5_000,
        10_000,
        20_000,
        50_000,
        100_000,
    ),
    seed: int | None = 0,
) -> pd.DataFrame:
    """Run MC for a single PricingInputs across multiple path counts."""
    bs_fn = _default_bs_price()
    mc_fn = _default_mc_price()
    bs = _to_float(bs_fn(p))

    rows = []
    for n in n_paths_list:
        out = mc_fn(p, n_paths=int(n), seed=seed)
        if isinstance(out, tuple) and len(out) >= 2:
            mc_val, se_val = _to_float(out[0]), _to_float(out[1])
        elif isinstance(out, dict) and "price" in out:
            mc_val = _to_float(out.get("price"))
            se_val = _to_float(out.get("std_err") or out.get("se") or out.get("stderr"))
        else:
            mc_val, se_val = _to_float(out), float("nan")
        err = mc_val - bs
        rows.append(
            {"n_paths": int(n), "mc": mc_val, "se": se_val, "bs": bs, "err": err}
        )

    df = pd.DataFrame(rows).sort_values("n_paths").reset_index(drop=True)
    # legacy aliases
    df["MC"] = df["mc"]
    df["SE"] = df["se"]
    df["BS"] = df["bs"]
    df["MC-BS"] = df["err"]
    return df


__all__ = ["run_mc_vs_bs_cases", "compare_table", "convergence_table"]
