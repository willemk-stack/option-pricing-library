from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from option_pricing import MarketData, OptionType
from option_pricing.data_generators.recipes import (
    LatentNoArbSyntheticResult,
    generate_synthetic_surface_latent_noarb,
)
from option_pricing.pricers.black_scholes import bs_price_from_ctx
from option_pricing.vol.surface_core import VolSurface

from ._capstone2_defaults import get_capstone2_defaults


def _coerce_float(value: object, *, name: str) -> float:
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"{name} must be numeric, got {type(value).__name__}")


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **val}
        else:
            out[key] = val
    return out


def apply_demo_overrides(
    cfg: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    if not overrides:
        return dict(cfg)

    out = dict(cfg)
    for key, val in overrides.items():
        if key == "RUN_*":
            continue
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _merge_dict(out[key], val)
        else:
            out[key] = val
    return out


def resolve_run_flag(
    *,
    name: str,
    default: bool,
    overrides: dict[str, Any] | None,
) -> bool:
    if overrides and isinstance(overrides.get("RUN_*"), dict):
        if name in overrides["RUN_*"]:
            return bool(overrides["RUN_*"][name])
    return bool(default)


@dataclass(frozen=True, slots=True)
class SharedDemoScenario:
    profile: str
    seed: int
    cfg: dict[str, Any]
    latent: LatentNoArbSyntheticResult
    market: MarketData
    quotes_df: pd.DataFrame
    quote_summary: pd.DataFrame
    price_quotes_df: pd.DataFrame
    rows_obs: list[tuple[float, float, float]]
    rows_true: list[tuple[float, float, float]]
    forward: Any
    discount: Any
    surface_true: VolSurface
    noarb_true: Any
    expiries: np.ndarray
    y_grid: np.ndarray
    meta: dict[str, Any]

    @property
    def synthetic_bundle(self) -> dict[str, Any]:
        return {
            "quotes_df": self.quotes_df,
            "price_quotes_df": self.price_quotes_df,
            "rows_obs": self.rows_obs,
            "rows_true": self.rows_true,
            "surface_true_grid": self.surface_true,
            "noarb_true_grid": self.noarb_true,
            "forward": self.forward,
            "df_curve": self.discount,
            "synth_cfg_used": self.latent.cfg_used,
            "synth_tuning_log": self.latent.tuning_log,
        }


def build_price_quotes_from_iv_df(
    quotes_df: pd.DataFrame,
    *,
    market: MarketData,
    iv_col: str = "iv_obs",
) -> pd.DataFrame:
    ctx = market.to_context()
    price_quotes = quotes_df.copy()
    is_call = price_quotes["y"].to_numpy(dtype=np.float64) >= 0.0

    prices = [
        bs_price_from_ctx(
            kind=OptionType.CALL if call_flag else OptionType.PUT,
            strike=_coerce_float(row.K, name="strike"),
            sigma=_coerce_float(getattr(row, iv_col), name=iv_col),
            tau=_coerce_float(row.T, name="expiry"),
            ctx=ctx,
        )
        for row, call_flag in zip(
            price_quotes.itertuples(index=False), is_call, strict=True
        )
    ]

    price_quotes["is_call"] = is_call
    price_quotes["kind"] = np.where(is_call, "call", "put")
    price_quotes["price_mkt"] = np.asarray(prices, dtype=np.float64)
    return price_quotes


def build_shared_demo_scenario(
    *,
    profile: str = "quick",
    seed: int = 7,
    overrides: dict[str, Any] | None = None,
) -> SharedDemoScenario:
    profile_name = str(profile).lower().strip()
    if profile_name not in {"quick", "full"}:
        raise ValueError("profile must be 'quick' or 'full'")

    cfg = apply_demo_overrides(get_capstone2_defaults(seed), overrides)
    latent = generate_synthetic_surface_latent_noarb(
        enforce=bool(cfg.get("ENFORCE_ARB_FREE_LATENT_TRUTH", True)),
        max_rounds=int(cfg.get("SYNTH_MAX_ROUNDS", 8)),
        **cfg["SYNTH_CFG"],
    )

    synthetic = latent.synthetic
    rows_obs = [(float(t), float(k), float(iv)) for t, k, iv in synthetic.rows_obs]
    quotes_df = (
        pd.DataFrame(
            {
                "T": synthetic.T,
                "x": synthetic.x,
                "K": synthetic.K,
                "F": synthetic.F,
                "iv_obs": synthetic.iv_obs,
                "iv_true": synthetic.iv_true,
            }
        )
        .sort_values(["T", "K"])
        .reset_index(drop=True)
    )
    quotes_df["y"] = np.log(quotes_df["K"] / quotes_df["F"])
    quotes_df["w_obs"] = quotes_df["T"] * quotes_df["iv_obs"] ** 2
    quotes_df["w_true"] = quotes_df["T"] * quotes_df["iv_true"] ** 2
    quotes_df["iv_noise_bp"] = 1e4 * (quotes_df["iv_obs"] - quotes_df["iv_true"])

    quote_summary = (
        quotes_df.groupby("T")
        .agg(
            n_quotes=("K", "size"),
            iv_obs_min=("iv_obs", "min"),
            iv_obs_max=("iv_obs", "max"),
            mean_abs_noise_bp=("iv_noise_bp", lambda s: float(np.mean(np.abs(s)))),
        )
        .reset_index()
    )

    market = MarketData(
        spot=_coerce_float(cfg["SYNTH_CFG"]["spot"], name="spot"),
        rate=_coerce_float(cfg["SYNTH_CFG"]["r"], name="r"),
        dividend_yield=_coerce_float(cfg["SYNTH_CFG"]["q"], name="q"),
    )
    price_quotes_df = build_price_quotes_from_iv_df(quotes_df, market=market)

    expiries = np.asarray(sorted({float(t) for t in quotes_df["T"]}), dtype=np.float64)
    y_grid = np.asarray(sorted({float(y) for y in quotes_df["y"]}), dtype=np.float64)
    meta = {
        "profile": profile_name,
        "seed": int(seed),
        "noise_level": _coerce_float(
            cfg["SYNTH_CFG"]["noise_level"],
            name="noise_level",
        ),
        "n_quotes": int(len(quotes_df)),
        "n_expiries": int(expiries.size),
    }

    return SharedDemoScenario(
        profile=profile_name,
        seed=int(seed),
        cfg=cfg,
        latent=latent,
        market=market,
        quotes_df=quotes_df,
        quote_summary=quote_summary,
        price_quotes_df=price_quotes_df,
        rows_obs=rows_obs,
        rows_true=latent.rows_true,
        forward=synthetic.forward,
        discount=synthetic.df,
        surface_true=latent.surface_true,
        noarb_true=latent.noarb_true,
        expiries=expiries,
        y_grid=y_grid,
        meta=meta,
    )


__all__ = [
    "SharedDemoScenario",
    "apply_demo_overrides",
    "build_price_quotes_from_iv_df",
    "build_shared_demo_scenario",
    "resolve_run_flag",
]
