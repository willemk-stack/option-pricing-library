"""Bias, runtime, and scheme-comparison diagnostics for Heston Monte Carlo."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from ...market.curves import PricingContext
from ...models.heston import HestonParams
from ...models.heston.simulation import HestonScheme
from ...monte_carlo import MCConfig, RandomConfig
from ...numerics.quadrature import CompositeRule, QuadratureConfig
from ...pricers.heston import HestonBackend, heston_price_from_ctx
from ...pricers.heston_mc import (
    heston_mc_price_from_ctx,
    heston_vanilla_control_from_ctx,
)
from ...types import OptionType

_SWEEP_COLUMNS = [
    "scheme",
    "n_steps",
    "dt",
    "repeat",
    "seed",
    "n_paths",
    "effective_n",
    "antithetic",
    "reference_price",
    "price",
    "stderr",
    "signed_error",
    "abs_error",
    "ci_low",
    "ci_high",
    "ci_half_width",
    "covered_reference",
    "runtime_seconds",
    "runtime_per_path",
]

_BIAS_SUMMARY_COLUMNS = [
    "scheme",
    "n_steps",
    "dt",
    "mean_price",
    "mean_signed_error",
    "mean_abs_error",
    "rmse",
    "mean_stderr",
    "mean_ci_half_width",
    "coverage_rate",
    "repeat_count",
]

_RUNTIME_SUMMARY_COLUMNS = [
    "scheme",
    "n_steps",
    "dt",
    "mean_runtime_seconds",
    "median_runtime_seconds",
    "mean_runtime_per_path",
    "mean_abs_error",
    "rmse",
    "mean_stderr",
    "error_per_second",
    "repeat_count",
]

_SCHEME_COMPARISON_COLUMNS = [
    "scheme",
    "best_n_steps_by_abs_error",
    "best_dt_by_abs_error",
    "best_mean_abs_error",
    "best_rmse",
    "runtime_at_best_error",
    "fastest_runtime",
    "stderr_at_best_error",
    "coverage_rate_at_best_error",
    "recommended_use",
]


@dataclass(frozen=True, slots=True)
class HestonMCComparisonCase:
    ctx: PricingContext
    params: HestonParams
    kind: OptionType
    strike: float
    tau: float
    reference_price: float | None = None


@dataclass(frozen=True, slots=True)
class HestonMCSweepConfig:
    schemes: tuple[HestonScheme, ...] = (
        "euler_full_truncation",
        "quadratic_exponential",
    )
    n_steps_grid: tuple[int, ...] = (8, 16, 32, 64)
    n_paths: int = 20_000
    seed: int | None = 12345
    antithetic: bool = True
    repeats: int = 1
    use_control_variate: bool = False


def _require_columns(df: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _validate_case(case: HestonMCComparisonCase) -> None:
    if not np.isfinite(float(case.strike)) or float(case.strike) <= 0.0:
        raise ValueError("strike must be finite and positive")
    if not np.isfinite(float(case.tau)) or float(case.tau) <= 0.0:
        raise ValueError("tau must be finite and positive")
    if case.reference_price is not None and not np.isfinite(
        float(case.reference_price)
    ):
        raise ValueError("reference_price must be finite when provided")


def _validate_config(config: HestonMCSweepConfig) -> None:
    if not config.schemes:
        raise ValueError("schemes must contain at least one scheme")
    if not config.n_steps_grid:
        raise ValueError("n_steps_grid must contain at least one step count")
    if int(config.n_paths) <= 0:
        raise ValueError("n_paths must be positive")
    if int(config.repeats) <= 0:
        raise ValueError("repeats must be positive")
    if any(int(n_steps) <= 0 for n_steps in config.n_steps_grid):
        raise ValueError("n_steps_grid values must all be positive")


def _resolve_reference_price(
    case: HestonMCComparisonCase,
    *,
    reference_backend: HestonBackend,
    quad_cfg: QuadratureConfig | None,
    rule: CompositeRule | None,
) -> float:
    if case.reference_price is not None:
        return float(case.reference_price)
    return float(
        heston_price_from_ctx(
            kind=case.kind,
            strike=float(case.strike),
            tau=float(case.tau),
            ctx=case.ctx,
            params=case.params,
            backend=reference_backend,
            quad_cfg=quad_cfg,
            rule=rule,
        )
    )


def _resolve_control_variate(
    case: HestonMCComparisonCase,
    config: HestonMCSweepConfig,
    *,
    reference_backend: HestonBackend,
    quad_cfg: QuadratureConfig | None,
    rule: CompositeRule | None,
):
    if not config.use_control_variate:
        return None
    return heston_vanilla_control_from_ctx(
        ctx=case.ctx,
        kind=case.kind,
        strike=float(case.strike),
        tau=float(case.tau),
        params=case.params,
        backend=reference_backend,
        quad_cfg=quad_cfg,
        rule=rule,
    )


def _seed_for_run(
    base_seed: int | None,
    *,
    scheme_index: int,
    step_index: int,
    repeat: int,
) -> int | None:
    if base_seed is None:
        return None
    return int(base_seed) + scheme_index * 1_000_000 + step_index * 10_000 + repeat


def _mc_config_for_run(config: HestonMCSweepConfig, seed: int | None) -> MCConfig:
    if seed is None:
        return MCConfig(
            n_paths=int(config.n_paths),
            antithetic=bool(config.antithetic),
            rng=np.random.default_rng(),
        )
    return MCConfig(
        n_paths=int(config.n_paths),
        antithetic=bool(config.antithetic),
        random=RandomConfig(seed=int(seed)),
    )


def _rmse(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    if numeric.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(numeric))))


def _recommendation_for_scheme(scheme: str) -> str:
    if scheme == "quadratic_exponential":
        return "production candidate"
    if scheme == "euler_full_truncation":
        return "baseline / educational"
    return "review evidence before using"


def run_heston_mc_comparison_sweep(
    case: HestonMCComparisonCase,
    config: HestonMCSweepConfig | None = None,
    *,
    reference_backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> pd.DataFrame:
    config = HestonMCSweepConfig() if config is None else config
    _validate_case(case)
    _validate_config(config)

    reference_price = _resolve_reference_price(
        case,
        reference_backend=reference_backend,
        quad_cfg=quad_cfg,
        rule=rule,
    )
    control = _resolve_control_variate(
        case,
        config,
        reference_backend=reference_backend,
        quad_cfg=quad_cfg,
        rule=rule,
    )

    rows: list[dict[str, object]] = []
    for scheme_index, scheme in enumerate(config.schemes):
        for step_index, n_steps in enumerate(config.n_steps_grid):
            n_steps_int = int(n_steps)
            dt = float(case.tau) / float(n_steps_int)

            for repeat in range(int(config.repeats)):
                seed = _seed_for_run(
                    config.seed,
                    scheme_index=scheme_index,
                    step_index=step_index,
                    repeat=repeat,
                )
                mc_cfg = _mc_config_for_run(config, seed)

                start = perf_counter()
                result = heston_mc_price_from_ctx(
                    ctx=case.ctx,
                    kind=case.kind,
                    strike=float(case.strike),
                    params=case.params,
                    tau=float(case.tau),
                    n_steps=n_steps_int,
                    scheme=scheme,
                    cfg=mc_cfg,
                    control=control,
                )
                runtime_seconds = float(perf_counter() - start)
                ci = result.confidence_interval()
                signed_error = float(result.price - reference_price)
                runtime_per_path = (
                    runtime_seconds / float(result.n_paths)
                    if int(result.n_paths) > 0
                    else float("nan")
                )

                rows.append(
                    {
                        "scheme": str(scheme),
                        "n_steps": n_steps_int,
                        "dt": dt,
                        "repeat": repeat,
                        "seed": result.seed,
                        "n_paths": int(result.n_paths),
                        "effective_n": int(result.effective_n),
                        "antithetic": bool(result.antithetic),
                        "reference_price": reference_price,
                        "price": float(result.price),
                        "stderr": float(result.stderr),
                        "signed_error": signed_error,
                        "abs_error": abs(signed_error),
                        "ci_low": float(ci.low),
                        "ci_high": float(ci.high),
                        "ci_half_width": float(ci.half_width),
                        "covered_reference": bool(ci.low <= reference_price <= ci.high),
                        "runtime_seconds": runtime_seconds,
                        "runtime_per_path": runtime_per_path,
                    }
                )

    return pd.DataFrame.from_records(rows, columns=_SWEEP_COLUMNS)


def summarize_bias_vs_timestep(sweep: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        sweep,
        [
            "scheme",
            "n_steps",
            "dt",
            "price",
            "signed_error",
            "abs_error",
            "stderr",
            "ci_half_width",
            "covered_reference",
        ],
        name="sweep",
    )
    if sweep.empty:
        return pd.DataFrame(columns=_BIAS_SUMMARY_COLUMNS)

    rows: list[dict[str, object]] = []
    grouped = sweep.groupby(["scheme", "n_steps", "dt"], sort=True, dropna=False)
    for (scheme, n_steps, dt), group in grouped:
        rows.append(
            {
                "scheme": str(scheme),
                "n_steps": int(n_steps),
                "dt": float(dt),
                "mean_price": float(group["price"].mean()),
                "mean_signed_error": float(group["signed_error"].mean()),
                "mean_abs_error": float(group["abs_error"].mean()),
                "rmse": _rmse(group["signed_error"]),
                "mean_stderr": float(group["stderr"].mean()),
                "mean_ci_half_width": float(group["ci_half_width"].mean()),
                "coverage_rate": float(group["covered_reference"].astype(float).mean()),
                "repeat_count": int(len(group)),
            }
        )

    return pd.DataFrame.from_records(rows, columns=_BIAS_SUMMARY_COLUMNS)


def summarize_runtime_vs_error(sweep: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        sweep,
        [
            "scheme",
            "n_steps",
            "dt",
            "runtime_seconds",
            "runtime_per_path",
            "abs_error",
            "signed_error",
            "stderr",
        ],
        name="sweep",
    )
    if sweep.empty:
        return pd.DataFrame(columns=_RUNTIME_SUMMARY_COLUMNS)

    rows: list[dict[str, object]] = []
    grouped = sweep.groupby(["scheme", "n_steps", "dt"], sort=True, dropna=False)
    for (scheme, n_steps, dt), group in grouped:
        mean_runtime_seconds = float(group["runtime_seconds"].mean())
        mean_abs_error = float(group["abs_error"].mean())
        error_per_second = (
            mean_abs_error / mean_runtime_seconds
            if mean_runtime_seconds > 0.0
            else float("nan")
        )
        rows.append(
            {
                "scheme": str(scheme),
                "n_steps": int(n_steps),
                "dt": float(dt),
                "mean_runtime_seconds": mean_runtime_seconds,
                "median_runtime_seconds": float(group["runtime_seconds"].median()),
                "mean_runtime_per_path": float(group["runtime_per_path"].mean()),
                "mean_abs_error": mean_abs_error,
                "rmse": _rmse(group["signed_error"]),
                "mean_stderr": float(group["stderr"].mean()),
                "error_per_second": error_per_second,
                "repeat_count": int(len(group)),
            }
        )

    return pd.DataFrame.from_records(rows, columns=_RUNTIME_SUMMARY_COLUMNS)


def compare_heston_mc_schemes(sweep: pd.DataFrame) -> pd.DataFrame:
    bias_summary = summarize_bias_vs_timestep(sweep)
    runtime_summary = summarize_runtime_vs_error(sweep)
    if runtime_summary.empty:
        return pd.DataFrame(columns=_SCHEME_COMPARISON_COLUMNS)

    merged = runtime_summary.merge(
        bias_summary[
            [
                "scheme",
                "n_steps",
                "dt",
                "coverage_rate",
            ]
        ],
        on=["scheme", "n_steps", "dt"],
        how="left",
    )

    best_rows = (
        merged.sort_values(
            ["scheme", "mean_abs_error", "rmse", "mean_runtime_seconds", "n_steps"],
            kind="stable",
        )
        .groupby("scheme", sort=True, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    fastest_runtime = runtime_summary.groupby("scheme", sort=True, as_index=False).agg(
        {"mean_runtime_seconds": "min"}
    )
    fastest_runtime = fastest_runtime.rename(
        columns={"mean_runtime_seconds": "fastest_runtime"}
    )

    comparison = best_rows.merge(fastest_runtime, on="scheme", how="left")
    comparison = comparison.rename(
        columns={
            "n_steps": "best_n_steps_by_abs_error",
            "dt": "best_dt_by_abs_error",
            "mean_abs_error": "best_mean_abs_error",
            "rmse": "best_rmse",
            "mean_runtime_seconds": "runtime_at_best_error",
            "mean_stderr": "stderr_at_best_error",
            "coverage_rate": "coverage_rate_at_best_error",
        }
    )
    comparison["recommended_use"] = [
        _recommendation_for_scheme(str(scheme))
        for scheme in comparison["scheme"].tolist()
    ]
    return comparison[_SCHEME_COMPARISON_COLUMNS].copy()
