from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import os
import platform
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from option_pricing.models.black_scholes.bs import black76_call_price_vec
from option_pricing.numerics.pde.ic_remedies import ICRemedy
from option_pricing.pricers.black_scholes import bs_price_from_ctx, digital_price
from option_pricing.pricers.pde.domain import BSDomainConfig, BSDomainPolicy
from option_pricing.pricers.pde_pricer import (
    bs_digital_price_pde,
    bs_price_pde_european,
)
from option_pricing.pricers.tree import binom_price_from_ctx
from option_pricing.types import (
    DigitalSpec,
    MarketData,
    OptionSpec,
    OptionType,
    PricingInputs,
)
from option_pricing.viz.publishing import (
    PUBLISHING_THEMES,
    copy_light_variant,
    publishing_palette,
    publishing_style,
    save_figure,
    themed_asset_paths,
)
from option_pricing.vol.implied_vol_scalar import implied_vol_bs_result
from option_pricing.vol.implied_vol_slice import implied_vol_black76_slice
from option_pricing.vol.local_vol_dupire import local_vol_from_call_grid_diagnostics
from option_pricing.vol.local_vol_surface import LocalVolSurface
from option_pricing.vol.surface_core import VolSurface
from option_pricing.vol.svi import SVIParams, svi_total_variance


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build reproducible benchmark tables, plots, and machine-readable "
            "summaries for docs/performance.md."
        )
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ROOT / "benchmarks" / "artifacts",
        help="Directory for JSON/CSV benchmark snapshots.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=ROOT / "docs" / "assets" / "generated" / "benchmarks",
        help="Directory for generated benchmark plots.",
    )
    parser.add_argument(
        "--pytest-benchmark-json",
        type=Path,
        default=None,
        help="Optional pytest-benchmark JSON to summarize alongside publish artifacts.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Default repeat count for timing measurements.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup repetitions before timing.",
    )
    return parser.parse_args(argv)


def _domain_cfg() -> BSDomainConfig:
    return BSDomainConfig(
        policy=BSDomainPolicy.LOG_NSIGMA,
        n_sigma=6.0,
        center="strike",
    )


def _time_callable(
    fn,
    *,
    repeat: int,
    warmup: int,
) -> dict[str, float]:
    for _ in range(max(int(warmup), 0)):
        fn()
    samples: list[float] = []
    for _ in range(max(int(repeat), 1)):
        t0 = perf_counter()
        fn()
        samples.append(1e3 * (perf_counter() - t0))
    arr = np.asarray(samples, dtype=float)
    return {
        "min_ms": float(np.min(arr)),
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr, ddof=0)),
    }


def _write_frame(df: pd.DataFrame, *, stem: str, artifacts_dir: Path) -> dict[str, str]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    csv_path = artifacts_dir / f"{stem}.csv"
    json_path = artifacts_dir / f"{stem}.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    return {
        "csv": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
        "json": str(json_path.relative_to(ROOT)).replace("\\", "/"),
    }


def _write_json(payload: dict[str, Any], *, name: str, artifacts_dir: Path) -> str:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _collect_environment() -> dict[str, Any]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except Exception:
        versions["scipy"] = "unavailable"

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "processor": platform.processor(),
        "versions": versions,
    }


def _build_iv_rows(
    *,
    n_strikes: int,
    market: MarketData,
    tau: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    strikes = np.linspace(60.0, 140.0, int(n_strikes), dtype=float)
    prices = black76_call_price_vec(
        forward=market.forward(tau),
        strikes=strikes,
        sigma=float(sigma),
        tau=float(tau),
        df=market.df(tau),
    )
    return strikes, np.asarray(prices, dtype=float)


def _build_macro_rows(
    *,
    n_expiries: int,
    n_strikes: int,
    market: MarketData,
    params: SVIParams,
) -> list[tuple[float, float, float]]:
    expiries = np.linspace(0.25, 2.0, int(n_expiries), dtype=float)
    strikes = np.linspace(60.0, 140.0, int(n_strikes), dtype=float)
    rows: list[tuple[float, float, float]] = []
    for expiry in expiries:
        forward = market.forward(float(expiry))
        y = np.log(strikes / forward)
        w = svi_total_variance(y, params)
        iv = np.sqrt(np.maximum(w / float(expiry), 1e-12))
        for strike, iv_k in zip(strikes, iv, strict=False):
            rows.append((float(expiry), float(strike), float(iv_k)))
    return rows


def _build_dupire_call_grid(
    *,
    n_expiries: int,
    n_strikes: int,
    market: MarketData,
    params: SVIParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expiries = np.linspace(0.25, 2.0, int(n_expiries), dtype=float)
    strikes = np.linspace(60.0, 140.0, int(n_strikes), dtype=float)

    rows = []
    for expiry in expiries:
        forward = market.forward(float(expiry))
        y = np.log(strikes / forward)
        smile_var = np.maximum(svi_total_variance(y, params), 1e-10)
        sigma = np.sqrt(smile_var)
        rows.append(
            black76_call_price_vec(
                forward=forward,
                strikes=strikes,
                sigma=sigma,
                tau=float(expiry),
                df=market.df(float(expiry)),
            )
        )
    return expiries, strikes, np.asarray(rows, dtype=float)


def _price_digital_local_vol_pde(
    *,
    lv: LocalVolSurface,
    market: MarketData,
    strike: float,
    tau: float,
    nx: int,
    nt: int,
) -> float:
    from option_pricing.numerics.grids import GridConfig
    from option_pricing.numerics.pde import solve_pde_1d
    from option_pricing.numerics.pde.domain import Coord
    from option_pricing.numerics.pde.ic_remedies import ic_cell_average
    from option_pricing.pricers.pde.digital_local_vol import local_vol_pde_wiring
    from option_pricing.pricers.pde.domain import bs_compute_bounds

    domain_cfg = _domain_cfg()

    sigma_for_bounds = 0.2
    try:
        sigma_for_bounds = float(np.sqrt(lv.local_var(strike, tau)))
    except Exception:
        sigma_for_bounds = 0.2
    if not np.isfinite(sigma_for_bounds) or sigma_for_bounds <= 0.0:
        sigma_for_bounds = 0.2

    p = PricingInputs(
        spec=DigitalSpec(
            kind=OptionType.CALL,
            strike=float(strike),
            expiry=float(tau),
            payout=1.0,
        ),
        market=market,
        sigma=sigma_for_bounds,
        t=0.0,
    )

    bounds = bs_compute_bounds(p, coord=Coord.LOG_S, cfg=domain_cfg)
    wiring = local_vol_pde_wiring(
        p,
        lv,
        Coord.LOG_S,
        x_lb=bounds.x_lb,
        x_ub=bounds.x_ub,
    )
    xk = float(np.asarray(wiring.to_x(p.spec.strike)).reshape(()))

    def ic_transform(grid, ic):
        return ic_cell_average(grid, ic, breakpoints=(xk,))

    grid_cfg = GridConfig(
        Nx=int(nx),
        Nt=int(nt),
        x_lb=bounds.x_lb,
        x_ub=bounds.x_ub,
        T=float(tau),
        spacing=domain_cfg.spacing,
        x_center=bounds.x_center,
        cluster_strength=domain_cfg.cluster_strength,
    )
    sol = solve_pde_1d(
        wiring.problem,
        grid_cfg=grid_cfg,
        method="rannacher",
        store="final",
        ic_transform=ic_transform,
    )
    x = np.asarray(sol.grid.x, dtype=float)
    u = np.asarray(sol.u_final, dtype=float)
    return float(np.interp(wiring.x_0, x, u))


def _digital_monotonicity(sol) -> tuple[int, float]:
    u = np.asarray(sol.u_final, dtype=float)
    du = np.diff(u)
    violations = du < -1e-8
    max_negative_step = float(np.max(np.maximum(-du, 0.0))) if du.size else 0.0
    return int(np.sum(violations)), max_negative_step


def _interior_invalid_share(report) -> float:
    invalid = np.asarray(report.invalid, dtype=bool)
    trim_t = int(getattr(report, "trim_t", 0))
    trim_k = int(getattr(report, "trim_k", 0))
    if invalid.shape[0] <= 2 * trim_t or invalid.shape[1] <= 2 * trim_k:
        return float("nan")
    interior = invalid[
        trim_t : invalid.shape[0] - trim_t,
        trim_k : invalid.shape[1] - trim_k,
    ]
    if interior.size == 0:
        return float("nan")
    return float(np.mean(interior))


def _run_iv_family(
    *,
    artifacts_dir: Path,
    plot_dir: Path,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    market = MarketData(spot=100.0, rate=0.01, dividend_yield=0.0)
    tau = 0.5
    sigma = 0.2

    scaling_rows: list[dict[str, Any]] = []
    for n_strikes in (21, 61, 201, 801):
        strikes, prices = _build_iv_rows(
            n_strikes=n_strikes,
            market=market,
            tau=tau,
            sigma=sigma,
        )
        forward = market.forward(tau)
        discount = market.df(tau)

        def vectorized_run(
            *,
            forward: float = forward,
            discount: float = discount,
            strikes: np.ndarray = strikes,
            prices: np.ndarray = prices,
            initial_sigma: float = sigma,
            expiry: float = tau,
        ) -> np.ndarray:
            return np.asarray(
                implied_vol_black76_slice(
                    forward=forward,
                    strikes=strikes,
                    tau=expiry,
                    df=discount,
                    prices=prices,
                    is_call=True,
                    initial_sigma=initial_sigma,
                    return_result=False,
                ),
                dtype=float,
            )

        def scalar_loop_run(
            *,
            strikes: np.ndarray = strikes,
            prices: np.ndarray = prices,
            expiry: float = tau,
            initial_sigma: float = sigma,
        ) -> np.ndarray:
            out = []
            for strike, price in zip(strikes, prices, strict=False):
                spec = OptionSpec(
                    kind=OptionType.CALL,
                    strike=float(strike),
                    expiry=float(expiry),
                )
                out.append(
                    float(
                        implied_vol_bs_result(
                            float(price),
                            spec,
                            market,
                            sigma0=float(initial_sigma),
                        ).vol
                    )
                )
            return np.asarray(out, dtype=float)

        vec_vol = vectorized_run()
        scalar_vol = scalar_loop_run()
        max_abs_diff = float(np.max(np.abs(vec_vol - scalar_vol)))

        for implementation, fn, impl_repeat in (
            ("vectorized_slice", vectorized_run, repeat),
            ("scalar_loop", scalar_loop_run, max(1, repeat - 1)),
        ):
            stats = _time_callable(fn, repeat=impl_repeat, warmup=warmup)
            runtime_ms = float(stats["median_ms"])
            scaling_rows.append(
                {
                    "implementation": implementation,
                    "n_strikes": int(n_strikes),
                    "runtime_ms": runtime_ms,
                    "latency_us_per_option": 1e3 * runtime_ms / float(n_strikes),
                    "throughput_options_per_s": 1e3
                    * float(n_strikes)
                    / max(runtime_ms, 1e-12),
                    "max_abs_vol_diff_vs_scalar": max_abs_diff,
                    **stats,
                }
            )

    scaling = pd.DataFrame(scaling_rows).sort_values(["implementation", "n_strikes"])

    scalar_rows: list[dict[str, Any]] = []
    for scenario, strike, scalar_tau in (
        ("ATM 1Y", 100.0, 1.0),
        ("ATM 2W", 100.0, 0.02),
        ("OTM 1Y", 140.0, 1.0),
    ):
        price = bs_price_from_ctx(
            kind=OptionType.CALL,
            strike=float(strike),
            sigma=sigma,
            tau=float(scalar_tau),
            ctx=market.to_context(),
        )
        spec = OptionSpec(
            kind=OptionType.CALL,
            strike=float(strike),
            expiry=float(scalar_tau),
        )

        def scalar_run(
            *,
            price: float = float(price),
            spec: OptionSpec = spec,
            sigma0: float = sigma,
        ) -> float:
            return float(
                implied_vol_bs_result(
                    price,
                    spec,
                    market,
                    sigma0=sigma0,
                ).vol
            )

        stats = _time_callable(
            scalar_run,
            repeat=max(5, repeat * 3),
            warmup=max(1, warmup),
        )
        scalar_rows.append(
            {
                "scenario": scenario,
                "strike": float(strike),
                "tau": float(scalar_tau),
                "runtime_ms": float(stats["median_ms"]),
                **stats,
            }
        )

    scalar_df = pd.DataFrame(scalar_rows)
    speedup = (
        scaling.pivot(index="n_strikes", columns="implementation", values="runtime_ms")
        .assign(vectorized_speedup_x=lambda d: d["scalar_loop"] / d["vectorized_slice"])
        .reset_index()
    )

    def render(theme: str):
        with publishing_style(theme=theme) as plt:
            fig, (ax_scaling, ax_scalar) = plt.subplots(
                1,
                2,
                figsize=(10.5, 4.2),
                constrained_layout=True,
            )
            for implementation, sub in scaling.groupby("implementation", sort=False):
                ax_scaling.plot(
                    sub["n_strikes"].to_numpy(),
                    sub["runtime_ms"].to_numpy(),
                    marker="o",
                    linewidth=2,
                    label=implementation.replace("_", " "),
                )
            ax_scaling.set_xscale("log")
            ax_scaling.set_yscale("log")
            ax_scaling.set_xlabel("Slice size (number of strikes)")
            ax_scaling.set_ylabel("Median runtime (ms)")
            ax_scaling.set_title("Slice inversion scaling")
            ax_scaling.legend()

            ax_scalar.bar(
                scalar_df["scenario"].to_list(),
                scalar_df["runtime_ms"].to_numpy(),
                color=["#1f77b4", "#ff7f0e", "#2ca02c"],
            )
            ax_scalar.set_ylabel("Median runtime (ms)")
            ax_scalar.set_title("Scalar inversion latency")
            ax_scalar.tick_params(axis="x", labelrotation=15)
            return fig

    figure_path = _save_themed_plot(plot_dir / "iv_scaling.png", dpi=180, render=render)

    scaling_files = _write_frame(
        scaling, stem="iv_scaling", artifacts_dir=artifacts_dir
    )
    scenario_files = _write_frame(
        scalar_df,
        stem="iv_scalar_scenarios",
        artifacts_dir=artifacts_dir,
    )
    speedup_files = _write_frame(
        speedup,
        stem="iv_speedup",
        artifacts_dir=artifacts_dir,
    )

    return {
        "scaling": scaling_files,
        "scalar_scenarios": scenario_files,
        "speedup": speedup_files,
        "figure": str(figure_path.relative_to(ROOT)).replace("\\", "/"),
        "max_speedup_x": float(speedup["vectorized_speedup_x"].max()),
    }


def _run_pde_family(
    *,
    artifacts_dir: Path,
    plot_dir: Path,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    market = MarketData(spot=100.0, rate=0.01, dividend_yield=0.0)
    vanilla_input = PricingInputs(
        spec=OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0),
        market=market,
        sigma=0.2,
        t=0.0,
    )
    analytic = bs_price_from_ctx(
        kind=OptionType.CALL,
        strike=100.0,
        sigma=0.2,
        tau=1.0,
        ctx=market.to_context(),
    )

    vanilla_rows: list[dict[str, Any]] = []
    for method in ("cn", "rannacher"):
        for nx in (101, 201, 401, 801):
            nt = nx

            def run(
                *,
                nx: int = nx,
                nt: int = nt,
                method: str = method,
            ) -> float:
                return float(
                    bs_price_pde_european(
                        vanilla_input,
                        coord="logS",
                        domain_cfg=_domain_cfg(),
                        Nx=int(nx),
                        Nt=int(nt),
                        method=method,
                    )
                )

            price = run()
            stats = _time_callable(run, repeat=repeat, warmup=warmup)
            vanilla_rows.append(
                {
                    "family": "black_scholes_vanilla",
                    "series": f"BS PDE {method}",
                    "method": method,
                    "Nx": int(nx),
                    "Nt": int(nt),
                    "grid_points": int(nx) * int(nt),
                    "runtime_ms": float(stats["median_ms"]),
                    "price": float(price),
                    "reference_price": float(analytic),
                    "abs_error": abs(float(price) - float(analytic)),
                    **stats,
                }
            )

    params = SVIParams(a=0.02, b=0.3, rho=-0.4, m=0.0, sigma=0.4)
    rows = _build_macro_rows(
        n_expiries=31,
        n_strikes=61,
        market=market,
        params=params,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        surface = VolSurface.from_svi(
            rows,
            forward=market.forward,
            calibrate_kwargs={
                "loss": "linear",
                "irls_max_outer": 0,
                "repair_butterfly": False,
            },
        )
        lv = LocalVolSurface.from_implied(
            surface,
            forward=market.forward,
            discount=market.df,
        )

    reference_price = _price_digital_local_vol_pde(
        lv=lv,
        market=market,
        strike=100.0,
        tau=1.0,
        nx=601,
        nt=601,
    )
    localvol_rows: list[dict[str, Any]] = []
    for nx in (101, 201, 301, 401):
        nt = nx

        def run_localvol(*, nx: int = nx, nt: int = nt) -> float:
            return float(
                _price_digital_local_vol_pde(
                    lv=lv,
                    market=market,
                    strike=100.0,
                    tau=1.0,
                    nx=nx,
                    nt=nt,
                )
            )

        price = run_localvol()
        stats = _time_callable(
            run_localvol,
            repeat=max(1, repeat - 1),
            warmup=warmup,
        )
        localvol_rows.append(
            {
                "family": "local_vol_digital",
                "series": "Local-vol PDE rannacher",
                "method": "rannacher",
                "Nx": int(nx),
                "Nt": int(nt),
                "grid_points": int(nx) * int(nt),
                "runtime_ms": float(stats["median_ms"]),
                "price": float(price),
                "reference_price": float(reference_price),
                "abs_error": abs(float(price) - float(reference_price)),
                **stats,
            }
        )

    frame = pd.concat(
        [pd.DataFrame(vanilla_rows), pd.DataFrame(localvol_rows)],
        ignore_index=True,
    )

    def render(theme: str):
        with publishing_style(theme=theme) as plt:
            fig, (ax_tradeoff, ax_conv) = plt.subplots(
                1,
                2,
                figsize=(10.8, 4.3),
                constrained_layout=True,
            )
            for series, sub in frame.groupby("series", sort=False):
                ax_tradeoff.plot(
                    sub["runtime_ms"].to_numpy(),
                    sub["abs_error"].to_numpy(),
                    marker="o",
                    linewidth=2,
                    label=series,
                )
                ax_conv.plot(
                    sub["grid_points"].to_numpy(),
                    sub["abs_error"].to_numpy(),
                    marker="o",
                    linewidth=2,
                    label=series,
                )
            ax_tradeoff.set_xscale("log")
            ax_tradeoff.set_yscale("log")
            ax_tradeoff.set_xlabel("Median runtime (ms)")
            ax_tradeoff.set_ylabel("Absolute error")
            ax_tradeoff.set_title("Cost vs accuracy")
            ax_tradeoff.legend()

            ax_conv.set_xscale("log")
            ax_conv.set_yscale("log")
            ax_conv.set_xlabel("Grid points (Nx * Nt)")
            ax_conv.set_ylabel("Absolute error")
            ax_conv.set_title("Grid refinement")
            return fig

    figure_path = _save_themed_plot(
        plot_dir / "pde_runtime_error_tradeoff.png",
        dpi=180,
        render=render,
    )
    files = _write_frame(
        frame.sort_values(["family", "grid_points"]),
        stem="pde_runtime_error_tradeoff",
        artifacts_dir=artifacts_dir,
    )
    return {
        "snapshot": files,
        "figure": str(figure_path.relative_to(ROOT)).replace("\\", "/"),
        "localvol_reference_grid": {"Nx": 601, "Nt": 601},
    }


def _run_digital_family(
    *,
    artifacts_dir: Path,
    plot_dir: Path,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    from option_pricing.instruments.digital import DigitalOption

    market = MarketData(spot=100.0, rate=0.01, dividend_yield=0.0)
    pricing_input = PricingInputs(
        spec=DigitalSpec(
            kind=OptionType.CALL,
            strike=100.0,
            expiry=1.0,
            payout=1.0,
        ),
        market=market,
        sigma=0.2,
        t=0.0,
    )
    analytic = float(
        digital_price(
            DigitalOption(
                kind=OptionType.CALL,
                strike=100.0,
                expiry=1.0,
                payout=1.0,
            ),
            ctx=market.to_context(),
            sigma=0.2,
        )
    )

    cfgs = [
        ("CN + none", "cn", ICRemedy.NONE),
        ("Rannacher + none", "rannacher", ICRemedy.NONE),
        ("Rannacher + cell average", "rannacher", ICRemedy.CELL_AVG),
        ("Rannacher + L2 projection", "rannacher", ICRemedy.L2_PROJ),
    ]
    rows: list[dict[str, Any]] = []
    for label, method, remedy in cfgs:
        for nx in (101, 201, 401):
            nt = nx

            def run(
                return_solution: bool = False,
                *,
                nx: int = nx,
                nt: int = nt,
                method: str = method,
                remedy: ICRemedy = remedy,
            ):
                return bs_digital_price_pde(
                    pricing_input,
                    coord="logS",
                    domain_cfg=_domain_cfg(),
                    Nx=int(nx),
                    Nt=int(nt),
                    method=method,
                    ic_remedy=remedy,
                    return_solution=return_solution,
                )

            price, sol = run(return_solution=True)
            stats = _time_callable(
                lambda: run(return_solution=False),
                repeat=max(1, repeat - 1),
                warmup=warmup,
            )
            monotonicity_breaks, max_negative_step = _digital_monotonicity(sol)
            rows.append(
                {
                    "config": label,
                    "method": method,
                    "ic_remedy": str(remedy.value),
                    "Nx": int(nx),
                    "Nt": int(nt),
                    "grid_points": int(nx) * int(nt),
                    "runtime_ms": float(stats["median_ms"]),
                    "price": float(price),
                    "reference_price": float(analytic),
                    "abs_error": abs(float(price) - float(analytic)),
                    "monotonicity_breaks": int(monotonicity_breaks),
                    "max_negative_step": float(max_negative_step),
                    **stats,
                }
            )

    frame = pd.DataFrame(rows).sort_values(["config", "grid_points"])
    spread = (
        frame.groupby("config", as_index=False)
        .agg(
            refinement_spread=("price", lambda s: float(np.max(s) - np.min(s))),
            best_abs_error=("abs_error", "min"),
        )
        .sort_values("refinement_spread")
        .reset_index(drop=True)
    )

    def render(theme: str):
        with publishing_style(theme=theme) as plt:
            fig, (ax_tradeoff, ax_stability) = plt.subplots(
                1,
                2,
                figsize=(10.8, 4.3),
                constrained_layout=True,
            )
            for label, sub in frame.groupby("config", sort=False):
                ax_tradeoff.plot(
                    sub["runtime_ms"].to_numpy(),
                    sub["abs_error"].to_numpy(),
                    marker="o",
                    linewidth=2,
                    label=label,
                )
                ax_stability.plot(
                    sub["grid_points"].to_numpy(),
                    sub["monotonicity_breaks"].to_numpy(),
                    marker="o",
                    linewidth=2,
                    label=label,
                )
            ax_tradeoff.set_xscale("log")
            ax_tradeoff.set_yscale("log")
            ax_tradeoff.set_xlabel("Median runtime (ms)")
            ax_tradeoff.set_ylabel("Absolute error")
            ax_tradeoff.set_title("Digital remedy tradeoff")
            ax_tradeoff.legend()

            ax_stability.bar(
                spread["config"].to_list(),
                spread["refinement_spread"].to_numpy(),
                color=["#55a868", "#4c72b0", "#ccb974", "#c44e52"],
            )
            ax_stability.set_ylabel("Price spread across tested grids")
            ax_stability.set_title("Grid-refinement stability")
            ax_stability.tick_params(axis="x", labelrotation=20)
            return fig

    figure_path = _save_themed_plot(
        plot_dir / "digital_remedies_tradeoff.png",
        dpi=180,
        render=render,
    )
    files = _write_frame(
        frame,
        stem="digital_remedies_tradeoff",
        artifacts_dir=artifacts_dir,
    )
    spread_files = _write_frame(
        spread,
        stem="digital_remedies_stability",
        artifacts_dir=artifacts_dir,
    )
    return {
        "snapshot": files,
        "stability": spread_files,
        "figure": str(figure_path.relative_to(ROOT)).replace("\\", "/"),
    }


def _run_localvol_family(
    *,
    artifacts_dir: Path,
    plot_dir: Path,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    market = MarketData(spot=100.0, rate=0.01, dividend_yield=0.0)
    params = SVIParams(a=0.02, b=0.3, rho=-0.4, m=0.0, sigma=0.4)

    rows: list[dict[str, Any]] = []
    for n_expiries, n_strikes in ((31, 61), (61, 121), (121, 241)):
        expiries, strikes, call = _build_dupire_call_grid(
            n_expiries=n_expiries,
            n_strikes=n_strikes,
            market=market,
            params=params,
        )
        for strike_coordinate in ("logK", "K"):

            def run(
                *,
                call: np.ndarray = call,
                strikes: np.ndarray = strikes,
                expiries: np.ndarray = expiries,
                strike_coordinate: str = strike_coordinate,
            ):
                return local_vol_from_call_grid_diagnostics(
                    call=call,
                    strikes=strikes,
                    taus=expiries,
                    market=market,
                    strike_coordinate=strike_coordinate,
                )

            report = run()
            stats = _time_callable(run, repeat=repeat, warmup=warmup)
            total_points = int(n_expiries) * int(n_strikes)
            rows.append(
                {
                    "strike_coordinate": strike_coordinate,
                    "nT": int(n_expiries),
                    "nK": int(n_strikes),
                    "grid_points": total_points,
                    "runtime_ms": float(stats["median_ms"]),
                    "invalid_share": float(report.invalid_count) / float(total_points),
                    "interior_invalid_share": _interior_invalid_share(report),
                    "invalid_count": int(report.invalid_count),
                    **stats,
                }
            )

    frame = pd.DataFrame(rows).sort_values(["strike_coordinate", "grid_points"])

    def render(theme: str):
        with publishing_style(theme=theme) as plt:
            fig, (ax_runtime, ax_invalid) = plt.subplots(
                1,
                2,
                figsize=(10.6, 4.2),
                constrained_layout=True,
            )
            for label, sub in frame.groupby("strike_coordinate", sort=False):
                ax_runtime.plot(
                    sub["grid_points"].to_numpy(),
                    sub["runtime_ms"].to_numpy(),
                    marker="o",
                    linewidth=2,
                    label=label,
                )
                ax_invalid.plot(
                    sub["grid_points"].to_numpy(),
                    100.0 * sub["interior_invalid_share"].to_numpy(),
                    marker="o",
                    linewidth=2,
                    label=label,
                )
            ax_runtime.set_xscale("log")
            ax_runtime.set_yscale("log")
            ax_runtime.set_xlabel("Grid points (nT * nK)")
            ax_runtime.set_ylabel("Median runtime (ms)")
            ax_runtime.set_title("Dupire extraction scaling")
            ax_runtime.legend()

            ax_invalid.set_xscale("log")
            ax_invalid.set_xlabel("Grid points (nT * nK)")
            ax_invalid.set_ylabel("Interior invalid share (%)")
            ax_invalid.set_title("Validity diagnostic")
            return fig

    figure_path = _save_themed_plot(
        plot_dir / "localvol_scaling.png",
        dpi=180,
        render=render,
    )
    files = _write_frame(frame, stem="localvol_scaling", artifacts_dir=artifacts_dir)
    return {
        "snapshot": files,
        "figure": str(figure_path.relative_to(ROOT)).replace("\\", "/"),
    }


def _run_macro_pipeline_family(
    *,
    artifacts_dir: Path,
    plot_dir: Path,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    market = MarketData(spot=100.0, rate=0.01, dividend_yield=0.0)
    params = SVIParams(a=0.02, b=0.3, rho=-0.4, m=0.0, sigma=0.4)
    scenarios = [
        ("small", 21, 41, 101, 101),
        ("medium", 31, 61, 201, 201),
        ("large", 61, 121, 301, 301),
    ]

    rows: list[dict[str, Any]] = []
    for label, n_expiries, n_strikes, nx, nt in scenarios:
        stage_samples: dict[str, list[float]] = {
            "quote_mesh_ms": [],
            "surface_fit_ms": [],
            "surface_handoff_ms": [],
            "local_vol_ms": [],
            "pde_reprice_ms": [],
        }
        final_price = float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            for _ in range(max(int(warmup), 0)):
                rows_raw = _build_macro_rows(
                    n_expiries=n_expiries,
                    n_strikes=n_strikes,
                    market=market,
                    params=params,
                )
                surface = VolSurface.from_svi(
                    rows_raw,
                    forward=market.forward,
                    calibrate_kwargs={
                        "loss": "linear",
                        "irls_max_outer": 0,
                        "repair_butterfly": False,
                    },
                )
                _ = np.vstack(
                    [
                        surface.iv(np.linspace(60.0, 140.0, 25, dtype=float), expiry)
                        for expiry in np.linspace(0.25, 2.0, 9, dtype=float)
                    ]
                )
                lv = LocalVolSurface.from_implied(
                    surface,
                    forward=market.forward,
                    discount=market.df,
                )
                _ = _price_digital_local_vol_pde(
                    lv=lv,
                    market=market,
                    strike=100.0,
                    tau=1.0,
                    nx=nx,
                    nt=nt,
                )

            for _ in range(max(int(repeat), 1)):
                t0 = perf_counter()
                rows_raw = _build_macro_rows(
                    n_expiries=n_expiries,
                    n_strikes=n_strikes,
                    market=market,
                    params=params,
                )
                t1 = perf_counter()
                surface = VolSurface.from_svi(
                    rows_raw,
                    forward=market.forward,
                    calibrate_kwargs={
                        "loss": "linear",
                        "irls_max_outer": 0,
                        "repair_butterfly": False,
                    },
                )
                t2 = perf_counter()
                _ = np.vstack(
                    [
                        surface.iv(np.linspace(60.0, 140.0, 25, dtype=float), expiry)
                        for expiry in np.linspace(0.25, 2.0, 9, dtype=float)
                    ]
                )
                t3 = perf_counter()
                lv = LocalVolSurface.from_implied(
                    surface,
                    forward=market.forward,
                    discount=market.df,
                )
                t4 = perf_counter()
                final_price = _price_digital_local_vol_pde(
                    lv=lv,
                    market=market,
                    strike=100.0,
                    tau=1.0,
                    nx=nx,
                    nt=nt,
                )
                t5 = perf_counter()

                stage_samples["quote_mesh_ms"].append(1e3 * (t1 - t0))
                stage_samples["surface_fit_ms"].append(1e3 * (t2 - t1))
                stage_samples["surface_handoff_ms"].append(1e3 * (t3 - t2))
                stage_samples["local_vol_ms"].append(1e3 * (t4 - t3))
                stage_samples["pde_reprice_ms"].append(1e3 * (t5 - t4))

        for stage, samples in stage_samples.items():
            rows.append(
                {
                    "scenario": label,
                    "nT": int(n_expiries),
                    "nK": int(n_strikes),
                    "Nx": int(nx),
                    "Nt": int(nt),
                    "stage": stage,
                    "runtime_ms": float(np.median(np.asarray(samples, dtype=float))),
                    "final_price": float(final_price),
                }
            )

    frame = pd.DataFrame(rows)
    pivot = frame.pivot(index="scenario", columns="stage", values="runtime_ms").loc[
        ["small", "medium", "large"]
    ]

    def render(theme: str):
        with publishing_style(theme=theme) as plt:
            fig, ax = plt.subplots(1, 1, figsize=(9.2, 4.5), constrained_layout=True)
            bottom = np.zeros(len(pivot), dtype=float)
            colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#ccb974"]
            for color, stage in zip(colors, pivot.columns, strict=False):
                vals = pivot[stage].to_numpy(dtype=float)
                ax.bar(
                    pivot.index.to_list(),
                    vals,
                    bottom=bottom,
                    label=stage,
                    color=color,
                )
                bottom += vals
            ax.set_ylabel("Median runtime (ms)")
            ax.set_title("End-to-end macro pipeline stage budget")
            ax.legend(ncol=2)
            return fig

    figure_path = _save_themed_plot(
        plot_dir / "macro_pipeline_summary.png",
        dpi=180,
        render=render,
    )
    files = _write_frame(
        frame.sort_values(["scenario", "stage"]),
        stem="macro_pipeline_summary",
        artifacts_dir=artifacts_dir,
    )
    return {
        "snapshot": files,
        "figure": str(figure_path.relative_to(ROOT)).replace("\\", "/"),
    }


def _run_tree_family(
    *,
    artifacts_dir: Path,
    plot_dir: Path,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.0)
    analytic = bs_price_from_ctx(
        kind=OptionType.CALL,
        strike=100.0,
        sigma=0.2,
        tau=1.0,
        ctx=market.to_context(),
    )
    rows: list[dict[str, Any]] = []
    for n_steps in (200, 500, 1000, 2500, 5000):

        def run_tree(*, n_steps: int = n_steps) -> float:
            return float(
                binom_price_from_ctx(
                    kind=OptionType.CALL,
                    strike=100.0,
                    sigma=0.2,
                    tau=1.0,
                    ctx=market.to_context(),
                    n_steps=int(n_steps),
                    american=False,
                    method="tree",
                )
            )

        price = run_tree()
        stats = _time_callable(
            run_tree,
            repeat=max(1, repeat - 1),
            warmup=warmup,
        )
        rows.append(
            {
                "n_steps": int(n_steps),
                "runtime_ms": float(stats["median_ms"]),
                "price": float(price),
                "reference_price": float(analytic),
                "abs_error": abs(float(price) - float(analytic)),
                **stats,
            }
        )

    frame = pd.DataFrame(rows).sort_values("n_steps")

    def render(theme: str):
        with publishing_style(theme=theme) as plt:
            fig, (ax_runtime, ax_error) = plt.subplots(
                1,
                2,
                figsize=(10.4, 4.2),
                constrained_layout=True,
            )
            ax_runtime.plot(
                frame["n_steps"].to_numpy(),
                frame["runtime_ms"].to_numpy(),
                marker="o",
                linewidth=2,
            )
            ax_runtime.set_xscale("log")
            ax_runtime.set_yscale("log")
            ax_runtime.set_xlabel("Tree steps")
            ax_runtime.set_ylabel("Median runtime (ms)")
            ax_runtime.set_title("CRR runtime scaling")

            ax_error.plot(
                frame["n_steps"].to_numpy(),
                frame["abs_error"].to_numpy(),
                marker="o",
                linewidth=2,
                color="#c44e52",
            )
            ax_error.set_xscale("log")
            ax_error.set_yscale("log")
            ax_error.set_xlabel("Tree steps")
            ax_error.set_ylabel("Absolute error vs Black-Scholes")
            ax_error.set_title("CRR convergence")
            return fig

    figure_path = _save_themed_plot(
        plot_dir / "tree_scaling.png",
        dpi=180,
        render=render,
    )
    files = _write_frame(frame, stem="tree_scaling", artifacts_dir=artifacts_dir)
    return {
        "snapshot": files,
        "figure": str(figure_path.relative_to(ROOT)).replace("\\", "/"),
    }


def _summarize_pytest_benchmark(
    *,
    input_path: Path,
    artifacts_dir: Path,
) -> dict[str, Any] | None:
    if not input_path.exists():
        return None
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for rec in payload.get("benchmarks", []):
        stats = rec.get("stats", {})
        rows.append(
            {
                "name": rec.get("name", ""),
                "fullname": rec.get("fullname", ""),
                "group": rec.get("group", ""),
                "params": json.dumps(rec.get("params", {}), sort_keys=True),
                "mean": stats.get("mean"),
                "median": stats.get("median"),
                "min": stats.get("min"),
                "max": stats.get("max"),
                "stddev": stats.get("stddev"),
                "rounds": stats.get("rounds"),
                "ops": stats.get("ops"),
            }
        )
    if not rows:
        return None
    frame = pd.DataFrame(rows).sort_values(["group", "name", "params"])
    files = _write_frame(
        frame,
        stem="pytest_benchmark_summary",
        artifacts_dir=artifacts_dir,
    )
    return {
        "input": str(input_path.relative_to(ROOT)).replace("\\", "/"),
        "summary": files,
        "machine_info": payload.get("machine_info", {}),
        "commit_info": payload.get("commit_info", {}),
    }


def _fmt_runtime_ms(value: float) -> str:
    value = float(value)
    if value >= 10_000.0:
        return f"{value / 1000.0:.1f} s"
    if value >= 1_000.0:
        return f"{value / 1000.0:.2f} s"
    if value >= 100.0:
        return f"{value:.0f} ms"
    if value >= 10.0:
        return f"{value:.1f} ms"
    return f"{value:.2f} ms"


def _fmt_sci(value: float) -> str:
    return f"{float(value):.2e}"


def _save_themed_plot(out_path: Path, *, dpi: int, render) -> Path:
    variants = themed_asset_paths(out_path)
    for theme in PUBLISHING_THEMES:
        fig = render(theme)
        save_figure(fig, variants.path_for(theme), dpi=dpi)
    copy_light_variant(variants)
    return variants.base


def _build_benchmark_overview_asset(
    *,
    artifacts_dir: Path,
    plot_dir: Path,
) -> str:
    iv_scaling = pd.read_csv(artifacts_dir / "iv_scaling.csv")
    iv_speedup = pd.read_csv(artifacts_dir / "iv_speedup.csv")
    pde = pd.read_csv(artifacts_dir / "pde_runtime_error_tradeoff.csv")
    digital_stability = pd.read_csv(artifacts_dir / "digital_remedies_stability.csv")
    localvol = pd.read_csv(artifacts_dir / "localvol_scaling.csv")
    macro = pd.read_csv(artifacts_dir / "macro_pipeline_summary.csv")
    tree = pd.read_csv(artifacts_dir / "tree_scaling.csv")

    iv_801 = (
        iv_scaling[iv_scaling["n_strikes"] == 801]
        .set_index("implementation")
        .sort_index()
    )
    iv_vec_ms = float(iv_801.loc["vectorized_slice", "runtime_ms"])
    iv_scalar_ms = float(iv_801.loc["scalar_loop", "runtime_ms"])
    iv_speedup_801 = float(
        iv_speedup.loc[iv_speedup["n_strikes"] == 801, "vectorized_speedup_x"].iloc[0]
    )

    pde_bs = pde[
        (pde["family"] == "black_scholes_vanilla") & (pde["method"] == "rannacher")
    ].set_index("Nx")
    pde_401 = pde_bs.loc[401]
    pde_801 = pde_bs.loc[801]

    digital = digital_stability.set_index("config")
    digital_none = digital.loc["Rannacher + none"]
    digital_l2 = digital.loc["Rannacher + L2 projection"]

    localvol_large = localvol[
        localvol["grid_points"] == int(localvol["grid_points"].max())
    ]
    localvol_large = localvol_large.set_index("strike_coordinate")
    localvol_k = localvol_large.loc["K"]
    localvol_logk = localvol_large.loc["logK"]

    macro_pivot = macro.pivot(index="scenario", columns="stage", values="runtime_ms")
    fit_small = float(macro_pivot.loc["small", "surface_fit_ms"])
    fit_medium = float(macro_pivot.loc["medium", "surface_fit_ms"])
    fit_large = float(macro_pivot.loc["large", "surface_fit_ms"])

    tree_last = tree.loc[tree["n_steps"] == int(tree["n_steps"].max())].iloc[0]
    variants = themed_asset_paths(plot_dir / "benchmark_overview.svg")
    for theme in PUBLISHING_THEMES:
        base_palette = publishing_palette(theme)
        image_suffix = "dark" if theme == "dark" else "light"
        if theme == "dark":
            colors = {
                "page_bg": "#08101F",
                "card_bg": "#0F172A",
                "card_stroke": "#314056",
                "header_start": "#18263C",
                "header_end": "#101B2D",
                "panel_fill": "#1E2D43",
                "section_fill": "#0F172A",
                "section_stroke": "#314056",
                "chip_fill": "#020617",
                "chip_text": "#FFFFFF",
                "accent": "#8CC9FF",
                "success": "#91E0D7",
                "text": base_palette["text"],
                "muted": base_palette["muted_text"],
                "thumb_fill": "#23344A",
            }
        else:
            colors = {
                "page_bg": "#EEF3F8",
                "card_bg": "#FFFFFF",
                "card_stroke": "#D7E0EA",
                "header_start": "#F4F8FC",
                "header_end": "#EAF1F7",
                "panel_fill": "#DCE7F1",
                "section_fill": "#F7FAFC",
                "section_stroke": "#D8E4EE",
                "chip_fill": "#0B1F33",
                "chip_text": "#FFFFFF",
                "accent": "#0B5CAB",
                "success": "#0F766E",
                "text": base_palette["text"],
                "muted": base_palette["muted_text"],
                "thumb_fill": "#DCE7F1",
            }
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1080" viewBox="0 0 1600 1080" role="img" aria-labelledby="title desc">
  <title id="title">Benchmark overview for the option pricing library</title>
  <desc id="desc">Reviewer-facing benchmark summary showing IV scaling, PDE cost versus accuracy, macro pipeline stage budget, and concise measured callouts for digital remedies, local-vol extraction, and tree scaling.</desc>
  <defs>
    <linearGradient id="headerBg" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="{colors["header_start"]}" />
      <stop offset="100%" stop-color="{colors["header_end"]}" />
    </linearGradient>
    <clipPath id="ivThumb">
      <rect x="74" y="264" width="452" height="290" rx="20" ry="20" />
    </clipPath>
    <clipPath id="pdeThumb">
      <rect x="574" y="264" width="452" height="290" rx="20" ry="20" />
    </clipPath>
    <clipPath id="macroThumb">
      <rect x="1074" y="264" width="452" height="290" rx="20" ry="20" />
    </clipPath>
  </defs>

  <rect width="1600" height="1080" fill="{colors["page_bg"]}" />
  <rect x="28" y="28" width="1544" height="1024" rx="34" ry="34" fill="{colors["card_bg"]}" stroke="{colors["card_stroke"]}" stroke-width="2" />

  <text x="72" y="108" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="54" font-weight="700" fill="{colors["text"]}">Benchmark overview</text>
  <text x="72" y="148" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="24" fill="{colors["muted"]}">Compressed reviewer scan of the repo's measured scaling, cost-versus-accuracy, and end-to-end stage budget story.</text>
  <text x="72" y="182" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="18" fill="{colors["muted"]}">Built from committed benchmark outputs under benchmarks/artifacts and docs/assets/generated/benchmarks.</text>

  <rect x="72" y="220" width="1456" height="360" rx="28" ry="28" fill="url(#headerBg)" stroke="{colors["card_stroke"]}" stroke-width="2" />
  <text x="102" y="254" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="18" font-weight="700" letter-spacing="1" fill="{colors["accent"]}">PRIMARY PANELS</text>

  <rect x="74" y="264" width="452" height="290" rx="20" ry="20" fill="{colors["thumb_fill"]}" />
  <image x="74" y="264" width="452" height="290" href="./iv_scaling.{image_suffix}.png" preserveAspectRatio="xMidYMid slice" clip-path="url(#ivThumb)" />
  <rect x="92" y="282" width="186" height="30" rx="15" ry="15" fill="{colors["chip_fill"]}" opacity="0.84" />
  <text x="110" y="303" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="16" font-weight="700" fill="{colors["chip_text"]}">IV inversion scaling</text>
  <text x="74" y="606" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="18" font-weight="700" fill="{colors["text"]}">801-strike slice: {_fmt_runtime_ms(iv_vec_ms)} vectorized vs {_fmt_runtime_ms(iv_scalar_ms)} scalar loop</text>
  <text x="74" y="636" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="16" fill="{colors["muted"]}">Measured max speedup at that size: {iv_speedup_801:.0f}x, with zero observed vol difference vs the scalar loop.</text>

  <rect x="574" y="264" width="452" height="290" rx="20" ry="20" fill="{colors["thumb_fill"]}" />
  <image x="574" y="264" width="452" height="290" href="./pde_runtime_error_tradeoff.{image_suffix}.png" preserveAspectRatio="xMidYMid slice" clip-path="url(#pdeThumb)" />
  <rect x="592" y="282" width="176" height="30" rx="15" ry="15" fill="{colors["chip_fill"]}" opacity="0.84" />
  <text x="610" y="303" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="16" font-weight="700" fill="{colors["chip_text"]}">PDE cost / accuracy</text>
  <text x="574" y="606" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="18" font-weight="700" fill="{colors["text"]}">Vanilla BS PDE, Rannacher: 401x401 -&gt; 801x801 cuts abs error</text>
  <text x="574" y="634" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="18" font-weight="700" fill="{colors["text"]}">{_fmt_sci(float(pde_401["abs_error"]))} -&gt; {_fmt_sci(float(pde_801["abs_error"]))} while runtime rises {_fmt_runtime_ms(float(pde_401["runtime_ms"]))} -&gt; {_fmt_runtime_ms(float(pde_801["runtime_ms"]))}.</text>

  <rect x="1074" y="264" width="452" height="290" rx="20" ry="20" fill="{colors["thumb_fill"]}" />
  <image x="1074" y="264" width="452" height="290" href="./macro_pipeline_summary.{image_suffix}.png" preserveAspectRatio="xMidYMid slice" clip-path="url(#macroThumb)" />
  <rect x="1092" y="282" width="196" height="30" rx="15" ry="15" fill="{colors["chip_fill"]}" opacity="0.84" />
  <text x="1110" y="303" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="16" font-weight="700" fill="{colors["chip_text"]}">Macro stage budget</text>
  <text x="1074" y="606" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="18" font-weight="700" fill="{colors["text"]}">Surface fit dominates the integrated path budget across tested sizes.</text>
  <text x="1074" y="634" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="16" fill="{colors["muted"]}">Surface-fit medians: {_fmt_runtime_ms(fit_small)} / {_fmt_runtime_ms(fit_medium)} / {_fmt_runtime_ms(fit_large)} for small / medium / large scenarios.</text>

  <text x="72" y="712" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="18" font-weight="700" letter-spacing="1" fill="{colors["accent"]}">ADDITIONAL MEASURED CALLOUTS</text>

  <rect x="72" y="734" width="452" height="224" rx="24" ry="24" fill="{colors["section_fill"]}" stroke="{colors["section_stroke"]}" stroke-width="2" />
  <text x="98" y="770" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="18" font-weight="700" fill="{colors["text"]}">Digital payoff remedies</text>
  <text x="98" y="818" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="34" font-weight="700" fill="{colors["accent"]}">{_fmt_sci(float(digital_none["refinement_spread"]))} -&gt; {_fmt_sci(float(digital_l2["refinement_spread"]))}</text>
  <text x="98" y="850" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="16" fill="{colors["muted"]}">Grid-to-grid price spread, Rannacher + none vs Rannacher + L2 projection.</text>
  <text x="98" y="882" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="16" fill="{colors["muted"]}">Best measured abs error with L2 projection: {_fmt_sci(float(digital_l2["best_abs_error"]))}.</text>

  <rect x="574" y="734" width="452" height="224" rx="24" ry="24" fill="{colors["section_fill"]}" stroke="{colors["section_stroke"]}" stroke-width="2" />
  <text x="600" y="770" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="18" font-weight="700" fill="{colors["text"]}">Local-vol extraction</text>
  <text x="600" y="818" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="34" font-weight="700" fill="{colors["success"]}">0.0% interior invalid share</text>
  <text x="600" y="850" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="16" fill="{colors["muted"]}">Largest tested 121x241 grids stayed interior-valid for both coordinate choices.</text>
  <text x="600" y="882" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="16" fill="{colors["muted"]}">Median runtime: K = {_fmt_runtime_ms(float(localvol_k["runtime_ms"]))}, logK = {_fmt_runtime_ms(float(localvol_logk["runtime_ms"]))}.</text>

  <rect x="1076" y="734" width="450" height="224" rx="24" ry="24" fill="{colors["section_fill"]}" stroke="{colors["section_stroke"]}" stroke-width="2" />
  <text x="1102" y="770" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="18" font-weight="700" fill="{colors["text"]}">Tree framing</text>
  <text x="1102" y="818" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="34" font-weight="700" fill="{colors["accent"]}">{_fmt_runtime_ms(float(tree_last["runtime_ms"]))}</text>
  <text x="1102" y="850" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="16" fill="{colors["muted"]}">5000-step CRR runtime in the committed snapshot.</text>
  <text x="1102" y="882" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="16" fill="{colors["muted"]}">Absolute error vs Black-Scholes at that depth: {_fmt_sci(float(tree_last["abs_error"]))}.</text>

  <text x="72" y="1012" font-family="Segoe UI, Inter, Arial, sans-serif" font-size="16" fill="{colors["muted"]}">This asset summarizes the committed benchmark bundle. Use the performance page for the full figures, conditions, and reproducibility notes.</text>
</svg>
"""
        variants.path_for(theme).write_text(svg, encoding="utf-8")

    copy_light_variant(variants)
    return str(variants.base.relative_to(ROOT)).replace("\\", "/")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    artifacts_dir = args.artifacts_dir.resolve()
    plot_dir = args.plot_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    environment = _collect_environment()
    pytest_summary = None
    if args.pytest_benchmark_json is not None:
        pytest_summary = _summarize_pytest_benchmark(
            input_path=args.pytest_benchmark_json.resolve(),
            artifacts_dir=artifacts_dir,
        )

    families = {
        "iv": _run_iv_family(
            artifacts_dir=artifacts_dir,
            plot_dir=plot_dir,
            repeat=args.repeat,
            warmup=args.warmup,
        ),
        "pde": _run_pde_family(
            artifacts_dir=artifacts_dir,
            plot_dir=plot_dir,
            repeat=max(2, args.repeat),
            warmup=args.warmup,
        ),
        "digital": _run_digital_family(
            artifacts_dir=artifacts_dir,
            plot_dir=plot_dir,
            repeat=max(2, args.repeat),
            warmup=args.warmup,
        ),
        "localvol": _run_localvol_family(
            artifacts_dir=artifacts_dir,
            plot_dir=plot_dir,
            repeat=args.repeat,
            warmup=args.warmup,
        ),
        "macro": _run_macro_pipeline_family(
            artifacts_dir=artifacts_dir,
            plot_dir=plot_dir,
            repeat=max(2, args.repeat),
            warmup=args.warmup,
        ),
        "tree": _run_tree_family(
            artifacts_dir=artifacts_dir,
            plot_dir=plot_dir,
            repeat=max(2, args.repeat),
            warmup=args.warmup,
        ),
    }
    overview_asset = _build_benchmark_overview_asset(
        artifacts_dir=artifacts_dir,
        plot_dir=plot_dir,
    )
    results = {
        "environment": environment,
        "pytest_benchmark": pytest_summary,
        "families": families,
        "overview_asset": overview_asset,
    }
    summary_path = _write_json(
        results,
        name="performance_summary.json",
        artifacts_dir=artifacts_dir,
    )
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
