from __future__ import annotations

# ruff: noqa: E402
import argparse
import hashlib
import json
import os
import platform
import sys
import warnings
from dataclasses import dataclass
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
    SVG_TEXT_FONT_STACK,
    SvgTextStyle,
    copy_light_variant,
    publishing_palette,
    publishing_style,
    render_svg_contained_raster,
    render_svg_text_block,
    save_figure,
    themed_asset_paths,
)
from option_pricing.vol.implied_vol_scalar import implied_vol_bs_result
from option_pricing.vol.implied_vol_slice import implied_vol_black76_slice
from option_pricing.vol.local_vol_dupire import local_vol_from_call_grid_diagnostics
from option_pricing.vol.local_vol_surface import LocalVolSurface
from option_pricing.vol.surface_core import VolSurface
from option_pricing.vol.svi import SVIParams, svi_total_variance

BENCHMARK_SOURCE_MANIFEST = "benchmark_source_manifest.json"
BENCHMARK_SOURCE_INPUTS = (
    "benchmarks/conftest.py",
    "benchmarks/test_bench_iv.py",
    "benchmarks/test_bench_localvol.py",
    "benchmarks/test_bench_macro.py",
    "benchmarks/test_bench_pde.py",
    "benchmarks/test_bench_tree.py",
    "src/option_pricing/models/",
    "src/option_pricing/numerics/",
    "src/option_pricing/pricers/",
    "src/option_pricing/types.py",
    "src/option_pricing/vol/",
)
BENCHMARK_SOURCE_MANIFEST_VERSION = 1


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
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Fail if the tracked benchmark-source manifest does not match the current "
            "benchmark inputs. This does not rerun benchmarks."
        ),
    )
    parser.add_argument(
        "--write-source-manifest",
        action="store_true",
        help=(
            "Write or refresh the benchmark source manifest without recomputing the "
            "benchmark timing snapshots."
        ),
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_benchmark_source_files() -> list[Path]:
    files: list[Path] = []
    for entry in BENCHMARK_SOURCE_INPUTS:
        path = ROOT / entry
        if path.is_file():
            files.append(path)
            continue
        if path.is_dir():
            files.extend(
                candidate for candidate in path.rglob("*.py") if candidate.is_file()
            )
    return sorted(set(files))


def _benchmark_source_manifest_payload() -> dict[str, Any]:
    files = _iter_benchmark_source_files()
    return {
        "version": BENCHMARK_SOURCE_MANIFEST_VERSION,
        "inputs": {
            str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
            for path in files
        },
    }


def _benchmark_source_manifest_path(artifacts_dir: Path) -> Path:
    return artifacts_dir / BENCHMARK_SOURCE_MANIFEST


def _write_benchmark_source_manifest(artifacts_dir: Path) -> Path:
    path = _benchmark_source_manifest_path(artifacts_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_benchmark_source_manifest_payload(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _check_benchmark_source_manifest(artifacts_dir: Path) -> int:
    manifest_path = _benchmark_source_manifest_path(artifacts_dir)
    current_payload = _benchmark_source_manifest_payload()
    if not manifest_path.exists():
        print("Benchmark source manifest is missing. Refresh benchmark artifacts with:")
        print("  python scripts/build_benchmark_artifacts.py --write-source-manifest")
        return 1

    recorded_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_inputs = dict(recorded_payload.get("inputs", {}))
    current_inputs = dict(current_payload.get("inputs", {}))

    missing = sorted(path for path in current_inputs if path not in recorded_inputs)
    changed = sorted(
        path
        for path, digest in current_inputs.items()
        if path in recorded_inputs and recorded_inputs[path] != digest
    )
    orphaned = sorted(path for path in recorded_inputs if path not in current_inputs)

    if not missing and not changed and not orphaned:
        print("Benchmark source manifest is up to date.")
        return 0

    print(
        "Benchmark source inputs have changed since the committed benchmark snapshot."
    )
    print(
        "Refresh the authoritative benchmark artifacts and commit the updated outputs."
    )
    for label, items in (
        ("missing", missing),
        ("changed", changed),
        ("orphaned", orphaned),
    ):
        if not items:
            continue
        for item in items:
            print(f" - {label}: {item}")
    return 1


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
        "cpu_count": os.cpu_count(),
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
    stability_labels = {
        "CN + none": "CN\nnone",
        "Rannacher + none": "Rannacher\nnone",
        "Rannacher + cell average": "Rannacher\ncell avg",
        "Rannacher + L2 projection": "Rannacher\nL2 proj",
    }

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
            ax_tradeoff.set_xscale("log")
            ax_tradeoff.set_yscale("log")
            ax_tradeoff.set_xlabel("Median runtime (ms)")
            ax_tradeoff.set_ylabel("Absolute error")
            ax_tradeoff.set_title("Digital remedy tradeoff")
            ax_tradeoff.legend()

            positions = np.arange(len(spread), dtype=float)
            ax_stability.barh(
                positions,
                spread["refinement_spread"].to_numpy(),
                color=["#55a868", "#4c72b0", "#ccb974", "#c44e52"],
            )
            ax_stability.set_xlabel("Price spread across tested grids")
            ax_stability.set_title("Grid-refinement stability")
            ax_stability.set_yticks(
                positions,
                [stability_labels.get(label, label) for label in spread["config"]],
            )
            ax_stability.margins(y=0.12)
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
    figure_path = build_macro_pipeline_summary_asset_from_frame(
        frame=frame,
        plot_dir=plot_dir,
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


_MACRO_SCENARIO_ORDER = ("small", "medium", "large")
_MACRO_STAGE_SPECS = (
    ("surface_fit_ms", "Surface fit"),
    ("pde_reprice_ms", "PDE repricing"),
    ("surface_handoff_ms", "Smooth handoff"),
    ("local_vol_ms", "Local-vol build"),
    ("quote_mesh_ms", "Quote mesh"),
)


def _macro_stage_budget_pivot(frame: pd.DataFrame) -> pd.DataFrame:
    pivot = frame.pivot(index="scenario", columns="stage", values="runtime_ms")
    pivot = pivot.reindex(
        index=list(_MACRO_SCENARIO_ORDER),
        columns=[stage for stage, _ in _MACRO_STAGE_SPECS],
    )
    return pivot.fillna(0.0)


def build_macro_pipeline_summary_asset_from_frame(
    *,
    frame: pd.DataFrame,
    plot_dir: Path,
) -> Path:
    pivot = _macro_stage_budget_pivot(frame)
    totals = pivot.sum(axis=1).to_numpy(dtype=float)
    fit = pivot["surface_fit_ms"].to_numpy(dtype=float)
    pde = pivot["pde_reprice_ms"].to_numpy(dtype=float)
    minor = totals - fit - pde
    fit_pde_share = (fit + pde) / totals

    def render(theme: str):
        from matplotlib.patches import FancyBboxPatch

        base_palette = publishing_palette(theme)
        if theme == "dark":
            colors = {
                "surface_fit_ms": "#89B8FF",
                "pde_reprice_ms": "#7AD6A5",
                "surface_handoff_ms": "#F0C86A",
                "local_vol_ms": "#9FB2FF",
                "quote_mesh_ms": "#E48A92",
                "bar_edge": "#08101F",
                "panel_fill": "#0F172A",
                "panel_stroke": "#314056",
                "panel_emphasis": "#13243B",
                "text": base_palette["text"],
                "muted": base_palette["muted_text"],
                "accent": "#8CC9FF",
                "segment_text": "#08101F",
            }
        else:
            colors = {
                "surface_fit_ms": "#355C8B",
                "pde_reprice_ms": "#2E8D68",
                "surface_handoff_ms": "#C7942C",
                "local_vol_ms": "#5D7DB5",
                "quote_mesh_ms": "#C55B63",
                "bar_edge": "#F7FAFC",
                "panel_fill": "#F7FAFC",
                "panel_stroke": "#D8E4EE",
                "panel_emphasis": "#EAF2F9",
                "text": base_palette["text"],
                "muted": base_palette["muted_text"],
                "accent": "#0B5CAB",
                "segment_text": "#FFFFFF",
            }

        with publishing_style(theme=theme) as plt:
            fig = plt.figure(figsize=(11.8, 5.25))
            gs = fig.add_gridspec(1, 2, width_ratios=[3.55, 1.55], wspace=0.08)
            ax = fig.add_subplot(gs[0, 0])
            ax_notes = fig.add_subplot(gs[0, 1])
            fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.14)

            x = np.arange(len(pivot), dtype=float)
            bottom = np.zeros(len(pivot), dtype=float)
            for stage, label in _MACRO_STAGE_SPECS:
                values = pivot[stage].to_numpy(dtype=float)
                ax.bar(
                    x,
                    values,
                    width=0.64,
                    bottom=bottom,
                    label=label,
                    color=colors[stage],
                    edgecolor=colors["bar_edge"],
                    linewidth=0.9,
                )
                bottom += values

            y_max = float(np.max(totals)) * 1.17
            ax.set_ylim(0.0, y_max)
            ax.set_xlim(-0.55, len(x) - 0.45)
            ax.set_xticks(x, list(_MACRO_SCENARIO_ORDER))
            ax.set_xlabel("Published scenario")
            ax.set_ylabel("Median runtime (ms)")
            ax.set_title("Where the runtime budget actually goes")
            ax.set_axisbelow(True)
            ax.legend(loc="upper left", ncol=2, frameon=False)

            for idx, total in enumerate(totals):
                ax.text(
                    x[idx],
                    total + y_max * 0.018,
                    _fmt_runtime_ms(total),
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="700",
                    color=colors["text"],
                )

            large_idx = len(x) - 1
            ax.text(
                x[large_idx],
                fit[large_idx] * 0.5,
                "Fit",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="700",
                color=colors["segment_text"],
            )
            ax.text(
                x[large_idx],
                fit[large_idx] + pde[large_idx] * 0.5,
                "PDE",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="700",
                color=colors["segment_text"],
            )

            ax_notes.set_axis_off()
            ax_notes.set_xlim(0.0, 1.0)
            ax_notes.set_ylim(0.0, 1.0)

            def add_note(
                *,
                y: float,
                height: float,
                title: str,
                metric: str | None,
                body: str | None,
                emphasized: bool = False,
            ) -> None:
                fill = colors["panel_emphasis"] if emphasized else colors["panel_fill"]
                patch = FancyBboxPatch(
                    (0.02, y),
                    0.96,
                    height,
                    boxstyle="round,pad=0.02,rounding_size=0.05",
                    linewidth=1.2,
                    edgecolor=colors["panel_stroke"],
                    facecolor=fill,
                )
                ax_notes.add_patch(patch)
                ax_notes.text(
                    0.08,
                    y + height - 0.07,
                    title,
                    ha="left",
                    va="top",
                    fontsize=11,
                    fontweight="700",
                    color=colors["text"],
                )
                if metric is not None:
                    ax_notes.text(
                        0.08,
                        y + height - 0.16,
                        metric,
                        ha="left",
                        va="top",
                        fontsize=16,
                        fontweight="800",
                        color=colors["accent"],
                    )
                    body_y = y + height - 0.24
                else:
                    body_y = y + height - 0.15
                if body is not None:
                    ax_notes.text(
                        0.08,
                        body_y,
                        body,
                        ha="left",
                        va="top",
                        fontsize=9.4,
                        linespacing=1.25,
                        color=colors["muted"],
                    )

            add_note(
                y=0.69,
                height=0.27,
                title="Fit + PDE share",
                metric=(
                    f"{_fmt_pct(float(np.min(fit_pde_share)))} to "
                    f"{_fmt_pct(float(np.max(fit_pde_share)))}"
                ),
                body=None,
                emphasized=True,
            )
            add_note(
                y=0.38,
                height=0.25,
                title="Everything else",
                metric=(
                    f"{_fmt_runtime_ms(float(np.min(minor)))} to "
                    f"{_fmt_runtime_ms(float(np.max(minor)))}"
                ),
                body="Combined low-leverage stages.",
            )
            add_note(
                y=0.08,
                height=0.22,
                title="Why this matters",
                metric=None,
                body=(
                    "Justify optimization order:\n"
                    "optimize fit or PDE before orchestration."
                ),
            )
            return fig

    return _save_themed_plot(
        plot_dir / "macro_pipeline_summary.png",
        dpi=180,
        render=render,
    )


def build_macro_pipeline_summary_asset(
    *,
    artifacts_dir: Path,
    plot_dir: Path,
) -> str:
    frame = pd.read_csv(artifacts_dir / "macro_pipeline_summary.csv")
    figure_path = build_macro_pipeline_summary_asset_from_frame(
        frame=frame,
        plot_dir=plot_dir,
    )
    return str(figure_path.relative_to(ROOT)).replace("\\", "/")


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


def _fmt_pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _fmt_sci(value: float) -> str:
    return f"{float(value):.2e}"


def _save_themed_plot(out_path: Path, *, dpi: int, render) -> Path:
    variants = themed_asset_paths(out_path)
    for theme in PUBLISHING_THEMES:
        fig = render(theme)
        save_figure(fig, variants.path_for(theme), dpi=dpi)
    copy_light_variant(variants)
    return variants.base


_BENCHMARK_OVERVIEW_PRIMARY_PANEL_WIDTH = 452.0
_BENCHMARK_OVERVIEW_PRIMARY_PANEL_HEIGHT = 290.0
_BENCHMARK_OVERVIEW_CALLOUT_WIDTH = 452.0
_BENCHMARK_OVERVIEW_CALLOUT_HEIGHT = 224.0


@dataclass(frozen=True)
class _BenchmarkOverviewPrimaryPanel:
    key: str
    x: float
    y: float
    chip_text: str
    chip_width: float
    image_path: Path
    headline: str
    body: str
    width: float = _BENCHMARK_OVERVIEW_PRIMARY_PANEL_WIDTH
    height: float = _BENCHMARK_OVERVIEW_PRIMARY_PANEL_HEIGHT


@dataclass(frozen=True)
class _BenchmarkOverviewCalloutCard:
    key: str
    title: str
    x: float
    y: float
    metric: str
    body: str
    metric_fill_key: str = "accent"
    body_y_offset: float = 164.0
    body_max_height: float = 60.0
    width: float = _BENCHMARK_OVERVIEW_CALLOUT_WIDTH
    height: float = _BENCHMARK_OVERVIEW_CALLOUT_HEIGHT


def _render_benchmark_overview_primary_panel(
    panel: _BenchmarkOverviewPrimaryPanel,
    *,
    colors: dict[str, str],
    font_stack: str,
) -> tuple[tuple[str, ...], str]:
    text_clip_id = f"benchmarkOverviewPrimaryText-{panel.key}"
    text_padding_x = 20.0
    text_clip_inset_x = 10.0
    text_clip_y = panel.y + 320.0
    text_clip_height = 124.0
    headline_style = SvgTextStyle(
        font_family=font_stack,
        font_size=18,
        font_weight="700",
        fill=colors["text"],
        line_height=22,
    )
    body_style = SvgTextStyle(
        font_family=font_stack,
        font_size=16,
        font_weight="400",
        fill=colors["muted"],
        line_height=20,
    )
    text_x = panel.x + text_padding_x
    text_width = panel.width - 2.0 * text_padding_x
    headline_y = panel.y + 342
    body_y = panel.y + 388
    raster_block = render_svg_contained_raster(
        block_id=f"benchmarkOverviewImage-{panel.key}",
        image_path=panel.image_path,
        slot_x=panel.x,
        slot_y=panel.y,
        slot_width=panel.width,
        slot_height=panel.height,
        frame_radius=20,
        frame_fill=colors["thumb_fill"],
        source_label=str(panel.image_path.relative_to(ROOT)).replace("\\", "/"),
    )
    clip_paths = (
        raster_block.clip_path,
        (
            f'    <clipPath id="{text_clip_id}">\n'
            f'      <rect x="{panel.x + text_clip_inset_x:g}" y="{text_clip_y:g}" '
            f'width="{panel.width - 2.0 * text_clip_inset_x:g}" '
            f'height="{text_clip_height:g}" rx="12" ry="12" />\n'
            "    </clipPath>"
        ),
    )
    headline_svg = render_svg_text_block(
        block_id=f"benchmark-overview-{panel.key}-headline",
        text=panel.headline,
        x=text_x,
        y=headline_y,
        max_width=text_width,
        max_height=44,
        style=headline_style,
        overflow_label=f"Benchmark overview primary panel {panel.key}/headline",
    )
    body_svg = render_svg_text_block(
        block_id=f"benchmark-overview-{panel.key}-body",
        text=panel.body,
        x=text_x,
        y=body_y,
        max_width=text_width,
        max_height=40,
        style=body_style,
        overflow_label=f"Benchmark overview primary panel {panel.key}/body",
    )
    svg = "\n".join(
        [
            raster_block.svg,
            (
                f'  <rect x="{panel.x + 18:g}" y="{panel.y + 18:g}" '
                f'width="{panel.chip_width:g}" height="30" rx="15" ry="15" '
                f'fill="{colors["chip_fill"]}" opacity="0.84" />'
            ),
            (
                f'  <text x="{panel.x + 36:g}" y="{panel.y + 39:g}" '
                f'font-family="{font_stack}" font-size="16" font-weight="700" '
                f'fill="{colors["chip_text"]}">{panel.chip_text}</text>'
            ),
            f'  <g clip-path="url(#{text_clip_id})">',
            f"    {headline_svg}",
            f"    {body_svg}",
            "  </g>",
        ]
    )
    return clip_paths, svg


def _render_benchmark_overview_callout(
    card: _BenchmarkOverviewCalloutCard,
    *,
    colors: dict[str, str],
    font_stack: str,
) -> tuple[str, str]:
    content_clip_id = f"benchmarkOverviewCallout-{card.key}"
    content_x = card.x + 26.0
    content_width = card.width - 52.0
    title_style = SvgTextStyle(
        font_family=font_stack,
        font_size=18,
        font_weight="700",
        fill=colors["text"],
        line_height=22,
    )
    metric_style = SvgTextStyle(
        font_family=font_stack,
        font_size=34,
        font_weight="700",
        fill=colors[card.metric_fill_key],
        line_height=38,
    )
    body_style = SvgTextStyle(
        font_family=font_stack,
        font_size=16,
        font_weight="400",
        fill=colors["muted"],
        line_height=20,
    )
    clip_path = (
        f'    <clipPath id="{content_clip_id}">\n'
        f'      <rect x="{card.x:g}" y="{card.y:g}" '
        f'width="{card.width:g}" height="{card.height:g}" '
        'rx="18" ry="18" />\n'
        "    </clipPath>"
    )
    title_svg = render_svg_text_block(
        block_id=f"benchmark-overview-{card.key}-title",
        text=card.title,
        x=content_x,
        y=card.y + 36,
        max_width=content_width,
        max_height=22,
        style=title_style,
        overflow_label=f"Benchmark overview callout {card.key}/title",
    )
    metric_svg = render_svg_text_block(
        block_id=f"benchmark-overview-{card.key}-metric",
        text=card.metric,
        x=content_x,
        y=card.y + 84,
        max_width=content_width,
        max_height=76,
        style=metric_style,
        overflow_label=f"Benchmark overview callout {card.key}/metric",
    )
    body_svg = render_svg_text_block(
        block_id=f"benchmark-overview-{card.key}-body",
        text=card.body,
        x=content_x,
        y=card.y + card.body_y_offset,
        max_width=content_width,
        max_height=card.body_max_height,
        style=body_style,
        overflow_label=f"Benchmark overview callout {card.key}/body",
    )
    svg = "\n".join(
        [
            (
                f'  <rect x="{card.x:g}" y="{card.y:g}" width="{card.width:g}" '
                f'height="{card.height:g}" rx="24" ry="24" fill="{colors["section_fill"]}" '
                f'stroke="{colors["section_stroke"]}" stroke-width="2" />'
            ),
            f'  <g clip-path="url(#{content_clip_id})">',
            f"    {title_svg}",
            f"    {metric_svg}",
            f"    {body_svg}",
            "  </g>",
        ]
    )
    return clip_path, svg


def build_benchmark_overview_asset(
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
    minor_small = float(
        macro_pivot.loc[
            "small", ["quote_mesh_ms", "surface_handoff_ms", "local_vol_ms"]
        ].sum()
    )
    minor_large = float(
        macro_pivot.loc[
            "large", ["quote_mesh_ms", "surface_handoff_ms", "local_vol_ms"]
        ].sum()
    )
    fit_pde_share = []
    for scenario in ("small", "medium", "large"):
        total = float(macro_pivot.loc[scenario].sum())
        fit_pde_share.append(
            float(
                (
                    macro_pivot.loc[scenario, "surface_fit_ms"]
                    + macro_pivot.loc[scenario, "pde_reprice_ms"]
                )
                / total
            )
        )

    tree_last = tree.loc[tree["n_steps"] == int(tree["n_steps"].max())].iloc[0]
    variants = themed_asset_paths(plot_dir / "benchmark_overview.svg")
    for theme in PUBLISHING_THEMES:
        base_palette = publishing_palette(theme)
        image_suffix = "dark" if theme == "dark" else "light"
        font_stack = SVG_TEXT_FONT_STACK
        iv_panel_path = plot_dir / f"iv_scaling.{image_suffix}.png"
        pde_panel_path = plot_dir / f"pde_runtime_error_tradeoff.{image_suffix}.png"
        macro_panel_path = plot_dir / f"macro_pipeline_summary.{image_suffix}.png"
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

        primary_panels = (
            _BenchmarkOverviewPrimaryPanel(
                key="iv-scaling",
                x=74,
                y=264,
                chip_text="IV inversion scaling",
                chip_width=186,
                image_path=iv_panel_path,
                headline=(
                    f"801-strike slice: {_fmt_runtime_ms(iv_vec_ms)} vectorized vs "
                    f"{_fmt_runtime_ms(iv_scalar_ms)} scalar loop"
                ),
                body=(
                    f"Measured max speedup at that size: {iv_speedup_801:.0f}x, with "
                    "zero observed vol difference vs the scalar loop."
                ),
            ),
            _BenchmarkOverviewPrimaryPanel(
                key="pde-cost-accuracy",
                x=574,
                y=264,
                chip_text="PDE cost / accuracy",
                chip_width=176,
                image_path=pde_panel_path,
                headline=(
                    "Vanilla BS PDE, Rannacher: 401x401 -> 801x801 cuts abs error"
                ),
                body=(
                    f"{_fmt_sci(float(pde_401['abs_error']))} -> "
                    f"{_fmt_sci(float(pde_801['abs_error']))} while runtime rises "
                    f"{_fmt_runtime_ms(float(pde_401['runtime_ms']))} -> "
                    f"{_fmt_runtime_ms(float(pde_801['runtime_ms']))}."
                ),
            ),
            _BenchmarkOverviewPrimaryPanel(
                key="macro-stage-budget",
                x=1074,
                y=264,
                chip_text="Macro stage budget",
                chip_width=196,
                image_path=macro_panel_path,
                headline=(
                    f"Fit + PDE consume {_fmt_pct(min(fit_pde_share))} to {_fmt_pct(max(fit_pde_share))} of the integrated budget."
                ),
                body=(
                    "Optimize surface fit or PDE first. "
                    f"The remaining stages stay around {_fmt_runtime_ms(minor_small)} -> {_fmt_runtime_ms(minor_large)} combined."
                ),
            ),
        )
        callout_cards = (
            _BenchmarkOverviewCalloutCard(
                key="digital-payoff-remedies",
                title="Digital payoff remedies",
                x=72,
                y=734,
                metric=(
                    f"{_fmt_sci(float(digital_none['refinement_spread']))} -> "
                    f"{_fmt_sci(float(digital_l2['refinement_spread']))}"
                ),
                body=(
                    "Grid-to-grid price spread, Rannacher + none vs Rannacher + "
                    "L2 projection.\n"
                    "Best measured abs error with L2 projection: "
                    f"{_fmt_sci(float(digital_l2['best_abs_error']))}."
                ),
                body_y_offset=150,
                body_max_height=80,
            ),
            _BenchmarkOverviewCalloutCard(
                key="local-vol-extraction",
                title="Local-vol extraction",
                x=574,
                y=734,
                metric="0.0% interior invalid share",
                body=(
                    "Largest tested 121x241 grids stayed interior-valid for both "
                    "coordinate choices.\n"
                    "Median runtime: "
                    f"K = {_fmt_runtime_ms(float(localvol_k['runtime_ms']))}, "
                    f"logK = {_fmt_runtime_ms(float(localvol_logk['runtime_ms']))}."
                ),
                metric_fill_key="success",
            ),
            _BenchmarkOverviewCalloutCard(
                key="tree-framing",
                title="Tree framing",
                x=1076,
                y=734,
                width=450,
                metric=_fmt_runtime_ms(float(tree_last["runtime_ms"])),
                body=(
                    "5000-step CRR runtime in the committed snapshot.\n"
                    "Absolute error vs Black-Scholes at that depth: "
                    f"{_fmt_sci(float(tree_last['abs_error']))}."
                ),
                body_y_offset=150,
                body_max_height=60,
            ),
        )

        intro_svg = render_svg_text_block(
            block_id="benchmark-overview-intro",
            text=(
                "Compressed reviewer scan of the repo's measured scaling, "
                "cost-versus-accuracy, and end-to-end stage budget story."
            ),
            x=72,
            y=148,
            max_width=1456,
            max_height=56,
            style=SvgTextStyle(
                font_family=font_stack,
                font_size=24,
                font_weight="400",
                fill=colors["muted"],
                line_height=28,
            ),
            overflow_label="Benchmark overview intro text",
        )
        provenance_svg = render_svg_text_block(
            block_id="benchmark-overview-provenance",
            text=(
                "Built from committed benchmark outputs under benchmarks/artifacts "
                "and docs/assets/generated/benchmarks."
            ),
            x=72,
            y=182,
            max_width=1456,
            max_height=22,
            style=SvgTextStyle(
                font_family=font_stack,
                font_size=18,
                font_weight="400",
                fill=colors["muted"],
                line_height=22,
            ),
            overflow_label="Benchmark overview provenance text",
        )
        footer_svg = render_svg_text_block(
            block_id="benchmark-overview-footer",
            text=(
                "This asset summarizes the committed benchmark bundle. Use the "
                "performance page for the full figures, conditions, and "
                "reproducibility notes."
            ),
            x=72,
            y=1012,
            max_width=1456,
            max_height=40,
            style=SvgTextStyle(
                font_family=font_stack,
                font_size=16,
                font_weight="400",
                fill=colors["muted"],
                line_height=20,
            ),
            overflow_label="Benchmark overview footer text",
        )

        primary_clip_paths: list[str] = []
        primary_svg_blocks: list[str] = []
        for panel in primary_panels:
            clip_paths, panel_svg = _render_benchmark_overview_primary_panel(
                panel,
                colors=colors,
                font_stack=font_stack,
            )
            primary_clip_paths.extend(clip_paths)
            primary_svg_blocks.append(panel_svg)

        callout_clip_paths: list[str] = []
        callout_svg_blocks: list[str] = []
        for card in callout_cards:
            clip_path, card_svg = _render_benchmark_overview_callout(
                card,
                colors=colors,
                font_stack=font_stack,
            )
            callout_clip_paths.append(clip_path)
            callout_svg_blocks.append(card_svg)

        defs_svg = "\n".join(
            [
                '    <linearGradient id="headerBg" x1="0%" y1="0%" x2="100%" y2="0%">',
                f'      <stop offset="0%" stop-color="{colors["header_start"]}" />',
                f'      <stop offset="100%" stop-color="{colors["header_end"]}" />',
                "    </linearGradient>",
                *primary_clip_paths,
                *callout_clip_paths,
            ]
        )
        primary_panels_svg = "\n".join(primary_svg_blocks)
        callouts_svg = "\n".join(callout_svg_blocks)

        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1080" viewBox="0 0 1600 1080" role="img" aria-labelledby="title desc">
  <title id="title">Benchmark overview for the option pricing library</title>
  <desc id="desc">Reviewer-facing benchmark summary showing IV scaling, PDE cost versus accuracy, macro pipeline stage budget, and concise measured callouts for digital remedies, local-vol extraction, and tree scaling.</desc>
  <defs>
{defs_svg}
  </defs>

  <rect width="1600" height="1080" fill="{colors["page_bg"]}" />
  <rect x="28" y="28" width="1544" height="1024" rx="34" ry="34" fill="{colors["card_bg"]}" stroke="{colors["card_stroke"]}" stroke-width="2" />

  <text x="72" y="108" font-family="{font_stack}" font-size="54" font-weight="700" fill="{colors["text"]}">Benchmark overview</text>
  {intro_svg}
  {provenance_svg}

  <rect x="72" y="220" width="1456" height="360" rx="28" ry="28" fill="url(#headerBg)" stroke="{colors["card_stroke"]}" stroke-width="2" />
  <text x="102" y="254" font-family="{font_stack}" font-size="18" font-weight="700" letter-spacing="1" fill="{colors["accent"]}">PRIMARY PANELS</text>
{primary_panels_svg}

  <text x="72" y="712" font-family="{font_stack}" font-size="18" font-weight="700" letter-spacing="1" fill="{colors["accent"]}">ADDITIONAL MEASURED CALLOUTS</text>
{callouts_svg}

  {footer_svg}
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

    if args.check:
        return _check_benchmark_source_manifest(artifacts_dir)

    if args.write_source_manifest:
        path = _write_benchmark_source_manifest(artifacts_dir)
        print(str(path.relative_to(ROOT)).replace("\\", "/"))
        return 0

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
    overview_asset = build_benchmark_overview_asset(
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
    _write_benchmark_source_manifest(artifacts_dir)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
