from __future__ import annotations

from dataclasses import replace

import matplotlib
import numpy as np
import pytest

import option_pricing.diagnostics.mc_vs_bs.tables as mc_vs_bs_tables
from option_pricing.diagnostics.heston import (
    HestonMCComparisonCase,
    HestonMCSweepConfig,
    compare_heston_mc_schemes,
    plot_heston_mc_bias_vs_timestep,
    plot_heston_mc_runtime_vs_error,
    plot_heston_mc_scheme_comparison,
    run_heston_mc_comparison_sweep,
    summarize_bias_vs_timestep,
    summarize_runtime_vs_error,
)
from option_pricing.diagnostics.mc_vs_bs.tables import convergence_table
from option_pricing.models.heston import HestonParams
from option_pricing.pricers.heston import heston_price_call_from_ctx
from option_pricing.pricers.heston_mc import heston_mc_price_call
from option_pricing.types import MarketData, OptionType

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _params() -> HestonParams:
    return HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05)


def _comparison_case() -> HestonMCComparisonCase:
    return HestonMCComparisonCase(
        ctx=MarketData(spot=100.0, rate=0.02, dividend_yield=0.0).to_context(),
        params=_params(),
        kind=OptionType.CALL,
        strike=100.0,
        tau=1.0,
    )


def _comparison_config(**overrides: object) -> HestonMCSweepConfig:
    defaults: dict[str, object] = {
        "schemes": ("euler_full_truncation", "quadratic_exponential"),
        "n_steps_grid": (4, 8),
        "n_paths": 128,
        "seed": 17,
        "antithetic": True,
        "repeats": 1,
        "use_control_variate": False,
    }
    defaults.update(overrides)
    return HestonMCSweepConfig(**defaults)


def _install_heston_mc_diagnostics(monkeypatch) -> None:
    params = _params()

    def _bs_like_price(p):
        return float(
            heston_price_call_from_ctx(
                strike=p.K,
                ctx=p.ctx,
                tau=p.tau,
                params=params,
            )
        )

    def _mc_price(p, *, cfg):
        return heston_mc_price_call(p, params=params, n_steps=64, cfg=cfg)

    monkeypatch.setattr(mc_vs_bs_tables, "_default_bs_price", lambda: _bs_like_price)
    monkeypatch.setattr(mc_vs_bs_tables, "_default_mc_price", lambda: _mc_price)


def test_heston_mc_convergence_table_has_expected_shape_and_columns(
    make_inputs,
    monkeypatch,
) -> None:
    _install_heston_mc_diagnostics(monkeypatch)
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.02,
        q=0.0,
        sigma=0.2,
        T=1.0,
        kind=OptionType.CALL,
    )

    table = convergence_table(p, n_paths_list=[512, 1_024, 2_048], seed=5)

    assert table.shape[0] == 3
    assert table["n_paths"].to_list() == [512, 1_024, 2_048]
    assert {"n_paths", "mc", "se", "bs", "err", "MC", "SE", "BS", "MC-BS"} <= set(
        table.columns
    )
    assert np.all(np.isfinite(table[["mc", "se", "bs", "err"]].to_numpy(dtype=float)))


def test_heston_mc_convergence_table_has_no_plotting_side_effects(
    make_inputs,
    monkeypatch,
) -> None:
    _install_heston_mc_diagnostics(monkeypatch)
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.02,
        q=0.0,
        sigma=0.2,
        T=1.0,
        kind=OptionType.CALL,
    )

    before = tuple(plt.get_fignums())
    convergence_table(p, n_paths_list=[256, 512], seed=7)
    after = tuple(plt.get_fignums())

    assert after == before


def test_mc_sweep_returns_expected_columns() -> None:
    case = _comparison_case()
    sweep = run_heston_mc_comparison_sweep(case, _comparison_config())

    assert list(sweep.columns) == [
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
    assert sweep.shape[0] == 4
    assert np.all(np.isfinite(sweep["price"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(sweep["stderr"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(sweep["runtime_seconds"].to_numpy(dtype=float)))
    assert np.all(sweep["runtime_seconds"].to_numpy(dtype=float) > 0.0)
    np.testing.assert_allclose(
        sweep["dt"].to_numpy(dtype=float),
        case.tau / sweep["n_steps"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_bias_summary_groups_by_scheme_and_timestep() -> None:
    summary = summarize_bias_vs_timestep(
        run_heston_mc_comparison_sweep(_comparison_case(), _comparison_config())
    )

    assert summary.shape[0] == 4
    assert {
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
    } == set(summary.columns)
    assert np.all(np.isfinite(summary["mean_price"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(summary["rmse"].to_numpy(dtype=float)))
    assert set(summary["repeat_count"].to_numpy(dtype=int)) == {1}


def test_runtime_summary_reports_positive_runtime() -> None:
    summary = summarize_runtime_vs_error(
        run_heston_mc_comparison_sweep(_comparison_case(), _comparison_config())
    )

    assert summary.shape[0] == 4
    assert np.all(summary["mean_runtime_seconds"].to_numpy(dtype=float) > 0.0)
    assert np.all(np.isfinite(summary["mean_runtime_per_path"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(summary["mean_abs_error"].to_numpy(dtype=float)))


def test_scheme_comparison_handles_euler_and_qe() -> None:
    comparison = compare_heston_mc_schemes(
        run_heston_mc_comparison_sweep(_comparison_case(), _comparison_config())
    )

    assert set(comparison["scheme"].tolist()) == {
        "euler_full_truncation",
        "quadratic_exponential",
    }
    assert set(comparison["recommended_use"].tolist()) == {
        "baseline / educational",
        "production candidate",
    }
    assert np.all(np.isfinite(comparison["best_mean_abs_error"].to_numpy(dtype=float)))
    assert np.all(
        np.isfinite(comparison["runtime_at_best_error"].to_numpy(dtype=float))
    )
    assert np.all(
        np.isfinite(comparison["coverage_rate_at_best_error"].to_numpy(dtype=float))
    )


def test_plot_helpers_accept_minimal_tables() -> None:
    sweep = run_heston_mc_comparison_sweep(_comparison_case(), _comparison_config())
    bias_summary = summarize_bias_vs_timestep(sweep)
    runtime_summary = summarize_runtime_vs_error(sweep)

    fig_bias, ax_bias = plot_heston_mc_bias_vs_timestep(bias_summary)
    fig_runtime, ax_runtime = plot_heston_mc_runtime_vs_error(runtime_summary)
    fig_compare, ax_compare = plot_heston_mc_scheme_comparison(sweep)

    try:
        assert fig_bias is ax_bias.figure
        assert fig_runtime is ax_runtime.figure
        assert fig_compare is ax_compare.figure
    finally:
        plt.close(fig_bias)
        plt.close(fig_runtime)
        plt.close(fig_compare)


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (_comparison_config(schemes=()), "schemes"),
        (_comparison_config(n_steps_grid=()), "n_steps_grid"),
        (_comparison_config(n_paths=0), "n_paths"),
        (_comparison_config(n_steps_grid=(0, 8)), "n_steps_grid"),
        (_comparison_config(repeats=0), "repeats"),
    ],
)
def test_invalid_sweep_config_raises(
    config: HestonMCSweepConfig,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        run_heston_mc_comparison_sweep(_comparison_case(), config)

    with pytest.raises(ValueError, match="tau"):
        run_heston_mc_comparison_sweep(
            replace(_comparison_case(), tau=0.0),
            _comparison_config(),
        )
