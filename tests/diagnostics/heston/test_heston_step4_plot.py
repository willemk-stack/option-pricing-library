from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

from option_pricing.diagnostics.heston import (
    compare_backend_slice,
    price_slice_with_diagnostics,
    run_heston_slice_diagnostics,
)
from option_pricing.diagnostics.heston.plot import (
    plot_backend_difference_by_strike,
    plot_cancellation_ratio_by_strike,
    plot_config_sweep,
    plot_panel_contributions,
    plot_smile_with_warning_overlay,
    plot_tail_fraction_by_strike,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _slice_table(*, all_clear: bool = False) -> pd.DataFrame:
    warning_count = np.array([0, 1, 0], dtype=np.int64)
    severity = np.array(["ok", "warning", "ok"], dtype=object)
    smoothness_flag = np.array([False, True, False], dtype=object)
    discontinuity_flag = np.array([False, False, True], dtype=object)
    suspicious_flag = np.array([False, True, True], dtype=object)
    suspicious_reasons = np.array(
        ["none", "integration warning severity >= warning", "discontinuity flag"],
        dtype=object,
    )
    combined_warning_labels = np.array(
        ["none", "large tail fraction", "oscillation spike"],
        dtype=object,
    )

    if all_clear:
        warning_count = np.zeros(3, dtype=np.int64)
        severity = np.array(["ok", "ok", "ok"], dtype=object)
        smoothness_flag = np.array([False, False, False], dtype=object)
        discontinuity_flag = np.array([False, False, False], dtype=object)
        suspicious_flag = np.array([False, False, False], dtype=object)
        suspicious_reasons = np.array(["none", "none", "none"], dtype=object)
        combined_warning_labels = np.array(["none", "none", "none"], dtype=object)

    return pd.DataFrame(
        {
            "strike": np.array([90.0, 100.0, 110.0], dtype=np.float64),
            "log_moneyness": np.array([0.10536052, 0.0, -0.09531018], dtype=np.float64),
            "price": np.array([12.5, 7.4, 4.0], dtype=np.float64),
            "implied_vol": np.array([0.23, 0.21, 0.22], dtype=np.float64),
            "warning_count": warning_count,
            "severity": severity,
            "tail_fraction": np.array([0.01, 0.06, 0.09], dtype=np.float64),
            "cancellation_ratio": np.array([1.2, 12.0, 20.0], dtype=np.float64),
            "backend_diff": np.array([0.0, 1.0e-4, 2.0e-4], dtype=np.float64),
            "smoothness_flag": smoothness_flag,
            "discontinuity_flag": discontinuity_flag,
            "tail_fraction_p0": np.array([0.01, 0.04, 0.03], dtype=np.float64),
            "tail_fraction_p1": np.array([0.0, 0.06, 0.09], dtype=np.float64),
            "cancellation_ratio_p0": np.array([1.0, 8.0, 10.0], dtype=np.float64),
            "cancellation_ratio_p1": np.array([1.2, 12.0, 20.0], dtype=np.float64),
            "warning_count_p0": np.array([0, 0, 0], dtype=np.int64),
            "warning_count_p1": warning_count,
            "severity_p0": np.array(["ok", "ok", "ok"], dtype=object),
            "severity_p1": severity,
            "combined_warning_labels": combined_warning_labels,
            "config_price_span": np.array([0.0, 2.0e-4, 5.0e-4], dtype=np.float64),
            "smoothness_signal": np.array([np.nan, 0.09, np.nan], dtype=np.float64),
            "discontinuity_signal": np.array([np.nan, np.nan, 0.18], dtype=np.float64),
            "perturbation_max_relative_price_change": np.array(
                [0.01, 0.12, 0.05],
                dtype=np.float64,
            ),
            "suspicious_flag": suspicious_flag,
            "suspicious_reasons": suspicious_reasons,
        }
    )


def _backend_compare_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strike": np.array([90.0, 100.0, 110.0], dtype=np.float64),
            "log_moneyness": np.array([0.10536052, 0.0, -0.09531018], dtype=np.float64),
            "backend_a": np.array(["gauss_legendre"] * 3, dtype=object),
            "backend_b": np.array(["quad"] * 3, dtype=object),
            "price_a": np.array([12.5, 7.4, 4.0], dtype=np.float64),
            "price_b": np.array([12.49, 7.3998, 3.9997], dtype=np.float64),
            "price_diff": np.array([0.01, 2.0e-4, 3.0e-4], dtype=np.float64),
            "abs_price_diff": np.array([0.01, 2.0e-4, 3.0e-4], dtype=np.float64),
            "implied_vol_a": np.array([0.23, 0.21, 0.22], dtype=np.float64),
            "implied_vol_b": np.array([0.229, 0.209, 0.219], dtype=np.float64),
            "implied_vol_diff": np.array([0.001, 0.001, 0.001], dtype=np.float64),
            "warning_count_a": np.array([0, 1, 0], dtype=np.int64),
            "warning_count_b": np.array([0, 0, 0], dtype=np.int64),
            "severity_a": np.array(["ok", "warning", "ok"], dtype=object),
            "severity_b": np.array(["ok", "ok", "ok"], dtype=object),
        }
    )


def _config_sweep_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "config_label": np.array(["primary", "comparison", "robust"], dtype=object),
            "backend": np.array(
                ["gauss_legendre", "quad", "gauss_legendre"],
                dtype=object,
            ),
            "p0_max_severity": np.array(["ok", "warning", "ok"], dtype=object),
            "p1_max_severity": np.array(["ok", "warning", "ok"], dtype=object),
            "p0_warning_point_count": np.array([0, 1, 0], dtype=np.int64),
            "p1_warning_point_count": np.array([0, 1, 0], dtype=np.int64),
            "max_abs_price_diff_vs_baseline": np.array(
                [0.0, 3.0e-4, 1.0e-4],
                dtype=np.float64,
            ),
            "mean_abs_price_diff_vs_baseline": np.array(
                [0.0, 2.0e-4, 6.0e-5],
                dtype=np.float64,
            ),
            "max_abs_probability_diff_p0": np.array(
                [0.0, 2.0e-4, 7.0e-5],
                dtype=np.float64,
            ),
            "max_abs_probability_diff_p1": np.array(
                [0.0, 2.0e-4, 7.0e-5],
                dtype=np.float64,
            ),
            "notes": np.array(["", "", ""], dtype=object),
        }
    )


def _sample_report(
    *,
    include_panel_detail: bool = True,
    include_config_prices: bool = True,
    all_clear: bool = False,
):
    slice_table = _slice_table(all_clear=all_clear)
    price_artifact = price_slice_with_diagnostics(
        slice_table=slice_table,
        arrays={
            "primary_implied_vol": slice_table["implied_vol"].to_numpy(
                dtype=np.float64
            ),
            "comparison_implied_vol": (
                slice_table["implied_vol"].to_numpy(dtype=np.float64) - 0.001
            ),
            "smoothness_signal": slice_table["smoothness_signal"].to_numpy(
                dtype=np.float64
            ),
            "discontinuity_signal": slice_table["discontinuity_signal"].to_numpy(
                dtype=np.float64
            ),
            "config_price_span": slice_table["config_price_span"].to_numpy(
                dtype=np.float64
            ),
            "perturbation_max_relative_price_change": (
                slice_table["perturbation_max_relative_price_change"].to_numpy(
                    dtype=np.float64
                )
            ),
            "suspicious_flag": slice_table["suspicious_flag"].to_numpy(dtype=bool),
        },
    )
    backend_artifact = compare_backend_slice(
        comparison_table=_backend_compare_table(),
    )

    arrays: dict[str, object] = {
        "probability_p0": {
            "panel_edges": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
            "log_moneyness": slice_table["log_moneyness"].to_numpy(dtype=np.float64),
        },
        "probability_p1": {
            "panel_edges": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
            "log_moneyness": slice_table["log_moneyness"].to_numpy(dtype=np.float64),
        },
    }
    if include_panel_detail:
        arrays["probability_p0"] = {
            **arrays["probability_p0"],
            "panel_contribs": np.array(
                [
                    [0.30, -0.05, 0.02],
                    [0.25, -0.06, 0.01],
                    [0.20, -0.04, 0.02],
                ],
                dtype=np.float64,
            ),
            "panel_invalid": np.array(
                [
                    [False, False, False],
                    [False, True, False],
                    [False, False, False],
                ],
                dtype=np.bool_,
            ),
        }
        arrays["probability_p1"] = {
            **arrays["probability_p1"],
            "panel_contribs": np.array(
                [
                    [0.22, -0.03, 0.01],
                    [0.18, -0.04, 0.02],
                    [0.16, -0.02, 0.01],
                ],
                dtype=np.float64,
            ),
            "panel_invalid": np.array(
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
                dtype=np.bool_,
            ),
        }

    if include_config_prices:
        arrays["config_sweep_prices"] = {
            "primary": np.array([12.5, 7.4, 4.0], dtype=np.float64),
            "comparison": np.array([12.49, 7.3998, 3.9997], dtype=np.float64),
            "robust": np.array([12.5001, 7.40005, 4.00001], dtype=np.float64),
        }

    return run_heston_slice_diagnostics(
        meta={"stage": "step4"},
        tables={
            "config_sweep": _config_sweep_table(),
            "worst_strikes": slice_table.head(2).copy(),
        },
        arrays=arrays,
        price=price_artifact,
        backend_comparison=backend_artifact,
    )


def test_plot_helpers_accept_reports_tables_and_presupplied_axes() -> None:
    report = _sample_report()

    fig_tail, ax_tail = plt.subplots()
    fig_cancel, ax_cancel = plt.subplots()
    fig_panel, ax_panel = plt.subplots()

    try:
        returned_fig_tail, returned_ax_tail = plot_tail_fraction_by_strike(
            report, ax=ax_tail
        )
        returned_fig_cancel, returned_ax_cancel = plot_cancellation_ratio_by_strike(
            report.tables["slice"][["strike", "cancellation_ratio"]],
            ax=ax_cancel,
        )
        returned_fig_panel, returned_ax_panel = plot_panel_contributions(
            report,
            probability_key="probability_p0",
            point_index=1,
            ax=ax_panel,
        )

        assert returned_fig_tail is fig_tail
        assert returned_ax_tail is ax_tail
        assert returned_fig_cancel is fig_cancel
        assert returned_ax_cancel is ax_cancel
        assert returned_fig_panel is fig_panel
        assert returned_ax_panel is ax_panel
        assert ax_tail.get_ylabel() == "Tail fraction"
        assert ax_cancel.get_ylabel() == "Cancellation ratio"
        assert "panel contributions" in ax_panel.get_title().lower()
    finally:
        plt.close(fig_tail)
        plt.close(fig_cancel)
        plt.close(fig_panel)


def test_plot_helpers_consume_existing_report_artifacts_without_repricing() -> None:
    report = _sample_report()

    fig_backend, ax_backend = plot_backend_difference_by_strike(report)
    fig_smile, ax_smile = plot_smile_with_warning_overlay(report)
    try:
        np.testing.assert_allclose(
            ax_backend.lines[0].get_ydata(),
            report.tables["backend_compare"]["abs_price_diff"].to_numpy(
                dtype=np.float64
            ),
        )
        np.testing.assert_allclose(
            ax_smile.lines[0].get_ydata(),
            report.tables["slice"]["implied_vol"].to_numpy(dtype=np.float64),
        )
    finally:
        plt.close(fig_backend)
        plt.close(fig_smile)


def test_plot_panel_and_config_sweep_handle_sparse_optional_arrays_gracefully() -> None:
    report = _sample_report(include_panel_detail=False, include_config_prices=False)

    fig_panel, ax_panel = plot_panel_contributions(
        report, probability_key="probability_p1"
    )
    fig_sweep, ax_sweep = plot_config_sweep(report)
    try:
        panel_text = " ".join(text.get_text() for text in ax_panel.texts)
        sweep_text = " ".join(text.get_text() for text in ax_sweep.texts)

        assert "unavailable" in panel_text.lower()
        assert "summary table only" in sweep_text.lower()
        assert len(ax_sweep.patches) == len(report.tables["config_sweep"])
    finally:
        plt.close(fig_panel)
        plt.close(fig_sweep)


def test_plot_smile_overlay_handles_all_clear_slice_readably() -> None:
    report = _sample_report(all_clear=True)

    fig, ax = plot_smile_with_warning_overlay(report)
    try:
        text = " ".join(text.get_text() for text in ax.texts)
        assert "no warning" in text.lower()
    finally:
        plt.close(fig)
