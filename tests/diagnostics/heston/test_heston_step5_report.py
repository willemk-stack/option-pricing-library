from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd
import pytest

from option_pricing.diagnostics.heston import (
    compare_backend_slice,
    price_slice_with_diagnostics,
    probability_slice_with_diagnostics,
    run_heston_slice_diagnostics,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _summary_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": np.array(
                [
                    "worst_strike",
                    "total_warnings",
                    "max_tail_fraction",
                    "max_cancellation_ratio",
                    "max_backend_discrepancy",
                    "suspicious_strike_count",
                ],
                dtype=object,
            ),
            "value": np.array([100.0, 1, 0.06, 12.0, 2.0e-4, 1], dtype=object),
            "notes": np.array(
                [
                    "Largest provisional concern.",
                    "Sum of warnings.",
                    "Tail fraction max.",
                    "Cancellation ratio max.",
                    "Primary vs comparison backend diff.",
                    "Approval required before finalizing suspicious-strike thresholds.",
                ],
                dtype=object,
            ),
            "severity": np.array(
                ["warning", "warning", "warning", "warning", "warning", "warning"],
                dtype=object,
            ),
        }
    )


def _slice_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strike": np.array([90.0, 100.0, 110.0], dtype=np.float64),
            "log_moneyness": np.array([0.10536052, 0.0, -0.09531018], dtype=np.float64),
            "price": np.array([12.5, 7.4, 4.0], dtype=np.float64),
            "implied_vol": np.array([0.23, 0.21, 0.22], dtype=np.float64),
            "warning_count": np.array([0, 1, 0], dtype=np.int64),
            "severity": np.array(["ok", "warning", "ok"], dtype=object),
            "tail_fraction": np.array([0.01, 0.06, 0.09], dtype=np.float64),
            "cancellation_ratio": np.array([1.2, 12.0, 20.0], dtype=np.float64),
            "backend_diff": np.array([0.0, 1.0e-4, 2.0e-4], dtype=np.float64),
            "smoothness_flag": np.array([False, True, False], dtype=object),
            "discontinuity_flag": np.array([False, False, True], dtype=object),
            "suspicious_flag": np.array([False, True, True], dtype=object),
            "suspicious_reasons": np.array(
                [
                    "none",
                    "integration warning severity >= warning",
                    "discontinuity flag",
                ],
                dtype=object,
            ),
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
        }
    )


def _report() -> object:
    slice_table = _slice_table()
    probability_artifact = probability_slice_with_diagnostics(
        probability_columns={
            "log_moneyness": slice_table["log_moneyness"].to_numpy(dtype=np.float64),
            "probability": np.array([0.42, 0.5, 0.61], dtype=np.float64),
            "warning_count": np.array([0, 0, 1], dtype=np.int64),
        },
        arrays={
            "log_moneyness": slice_table["log_moneyness"].to_numpy(dtype=np.float64),
            "probability": np.array([0.42, 0.5, 0.61], dtype=np.float64),
        },
    )
    price_artifact = price_slice_with_diagnostics(
        slice_table=slice_table,
        arrays={
            "primary_price": slice_table["price"].to_numpy(dtype=np.float64),
            "smoothness_signal": np.array([np.nan, 0.09, np.nan], dtype=np.float64),
            "discontinuity_signal": np.array([np.nan, np.nan, 0.18], dtype=np.float64),
            "config_price_span": np.array([0.0, 2.0e-4, 5.0e-4], dtype=np.float64),
            "perturbation_max_relative_price_change": np.array(
                [0.01, 0.12, 0.05],
                dtype=np.float64,
            ),
            "suspicious_flag": np.array([False, True, True], dtype=np.bool_),
        },
    )
    backend_artifact = compare_backend_slice(
        comparison_table=_backend_compare_table(),
        arrays={"abs_price_diff": np.array([0.01, 2.0e-4, 3.0e-4], dtype=np.float64)},
    )

    return run_heston_slice_diagnostics(
        meta={"stage": "step5"},
        tables={
            "summary": _summary_table(),
            "worst_strikes": slice_table.iloc[[1, 2]].copy(),
            "config_sweep": pd.DataFrame(
                {
                    "config_label": np.array(["primary", "comparison"], dtype=object),
                    "backend": np.array(["gauss_legendre", "quad"], dtype=object),
                    "p0_max_severity": np.array(["ok", "warning"], dtype=object),
                    "p1_max_severity": np.array(["ok", "warning"], dtype=object),
                    "p0_warning_point_count": np.array([0, 1], dtype=np.int64),
                    "p1_warning_point_count": np.array([0, 1], dtype=np.int64),
                    "max_abs_price_diff_vs_baseline": np.array(
                        [0.0, 3.0e-4],
                        dtype=np.float64,
                    ),
                    "mean_abs_price_diff_vs_baseline": np.array(
                        [0.0, 2.0e-4],
                        dtype=np.float64,
                    ),
                    "max_abs_probability_diff_p0": np.array(
                        [0.0, 2.0e-4],
                        dtype=np.float64,
                    ),
                    "max_abs_probability_diff_p1": np.array(
                        [0.0, 2.0e-4],
                        dtype=np.float64,
                    ),
                    "notes": np.array(["", ""], dtype=object),
                }
            ),
        },
        arrays={
            "probability_p0": {
                "probability": np.array([0.4, 0.5, 0.6], dtype=np.float64)
            },
            "parameter_perturbation_table": {
                "label": np.array(["eta_up"], dtype=object),
                "max_relative_price_change": np.array([0.12], dtype=np.float64),
            },
        },
        probability=probability_artifact,
        price=price_artifact,
        backend_comparison=backend_artifact,
    )


def test_run_heston_slice_diagnostics_keeps_step1_shape_and_optional_array_contract() -> (
    None
):
    report = _report()
    payload = report.to_dict()

    assert set(payload) == {"meta", "tables", "arrays"}
    assert {
        "summary",
        "slice",
        "worst_strikes",
        "backend_compare",
        "config_sweep",
        "worst_panels_p0",
        "worst_panels_p1",
    } <= set(report.tables)
    assert "probability" in report.tables
    assert {"probability", "slice", "backend_compare"} <= set(report.arrays)
    assert "parameter_perturbation_table" in report.arrays
    assert "config_sweep_prices" in report.meta["optional_array_groups"]
    assert "plotting_contract" in report.meta


def test_run_heston_slice_diagnostics_default_tables_use_plot_friendly_columns() -> (
    None
):
    report = run_heston_slice_diagnostics()

    assert "abs_price_diff" in report.tables["backend_compare"].columns
    assert "max_abs_price_diff_vs_baseline" in report.tables["config_sweep"].columns
    assert "detail_available" in report.tables["worst_panels_p0"].columns
    assert "suspicious_reasons" in report.tables["worst_strikes"].columns


def test_run_heston_slice_diagnostics_rejects_matplotlib_objects_in_payload() -> None:
    fig, ax = plt.subplots()
    try:
        with pytest.raises(TypeError, match="matplotlib"):
            run_heston_slice_diagnostics(arrays={"figure": fig})
        with pytest.raises(TypeError, match="matplotlib"):
            run_heston_slice_diagnostics(meta={"axes": ax})
    finally:
        plt.close(fig)


def test_report_serialization_remains_plain_data_friendly() -> None:
    report = _report()

    payload = json.loads(report.to_json())

    assert set(payload) == {"meta", "tables", "arrays"}
    assert "figure" not in payload["arrays"]
    assert payload["arrays"]["slice"]["config_price_span"] == [0.0, 0.0002, 0.0005]
