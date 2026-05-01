from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from option_pricing.diagnostics.heston import (
    HestonCalibrationBenchmarkDiagnostics,
    run_heston_calibration_benchmark_diagnostics,
)
from option_pricing.diagnostics.heston.contracts import (
    HESTON_CALIBRATION_BENCHMARK_PARAMETER_COLUMNS,
    HESTON_CALIBRATION_BENCHMARK_REQUIRED_TABLES,
    HESTON_CALIBRATION_BENCHMARK_RESIDUAL_COLUMNS,
    HESTON_CALIBRATION_BENCHMARK_RUN_COLUMNS,
    HESTON_CALIBRATION_BENCHMARK_SUMMARY_COLUMNS,
)
from option_pricing.models.heston.params import HestonParams


@pytest.fixture(scope="module")
def calibration_report() -> HestonCalibrationBenchmarkDiagnostics:
    return run_heston_calibration_benchmark_diagnostics(repeat=1, warmup=0)


def test_runner_returns_calibration_benchmark_diagnostics(
    calibration_report: HestonCalibrationBenchmarkDiagnostics,
) -> None:
    assert isinstance(calibration_report, HestonCalibrationBenchmarkDiagnostics)
    assert set(HESTON_CALIBRATION_BENCHMARK_REQUIRED_TABLES) <= set(
        calibration_report.tables
    )
    assert calibration_report.meta["benchmark_label"] == "smoke"
    assert calibration_report.meta["objective"].startswith("existing HestonObjective")


def test_required_tables_exist_and_columns_are_ordered_first(
    calibration_report: HestonCalibrationBenchmarkDiagnostics,
) -> None:
    expected_columns = {
        "runs": HESTON_CALIBRATION_BENCHMARK_RUN_COLUMNS,
        "summary": HESTON_CALIBRATION_BENCHMARK_SUMMARY_COLUMNS,
        "parameter_recovery": HESTON_CALIBRATION_BENCHMARK_PARAMETER_COLUMNS,
        "residuals": HESTON_CALIBRATION_BENCHMARK_RESIDUAL_COLUMNS,
    }
    for table_name, columns in expected_columns.items():
        table = calibration_report.tables[table_name]
        assert list(table.columns[: len(columns)]) == list(columns)


def test_serialization_roundtrip_preserves_tables_and_arrays(
    calibration_report: HestonCalibrationBenchmarkDiagnostics,
) -> None:
    restored = HestonCalibrationBenchmarkDiagnostics.from_json(
        calibration_report.to_json()
    )

    assert restored.meta == calibration_report.meta
    for table_name in HESTON_CALIBRATION_BENCHMARK_REQUIRED_TABLES:
        pd.testing.assert_frame_equal(
            restored.tables[table_name],
            calibration_report.tables[table_name],
            check_dtype=False,
        )
    np.testing.assert_allclose(
        restored.arrays["quotes"]["strike"],
        calibration_report.arrays["quotes"]["strike"],
    )


def test_report_rejects_matplotlib_like_objects(
    calibration_report: HestonCalibrationBenchmarkDiagnostics,
) -> None:
    matplotlib_like = type("Figure", (), {"__module__": "matplotlib.figure"})()
    tables = {name: table.copy() for name, table in calibration_report.tables.items()}

    with pytest.raises(TypeError, match="matplotlib"):
        HestonCalibrationBenchmarkDiagnostics(
            meta={"figure": matplotlib_like},
            tables=tables,
            arrays={},
        )
    with pytest.raises(TypeError, match="matplotlib"):
        HestonCalibrationBenchmarkDiagnostics(
            meta={},
            tables=tables,
            arrays={"figure": matplotlib_like},
        )


def test_analytic_and_finite_difference_modes_are_present(
    calibration_report: HestonCalibrationBenchmarkDiagnostics,
) -> None:
    expected = {"analytic", "finite_difference"}
    assert set(calibration_report.tables["runs"]["mode"]) == expected
    assert set(calibration_report.tables["summary"]["mode"]) == expected
    assert set(calibration_report.tables["parameter_recovery"]["mode"]) == expected
    assert set(calibration_report.tables["residuals"]["mode"]) == expected


def test_speedup_columns_are_finite_or_explained(
    calibration_report: HestonCalibrationBenchmarkDiagnostics,
) -> None:
    summary = calibration_report.tables["summary"]
    for column in (
        "speedup_vs_finite_difference",
        "nfev_ratio_vs_finite_difference",
    ):
        for _, row in summary.iterrows():
            value = row[column]
            if pd.isna(value):
                assert str(row["notes"])
            else:
                assert np.isfinite(float(value))
                assert float(value) > 0.0


def test_residual_norms_are_finite(
    calibration_report: HestonCalibrationBenchmarkDiagnostics,
) -> None:
    runs = calibration_report.tables["runs"]
    residuals = calibration_report.tables["residuals"]

    assert np.isfinite(runs["residual_norm"].to_numpy(dtype=np.float64)).all()
    assert np.isfinite(runs["max_abs_residual"].to_numpy(dtype=np.float64)).all()
    assert np.isfinite(residuals["residual"].to_numpy(dtype=np.float64)).all()


def test_runner_does_not_mutate_inputs() -> None:
    true_params = HestonParams(kappa=1.5, vbar=0.04, eta=0.45, rho=-0.55, v=0.045)
    seed_params = HestonParams(kappa=1.1, vbar=0.035, eta=0.35, rho=-0.35, v=0.04)
    true_before = true_params.as_array()
    seed_before = seed_params.as_array()
    expiries = np.array([0.5], dtype=np.float64)
    log_moneyness = np.array([-0.05, 0.05], dtype=np.float64)
    expiries_before = expiries.copy()
    log_moneyness_before = log_moneyness.copy()

    run_heston_calibration_benchmark_diagnostics(
        true_params=true_params,
        seed_params=seed_params,
        expiries=expiries,
        log_moneyness=log_moneyness,
        repeat=1,
        warmup=0,
    )

    np.testing.assert_allclose(true_params.as_array(), true_before)
    np.testing.assert_allclose(seed_params.as_array(), seed_before)
    np.testing.assert_array_equal(expiries, expiries_before)
    np.testing.assert_array_equal(log_moneyness, log_moneyness_before)


def test_lm_rejects_non_linear_loss() -> None:
    with pytest.raises(ValueError, match="method='lm'.*loss='linear'"):
        run_heston_calibration_benchmark_diagnostics(
            method="lm",
            loss="soft_l1",
            repeat=1,
            warmup=0,
        )


def test_quad_backend_reports_missing_analytic_jacobian() -> None:
    with pytest.raises(NotImplementedError, match="quad backend.*analytic"):
        run_heston_calibration_benchmark_diagnostics(
            backend="quad",
            repeat=1,
            warmup=0,
        )
