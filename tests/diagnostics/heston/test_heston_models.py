from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from option_pricing.diagnostics.heston.contracts import HESTON_SUMMARY_METRICS
from option_pricing.diagnostics.heston.integration import (
    probability_slice_with_diagnostics,
)
from option_pricing.diagnostics.heston.models import (
    HestonDiagnosticsReport,
    HestonIntegrationDiagnosticsBundle,
    HestonProbabilitySliceDiagnostics,
)
from option_pricing.diagnostics.heston.pricing import price_slice_with_diagnostics
from option_pricing.diagnostics.heston.report import run_heston_slice_diagnostics


def _summary_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": np.array(HESTON_SUMMARY_METRICS, dtype=object),
            "value": np.array([100.0, 2, 0.08, 12.0, 2.0e-4, 1], dtype=object),
            "notes": np.array([""] * len(HESTON_SUMMARY_METRICS), dtype=object),
            "severity": np.array(
                ["ok", "warning", "warning", "warning", "warning", "warning"],
                dtype=object,
            ),
        }
    )


def _slice_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "extra_column": np.array(["a", "b", "c"], dtype=object),
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
        }
    )


def test_probability_slice_artifact_roundtrip_preserves_meta_table_and_arrays() -> None:
    artifact = probability_slice_with_diagnostics(
        probability_columns={
            "log_moneyness": np.array([-0.1, 0.0, 0.1], dtype=np.float64),
            "probability": np.array([0.42, 0.50, 0.61], dtype=np.float64),
            "warning_count": np.array([0, 0, 1], dtype=np.int64),
        },
        meta={"backend": "gauss_legendre", "probability_index": 1},
        arrays={"panel_contribs": np.array([[0.1, 0.2], [0.2, 0.1], [0.15, 0.05]])},
    )

    restored = HestonProbabilitySliceDiagnostics.from_dict(artifact.to_dict())

    assert restored.meta["backend"] == "gauss_legendre"
    assert restored.meta["probability_index"] == 1
    np.testing.assert_allclose(
        restored.table["probability"].to_numpy(dtype=np.float64),
        np.array([0.42, 0.50, 0.61], dtype=np.float64),
    )
    np.testing.assert_allclose(
        restored.arrays["panel_contribs"],
        np.array([[0.1, 0.2], [0.2, 0.1], [0.15, 0.05]], dtype=np.float64),
    )


def test_price_slice_artifact_reorders_canonical_columns_and_keeps_extra_columns() -> (
    None
):
    artifact = price_slice_with_diagnostics(slice_table=_slice_table())

    assert list(artifact.table.columns[:11]) == [
        "strike",
        "log_moneyness",
        "price",
        "implied_vol",
        "warning_count",
        "severity",
        "tail_fraction",
        "cancellation_ratio",
        "backend_diff",
        "smoothness_flag",
        "discontinuity_flag",
    ]
    assert artifact.table.columns[-1] == "extra_column"


def test_heston_integration_bundle_requires_standard_tables() -> None:
    probability = probability_slice_with_diagnostics(
        probability_columns={
            "log_moneyness": np.array([0.0], dtype=np.float64),
            "probability": np.array([0.5], dtype=np.float64),
            "warning_count": np.array([0], dtype=np.int64),
        }
    )

    with pytest.raises(ValueError, match="missing required table"):
        HestonIntegrationDiagnosticsBundle(
            meta={},
            probability=probability,
            tables={
                "panels": pd.DataFrame(),
                "warning_summary": pd.DataFrame(),
            },
            arrays={},
        )


def test_heston_report_from_dict_requires_exact_top_level_keys() -> None:
    with pytest.raises(ValueError, match="exactly 'meta', 'tables', and 'arrays'"):
        HestonDiagnosticsReport.from_dict(
            {
                "meta": {},
                "tables": {},
                "arrays": {},
                "extra": {},
            }
        )


def test_heston_report_validates_slice_severity_values() -> None:
    report = run_heston_slice_diagnostics()
    tables = {name: table.copy() for name, table in report.tables.items()}
    tables["summary"] = _summary_table()
    tables["slice"] = _slice_table()
    tables["slice"].loc[0, "severity"] = "not-a-valid-severity"

    with pytest.raises(ValueError, match="unsupported severity values"):
        HestonDiagnosticsReport(meta={}, tables=tables, arrays={})
