from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from option_pricing.diagnostics.heston import (
    HestonBackendComparisonDiagnostics,
    HestonDiagnosticsReport,
    HestonPriceSliceDiagnostics,
    HestonProbabilitySliceDiagnostics,
    compare_backend_slice,
    price_slice_with_diagnostics,
    probability_slice_with_diagnostics,
    run_heston_slice_diagnostics,
)
from option_pricing.models.heston.fourier import P_j_batch_with_diagnostics
from option_pricing.models.heston.params import HestonParams
from option_pricing.numerics.quadrature import QuadratureConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _report_schema() -> dict[str, object]:
    schema_path = (
        _repo_root() / "heston_step1_codex_resource_pack" / "report_schema.json"
    )
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _params() -> HestonParams:
    return HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05)


def _canonical_slice_columns() -> dict[str, object]:
    return {
        "strike": np.array([90.0, 100.0, 110.0], dtype=np.float64),
        "log_moneyness": np.array([0.10536052, 0.0, -0.09531018], dtype=np.float64),
        "price": np.array([12.5, 7.4, 4.0], dtype=np.float64),
        "implied_vol": np.array([0.23, 0.21, 0.22], dtype=np.float64),
        "warning_count": np.array([0, 1, 2], dtype=np.int64),
        "severity": np.array(["ok", "warning", "severe"], dtype=object),
        "tail_fraction": np.array([0.01, 0.06, 0.09], dtype=np.float64),
        "cancellation_ratio": np.array([1.2, 12.0, 20.0], dtype=np.float64),
        "backend_diff": np.array([0.0, 1.0e-4, 2.0e-4], dtype=np.float64),
        "smoothness_flag": np.array([False, None, True], dtype=object),
        "discontinuity_flag": np.array([False, False, True], dtype=object),
    }


def test_run_heston_slice_diagnostics_default_shape_matches_schema() -> None:
    schema = _report_schema()
    required_tables = schema["properties"]["tables"]["required"]
    assert isinstance(required_tables, list)

    report = run_heston_slice_diagnostics()
    payload = report.to_dict()

    assert set(payload.keys()) == {"meta", "tables", "arrays"}
    assert set(payload["tables"].keys()) >= set(required_tables)
    assert callable(probability_slice_with_diagnostics)
    assert callable(price_slice_with_diagnostics)
    assert callable(compare_backend_slice)
    assert callable(run_heston_slice_diagnostics)

    assert payload["tables"]["summary"]["columns"] == [
        "metric",
        "value",
        "notes",
        "severity",
    ]
    assert payload["tables"]["slice"]["columns"] == [
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

    summary_metrics = [row["metric"] for row in payload["tables"]["summary"]["records"]]
    assert summary_metrics == [
        "worst_strike",
        "total_warnings",
        "max_tail_fraction",
        "max_cancellation_ratio",
        "max_backend_discrepancy",
        "suspicious_strike_count",
    ]


def test_price_slice_with_diagnostics_freezes_canonical_columns() -> None:
    artifact = price_slice_with_diagnostics(
        slice_columns=_canonical_slice_columns(),
        meta={"source": "test"},
        arrays={"strike_grid": np.array([90.0, 100.0, 110.0], dtype=np.float64)},
    )

    assert isinstance(artifact, HestonPriceSliceDiagnostics)
    assert list(artifact.table.columns) == [
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
    assert artifact.meta["source"] == "test"
    np.testing.assert_allclose(
        artifact.arrays["strike_grid"],
        np.array([90.0, 100.0, 110.0]),
    )

    with pytest.raises(ValueError, match="canonical slice columns"):
        price_slice_with_diagnostics(
            slice_columns={
                key: value
                for key, value in _canonical_slice_columns().items()
                if key != "backend_diff"
            }
        )


def test_probability_slice_with_diagnostics_packages_existing_fourier_batch_output() -> (
    None
):
    x = np.array([-0.15, 0.0, 0.15], dtype=np.float64)
    strike = np.array([115.0, 100.0, 87.0], dtype=np.float64)
    diag = P_j_batch_with_diagnostics(
        x=x,
        tau=0.75,
        params=_params(),
        j=1,
        backend="gauss_legendre",
        quad_cfg=QuadratureConfig(u_max=120.0, n_panels=12, nodes_per_panel=12),
    )

    artifact = probability_slice_with_diagnostics(
        probability_diagnostics=diag,
        strike=strike,
    )

    assert isinstance(artifact, HestonProbabilitySliceDiagnostics)
    assert artifact.meta["backend"] == "gauss_legendre"
    assert artifact.meta["probability_index"] == 1
    assert "probability" in artifact.table.columns
    assert "warning_count" in artifact.table.columns
    assert "strike" in artifact.table.columns
    assert "panel_contribs" in artifact.arrays
    assert "panel_reason" in artifact.arrays
    np.testing.assert_allclose(artifact.table["log_moneyness"], x)
    np.testing.assert_allclose(artifact.table["strike"], strike)


def test_report_roundtrip_serialization_keeps_step1_contract() -> None:
    price_artifact = price_slice_with_diagnostics(
        slice_columns=_canonical_slice_columns()
    )
    backend_artifact = compare_backend_slice(
        comparison_columns={
            "strike": np.array([90.0, 100.0, 110.0], dtype=np.float64),
            "backend_a": np.array(["gauss_legendre"] * 3, dtype=object),
            "backend_b": np.array(["quad"] * 3, dtype=object),
            "price_diff": np.array([0.0, 1.0e-4, 2.0e-4], dtype=np.float64),
        },
        meta={"purpose": "price-diff comparison"},
    )
    probability_artifact = probability_slice_with_diagnostics(
        probability_columns={
            "log_moneyness": np.array([-0.1, 0.0, 0.1], dtype=np.float64),
            "probability": np.array([0.42, 0.5, 0.61], dtype=np.float64),
            "warning_count": np.array([0, 0, 1], dtype=np.int64),
        }
    )

    report = run_heston_slice_diagnostics(
        meta={"stage": "step1"},
        tables={
            "config_sweep": pd.DataFrame(
                [
                    {
                        "config_label": "balanced",
                        "metric": "status",
                        "value": "placeholder",
                    }
                ]
            )
        },
        arrays={"user_array": np.array([1.0, 2.0, 3.0], dtype=np.float64)},
        probability=probability_artifact,
        price=price_artifact,
        backend_comparison=backend_artifact,
    )

    payload = report.to_dict()
    restored = HestonDiagnosticsReport.from_dict(payload)

    assert isinstance(backend_artifact, HestonBackendComparisonDiagnostics)
    assert isinstance(restored, HestonDiagnosticsReport)
    assert restored.meta["stage"] == "step1"
    assert "probability" in restored.tables
    assert "backend_compare" in restored.tables
    assert "slice" in restored.tables
    assert list(restored.tables["slice"].columns) == list(price_artifact.table.columns)
    np.testing.assert_allclose(
        restored.arrays["user_array"],
        np.array([1.0, 2.0, 3.0]),
    )
