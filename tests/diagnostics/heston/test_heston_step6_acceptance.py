from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest

import option_pricing.diagnostics.heston as public_heston
from option_pricing.diagnostics.heston import (
    HestonDiagnosticsReport,
    compare_backend_slice,
    price_slice_with_diagnostics,
    probability_slice_with_diagnostics,
    run_heston_slice_diagnostics,
)
from option_pricing.diagnostics.heston.contracts import (
    HESTON_REQUIRED_REPORT_TABLES,
    HESTON_SEVERITY_ORDER,
    HESTON_SLICE_TABLE_COLUMNS,
)
from option_pricing.diagnostics.heston.integration import integration_diagnostics
from option_pricing.diagnostics.heston.plot import (
    plot_backend_difference_by_strike,
    plot_config_sweep,
    plot_panel_contributions,
    plot_smile_with_warning_overlay,
    plot_tail_fraction_by_strike,
)

from ._step6_policy import (
    PROVISIONAL_GAUSS_LEGENDRE_CFG,
    PROVISIONAL_MARKET,
    PROVISIONAL_STRESS_CASES,
    PROVISIONAL_STRIKE_GRID,
    STEP6_OWNER_APPROVAL_NOTE,
    build_pricing_report,
    provisional_stress_case,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_SEVERITY_RANK = {
    severity: rank for rank, severity in enumerate(HESTON_SEVERITY_ORDER, start=0)
}


def _summary_value(report: HestonDiagnosticsReport, metric: str) -> Any:
    summary = report.tables["summary"]
    row = summary.loc[summary["metric"] == metric]
    assert not row.empty, f"Missing summary metric: {metric!r}."
    return row.iloc[0]["value"]


def _severity_rank(value: Any) -> int:
    return _SEVERITY_RANK.get(str(value), -1)


def _assert_plain_runtime_payload(value: Any) -> None:
    module = type(value).__module__
    disallowed_modules = ("matplotlib", "IPython", "ipykernel", "ipywidgets")

    if module.startswith(disallowed_modules):
        raise AssertionError(f"Unexpected notebook/display object from {module!r}.")

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            for item in value.tolist():
                _assert_plain_runtime_payload(item)
        return

    if isinstance(value, Mapping):
        for item in value.values():
            _assert_plain_runtime_payload(item)
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _assert_plain_runtime_payload(item)


def _assert_sorted_by_severity(
    table: pd.DataFrame, *, column: str = "severity"
) -> None:
    if table.empty or column not in table.columns:
        return

    ranks = [_severity_rank(value) for value in table[column].astype(str)]
    assert ranks == sorted(ranks, reverse=True)


def _count_true(values: pd.Series) -> int:
    normalized = values.fillna(False).astype(bool)
    return int(normalized.sum())


def test_step6_public_api_surface_exports_contract_entrypoints() -> None:
    # Step 1 freezes the packager surface, while Step 3 adds the computed
    # one-call entrypoint. Step 6 acceptance coverage keeps both explicit.
    expected_callables = {
        "probability_slice_with_diagnostics",
        "price_slice_with_diagnostics",
        "compare_backend_slice",
        "run_heston_slice_diagnostics",
        "run_heston_pricing_diagnostics",
    }

    assert "Owner review required" in STEP6_OWNER_APPROVAL_NOTE
    assert expected_callables <= set(public_heston.__all__)
    for name in expected_callables:
        assert callable(getattr(public_heston, name))


def test_step6_public_entrypoints_return_notebook_usable_artifacts() -> None:
    case = provisional_stress_case("normal_case")
    report = build_pricing_report(case)

    slice_table = report.tables["slice"]
    probability_bundle = integration_diagnostics(
        x=slice_table["log_moneyness"].to_numpy(dtype=np.float64),
        tau=case.tau,
        params=case.params,
        j=1,
        backend="gauss_legendre",
        quad_cfg=PROVISIONAL_GAUSS_LEGENDRE_CFG,
        strike=slice_table["strike"].to_numpy(dtype=np.float64),
    )
    probability = probability_slice_with_diagnostics(
        probability_table=probability_bundle.probability.table,
        meta=probability_bundle.probability.meta,
        arrays=probability_bundle.probability.arrays,
    )
    price = price_slice_with_diagnostics(
        slice_table=slice_table,
        meta=report.meta["price"],
        arrays=report.arrays["slice"],
    )
    backend = compare_backend_slice(
        comparison_table=report.tables["backend_compare"],
        meta=report.meta["backend_comparison"],
        arrays=report.arrays["backend_compare"],
    )
    packaged = run_heston_slice_diagnostics(
        meta={"acceptance_case": case.label},
        tables={
            "summary": report.tables["summary"],
            "worst_strikes": report.tables["worst_strikes"],
            "config_sweep": report.tables["config_sweep"],
            "worst_panels_p0": report.tables["worst_panels_p0"],
            "worst_panels_p1": report.tables["worst_panels_p1"],
        },
        arrays=report.arrays,
        probability=probability,
        price=price,
        backend_comparison=backend,
    )

    assert isinstance(probability.table, pd.DataFrame)
    assert isinstance(price.table, pd.DataFrame)
    assert isinstance(backend.table, pd.DataFrame)
    assert set(packaged.to_dict()) == {"meta", "tables", "arrays"}
    assert {
        "summary",
        "slice",
        "backend_compare",
        "config_sweep",
        "worst_panels_p0",
        "worst_panels_p1",
        "probability",
    } <= set(packaged.tables)
    json.dumps(packaged.to_dict())


def test_step6_actual_report_round_trip_is_plain_data_friendly() -> None:
    report = build_pricing_report(provisional_stress_case("high_vol_of_vol"))
    payload = report.to_dict()
    restored = HestonDiagnosticsReport.from_dict(payload)

    for table in report.tables.values():
        assert isinstance(table, pd.DataFrame)
    _assert_plain_runtime_payload(report.meta)
    _assert_plain_runtime_payload(report.arrays)
    _assert_plain_runtime_payload(payload)
    json.dumps(payload)

    assert set(restored.tables) == set(HESTON_REQUIRED_REPORT_TABLES)
    assert list(
        restored.tables["slice"].columns[: len(HESTON_SLICE_TABLE_COLUMNS)]
    ) == list(HESTON_SLICE_TABLE_COLUMNS)
    np.testing.assert_allclose(
        restored.arrays["slice"]["config_price_span"],
        report.arrays["slice"]["config_price_span"],
    )
    np.testing.assert_allclose(
        restored.arrays["backend_compare"]["abs_price_diff"],
        report.arrays["backend_compare"]["abs_price_diff"],
    )


@pytest.mark.parametrize(
    "case",
    PROVISIONAL_STRESS_CASES,
    ids=lambda case: case.label,
)
def test_step6_stressed_regime_reports_preserve_contract_and_summary_consistency(
    case,
) -> None:
    report = build_pricing_report(case)
    slice_table = report.tables["slice"]
    backend_compare = report.tables["backend_compare"]
    worst_strikes = report.tables["worst_strikes"]
    config_sweep = report.tables["config_sweep"]
    perturbation_table = report.arrays["parameter_perturbation_table"]

    assert set(report.tables) == set(HESTON_REQUIRED_REPORT_TABLES)
    assert list(slice_table.columns[: len(HESTON_SLICE_TABLE_COLUMNS)]) == list(
        HESTON_SLICE_TABLE_COLUMNS
    )
    assert len(slice_table) == len(PROVISIONAL_STRIKE_GRID)
    assert len(backend_compare) == len(slice_table)
    assert not config_sweep.empty
    assert not report.tables["worst_panels_p0"].empty
    assert not report.tables["worst_panels_p1"].empty

    np.testing.assert_allclose(
        backend_compare["abs_price_diff"].to_numpy(dtype=np.float64),
        np.abs(backend_compare["price_diff"].to_numpy(dtype=np.float64)),
    )
    np.testing.assert_allclose(
        slice_table["backend_diff"].to_numpy(dtype=np.float64),
        backend_compare["abs_price_diff"].to_numpy(dtype=np.float64),
    )
    assert np.isfinite(slice_table["price"].to_numpy(dtype=np.float64)).all()

    assert {
        "probability_p0",
        "probability_p1",
        "comparison_probability_p0",
        "comparison_probability_p1",
        "config_sweep_prices",
        "parameter_perturbation_prices",
        "parameter_perturbation_table",
        "slice",
        "backend_compare",
    } <= set(report.arrays)
    assert len(report.arrays["slice"]["config_price_span"]) == len(slice_table)
    assert len(report.arrays["backend_compare"]["abs_price_diff"]) == len(slice_table)

    assert int(_summary_value(report, "total_warnings")) == int(
        slice_table["warning_count"].sum()
    )
    assert int(_summary_value(report, "suspicious_strike_count")) == _count_true(
        slice_table["suspicious_flag"]
    )
    assert float(_summary_value(report, "max_backend_discrepancy")) == pytest.approx(
        float(backend_compare["abs_price_diff"].max())
    )

    if not worst_strikes.empty:
        assert float(_summary_value(report, "worst_strike")) == pytest.approx(
            float(worst_strikes.iloc[0]["strike"])
        )
        if _count_true(slice_table["suspicious_flag"]) > 0:
            assert bool(worst_strikes.iloc[0]["suspicious_flag"])

    assert (
        report.meta["config_sweep_labels"]
        == config_sweep["config_label"].astype(str).tolist()
    )
    assert set(report.meta["config_sweep_labels"]) == set(
        report.arrays["config_sweep_prices"]
    )
    assert [str(label) for label in perturbation_table["label"]] == [
        item["label"] for item in report.meta["parameter_perturbations"]
    ]

    if case.expect_any_non_ok_severity:
        assert (slice_table["severity"].astype(str) != "ok").any()
    if case.expect_any_suspicious:
        assert _count_true(slice_table["suspicious_flag"]) > 0


def test_step6_provisional_policy_and_owner_review_notes_stay_explicit() -> None:
    report = build_pricing_report(provisional_stress_case("normal_case"))
    policy = report.meta["provisional_policy"]
    suspicious_row = (
        report.tables["summary"]
        .loc[report.tables["summary"]["metric"] == "suspicious_strike_count"]
        .iloc[0]
    )
    perturbation_notes = [
        str(note).lower()
        for note in report.arrays["parameter_perturbation_table"]["notes"]
    ]

    assert {
        "backend_price_diff_abs",
        "smoothness_linear_residual",
        "discontinuity_local_jump",
        "config_price_span_abs",
        "perturbation_relative_price_change",
    } <= set(policy)
    assert report.meta["price"]["policy"] == "provisional_owner_approval_required"
    assert report.meta["backend_comparison"]["primary_metric"] == "price difference"
    assert "approval required" in str(suspicious_row["notes"]).lower()
    assert all("owner approval required" in note for note in perturbation_notes)


@pytest.mark.parametrize(
    ("backend", "expect_panel_detail"),
    [("gauss_legendre", True), ("quad", False)],
)
def test_step6_integration_wrappers_remain_readable_across_supported_backends(
    backend: str,
    expect_panel_detail: bool,
) -> None:
    case = provisional_stress_case("high_vol_of_vol")
    forward = float(PROVISIONAL_MARKET.to_context().fwd(case.tau))
    strike = PROVISIONAL_STRIKE_GRID[1:4]
    x = np.log(forward / strike)

    bundle = integration_diagnostics(
        x=x,
        tau=case.tau,
        params=case.params,
        j=0,
        backend=backend,
        quad_cfg=(
            PROVISIONAL_GAUSS_LEGENDRE_CFG if backend == "gauss_legendre" else None
        ),
        strike=strike,
    )

    assert {
        "panels",
        "warning_summary",
        "worst_panels",
        "reason_counts",
    } == set(bundle.tables)
    assert {"warning_label", "severity", "count"} <= set(
        bundle.tables["warning_summary"].columns
    )
    assert {"reason_label", "severity", "count"} <= set(
        bundle.tables["reason_counts"].columns
    )
    assert {"rank", "reason_labels", "severity", "detail_available"} <= set(
        bundle.tables["worst_panels"].columns
    )

    assert set(bundle.tables["warning_summary"]["severity"]).issubset(
        set(HESTON_SEVERITY_ORDER)
    )
    assert set(bundle.tables["reason_counts"]["severity"]).issubset(
        set(HESTON_SEVERITY_ORDER)
    )
    _assert_sorted_by_severity(bundle.tables["warning_summary"])
    _assert_sorted_by_severity(bundle.tables["reason_counts"])
    assert (
        bundle.tables["warning_summary"]["warning_label"]
        .astype(str)
        .str.len()
        .gt(0)
        .all()
    )
    assert (
        bundle.tables["reason_counts"]["reason_label"].astype(str).str.len().gt(0).all()
    )

    panels = bundle.tables["panels"]
    if expect_panel_detail:
        assert bool(panels["detail_available"].any())
        assert bundle.meta["panel_detail_available"] is True
    else:
        assert not bool(panels["detail_available"].any())
        assert panels["notes"].astype(str).str.contains("unavailable", case=False).all()
        assert bundle.meta["panel_detail_available"] is False


def test_step6_plot_helpers_accept_actual_reports_and_sparse_optional_arrays() -> None:
    report = build_pricing_report(provisional_stress_case("normal_case"))
    sparse_payload = report.to_dict()
    sparse_payload["arrays"]["probability_p1"] = {
        "log_moneyness": sparse_payload["arrays"]["probability_p1"]["log_moneyness"],
    }
    sparse_payload["arrays"].pop("config_sweep_prices", None)
    sparse_report = HestonDiagnosticsReport.from_dict(sparse_payload)

    figures: list[plt.Figure] = []
    try:
        fig_tail, ax_tail = plot_tail_fraction_by_strike(report)
        figures.append(fig_tail)
        fig_backend, ax_backend = plot_backend_difference_by_strike(report)
        figures.append(fig_backend)
        fig_smile, ax_smile = plot_smile_with_warning_overlay(report)
        figures.append(fig_smile)
        fig_panel, ax_panel = plot_panel_contributions(
            sparse_report,
            probability_key="probability_p1",
        )
        figures.append(fig_panel)
        fig_sweep, ax_sweep = plot_config_sweep(sparse_report)
        figures.append(fig_sweep)

        np.testing.assert_allclose(
            ax_tail.lines[-1].get_ydata(),
            report.tables["slice"]["tail_fraction"].to_numpy(dtype=np.float64),
        )
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

        panel_text = " ".join(text.get_text() for text in ax_panel.texts)
        sweep_text = " ".join(text.get_text() for text in ax_sweep.texts)
        assert "unavailable" in panel_text.lower()
        assert "summary table only" in sweep_text.lower()
        assert len(ax_sweep.patches) == len(report.tables["config_sweep"])
    finally:
        for figure in figures:
            plt.close(figure)
