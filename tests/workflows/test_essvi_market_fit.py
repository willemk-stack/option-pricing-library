from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import option_pricing.workflows as workflows
import option_pricing.workflows.surface_fit as surface_fit
from option_pricing.marketdata import prepare_essvi_market_fit
from option_pricing.marketdata.schemas import SURFACE_INPUTS_COLUMNS
from option_pricing.marketdata.surface_ready import (
    PreparedESSVIMarketFit,
    SurfaceReadyStats,
)
from option_pricing.types import MarketData, OptionType
from option_pricing.vol.ssvi import (
    ATMThetaDiagnostics,
    ESSVIFitDiagnostics,
    ESSVIFitResult,
    ESSVIGlobalCalibrationConfig,
    ESSVINodalSurface,
    ESSVINodeSet,
    validate_essvi_nodes,
)
from option_pricing.workflows import (
    ESSVIMarketFitConfig,
    ESSVIMarketFitResult,
    HestonCalibrationConfig,
    HestonMarketFitResult,
    SVIMarketFitConfig,
    SVIMarketFitResult,
    fit_essvi_from_bundle,
    fit_essvi_market,
    fit_heston_market,
    fit_market_model,
    fit_svi_market,
)

ASOF = "2026-05-22T00:00:00Z"


def _market_data() -> MarketData:
    return MarketData(spot=100.0, rate=0.01, dividend_yield=0.0)


def _expiry_from_asof(days: float) -> str:
    timestamp = pd.Timestamp(ASOF).tz_convert("UTC").normalize()
    return (timestamp + pd.Timedelta(days=days)).strftime("%Y-%m-%d")


def _synthetic_nodes() -> ESSVINodeSet:
    return ESSVINodeSet(
        expiries=np.array([30.0 / 365.0, 60.0 / 365.0, 120.0 / 365.0]),
        theta=np.array([0.018, 0.028, 0.045]),
        psi=np.array([0.12, 0.16, 0.20]),
        rho=np.array([-0.35, -0.25, -0.15]),
    )


def _surface_inputs() -> pd.DataFrame:
    market = _market_data()
    nodes = _synthetic_nodes()
    surface = ESSVINodalSurface(nodes)
    y_grid = np.array([-0.30, -0.15, 0.0, 0.15, 0.30], dtype=np.float64)
    records: list[dict[str, object]] = []

    for expiry_index, tau in enumerate(nodes.expiries):
        expiry_days = float(tau) * 365.0
        forward = market.forward(float(tau))
        for point_index, y_value in enumerate(y_grid):
            strike = forward * math.exp(float(y_value))
            right = "call" if y_value >= 0.0 else "put"
            kind = OptionType.CALL if right == "call" else OptionType.PUT
            price = surface.slice(float(tau)).price_at(
                kind=kind,
                strike=strike,
                market=market,
            )
            total_variance = float(surface.w(np.array([y_value]), float(tau))[0])
            records.append(
                {
                    "underlying": "SYNTH",
                    "quote_id": f"essvi-{expiry_index}-{point_index}",
                    "asof": ASOF,
                    "expiry": _expiry_from_asof(expiry_days),
                    "expiry_years": float(tau),
                    "strike": strike,
                    "right": right,
                    "mid": float(np.asarray(price)),
                    "iv": math.sqrt(total_variance / float(tau)),
                    "source": "unit_test",
                    "cleaning_policy": "quote_cleaning_policy.v1",
                }
            )

    return pd.DataFrame(records, columns=SURFACE_INPUTS_COLUMNS)


def _bundle(surface_inputs: pd.DataFrame | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        surface_inputs=_surface_inputs() if surface_inputs is None else surface_inputs,
        market_data=_market_data(),
    )


def _prepared_ready() -> PreparedESSVIMarketFit:
    return prepare_essvi_market_fit(_bundle(), min_expiry_days=1.0)


def _prepared(
    selected_points: pd.DataFrame,
    *,
    status: str,
    warnings: tuple[str, ...] = (),
) -> PreparedESSVIMarketFit:
    selected = selected_points.reset_index(drop=True)
    expiry_days = (
        pd.to_numeric(selected["expiry_days"], errors="coerce").dropna()
        if "expiry_days" in selected
        else pd.Series(dtype=float)
    )
    stats = SurfaceReadyStats(
        input_point_count=int(len(selected)),
        selected_point_count=int(len(selected)),
        rejected_point_count=0,
        rejection_counts={},
        expiry_count=int(selected["T"].nunique()) if "T" in selected else 0,
        min_expiry_days=None if expiry_days.empty else float(expiry_days.min()),
        max_expiry_days=None if expiry_days.empty else float(expiry_days.max()),
        warnings=warnings,
    )
    return PreparedESSVIMarketFit(
        model_name="essvi",
        market_data=_market_data(),
        selected_points=selected,
        rejected_points=pd.DataFrame(),
        stats=stats,
        status=status,
        warnings=warnings,
    )


def _fake_fit_result(prepared: PreparedESSVIMarketFit) -> ESSVIFitResult:
    expiries = np.unique(
        pd.to_numeric(prepared.selected_points["T"], errors="coerce").to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )
    )
    expiries = expiries[np.isfinite(expiries)]
    nodes = ESSVINodeSet(
        expiries=expiries,
        theta=np.linspace(0.02, 0.05, expiries.size, dtype=np.float64),
        psi=np.linspace(0.10, 0.18, expiries.size, dtype=np.float64),
        rho=np.linspace(-0.30, -0.10, expiries.size, dtype=np.float64),
    )
    node_report = validate_essvi_nodes(nodes)
    theta_diag = ATMThetaDiagnostics(
        expiries=expiries,
        theta_raw=nodes.theta,
        theta_isotonic=nodes.theta,
        extraction_methods=tuple("exact" for _ in expiries),
    )
    diag = ESSVIFitDiagnostics(
        success=True,
        status=1,
        message="synthetic success",
        nfev=1,
        cost=0.0,
        x0=np.zeros(3 * expiries.size, dtype=np.float64),
        x_opt=np.zeros(3 * expiries.size, dtype=np.float64),
        theta=theta_diag,
        node_validation=node_report,
        price_rmse=0.0,
        max_abs_price_error=0.0,
        invalid_candidate_count=0,
        last_invalid_reason=None,
    )
    return ESSVIFitResult(nodes=nodes, diag=diag)


def test_fit_essvi_market_happy_path_synthetic_surface() -> None:
    prepared = _prepared_ready()

    result = fit_essvi_market(
        prepared,
        fit_config=ESSVIMarketFitConfig(
            calibration_config=ESSVIGlobalCalibrationConfig(
                max_nfev=2_000,
                strict_validation=True,
            )
        ),
    )

    assert isinstance(result, ESSVIMarketFitResult)
    assert result.status == "ok"
    assert result.fit_result is not None
    assert result.validation is not None
    assert result.validation.ok
    assert result.surface is not None
    assert result.summary["model_name"] == "essvi"
    assert result.summary["calibration_status"] == "ok"
    assert result.summary["selected_point_count"] == len(prepared.selected_points)
    assert result.summary["node_count"] == 3
    assert result.summary["node_validation_ok"] is True
    assert result.summary["surface_noarb_ok"] is True
    assert result.summary["price_rmse"] < 1.0e-3
    pd.testing.assert_frame_equal(result.selected_points, prepared.selected_points)
    pd.testing.assert_frame_equal(result.rejected_points, prepared.rejected_points)


def test_fit_essvi_from_bundle_one_shot_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_ready()
    bundle = SimpleNamespace(name="loaded-bundle")
    calls: list[tuple[str, object, dict[str, object]]] = []

    def fake_load_model_validation_bundle(path: object) -> object:
        calls.append(("load", path, {}))
        return bundle

    def fake_prepare_essvi_market_fit(
        loaded: object,
        **kwargs: object,
    ) -> PreparedESSVIMarketFit:
        calls.append(("prepare", loaded, kwargs))
        return prepared

    def fake_calibrate_essvi_global(**kwargs: object) -> ESSVIFitResult:
        calls.append(("calibrate", kwargs["market"], kwargs))
        return _fake_fit_result(prepared)

    monkeypatch.setattr(
        surface_fit,
        "load_model_validation_bundle",
        fake_load_model_validation_bundle,
    )
    monkeypatch.setattr(
        surface_fit,
        "prepare_essvi_market_fit",
        fake_prepare_essvi_market_fit,
    )
    monkeypatch.setattr(
        surface_fit,
        "calibrate_essvi_global",
        fake_calibrate_essvi_global,
    )

    result = fit_essvi_from_bundle(
        "bundle-root",
        min_expiry_days=3.0,
        min_points_per_expiry=4,
        min_expiry_count=2,
    )

    assert result.status == "ok"
    assert calls[0] == ("load", "bundle-root", {})
    assert calls[1][0] == "prepare"
    assert calls[1][1] is bundle
    assert calls[1][2]["min_expiry_days"] == 3.0
    assert calls[1][2]["min_points_per_expiry"] == 4
    assert calls[1][2]["min_expiry_count"] == 2
    assert calls[2][0] == "calibrate"


def test_empty_prepared_result_skips_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = _prepared(
        _prepared_ready().selected_points.iloc[0:0],
        status="empty",
    )

    def unexpected_calibration(**_kwargs: object) -> ESSVIFitResult:
        raise AssertionError("calibration should not run")

    monkeypatch.setattr(
        surface_fit,
        "calibrate_essvi_global",
        unexpected_calibration,
    )

    result = fit_essvi_market(prepared)

    assert result.status == "empty"
    assert result.fit_result is None
    assert result.validation is None
    assert result.surface is None
    assert result.summary["calibration_status"] == "empty"
    assert result.errors


def test_blocked_prepared_result_skips_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    ready = _prepared_ready()
    prepared = replace(
        ready,
        status="blocked",
        warnings=("forced cross-maturity block",),
    )

    def unexpected_calibration(**_kwargs: object) -> ESSVIFitResult:
        raise AssertionError("calibration should not run")

    monkeypatch.setattr(
        surface_fit,
        "calibrate_essvi_global",
        unexpected_calibration,
    )

    result = fit_essvi_market(prepared)

    assert result.status == "blocked"
    assert result.fit_result is None
    assert result.validation is None
    assert result.surface is None
    assert "forced cross-maturity block" in result.warnings
    assert result.errors


def test_failure_returns_failed_unless_raise_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_ready()

    def fail_calibration(**_kwargs: object) -> ESSVIFitResult:
        raise RuntimeError("synthetic eSSVI calibration failure")

    monkeypatch.setattr(surface_fit, "calibrate_essvi_global", fail_calibration)

    result = fit_essvi_market(prepared)

    assert result.status == "failed"
    assert result.fit_result is None
    assert result.surface is None
    assert "synthetic eSSVI calibration failure" in result.errors[0]

    with pytest.raises(RuntimeError, match="synthetic eSSVI calibration failure"):
        fit_essvi_market(prepared, raise_on_failure=True)


def test_fit_essvi_market_requires_prepared_helper_guidance() -> None:
    with pytest.raises(TypeError) as exc_info:
        fit_essvi_market(_bundle())  # type: ignore[arg-type]

    message = str(exc_info.value)
    assert "PreparedESSVIMarketFit" in message
    assert "prepare_essvi_market_fit(bundle)" in message
    assert "fit_essvi_from_bundle(path)" in message


def test_public_workflow_exports_expose_essvi_helpers() -> None:
    assert "ESSVIMarketFitConfig" in workflows.__all__
    assert "ESSVIMarketFitResult" in workflows.__all__
    assert "fit_essvi_market" in workflows.__all__
    assert "fit_essvi_from_bundle" in workflows.__all__
    assert "fit_market_model" in workflows.__all__
    assert workflows.ESSVIMarketFitConfig is ESSVIMarketFitConfig
    assert workflows.ESSVIMarketFitResult is ESSVIMarketFitResult
    assert workflows.fit_essvi_market is fit_essvi_market
    assert workflows.fit_essvi_from_bundle is fit_essvi_from_bundle
    assert workflows.fit_market_model is fit_market_model


def test_svi_and_heston_workflow_imports_still_work() -> None:
    assert workflows.SVIMarketFitConfig is SVIMarketFitConfig
    assert workflows.SVIMarketFitResult is SVIMarketFitResult
    assert workflows.fit_svi_market is fit_svi_market
    assert workflows.HestonCalibrationConfig is HestonCalibrationConfig
    assert workflows.HestonMarketFitResult is HestonMarketFitResult
    assert workflows.fit_heston_market is fit_heston_market
