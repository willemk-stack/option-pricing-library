from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

import option_pricing.workflows as workflows
import option_pricing.workflows.market_fit as market_fit
from option_pricing.marketdata.bundles import (
    HestonSmokeResult,
    LoadedModelValidationBundle,
    build_model_validation_manifest,
    load_model_validation_bundle,
)
from option_pricing.marketdata.gold import GoldMarketDataSnapshot
from option_pricing.marketdata.model_ready import (
    PreparedHestonMarketFit,
    prepare_heston_market_fit,
)
from option_pricing.marketdata.schemas import DatasetName
from option_pricing.marketdata.validation import coerce_frame
from option_pricing.models.heston.calibration import (
    HestonCalibrationBounds,
    HestonCalibrationRun,
    HestonMultistartResult,
)
from option_pricing.models.heston.calibration.preflight import HestonQuotePreflight
from option_pricing.models.heston.params import HestonParams
from option_pricing.types import MarketData
from option_pricing.workflows import (
    HestonCalibrationConfig,
    HestonMarketFitError,
    HestonMarketFitResult,
    fit_heston_from_bundle,
    fit_heston_market,
)

ASOF = "2026-05-22T15:30:00Z"
EXPECTED_ARTIFACTS = {
    "market_data": "market_data.json",
    "cleaned_quotes": "cleaned_quotes.parquet",
    "rejected_quotes": "rejected_quotes.parquet",
    "heston_quotes": "heston_quotes.parquet",
    "surface_inputs": "surface_inputs.parquet",
    "heston_fit_summary": "heston_fit_summary.csv",
    "warnings": "warnings.json",
}


@pytest.fixture
def fake_parquet(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_to_parquet(
        self: pd.DataFrame,
        path: str | Path,
        compression: str | None = None,
        index: bool = False,
    ) -> None:
        del compression
        payload = self if index else self.reset_index(drop=True)
        payload.to_pickle(path)

    def _fake_read_parquet(
        path: str | Path,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        frame = cast(pd.DataFrame, pd.read_pickle(path))
        if columns is None:
            return frame
        return cast(pd.DataFrame, frame.loc[:, columns])

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet)
    monkeypatch.setattr(pd, "read_parquet", _fake_read_parquet)


def _market_data() -> MarketData:
    return MarketData(spot=100.0, rate=0.02, dividend_yield=0.0)


def _market_data_payload() -> dict[str, object]:
    return {
        "schema_version": "gold_market_data.v1",
        "underlying": "SYNTH",
        "valuation_timestamp_utc": ASOF,
        "run_id": "test-run",
        "snapshot_id": "snapshot-001",
        "market_data": {
            "spot": 100.0,
            "rate": 0.02,
            "dividend_yield": 0.0,
        },
        "sources": {
            "spot_source": "local_fixture",
            "rate_source": "local_fixture",
            "dividend_yield_source": "assumption",
        },
        "rate_compounding": "continuous",
        "day_count": "ACT/365",
        "quote_cleaning_policy": "quote_cleaning_policy.v1",
        "library_commit": "abc123",
    }


def _manifest() -> dict[str, object]:
    return build_model_validation_manifest(
        run_id="test-run",
        snapshot_id="snapshot-001",
        underlying="SYNTH",
        valuation_timestamp_utc=ASOF,
        market_data_payload=_market_data_payload(),
        rows={
            "market_inputs": 1,
            "cleaned_quotes": 3,
            "rejected_quotes": 0,
            "heston_quotes": 3,
            "surface_inputs": 3,
        },
        reason_counts={},
        warnings=[],
        artifacts=EXPECTED_ARTIFACTS,
        heston_smoke=HestonSmokeResult(
            status="skipped",
            message="not run in workflow test",
            objective_type="price_rmse",
            quote_count=3,
        ),
        library_commit="abc123",
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _heston_quotes(
    row_overrides: dict[int, dict[str, object]] | None = None,
) -> pd.DataFrame:
    records: list[dict[str, object]] = [
        {
            "underlying": "SYNTH",
            "contract_symbol": "SYNTH260621C00095000",
            "quote_id": "quote-call-95",
            "asof": ASOF,
            "expiry": "2026-06-21",
            "expiry_years": 30.0 / 365.0,
            "strike": 95.0,
            "right": "call",
            "mid": 7.0,
            "bid": 6.9,
            "ask": 7.1,
            "iv": 0.25,
            "vega": 0.08,
            "option_type": "call",
            "label": "SYNTH260621C00095000",
            "source": "unit_test",
            "cleaning_policy": "quote_cleaning_policy.v1",
        },
        {
            "underlying": "SYNTH",
            "contract_symbol": "SYNTH260721P00105000",
            "quote_id": "quote-put-105",
            "asof": ASOF,
            "expiry": "2026-07-21",
            "expiry_years": 60.0 / 365.0,
            "strike": 105.0,
            "right": "put",
            "mid": 6.5,
            "bid": 6.4,
            "ask": 6.6,
            "iv": 0.28,
            "vega": 0.09,
            "option_type": "put",
            "label": "SYNTH260721P00105000",
            "source": "unit_test",
            "cleaning_policy": "quote_cleaning_policy.v1",
        },
        {
            "underlying": "SYNTH",
            "contract_symbol": "SYNTH260820C00100000",
            "quote_id": "quote-call-100",
            "asof": ASOF,
            "expiry": "2026-08-20",
            "expiry_years": 90.0 / 365.0,
            "strike": 100.0,
            "right": "call",
            "mid": 4.2,
            "bid": 4.1,
            "ask": 4.3,
            "iv": 0.24,
            "vega": 0.07,
            "option_type": "call",
            "label": "SYNTH260820C00100000",
            "source": "unit_test",
            "cleaning_policy": "quote_cleaning_policy.v1",
        },
    ]
    for row_index, overrides in (row_overrides or {}).items():
        records[row_index].update(overrides)
    return coerce_frame(
        pd.DataFrame(records),
        DatasetName.HESTON_QUOTES,
        allow_extra=False,
    )


def _write_bundle(root: Path) -> Path:
    root.mkdir(parents=True)
    _write_json(root / "manifest.json", _manifest())
    _write_json(root / "market_data.json", _market_data_payload())
    _write_json(root / "warnings.json", {"warnings": []})
    quotes = _heston_quotes()
    quotes.to_parquet(root / "heston_quotes.parquet", index=False)
    quotes.to_parquet(root / "cleaned_quotes.parquet", index=False)
    quotes.iloc[0:0].to_parquet(root / "rejected_quotes.parquet", index=False)
    quotes.to_parquet(root / "surface_inputs.parquet", index=False)
    pd.DataFrame({"status": ["skipped"]}).to_csv(
        root / "heston_fit_summary.csv",
        index=False,
    )
    return root


def _loaded_bundle(
    tmp_path: Path, quotes: pd.DataFrame | None = None
) -> LoadedModelValidationBundle:
    market_data = _market_data()
    return LoadedModelValidationBundle(
        root=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        manifest={"artifact_schema_version": "model_validation_bundle.v1"},
        warnings={"warnings": []},
        market_snapshot=GoldMarketDataSnapshot(market_data=market_data, metadata={}),
        market_data=market_data,
        cleaned_quotes=pd.DataFrame(),
        rejected_quotes=pd.DataFrame(),
        heston_quotes=_heston_quotes() if quotes is None else quotes,
        surface_inputs=pd.DataFrame(),
        heston_fit_summary=pd.DataFrame(),
    )


def _prepared(
    tmp_path: Path, *, objective_type: str = "price_rmse"
) -> PreparedHestonMarketFit:
    return prepare_heston_market_fit(
        _loaded_bundle(tmp_path),
        objective_type=objective_type,
    )


def _heston_params() -> HestonParams:
    return HestonParams(kappa=1.4, vbar=0.045, eta=0.55, rho=-0.35, v=0.04)


def _fake_multistart_result(
    *,
    quote_count: int,
    objective_type: str = "price_rmse",
    cost: float = 0.125,
) -> HestonMultistartResult:
    params = _heston_params()
    best_run = HestonCalibrationRun(
        seed_index=0,
        seed_params=params,
        fitted_params=params,
        success=True,
        cost=cost,
        optimality=1.0e-8,
        nfev=1,
        njev=1,
        status=1,
        message="synthetic success",
        raw_x=None,
    )
    return HestonMultistartResult(
        best_params=params,
        best_run=best_run,
        runs=(best_run,),
        objective_type=objective_type,  # type: ignore[arg-type]
        parameter_transform="bounded",
        backend="gauss_legendre",
        quote_count=quote_count,
        success_count=1,
        failure_count=0,
        jacobian_mode="analytic",
        analytic_jacobian_eta_min=None,
    )


def _patch_successful_calibration(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[dict[str, object]],
) -> None:
    def fake_calibrate_heston_multistart(
        quotes: object,
        **kwargs: object,
    ) -> HestonMultistartResult:
        calls.append({"quotes": quotes, **kwargs})
        quote_count = int(quotes.n_quotes)
        objective_type = str(kwargs["objective_type"])
        return _fake_multistart_result(
            quote_count=quote_count,
            objective_type=objective_type,
        )

    monkeypatch.setattr(
        market_fit,
        "calibrate_heston_multistart",
        fake_calibrate_heston_multistart,
    )


def test_fit_heston_from_bundle_accepts_bundle_root_path(
    tmp_path: Path,
    fake_parquet: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _patch_successful_calibration(monkeypatch, calls)
    root = _write_bundle(tmp_path / "bundle")

    result = fit_heston_from_bundle(root)

    assert result.status == "ok"
    assert result.prepared.objective_type == "price_rmse"
    assert len(calls) == 1


def test_workflow_public_imports_expose_canonical_heston_helpers() -> None:
    assert tuple(workflows.__all__) == (
        "HestonCalibrationConfig",
        "HestonMarketFitError",
        "HestonMarketFitResult",
        "SVIMarketFitConfig",
        "SVIMarketFitResult",
        "SVISliceMarketFitResult",
        "fit_heston_from_bundle",
        "fit_heston_market",
        "fit_svi_from_bundle",
        "fit_svi_market",
    )
    assert workflows.HestonCalibrationConfig is HestonCalibrationConfig
    assert workflows.HestonMarketFitError is HestonMarketFitError
    assert workflows.HestonMarketFitResult is HestonMarketFitResult
    assert workflows.fit_heston_from_bundle is fit_heston_from_bundle
    assert workflows.fit_heston_market is fit_heston_market


def test_loaded_prepared_and_fit_summary_counts_stay_coherent(
    tmp_path: Path,
    fake_parquet: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _patch_successful_calibration(monkeypatch, calls)
    root = _write_bundle(tmp_path / "bundle")
    quotes = _heston_quotes({0: {"iv": pd.NA}})
    quotes.to_parquet(root / "heston_quotes.parquet", index=False)

    loaded = load_model_validation_bundle(root)
    prepared = prepare_heston_market_fit(loaded)
    ladder_result = fit_heston_market(prepared)
    one_shot_result = fit_heston_from_bundle(root)

    assert prepared.stats.input_quote_count == len(loaded.heston_quotes) == 3
    assert prepared.stats.selected_quote_count == 2
    assert prepared.stats.rejected_quote_count == 1
    assert len(prepared.selected_quotes) + len(prepared.rejected_quotes) == len(
        loaded.heston_quotes
    )
    assert ladder_result.prepared is prepared
    assert ladder_result.summary["input_quote_count"] == 3
    assert ladder_result.summary["selected_quote_count"] == 2
    assert ladder_result.summary["rejected_quote_count"] == 1
    assert ladder_result.summary["calibration_quote_count"] == 2
    assert one_shot_result.status == ladder_result.status == "ok"
    assert one_shot_result.summary["selected_quote_count"] == 2
    assert calls[0]["quotes"] is prepared.quote_set
    assert calls[0]["quotes"].n_quotes == 2


def test_fit_heston_from_bundle_accepts_manifest_path(
    tmp_path: Path,
    fake_parquet: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _patch_successful_calibration(monkeypatch, calls)
    root = _write_bundle(tmp_path / "bundle")

    result = fit_heston_from_bundle(root / "manifest.json")

    assert result.status == "ok"
    assert result.prepared.objective_type == "price_rmse"
    assert len(calls) == 1


def test_fit_heston_from_bundle_accepts_loaded_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _patch_successful_calibration(monkeypatch, calls)
    bundle = _loaded_bundle(tmp_path)

    result = fit_heston_from_bundle(bundle)

    assert result.status == "ok"
    assert result.prepared.market_data is bundle.market_data
    assert len(calls) == 1


def test_fit_heston_from_bundle_calls_preparation_with_default_price_rmse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _loaded_bundle(tmp_path)
    prepared = _prepared(tmp_path)
    seen_objectives: list[object] = []

    def fake_prepare_heston_market_fit(
        loaded: LoadedModelValidationBundle,
        **kwargs: object,
    ) -> PreparedHestonMarketFit:
        assert loaded is bundle
        seen_objectives.append(kwargs["objective_type"])
        return prepared

    monkeypatch.setattr(
        market_fit, "prepare_heston_market_fit", fake_prepare_heston_market_fit
    )
    calls: list[dict[str, object]] = []
    _patch_successful_calibration(monkeypatch, calls)

    result = fit_heston_from_bundle(bundle)

    assert result.status == "ok"
    assert seen_objectives == ["price_rmse"]


def test_fit_heston_market_uses_objective_from_prepared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _patch_successful_calibration(monkeypatch, calls)
    prepared = _prepared(tmp_path, objective_type="relative_price_rmse")

    result = fit_heston_market(prepared)

    assert result.status == "ok"
    assert calls[0]["objective_type"] == "relative_price_rmse"
    assert result.summary["objective_type"] == "relative_price_rmse"


def test_empty_prepared_universe_returns_empty_without_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _patch_successful_calibration(monkeypatch, calls)
    quotes = _heston_quotes(
        {
            0: {"expiry_years": 1.0 / 365.0},
            1: {"expiry_years": 2.0 / 365.0},
            2: {"expiry_years": 3.0 / 365.0},
        }
    )
    prepared = prepare_heston_market_fit(_loaded_bundle(tmp_path, quotes))

    result = fit_heston_market(prepared)

    assert result.status == "empty"
    assert result.calibration_result is None
    assert result.best_params is None
    assert calls == []
    assert result.summary["selected_quote_count"] == 0
    assert result.errors


def test_blocked_prepared_universe_returns_blocked_without_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _patch_successful_calibration(monkeypatch, calls)
    prepared = _blocked_prepared(_prepared(tmp_path))

    result = fit_heston_market(prepared)

    assert result.status == "blocked"
    assert result.calibration_result is None
    assert calls == []
    assert result.summary["preflight_recommendation"] == "block"


def test_blocked_prepared_universe_calibrates_when_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _patch_successful_calibration(monkeypatch, calls)
    prepared = _blocked_prepared(_prepared(tmp_path))

    result = fit_heston_market(prepared, allow_blocked=True)

    assert result.status == "ok"
    assert len(calls) == 1
    assert any("despite blocked" in warning for warning in result.warnings)


def test_calibration_exception_returns_failed_when_not_raising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_calibration(*_args: object, **_kwargs: object) -> HestonMultistartResult:
        raise RuntimeError("synthetic calibration failure")

    monkeypatch.setattr(market_fit, "calibrate_heston_multistart", fail_calibration)
    prepared = _prepared(tmp_path)

    result = fit_heston_market(prepared)

    assert result.status == "failed"
    assert result.calibration_result is None
    assert "selected_quote_count=3" in result.errors[0]
    assert "objective_type='price_rmse'" in result.errors[0]
    assert "spot=100.0" in result.errors[0]
    assert result.summary["calibration_status"] == "failed"


def test_calibration_exception_raises_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_calibration(*_args: object, **_kwargs: object) -> HestonMultistartResult:
        raise RuntimeError("synthetic calibration failure")

    monkeypatch.setattr(market_fit, "calibrate_heston_multistart", fail_calibration)
    prepared = _prepared(tmp_path)

    with pytest.raises(HestonMarketFitError, match="synthetic calibration failure"):
        fit_heston_market(prepared, raise_on_failure=True)


def test_fit_heston_market_requires_prepared_helper_guidance(tmp_path: Path) -> None:
    bundle = _loaded_bundle(tmp_path)

    with pytest.raises(TypeError) as exc_info:
        fit_heston_market(bundle)  # type: ignore[arg-type]

    message = str(exc_info.value)
    assert "PreparedHestonMarketFit" in message
    assert "prepare_heston_market_fit(bundle)" in message
    assert "fit_heston_from_bundle(path)" in message


def test_empty_result_error_mentions_public_recovery_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _patch_successful_calibration(monkeypatch, calls)
    quotes = _heston_quotes(
        {
            0: {"expiry_years": 1.0 / 365.0},
            1: {"expiry_years": 2.0 / 365.0},
            2: {"expiry_years": 3.0 / 365.0},
        }
    )
    prepared = prepare_heston_market_fit(_loaded_bundle(tmp_path, quotes))

    result = fit_heston_market(prepared)

    assert result.status == "empty"
    assert "prepared.rejected_quotes" in result.errors[0]
    assert "fit_heston_from_bundle(path)" in result.errors[0]


def test_blocked_result_error_preserves_preflight_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _patch_successful_calibration(monkeypatch, calls)
    prepared = _blocked_prepared(_prepared(tmp_path))

    result = fit_heston_market(prepared)

    assert result.status == "blocked"
    assert "prepared.preflight" in result.errors[0]
    assert "allow_blocked=True" in result.errors[0]
    assert "fit_heston_from_bundle(path)" in result.errors[0]


def test_failed_result_error_mentions_one_shot_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_calibration(*_args: object, **_kwargs: object) -> HestonMultistartResult:
        raise RuntimeError("synthetic calibration failure")

    monkeypatch.setattr(market_fit, "calibrate_heston_multistart", fail_calibration)
    prepared = _prepared(tmp_path)

    result = fit_heston_market(prepared)

    assert result.status == "failed"
    assert "selected_quote_count=3" in result.errors[0]
    assert "fit_heston_from_bundle(path)" in result.errors[0]


def test_successful_synthetic_calibration_returns_ok(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _patch_successful_calibration(monkeypatch, calls)
    prepared = _prepared(tmp_path)

    result = fit_heston_market(prepared)

    assert isinstance(result, HestonMarketFitResult)
    assert result.status == "ok"
    assert result.calibration_result is not None
    assert result.best_params == _heston_params()
    assert result.summary["best_objective_value"] == pytest.approx(0.125)
    assert result.summary["best_params"] == {
        "kappa": 1.4,
        "vbar": 0.045,
        "eta": 0.55,
        "rho": -0.35,
        "v": 0.04,
    }


def test_result_preserves_selected_and_rejected_quotes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _patch_successful_calibration(monkeypatch, calls)
    quotes = _heston_quotes({0: {"iv": pd.NA}})
    prepared = prepare_heston_market_fit(_loaded_bundle(tmp_path, quotes))

    result = fit_heston_market(prepared)

    assert result.prepared is prepared
    pd.testing.assert_frame_equal(
        result.prepared.selected_quotes, prepared.selected_quotes
    )
    pd.testing.assert_frame_equal(
        result.prepared.rejected_quotes, prepared.rejected_quotes
    )


def test_fit_heston_market_does_no_additional_filtering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _patch_successful_calibration(monkeypatch, calls)
    prepared = _prepared(tmp_path)

    fit_heston_market(prepared)

    assert calls[0]["quotes"] is prepared.quote_set
    assert calls[0]["quotes"].n_quotes == len(prepared.selected_quotes)


def test_fit_heston_market_does_not_mutate_prepared_or_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _patch_successful_calibration(monkeypatch, calls)
    prepared = _prepared(tmp_path)
    selected = prepared.selected_quotes.copy(deep=True)
    rejected = prepared.rejected_quotes.copy(deep=True)
    stats = replace(
        prepared.stats,
        rejection_counts=dict(prepared.stats.rejection_counts),
    )

    result = fit_heston_market(prepared)

    assert result.prepared is prepared
    assert prepared.stats == stats
    pd.testing.assert_frame_equal(prepared.selected_quotes, selected)
    pd.testing.assert_frame_equal(prepared.rejected_quotes, rejected)


def test_fit_heston_from_bundle_uses_local_loader_without_provider_calls(
    tmp_path: Path,
    fake_parquet: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _patch_successful_calibration(monkeypatch, calls)
    root = _write_bundle(tmp_path / "bundle")
    loaded = load_model_validation_bundle(root)

    result = fit_heston_from_bundle(loaded)

    assert result.status == "ok"
    assert result.prepared.market_data.spot == 100.0
    assert len(calls) == 1


def test_result_summary_uses_aggregate_counts_not_private_quote_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _patch_successful_calibration(monkeypatch, calls)
    prepared = _prepared(tmp_path)

    result = fit_heston_market(prepared)

    assert result.summary["selected_quote_count"] == 3
    assert result.summary["rejected_quote_count"] == 0
    assert "selected_quotes" not in result.summary
    assert "rejected_quotes" not in result.summary
    assert "cleaned_quotes" not in result.summary


def test_default_workflow_does_not_override_heston_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _patch_successful_calibration(monkeypatch, calls)
    prepared = _prepared(tmp_path)

    fit_heston_market(prepared)

    assert "bounds" not in calls[0]
    assert calls[0]["max_seeds"] == 8


def test_explicit_calibration_config_is_forwarded_without_hidden_bounds_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _patch_successful_calibration(monkeypatch, calls)
    prepared = _prepared(tmp_path)
    bounds = HestonCalibrationBounds()

    fit_heston_market(
        prepared,
        calibration_config=HestonCalibrationConfig(bounds=bounds, max_nfev=3),
        max_seeds=None,
    )

    assert calls[0]["bounds"] is bounds
    assert calls[0]["max_nfev"] == 3
    assert "max_seeds" not in calls[0]


def _blocked_prepared(prepared: PreparedHestonMarketFit) -> PreparedHestonMarketFit:
    assert prepared.quote_set is not None
    return replace(
        prepared,
        status="blocked",
        preflight=HestonQuotePreflight(
            quote_count=int(prepared.quote_set.n_quotes),
            price_bound_violation_count=1,
            mid_outside_bid_ask_count=0,
            recommendation="block",
            messages=("forced preflight block",),
        ),
        warnings=("forced preflight block",),
    )
