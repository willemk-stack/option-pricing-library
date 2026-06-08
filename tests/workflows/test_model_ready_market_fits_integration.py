from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from option_pricing.marketdata.bundles import (
    LoadedModelValidationBundle,
    ModelValidationBundleConfig,
    load_model_validation_bundle,
)
from option_pricing.marketdata.model_ready import (
    PreparedHestonMarketFit,
    prepare_heston_market_fit,
)
from option_pricing.marketdata.pipeline import run_local_model_validation_pipeline
from option_pricing.marketdata.surface_ready import (
    PreparedESSVIMarketFit,
    PreparedSVIMarketFit,
    prepare_essvi_market_fit,
    prepare_svi_market_fit,
)
from option_pricing.vol.ssvi import ESSVIGlobalCalibrationConfig
from option_pricing.workflows.surface_fit import (
    ESSVIMarketFitConfig,
    SVIMarketFitConfig,
    fit_essvi_market,
    fit_svi_market,
)

FIXTURE_NAME = "local_snapshot_surface_fit_v1"
RUN_ID = "model-ready-market-fits"
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


def test_synthetic_local_bundle_feeds_model_ready_market_fit_workflows(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    pipeline_result = run_local_model_validation_pipeline(
        storage=tmp_path,
        run_id=RUN_ID,
        fixture_name=FIXTURE_NAME,
        bundle_config=ModelValidationBundleConfig(run_heston_smoke=False),
        library_commit="synthetic-integration-test",
    )
    bundle = load_model_validation_bundle(
        pipeline_result.model_validation_bundle.manifest_path
    )

    _assert_loaded_bundle_artifacts(bundle)
    assert pipeline_result.quote_cleaning.reason_counts == {}
    assert pipeline_result.quote_cleaning.rejected_quotes.empty

    heston_prepared = prepare_heston_market_fit(bundle)
    svi_prepared = prepare_svi_market_fit(bundle)
    essvi_prepared = prepare_essvi_market_fit(bundle)

    _assert_heston_preparation_coherent(heston_prepared)
    _assert_svi_preparation_coherent(svi_prepared)
    _assert_essvi_preparation_coherent(essvi_prepared)

    assert heston_prepared.status == "ready"
    assert heston_prepared.preflight is not None
    assert heston_prepared.preflight.recommendation == "ok"
    assert heston_prepared.stats.expiry_count == 3

    assert svi_prepared.status == "ready"
    svi_result = fit_svi_market(
        svi_prepared,
        fit_config=SVIMarketFitConfig(loss="linear"),
    )
    assert svi_result.status == "ok"
    assert svi_result.surface is not None
    assert svi_result.summary["selected_point_count"] == len(
        svi_prepared.selected_points
    )
    assert svi_result.summary["fitted_expiry_count"] == 3
    assert svi_result.errors == ()

    assert essvi_prepared.status == "ready"
    essvi_result = fit_essvi_market(
        essvi_prepared,
        fit_config=ESSVIMarketFitConfig(
            calibration_config=ESSVIGlobalCalibrationConfig(
                max_nfev=2_000,
                strict_validation=True,
            )
        ),
    )
    assert essvi_result.status == "ok"
    assert essvi_result.fit_result is not None
    assert essvi_result.surface is not None
    assert essvi_result.validation is not None
    assert essvi_result.validation.ok
    assert essvi_result.summary["selected_point_count"] == len(
        essvi_prepared.selected_points
    )
    assert essvi_result.summary["node_count"] == 3
    assert essvi_result.errors == ()


def _assert_loaded_bundle_artifacts(bundle: LoadedModelValidationBundle) -> None:
    artifacts = bundle.manifest.get("artifacts")
    assert isinstance(artifacts, dict)
    assert artifacts == EXPECTED_ARTIFACTS
    assert bundle.manifest_path.is_file()
    for filename in artifacts.values():
        assert (bundle.root / str(filename)).is_file()

    rows = bundle.manifest["rows"]
    assert isinstance(rows, dict)
    assert rows["cleaned_quotes"] == len(bundle.cleaned_quotes)
    assert rows["rejected_quotes"] == len(bundle.rejected_quotes)
    assert rows["heston_quotes"] == len(bundle.heston_quotes)
    assert rows["surface_inputs"] == len(bundle.surface_inputs)

    assert bundle.market_data.spot == pytest.approx(100.0)
    assert not bundle.cleaned_quotes.empty
    assert bundle.rejected_quotes.empty
    assert not bundle.heston_quotes.empty
    assert not bundle.surface_inputs.empty
    assert not bundle.heston_fit_summary.empty
    assert set(bundle.cleaned_quotes["underlying"].astype(str).unique()) == {"SYNTH"}


def _assert_heston_preparation_coherent(
    prepared: PreparedHestonMarketFit,
) -> None:
    assert prepared.status in {"ready", "empty", "blocked"}
    assert prepared.model_name == "heston"
    assert prepared.stats.input_quote_count == (
        len(prepared.selected_quotes) + len(prepared.rejected_quotes)
    )
    assert prepared.stats.selected_quote_count == len(prepared.selected_quotes)
    assert prepared.stats.rejected_quote_count == len(prepared.rejected_quotes)
    assert isinstance(prepared.stats.rejection_counts, dict)
    assert prepared.stats.warnings == prepared.warnings
    if prepared.status == "empty":
        assert prepared.selected_quotes.empty
        assert prepared.quote_set is None
    else:
        assert not prepared.selected_quotes.empty
        assert prepared.quote_set is not None
        assert prepared.preflight is not None


def _assert_svi_preparation_coherent(prepared: PreparedSVIMarketFit) -> None:
    assert prepared.status in {"ready", "empty", "blocked"}
    assert prepared.model_name == "svi"
    assert prepared.stats.input_point_count == (
        len(prepared.selected_points) + len(prepared.rejected_points)
    )
    assert prepared.stats.selected_point_count == len(prepared.selected_points)
    assert prepared.stats.rejected_point_count == len(prepared.rejected_points)
    assert isinstance(prepared.stats.rejection_counts, dict)
    assert prepared.stats.warnings == prepared.warnings
    if prepared.status == "empty":
        assert prepared.selected_points.empty
    else:
        assert not prepared.selected_points.empty
        assert {"log_moneyness", "total_variance", "sqrt_weight"} <= set(
            prepared.selected_points.columns
        )


def _assert_essvi_preparation_coherent(
    prepared: PreparedESSVIMarketFit,
) -> None:
    assert prepared.status in {"ready", "empty", "blocked"}
    assert prepared.model_name == "essvi"
    assert prepared.stats.input_point_count == (
        len(prepared.selected_points) + len(prepared.rejected_points)
    )
    assert prepared.stats.selected_point_count == len(prepared.selected_points)
    assert prepared.stats.rejected_point_count == len(prepared.rejected_points)
    assert isinstance(prepared.stats.rejection_counts, dict)
    assert prepared.stats.warnings == prepared.warnings
    if prepared.status == "empty":
        assert prepared.selected_points.empty
    elif prepared.status == "blocked":
        assert prepared.warnings or prepared.stats.rejection_counts
    else:
        assert not prepared.selected_points.empty
        assert {"y", "T", "price_mkt", "sqrt_weight", "is_call"} <= set(
            prepared.selected_points.columns
        )
