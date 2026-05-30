from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

import option_pricing.marketdata.bundles as bundles_module
from option_pricing.marketdata.bundles import (
    ModelValidationBundleConfig,
    build_surface_inputs,
)
from option_pricing.marketdata.cleaning import clean_option_quotes
from option_pricing.marketdata.config import StorageConfig
from option_pricing.marketdata.manifests import validate_model_validation_manifest
from option_pricing.marketdata.normalize import (
    normalize_market_inputs,
    normalize_option_chain,
)
from option_pricing.marketdata.schemas import (
    HESTON_QUOTES_COLUMNS,
    SURFACE_INPUTS_COLUMNS,
    DatasetName,
)
from option_pricing.marketdata.storage import LocalStorage
from option_pricing.marketdata.validation import coerce_frame, validate_dtypes
from option_pricing.models.heston.calibration.heston_types import (
    HestonCalibrationRun,
    HestonMultistartResult,
)
from option_pricing.models.heston.params import HestonParams

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "local_snapshot_synth_schema_v1"
FIXTURE_NAME = "local_snapshot_synth_schema_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLES_FILE = REPO_ROOT / "src/option_pricing/marketdata/bundles.py"
EXPECTED_ARTIFACTS = {
    "market_data": "market_data.json",
    "cleaned_quotes": "cleaned_quotes.parquet",
    "rejected_quotes": "rejected_quotes.parquet",
    "heston_quotes": "heston_quotes.parquet",
    "surface_inputs": "surface_inputs.parquet",
    "heston_fit_summary": "heston_fit_summary.csv",
    "warnings": "warnings.json",
}
EXPECTED_FILES = ("manifest.json", *EXPECTED_ARTIFACTS.values())
EXPECTED_HESTON_FIT_SUMMARY_COLUMNS = (
    "status",
    "message",
    "objective_type",
    "quote_count",
    "success_count",
    "failure_count",
    "best_cost",
    "kappa",
    "vbar",
    "eta",
    "rho",
    "v",
    "jacobian_mode",
    "backend",
    "max_seeds",
    "max_nfev",
)


@dataclass(frozen=True, slots=True)
class _LocalSnapshotStub:
    fixture_name: str
    snapshot_id: str
    run_id: str | None
    underlying: str
    asof: pd.Timestamp


@dataclass(frozen=True, slots=True)
class _A3Outputs:
    local_snapshot: _LocalSnapshotStub
    market_inputs: pd.DataFrame
    cleaned_quotes: pd.DataFrame
    rejected_quotes: pd.DataFrame
    reason_counts: dict[str, int]
    warnings: tuple[str, ...]


@pytest.fixture
def fake_parquet(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_to_parquet(
        self: pd.DataFrame,
        path: str,
        compression: str | None = None,
        index: bool = False,
    ) -> None:
        del compression
        payload = self if index else self.reset_index(drop=True)
        payload.to_pickle(path)

    def _fake_read_parquet(
        path: str,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        frame = cast(pd.DataFrame, pd.read_pickle(path))
        if columns is None:
            return frame
        return cast(pd.DataFrame, frame.loc[:, columns])

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet)
    monkeypatch.setattr(pd, "read_parquet", _fake_read_parquet)


def _a3_outputs() -> _A3Outputs:
    market_inputs = normalize_market_inputs(
        pd.read_csv(FIXTURE_ROOT / "market_inputs.csv")
    )
    option_chain = normalize_option_chain(
        pd.read_csv(FIXTURE_ROOT / "option_chain.csv")
    )
    result = clean_option_quotes(option_chain, market_inputs)
    asof = pd.Timestamp(market_inputs.iloc[0]["asof"])
    local_snapshot = _LocalSnapshotStub(
        fixture_name=FIXTURE_NAME,
        snapshot_id=f"{FIXTURE_NAME}:SYNTH:{asof.isoformat()}",
        run_id="test-run",
        underlying="SYNTH",
        asof=asof,
    )
    return _A3Outputs(
        local_snapshot=local_snapshot,
        market_inputs=market_inputs,
        cleaned_quotes=result.cleaned_quotes,
        rejected_quotes=result.rejected_quotes,
        reason_counts=result.reason_counts,
        warnings=result.warnings,
    )


def _storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(StorageConfig(root=tmp_path))


def _partitions() -> dict[str, str | date]:
    return {
        "underlying": "SYNTH",
        "date": date(2026, 5, 22),
        "run_id": "test-run",
    }


def _bundle_root(root: Path) -> Path:
    return (
        root
        / "gold"
        / DatasetName.MODEL_VALIDATION_BUNDLE.value
        / "underlying=SYNTH"
        / "date=2026-05-22"
        / "run_id=test-run"
    )


def _bundle_path(root: Path, filename: str) -> Path:
    return _bundle_root(root) / filename


def _write_bundle(
    storage: LocalStorage,
    outputs: _A3Outputs,
    *,
    rejected_quotes: pd.DataFrame | None = None,
    reason_counts: dict[str, int] | None = None,
    warnings: tuple[str, ...] | None = None,
    config: ModelValidationBundleConfig | None = None,
    overwrite: bool = False,
    library_commit: str | None = "abc123",
):
    effective_config = (
        ModelValidationBundleConfig(run_heston_smoke=False)
        if config is None
        else config
    )
    return bundles_module._write_model_validation_bundle_artifacts(
        storage,
        local_snapshot=outputs.local_snapshot,
        market_inputs=outputs.market_inputs,
        cleaned_quotes=outputs.cleaned_quotes,
        rejected_quotes=(
            outputs.rejected_quotes if rejected_quotes is None else rejected_quotes
        ),
        reason_counts=outputs.reason_counts if reason_counts is None else reason_counts,
        warnings=outputs.warnings if warnings is None else warnings,
        config=effective_config,
        overwrite=overwrite,
        library_commit=library_commit,
    )


def _rejected_quotes_with_unique_detail(outputs: _A3Outputs) -> pd.DataFrame:
    row = outputs.cleaned_quotes.iloc[0]
    return coerce_frame(
        pd.DataFrame(
            [
                {
                    "underlying": row["underlying"],
                    "contract_symbol": row["contract_symbol"],
                    "quote_id": "rejected-quote-not-in-manifest",
                    "quote_ts": row["quote_ts"],
                    "asof": row["asof"],
                    "expiry": row["expiry"],
                    "strike": row["strike"],
                    "right": row["right"],
                    "bid": row["bid"],
                    "ask": row["ask"],
                    "mid": row["mid"],
                    "iv": row["iv"],
                    "vega": row["vega"],
                    "source": row["source"],
                    "rejection_reason": "unit_reject",
                    "rejection_detail": "do-not-embed-rejected-row-detail",
                    "cleaning_policy": row["cleaning_policy"],
                }
            ]
        ),
        DatasetName.REJECTED_QUOTES,
        allow_extra=False,
    )


def _heston_params() -> HestonParams:
    return HestonParams(kappa=1.4, vbar=0.045, eta=0.55, rho=-0.35, v=0.04)


def _fake_multistart_result(
    *,
    quote_count: int,
    objective_type: str = "price_rmse",
    cost: float = 0.125,
    success_count: int = 1,
    failure_count: int = 0,
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
        message="synthetic smoke success",
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
        success_count=success_count,
        failure_count=failure_count,
        jacobian_mode="analytic",
        analytic_jacobian_eta_min=None,
    )


def test_public_contracts_include_build_surface_inputs() -> None:
    assert "build_surface_inputs" in bundles_module.__all__
    assert bundles_module.build_surface_inputs is build_surface_inputs


def test_build_surface_inputs_exact_columns() -> None:
    surface_inputs = build_surface_inputs(_a3_outputs().cleaned_quotes)

    assert tuple(surface_inputs.columns) == SURFACE_INPUTS_COLUMNS


def test_build_surface_inputs_validates_dtypes_against_surface_schema() -> None:
    surface_inputs = build_surface_inputs(_a3_outputs().cleaned_quotes)

    validate_dtypes(surface_inputs, DatasetName.SURFACE_INPUTS, allow_extra=False)


def test_build_surface_inputs_preserves_expiry_years() -> None:
    cleaned_quotes = _a3_outputs().cleaned_quotes.copy()
    cleaned_quotes.loc[0, "expiry_years"] = 0.314159265

    surface_inputs = build_surface_inputs(cleaned_quotes)

    pd.testing.assert_series_equal(
        surface_inputs["expiry_years"],
        cleaned_quotes["expiry_years"].reset_index(drop=True),
        check_names=False,
    )


def test_build_surface_inputs_rejects_invalid_cleaned_quotes() -> None:
    cleaned_quotes = _a3_outputs().cleaned_quotes.assign(extra_column="nope")

    with pytest.raises(ValueError, match="unexpected extra columns"):
        build_surface_inputs(cleaned_quotes)


def test_packaging_writes_all_expected_files(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    storage = _storage(tmp_path)
    result = _write_bundle(storage, _a3_outputs())
    root = _bundle_root(tmp_path)

    assert result.manifest_path == root / "manifest.json"
    assert all((root / filename).exists() for filename in EXPECTED_FILES)
    assert result.manifest_path.exists()
    assert {path.name for path in result.artifact_paths} == set(
        EXPECTED_ARTIFACTS.values()
    )
    assert "gold/model_validation_bundle" in root.as_posix()

    heston_quotes = pd.read_parquet(root / "heston_quotes.parquet")
    surface_inputs = pd.read_parquet(root / "surface_inputs.parquet")
    validate_dtypes(heston_quotes, DatasetName.HESTON_QUOTES, allow_extra=False)
    validate_dtypes(surface_inputs, DatasetName.SURFACE_INPUTS, allow_extra=False)
    assert tuple(heston_quotes.columns) == HESTON_QUOTES_COLUMNS
    assert tuple(surface_inputs.columns) == SURFACE_INPUTS_COLUMNS


def test_packaging_manifest_validates(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    storage = _storage(tmp_path)
    _write_bundle(storage, _a3_outputs())

    manifest = storage.read_json(
        dataset=DatasetName.MODEL_VALIDATION_BUNDLE.value,
        layer="gold",
        partitions=_partitions(),
        filename="manifest.json",
    )

    validate_model_validation_manifest(manifest)
    assert manifest["dataset"] == DatasetName.MODEL_VALIDATION_BUNDLE.value
    assert manifest["layer"] == "gold"


def test_manifest_artifact_references_are_relative_filenames(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    storage = _storage(tmp_path)
    _write_bundle(storage, _a3_outputs())

    manifest = json.loads(_bundle_path(tmp_path, "manifest.json").read_text())
    assert manifest["artifacts"] == EXPECTED_ARTIFACTS
    assert all(
        Path(filename).name == filename and "/" not in filename and "\\" not in filename
        for filename in manifest["artifacts"].values()
    )


def test_rejected_quote_rows_are_not_embedded_in_manifest_json(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    storage = _storage(tmp_path)
    outputs = _a3_outputs()

    _write_bundle(
        storage,
        outputs,
        rejected_quotes=_rejected_quotes_with_unique_detail(outputs),
        reason_counts={"unit_reject": 1},
    )

    manifest_text = _bundle_path(tmp_path, "manifest.json").read_text()
    assert "do-not-embed-rejected-row-detail" not in manifest_text
    assert "rejected-quote-not-in-manifest" not in manifest_text
    assert "rejection_detail" not in manifest_text
    assert "rejected_quote_rows" not in manifest_text


def test_heston_smoke_success_records_manifest_and_summary(
    tmp_path: Path,
    fake_parquet: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage(tmp_path)
    outputs = _a3_outputs()
    calls: list[dict[str, object]] = []

    def fake_calibrate_heston_multistart(
        quotes: object,
        **kwargs: object,
    ) -> HestonMultistartResult:
        calls.append({"quotes": quotes, **kwargs})
        return _fake_multistart_result(
            quote_count=int(quotes.n_quotes),
            objective_type=str(kwargs["objective_type"]),
            cost=0.25,
            success_count=1,
            failure_count=0,
        )

    monkeypatch.setattr(
        bundles_module,
        "calibrate_heston_multistart",
        fake_calibrate_heston_multistart,
    )

    _write_bundle(
        storage,
        outputs,
        config=ModelValidationBundleConfig(run_heston_smoke=True),
    )

    manifest = json.loads(_bundle_path(tmp_path, "manifest.json").read_text())
    smoke = manifest["heston_smoke"]
    assert smoke["status"] == "success"
    assert smoke["objective_type"] == "price_rmse"
    assert smoke["quote_count"] == len(outputs.cleaned_quotes)
    assert smoke["success_count"] == 1
    assert smoke["failure_count"] == 0
    assert smoke["best_cost"] == 0.25
    assert smoke["parameters"] == {
        "eta": 0.55,
        "kappa": 1.4,
        "rho": -0.35,
        "v": 0.04,
        "vbar": 0.045,
    }
    assert "completed" in str(smoke["message"])
    assert int(calls[0]["quotes"].n_quotes) == len(outputs.cleaned_quotes)

    summary = pd.read_csv(_bundle_path(tmp_path, "heston_fit_summary.csv"))
    assert tuple(summary.columns) == EXPECTED_HESTON_FIT_SUMMARY_COLUMNS
    assert summary.loc[0, "status"] == "success"
    assert int(cast(int, summary.loc[0, "success_count"])) == 1
    assert int(cast(int, summary.loc[0, "failure_count"])) == 0
    assert float(cast(float, summary.loc[0, "best_cost"])) == pytest.approx(0.25)
    assert float(cast(float, summary.loc[0, "kappa"])) == pytest.approx(1.4)
    assert float(cast(float, summary.loc[0, "vbar"])) == pytest.approx(0.045)
    assert float(cast(float, summary.loc[0, "eta"])) == pytest.approx(0.55)
    assert float(cast(float, summary.loc[0, "rho"])) == pytest.approx(-0.35)
    assert float(cast(float, summary.loc[0, "v"])) == pytest.approx(0.04)
    assert summary.loc[0, "jacobian_mode"] == "analytic"
    assert summary.loc[0, "backend"] == "gauss_legendre"


def test_heston_smoke_failure_writes_auditable_bundle_by_default(
    tmp_path: Path,
    fake_parquet: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage(tmp_path)
    outputs = _a3_outputs()

    def fake_calibrate_heston_multistart(
        _quotes: object,
        **_kwargs: object,
    ) -> HestonMultistartResult:
        raise ValueError("synthetic calibration failure")

    monkeypatch.setattr(
        bundles_module,
        "calibrate_heston_multistart",
        fake_calibrate_heston_multistart,
    )

    _write_bundle(
        storage,
        outputs,
        config=ModelValidationBundleConfig(run_heston_smoke=True),
    )

    manifest = json.loads(_bundle_path(tmp_path, "manifest.json").read_text())
    smoke = manifest["heston_smoke"]
    assert smoke["status"] == "failed"
    assert smoke["message"] == "ValueError: synthetic calibration failure"
    assert smoke["objective_type"] == "price_rmse"
    assert smoke["quote_count"] == len(outputs.cleaned_quotes)
    assert smoke["success_count"] is None
    assert smoke["failure_count"] is None
    assert smoke["best_cost"] is None
    assert smoke["parameters"] is None

    summary = pd.read_csv(_bundle_path(tmp_path, "heston_fit_summary.csv"))
    assert tuple(summary.columns) == EXPECTED_HESTON_FIT_SUMMARY_COLUMNS
    assert summary.loc[0, "status"] == "failed"
    assert summary.loc[0, "message"] == "ValueError: synthetic calibration failure"
    assert pd.isna(summary.loc[0, "success_count"])
    assert pd.isna(summary.loc[0, "failure_count"])
    assert pd.isna(summary.loc[0, "best_cost"])
    assert pd.isna(summary.loc[0, "kappa"])

    warnings_payload = json.loads(_bundle_path(tmp_path, "warnings.json").read_text())
    assert warnings_payload["heston_smoke"] == [
        "ValueError: synthetic calibration failure"
    ]


def test_heston_smoke_skipped_when_config_disables_it(
    tmp_path: Path,
    fake_parquet: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage(tmp_path)
    outputs = _a3_outputs()
    calls: list[object] = []

    def fake_calibrate_heston_multistart(
        quotes: object,
        **_kwargs: object,
    ) -> HestonMultistartResult:
        calls.append(quotes)
        return _fake_multistart_result(quote_count=1)

    monkeypatch.setattr(
        bundles_module,
        "calibrate_heston_multistart",
        fake_calibrate_heston_multistart,
    )

    _write_bundle(
        storage,
        outputs,
        config=ModelValidationBundleConfig(
            run_heston_smoke=False,
            heston_objective_type="iv_rmse",
        ),
    )

    assert calls == []
    manifest = json.loads(_bundle_path(tmp_path, "manifest.json").read_text())
    smoke = manifest["heston_smoke"]
    assert smoke["status"] == "skipped"
    assert "config.run_heston_smoke is False" in str(smoke["message"])
    assert smoke["objective_type"] == "iv_rmse"
    assert smoke["quote_count"] == len(outputs.cleaned_quotes)

    summary = pd.read_csv(_bundle_path(tmp_path, "heston_fit_summary.csv"))
    assert tuple(summary.columns) == EXPECTED_HESTON_FIT_SUMMARY_COLUMNS
    assert summary.loc[0, "status"] == "skipped"
    assert summary.loc[0, "objective_type"] == "iv_rmse"
    assert int(cast(int, summary.loc[0, "quote_count"])) == len(outputs.cleaned_quotes)
    assert pd.isna(summary.loc[0, "success_count"])
    assert pd.isna(summary.loc[0, "max_seeds"])


def test_heston_smoke_failure_can_fail_fast_without_writing_bundle(
    tmp_path: Path,
    fake_parquet: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage(tmp_path)

    def fake_calibrate_heston_multistart(
        _quotes: object,
        **_kwargs: object,
    ) -> HestonMultistartResult:
        raise RuntimeError("synthetic smoke failure")

    monkeypatch.setattr(
        bundles_module,
        "calibrate_heston_multistart",
        fake_calibrate_heston_multistart,
    )

    with pytest.raises(
        RuntimeError,
        match="Heston smoke failed: RuntimeError: synthetic smoke failure",
    ):
        _write_bundle(
            storage,
            _a3_outputs(),
            config=ModelValidationBundleConfig(
                run_heston_smoke=True,
                fail_on_heston_smoke_failure=True,
            ),
        )

    assert not _bundle_path(tmp_path, "manifest.json").exists()
    assert not any(
        _bundle_path(tmp_path, filename).exists() for filename in EXPECTED_FILES
    )
    assert not (tmp_path / "_meta" / "artifacts.jsonl").exists()


def test_heston_smoke_forwards_config_to_calibrator(
    tmp_path: Path,
    fake_parquet: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage(tmp_path)
    captured: dict[str, object] = {}

    def fake_calibrate_heston_multistart(
        quotes: object,
        **kwargs: object,
    ) -> HestonMultistartResult:
        captured.update(kwargs)
        return _fake_multistart_result(
            quote_count=int(quotes.n_quotes),
            objective_type=str(kwargs["objective_type"]),
        )

    monkeypatch.setattr(
        bundles_module,
        "calibrate_heston_multistart",
        fake_calibrate_heston_multistart,
    )

    _write_bundle(
        storage,
        _a3_outputs(),
        config=ModelValidationBundleConfig(
            run_heston_smoke=True,
            heston_objective_type="relative_price_rmse",
            heston_max_seeds=3,
            heston_max_nfev=7,
        ),
    )

    assert captured["objective_type"] == "relative_price_rmse"
    assert captured["max_seeds"] == 3
    assert captured["max_nfev"] == 7


@pytest.mark.parametrize("filename", EXPECTED_FILES)
def test_overwrite_false_fails_before_partial_overwrite(
    tmp_path: Path,
    fake_parquet: None,
    filename: str,
) -> None:
    storage = _storage(tmp_path)
    existing_target = _bundle_path(tmp_path, filename)
    existing_target.parent.mkdir(parents=True)
    existing_target.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        _write_bundle(storage, _a3_outputs())

    assert existing_target.read_text(encoding="utf-8") == "existing"
    for expected_filename in EXPECTED_FILES:
        target = _bundle_path(tmp_path, expected_filename)
        if target == existing_target:
            continue
        assert not target.exists()
    assert not (tmp_path / "_meta" / "artifacts.jsonl").exists()


def test_overwrite_true_replaces_deterministic_outputs(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    storage = _storage(tmp_path)
    first_outputs = _a3_outputs()
    first_result = _write_bundle(storage, first_outputs, warnings=("first-warning",))

    replacement_quotes = first_outputs.cleaned_quotes.copy()
    replacement_quotes.loc[0, "mid"] = 9.99
    replacement_outputs = _A3Outputs(
        local_snapshot=first_outputs.local_snapshot,
        market_inputs=first_outputs.market_inputs,
        cleaned_quotes=replacement_quotes,
        rejected_quotes=first_outputs.rejected_quotes,
        reason_counts=first_outputs.reason_counts,
        warnings=first_outputs.warnings,
    )

    replacement_result = _write_bundle(
        storage,
        replacement_outputs,
        warnings=("replacement-warning",),
        overwrite=True,
        library_commit="replacement",
    )

    assert replacement_result.manifest_path == first_result.manifest_path
    heston_quotes = pd.read_parquet(_bundle_path(tmp_path, "heston_quotes.parquet"))
    assert float(cast(float, heston_quotes.loc[0, "mid"])) == pytest.approx(9.99)
    warnings_payload = json.loads(_bundle_path(tmp_path, "warnings.json").read_text())
    assert warnings_payload["warnings"] == ["replacement-warning"]
    market_payload = json.loads(_bundle_path(tmp_path, "market_data.json").read_text())
    assert market_payload["library_commit"] == "replacement"


def _import_root(name: str) -> str:
    return name.split(".", maxsplit=1)[0]


def _imported_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    names: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.append(node.module)
            names.extend(
                f"{node.module}.{alias.name}"
                for alias in node.names
                if alias.name != "*"
            )

    return names


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _called_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    return [
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and (name := _call_name(node.func)) is not None
    ]


def _string_constants(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _is_disallowed_import(name: str) -> bool:
    allowed_calibration_imports = {
        "option_pricing.models.heston.calibration",
        "option_pricing.models.heston.calibration.calibrate_heston_multistart",
    }
    disallowed_roots = {
        "alpaca",
        "argparse",
        "click",
        "fredapi",
        "requests",
        "yfinance",
    }
    lowered_parts = {part.lower() for part in name.split(".")}

    if _import_root(name) in disallowed_roots:
        return True
    if name.startswith("option_pricing.marketdata.providers"):
        return True
    if name.startswith("option_pricing.diagnostics.heston"):
        return True
    if name in allowed_calibration_imports:
        return False
    if name.startswith("option_pricing.models.heston.calibration"):
        return True
    if "research" in lowered_parts:
        return True
    return False


def test_boundary_guard_excludes_out_of_scope_work() -> None:
    forbidden_imports = [
        name for name in _imported_names(BUNDLES_FILE) if _is_disallowed_import(name)
    ]
    forbidden_calls = [
        name
        for name in _called_names(BUNDLES_FILE)
        if ("calibr" in name.lower() and name != "calibrate_heston_multistart")
        or name
        in {
            "refresh_providers",
            "run_provider_refresh",
            "write_research_export",
        }
    ]
    forbidden_strings = [
        value
        for value in _string_constants(BUNDLES_FILE)
        if any(
            term in value.lower()
            for term in (
                "api_key",
                "secret_key",
                "credential",
                "research_export",
                "provider_refresh",
            )
        )
    ]

    assert forbidden_imports == []
    assert forbidden_calls == []
    assert forbidden_strings == []
