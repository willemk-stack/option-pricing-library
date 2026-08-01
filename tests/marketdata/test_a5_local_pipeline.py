from __future__ import annotations

import ast
import json
import shutil
from dataclasses import fields
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

import option_pricing.marketdata.bundles as bundles_module
from option_pricing.marketdata.bundles import (
    ModelValidationBundleConfig,
    write_model_validation_bundle_artifacts,
)
from option_pricing.marketdata.cleaning import QuoteCleaningPolicyV1
from option_pricing.marketdata.config import StorageConfig
from option_pricing.marketdata.pipeline import (
    LocalModelValidationPipelineResult,
    run_local_model_validation_pipeline,
)
from option_pricing.marketdata.providers.local import (
    LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1,
    LOCAL_SNAPSHOT_SYNTH_WITH_REJECTIONS_V1,
)
from option_pricing.marketdata.schemas import (
    HESTON_QUOTES_COLUMNS,
    SURFACE_INPUTS_COLUMNS,
    DatasetName,
)
from option_pricing.marketdata.storage import LocalStorage
from option_pricing.marketdata.validation import validate_dtypes

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests/marketdata/fixtures"
BUNDLES_FILE = REPO_ROOT / "src/option_pricing/marketdata/bundles.py"
PIPELINE_FILE = REPO_ROOT / "src/option_pricing/marketdata/pipeline.py"
LOCAL_BUNDLE_CONFIG = ModelValidationBundleConfig(run_heston_smoke=False)
EXPECTED_BUNDLE_FILES = {
    "manifest.json",
    "market_data.json",
    "cleaned_quotes.parquet",
    "rejected_quotes.parquet",
    "heston_quotes.parquet",
    "surface_inputs.parquet",
    "heston_fit_summary.csv",
    "warnings.json",
}


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


def _run_pipeline(
    tmp_path: Path,
    *,
    run_id: str = "test-run",
    fixture_name: str = LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1,
    cleaning_policy: QuoteCleaningPolicyV1 | None = None,
    overwrite: bool = False,
    library_commit: str | None = "abc123",
) -> LocalModelValidationPipelineResult:
    return run_local_model_validation_pipeline(
        storage=tmp_path,
        run_id=run_id,
        fixture_name=fixture_name,
        cleaning_policy=cleaning_policy,
        bundle_config=LOCAL_BUNDLE_CONFIG,
        overwrite=overwrite,
        library_commit=library_commit,
    )


def _bundle_root(root: Path, *, run_id: str = "test-run") -> Path:
    return (
        root
        / "gold"
        / DatasetName.MODEL_VALIDATION_BUNDLE.value
        / "underlying=SYNTH"
        / "date=2026-05-22"
        / f"run_id={run_id}"
    )


def _bronze_root(root: Path, *, run_id: str = "test-run") -> Path:
    return (
        root
        / "bronze"
        / "local_snapshot"
        / "underlying=SYNTH"
        / "date=2026-05-22"
        / f"run_id={run_id}"
    )


def _silver_root(
    root: Path,
    dataset: DatasetName,
    *,
    run_id: str = "test-run",
) -> Path:
    return (
        root
        / "silver"
        / dataset.value
        / "underlying=SYNTH"
        / "date=2026-05-22"
        / f"run_id={run_id}"
    )


def _gold_root(
    root: Path,
    dataset: DatasetName,
    *,
    run_id: str = "test-run",
) -> Path:
    return (
        root
        / "gold"
        / dataset.value
        / "underlying=SYNTH"
        / "date=2026-05-22"
        / f"run_id={run_id}"
    )


def _silver_path(root: Path, dataset: DatasetName, filename: str) -> Path:
    return _silver_root(root, dataset) / filename


def _gold_path(root: Path, dataset: DatasetName, filename: str) -> Path:
    return _gold_root(root, dataset) / filename


def _known_pipeline_targets(root: Path) -> tuple[Path, ...]:
    bronze_root = _bronze_root(root)
    bundle_root = _bundle_root(root)
    return (
        bronze_root / "market_inputs.parquet",
        bronze_root / "option_chain.parquet",
        bronze_root / "manifest.json",
        _silver_path(root, DatasetName.MARKET_INPUTS, "market_inputs.parquet"),
        _silver_path(root, DatasetName.CLEANED_QUOTES, "cleaned_quotes.parquet"),
        _silver_path(root, DatasetName.REJECTED_QUOTES, "rejected_quotes.parquet"),
        _silver_path(root, DatasetName.CLEANED_QUOTES, "manifest.json"),
        _gold_path(root, DatasetName.MARKET_SNAPSHOT, "market_data.json"),
        _gold_path(root, DatasetName.MARKET_SNAPSHOT, "manifest.json"),
        _gold_path(root, DatasetName.HESTON_QUOTES, "heston_quotes.parquet"),
        _gold_path(root, DatasetName.HESTON_QUOTES, "manifest.json"),
        bundle_root / "manifest.json",
        bundle_root / "market_data.json",
        bundle_root / "cleaned_quotes.parquet",
        bundle_root / "rejected_quotes.parquet",
        bundle_root / "heston_quotes.parquet",
        bundle_root / "surface_inputs.parquet",
        bundle_root / "heston_fit_summary.csv",
        bundle_root / "warnings.json",
    )


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _precreate_text(path: Path, text: str = "existing") -> None:
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")


def _assert_only_existing_target_remains(root: Path, existing_target: Path) -> None:
    assert existing_target.read_text(encoding="utf-8") == "existing"
    for target in _known_pipeline_targets(root):
        if target == existing_target:
            continue
        assert not target.exists()
    assert not (root / "_meta" / "artifacts.jsonl").exists()


def test_local_model_validation_pipeline_result_fields_are_stable() -> None:
    assert tuple(
        field.name for field in fields(LocalModelValidationPipelineResult)
    ) == (
        "local_snapshot",
        "market_inputs",
        "option_chain",
        "quote_cleaning",
        "bronze_paths",
        "silver_paths",
        "gold_paths",
        "model_validation_bundle",
    )


def test_local_model_validation_pipeline_writes_bronze_silver_gold_and_bundle(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    result = _run_pipeline(tmp_path)
    bundle_root = _bundle_root(tmp_path)

    assert result.local_snapshot.run_id == "test-run"
    assert result.local_snapshot.underlying == "SYNTH"
    assert not result.market_inputs.empty
    assert not result.option_chain.empty
    assert not result.quote_cleaning.cleaned_quotes.empty

    assert result.bronze_paths.manifest.exists()
    assert result.bronze_paths.market_inputs.exists()
    assert result.bronze_paths.option_chain.exists()

    assert result.silver_paths.market_inputs.exists()
    assert result.silver_paths.cleaned_quotes.exists()
    assert result.silver_paths.rejected_quotes.exists()
    assert result.silver_paths.manifest.exists()

    assert result.gold_paths.market_data.exists()
    assert result.gold_paths.market_manifest.exists()
    assert result.gold_paths.heston_quotes.exists()
    assert result.gold_paths.heston_manifest.exists()

    assert result.model_validation_bundle.manifest_path == bundle_root / "manifest.json"
    assert {path.name for path in bundle_root.iterdir()} == EXPECTED_BUNDLE_FILES
    assert {path.name for path in result.model_validation_bundle.artifact_paths} == (
        EXPECTED_BUNDLE_FILES - {"manifest.json"}
    )
    assert result.model_validation_bundle.manifest["heston_smoke"] == {
        "status": "skipped",
        "message": ("Heston smoke skipped because config.run_heston_smoke is False."),
        "objective_type": "price_rmse",
        "quote_count": len(result.quote_cleaning.cleaned_quotes),
        "success_count": None,
        "failure_count": None,
        "best_cost": None,
        "parameters": None,
    }


def test_local_snapshot_pipeline_round_trip_preserves_expiry_timestamp(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_path = fixture_root / LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1
    shutil.copytree(FIXTURE_ROOT / LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1, fixture_path)
    option_chain_path = fixture_path / "option_chain.csv"
    option_chain = pd.read_csv(option_chain_path)
    option_chain["expiry"] = "2026-06-19T21:00:00"
    option_chain.to_csv(option_chain_path, index=False)

    result = run_local_model_validation_pipeline(
        storage=tmp_path / "storage",
        run_id="timestamp-round-trip",
        fixture_name=LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1,
        fixture_root=fixture_root,
        bundle_config=LOCAL_BUNDLE_CONFIG,
        library_commit="abc123",
    )
    loaded = bundles_module.load_model_validation_bundle(
        result.model_validation_bundle.manifest_path.parent
    )

    expected = (
        pd.Timestamp("2026-06-19T21:00:00Z") - pd.Timestamp("2026-05-22T15:30:00Z")
    ).total_seconds() / (365.0 * 86400.0)
    assert set(result.option_chain["expiry"]) == {pd.Timestamp("2026-06-19 21:00:00")}
    for field in ("expiry_years", "time_to_expiry_years"):
        observed = pd.to_numeric(loaded.cleaned_quotes[field], errors="raise")
        assert observed.to_numpy(dtype=float) == pytest.approx(
            [expected] * len(observed),
            abs=1.0e-15,
        )


def test_rejected_quote_fixture_flows_evidence_through_pipeline_and_bundle(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    result = _run_pipeline(
        tmp_path,
        fixture_name=LOCAL_SNAPSHOT_SYNTH_WITH_REJECTIONS_V1,
    )

    assert not result.quote_cleaning.cleaned_quotes.empty
    assert not result.quote_cleaning.rejected_quotes.empty
    assert result.quote_cleaning.reason_counts == {"crossed_bid_ask": 1}
    assert result.silver_paths.rejected_quotes.exists()

    bundle_root = _bundle_root(tmp_path)
    bundle_rejected = bundle_root / "rejected_quotes.parquet"
    assert bundle_rejected.exists()
    assert pd.read_parquet(bundle_rejected).equals(
        result.quote_cleaning.rejected_quotes.reset_index(drop=True)
    )

    manifest = _read_json(result.model_validation_bundle.manifest_path)
    assert manifest["rows"]["cleaned_quotes"] == len(
        result.quote_cleaning.cleaned_quotes
    )
    assert manifest["rows"]["rejected_quotes"] == len(
        result.quote_cleaning.rejected_quotes
    )
    assert manifest["reason_counts"] == {"crossed_bid_ask": 1}
    manifest_text = result.model_validation_bundle.manifest_path.read_text(
        encoding="utf-8"
    )
    assert "rejection_detail" not in manifest_text
    assert "ask must be >= bid" not in manifest_text
    assert "rejected_quote_rows" not in manifest_text


def test_all_quotes_rejected_pipeline_writes_auditable_empty_gold_and_bundle(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    result = _run_pipeline(
        tmp_path,
        fixture_name=LOCAL_SNAPSHOT_SYNTH_WITH_REJECTIONS_V1,
        cleaning_policy=QuoteCleaningPolicyV1(max_relative_spread=0.001),
    )
    bundle_root = _bundle_root(tmp_path)

    assert result.quote_cleaning.cleaned_quotes.empty
    assert not result.quote_cleaning.rejected_quotes.empty
    assert result.quote_cleaning.reason_counts == {
        "crossed_bid_ask": 1,
        "nonfinite_numeric_field": 1,
    }
    assert result.quote_cleaning.warnings == ("all_quotes_rejected",)
    assert {path.name for path in bundle_root.iterdir()} == EXPECTED_BUNDLE_FILES

    heston_quotes = pd.read_parquet(bundle_root / "heston_quotes.parquet")
    surface_inputs = pd.read_parquet(bundle_root / "surface_inputs.parquet")
    assert heston_quotes.empty
    assert surface_inputs.empty
    assert tuple(heston_quotes.columns) == HESTON_QUOTES_COLUMNS
    assert tuple(surface_inputs.columns) == SURFACE_INPUTS_COLUMNS
    validate_dtypes(heston_quotes, DatasetName.HESTON_QUOTES, allow_extra=False)
    validate_dtypes(surface_inputs, DatasetName.SURFACE_INPUTS, allow_extra=False)

    manifest = _read_json(result.model_validation_bundle.manifest_path)
    assert manifest["rows"] == {
        "market_inputs": 1,
        "cleaned_quotes": 0,
        "rejected_quotes": len(result.quote_cleaning.rejected_quotes),
        "heston_quotes": 0,
        "surface_inputs": 0,
    }
    assert manifest["reason_counts"] == result.quote_cleaning.reason_counts
    assert manifest["heston_smoke"]["status"] == "skipped"
    assert manifest["heston_smoke"]["message"] == (
        "Heston smoke skipped because no cleaned quotes are available."
    )
    assert result.gold_paths.market_data.exists()
    assert result.gold_paths.heston_quotes.exists()
    assert result.model_validation_bundle.manifest_path.exists()


@pytest.mark.parametrize("storage_kind", ["path", "storage_config", "local_storage"])
def test_run_local_model_validation_pipeline_accepts_local_storage_inputs(
    tmp_path: Path,
    fake_parquet: None,
    storage_kind: str,
) -> None:
    storage_root = tmp_path / storage_kind
    if storage_kind == "path":
        storage: Path | StorageConfig | LocalStorage = storage_root
    elif storage_kind == "storage_config":
        storage = StorageConfig(root=storage_root)
    else:
        storage = LocalStorage(StorageConfig(root=storage_root))

    result = run_local_model_validation_pipeline(
        storage=storage,
        run_id="test-run",
        bundle_config=LOCAL_BUNDLE_CONFIG,
    )

    assert result.bronze_paths.manifest.exists()
    assert result.model_validation_bundle.manifest_path.exists()


def test_run_local_model_validation_pipeline_rejects_invalid_storage_type() -> None:
    with pytest.raises(
        TypeError,
        match="storage must be a LocalStorage, StorageConfig, or pathlib.Path, got str",
    ):
        run_local_model_validation_pipeline(
            storage="not-local-storage",  # type: ignore[arg-type]
            run_id="test-run",
            bundle_config=LOCAL_BUNDLE_CONFIG,
        )


def test_run_local_model_validation_pipeline_rejects_invalid_cleaning_policy(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TypeError,
        match="cleaning_policy must be a QuoteCleaningPolicyV1, got object",
    ):
        run_local_model_validation_pipeline(
            storage=tmp_path,
            run_id="test-run",
            cleaning_policy=object(),  # type: ignore[arg-type]
            bundle_config=LOCAL_BUNDLE_CONFIG,
        )

    assert not (tmp_path / "bronze").exists()
    assert not (tmp_path / "silver").exists()
    assert not (tmp_path / "gold").exists()


def test_run_local_model_validation_pipeline_requires_run_id(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    with pytest.raises(ValueError, match="run_id is required"):
        run_local_model_validation_pipeline(
            storage=tmp_path,
            run_id=" ",
            bundle_config=LOCAL_BUNDLE_CONFIG,
        )

    assert not (tmp_path / "bronze").exists()
    assert not (tmp_path / "silver").exists()
    assert not (tmp_path / "gold").exists()


def test_local_model_validation_pipeline_overwrite_false_preflights_existing_run(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    first = _run_pipeline(tmp_path, library_commit="first")
    manifest_text = first.bronze_paths.manifest.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        _run_pipeline(tmp_path, library_commit="replacement")

    assert first.bronze_paths.manifest.read_text(encoding="utf-8") == manifest_text
    assert (
        _read_json(first.model_validation_bundle.manifest_path)["library_commit"]
        == "first"
    )


def test_local_model_validation_pipeline_preflights_silver_conflict(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    existing_target = _silver_path(
        tmp_path,
        DatasetName.CLEANED_QUOTES,
        "cleaned_quotes.parquet",
    )
    _precreate_text(existing_target)

    with pytest.raises(FileExistsError, match="overwrite=True"):
        _run_pipeline(tmp_path, library_commit="replacement")

    _assert_only_existing_target_remains(tmp_path, existing_target)


def test_local_model_validation_pipeline_preflights_gold_conflict(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    existing_target = _gold_path(
        tmp_path,
        DatasetName.HESTON_QUOTES,
        "heston_quotes.parquet",
    )
    _precreate_text(existing_target)

    with pytest.raises(FileExistsError, match="overwrite=True"):
        _run_pipeline(tmp_path, library_commit="replacement")

    _assert_only_existing_target_remains(tmp_path, existing_target)


def test_local_model_validation_pipeline_preflights_bundle_conflict(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    existing_target = _bundle_root(tmp_path) / "warnings.json"
    _precreate_text(existing_target)

    with pytest.raises(FileExistsError, match="overwrite=True"):
        _run_pipeline(tmp_path, library_commit="replacement")

    _assert_only_existing_target_remains(tmp_path, existing_target)


def test_all_quotes_rejected_pipeline_preflight_conflict_has_no_partial_writes(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    existing_target = _bundle_root(tmp_path) / "heston_quotes.parquet"
    _precreate_text(existing_target)

    with pytest.raises(FileExistsError, match="overwrite=True"):
        _run_pipeline(
            tmp_path,
            fixture_name=LOCAL_SNAPSHOT_SYNTH_WITH_REJECTIONS_V1,
            cleaning_policy=QuoteCleaningPolicyV1(max_relative_spread=0.001),
            library_commit="replacement",
        )

    _assert_only_existing_target_remains(tmp_path, existing_target)


def test_local_model_validation_pipeline_overwrite_true_replaces_outputs(
    tmp_path: Path,
    fake_parquet: None,
) -> None:
    first = _run_pipeline(tmp_path, library_commit="first")
    replacement = _run_pipeline(
        tmp_path,
        overwrite=True,
        library_commit="replacement",
    )

    assert replacement.bronze_paths == first.bronze_paths
    assert replacement.silver_paths == first.silver_paths
    assert replacement.gold_paths == first.gold_paths
    assert replacement.model_validation_bundle.manifest_path == (
        first.model_validation_bundle.manifest_path
    )
    assert _read_json(replacement.bronze_paths.manifest)["library_commit"] == (
        "replacement"
    )
    assert _read_json(replacement.gold_paths.market_data)["library_commit"] == (
        "replacement"
    )
    assert (
        _read_json(replacement.model_validation_bundle.manifest_path)["library_commit"]
        == "replacement"
    )


def test_model_validation_bundle_public_wrapper_is_exported() -> None:
    assert "write_model_validation_bundle_artifacts" in bundles_module.__all__
    assert (
        bundles_module.write_model_validation_bundle_artifacts
        is write_model_validation_bundle_artifacts
    )


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


def _is_disallowed_import(name: str) -> bool:
    disallowed_roots = {
        "alpaca",
        "argparse",
        "click",
        "duckdb",
        "fredapi",
        "requests",
        "yfinance",
    }
    lowered_parts = {part.lower() for part in name.split(".")}

    if _import_root(name) in disallowed_roots:
        return True
    if name.startswith("option_pricing.marketdata.providers.") and not name.startswith(
        "option_pricing.marketdata.providers.local"
    ):
        return True
    if "research" in lowered_parts:
        return True
    if "refresh" in lowered_parts:
        return True
    return False


def test_a5_modules_do_not_import_live_providers_cli_research_or_refresh() -> None:
    forbidden = {
        path.as_posix(): [
            name for name in _imported_names(path) if _is_disallowed_import(name)
        ]
        for path in (PIPELINE_FILE, BUNDLES_FILE)
    }

    assert forbidden == {
        PIPELINE_FILE.as_posix(): [],
        BUNDLES_FILE.as_posix(): [],
    }
