from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

import option_pricing.marketdata.bundles as bundles_module
from option_pricing.marketdata.bundles import (
    ModelValidationBundleConfig,
    write_model_validation_bundle_artifacts,
)
from option_pricing.marketdata.pipeline import (
    LocalModelValidationPipelineResult,
    run_local_model_validation_pipeline,
)
from option_pricing.marketdata.schemas import DatasetName

REPO_ROOT = Path(__file__).resolve().parents[2]
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
    overwrite: bool = False,
    library_commit: str | None = "abc123",
) -> LocalModelValidationPipelineResult:
    return run_local_model_validation_pipeline(
        storage=tmp_path,
        run_id=run_id,
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


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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
