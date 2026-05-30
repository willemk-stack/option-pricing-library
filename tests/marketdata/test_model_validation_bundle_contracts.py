from __future__ import annotations

import ast
import json
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import pytest

import option_pricing.marketdata.bundles as bundles_module
from option_pricing.marketdata.bundles import (
    HestonSmokeResult,
    ModelValidationBundleConfig,
    ModelValidationBundlePaths,
    build_model_validation_manifest,
)
from option_pricing.marketdata.contracts import ModelValidationBundleResult
from option_pricing.marketdata.manifests import validate_model_validation_manifest
from option_pricing.marketdata.schemas import MODEL_VALIDATION_BUNDLE_VERSION

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


def _market_data_payload() -> dict[str, object]:
    return {
        "schema_version": "gold_market_data.v1",
        "underlying": "SYNTH",
        "valuation_timestamp_utc": "2026-05-22T15:30:00Z",
        "run_id": "test-run",
        "snapshot_id": "snapshot-001",
        "market_data": {
            "spot": 100.0,
            "rate": 0.04,
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


def _heston_smoke() -> HestonSmokeResult:
    return HestonSmokeResult(
        status="skipped",
        message="not run in A5-S1",
        objective_type="price_rmse",
        quote_count=12,
    )


def _manifest(
    *,
    market_data_payload: dict[str, object] | None = None,
    artifacts: dict[str, str] | None = None,
) -> dict[str, object]:
    return build_model_validation_manifest(
        run_id="test-run",
        snapshot_id="snapshot-001",
        underlying="SYNTH",
        valuation_timestamp_utc="2026-05-22T15:30:00Z",
        market_data_payload=(
            _market_data_payload()
            if market_data_payload is None
            else market_data_payload
        ),
        rows={
            "market_inputs": 1,
            "cleaned_quotes": 12,
            "rejected_quotes": 3,
            "heston_quotes": 12,
            "surface_inputs": 0,
        },
        reason_counts={"expired": 1, "bad_mid": 2},
        warnings=["surface grid is sparse"],
        artifacts=EXPECTED_ARTIFACTS if artifacts is None else artifacts,
        heston_smoke=_heston_smoke(),
        library_commit="abc123",
    )


def test_public_dataclasses_have_exact_expected_fields() -> None:
    assert tuple(field.name for field in fields(ModelValidationBundlePaths)) == (
        "root",
        "manifest",
        "market_data",
        "cleaned_quotes",
        "rejected_quotes",
        "heston_quotes",
        "surface_inputs",
        "heston_fit_summary",
        "warnings",
    )
    assert tuple(field.name for field in fields(HestonSmokeResult)) == (
        "status",
        "message",
        "objective_type",
        "quote_count",
        "success_count",
        "failure_count",
        "best_cost",
        "parameters",
    )
    assert tuple(field.name for field in fields(ModelValidationBundleConfig)) == (
        "run_heston_smoke",
        "heston_objective_type",
        "heston_max_seeds",
        "heston_max_nfev",
        "fail_on_heston_smoke_failure",
    )


def test_bundle_all_exposes_intended_a5_s1_api() -> None:
    assert tuple(bundles_module.__all__) == (
        "HestonSmokeResult",
        "ModelValidationBundleConfig",
        "ModelValidationBundlePaths",
        "ModelValidationBundleResult",
        "build_model_validation_manifest",
    )
    assert bundles_module.ModelValidationBundleResult is ModelValidationBundleResult
    for symbol in bundles_module.__all__:
        assert getattr(bundles_module, symbol) is not None


def test_model_validation_bundle_config_defaults_are_frozen() -> None:
    config = ModelValidationBundleConfig()

    assert config.run_heston_smoke is True
    assert config.heston_objective_type == "price_rmse"
    assert config.heston_max_seeds == 1
    assert config.heston_max_nfev == 1
    assert config.fail_on_heston_smoke_failure is False


def test_build_model_validation_manifest_returns_s1_shape() -> None:
    manifest = _manifest()

    assert manifest["artifact_schema_version"] == MODEL_VALIDATION_BUNDLE_VERSION
    assert manifest["run_id"] == "test-run"
    assert manifest["snapshot_id"] == "snapshot-001"
    assert isinstance(manifest["created_at_utc"], str)
    datetime.fromisoformat(str(manifest["created_at_utc"]).replace("Z", "+00:00"))
    assert manifest["library_commit"] == "abc123"
    assert manifest["underlying"] == "SYNTH"
    assert manifest["valuation_timestamp_utc"] == "2026-05-22T15:30:00Z"
    assert manifest["spot_source"] == "local_fixture"
    assert manifest["rate_source"] == "local_fixture"
    assert manifest["rate_compounding"] == "continuous"
    assert manifest["dividend_yield_source"] == "assumption"
    assert manifest["day_count"] == "ACT/365"
    assert manifest["quote_cleaning_policy"] == "quote_cleaning_policy.v1"
    assert manifest["rows"] == {
        "market_inputs": 1,
        "cleaned_quotes": 12,
        "rejected_quotes": 3,
        "heston_quotes": 12,
        "surface_inputs": 0,
    }
    assert manifest["reason_counts"] == {"bad_mid": 2, "expired": 1}
    assert manifest["warnings"] == ["surface grid is sparse"]
    assert manifest["artifacts"] == EXPECTED_ARTIFACTS
    assert manifest["heston_smoke"] == {
        "status": "skipped",
        "message": "not run in A5-S1",
        "objective_type": "price_rmse",
        "quote_count": 12,
        "success_count": None,
        "failure_count": None,
        "best_cost": None,
        "parameters": None,
    }
    assert "schema_version" not in manifest


def test_model_validation_manifest_passes_existing_validator() -> None:
    validate_model_validation_manifest(_manifest())


def test_artifact_map_uses_frozen_bundle_local_filenames() -> None:
    manifest = _manifest()

    assert manifest["artifacts"] == {
        "market_data": "market_data.json",
        "cleaned_quotes": "cleaned_quotes.parquet",
        "rejected_quotes": "rejected_quotes.parquet",
        "heston_quotes": "heston_quotes.parquet",
        "surface_inputs": "surface_inputs.parquet",
        "heston_fit_summary": "heston_fit_summary.csv",
        "warnings": "warnings.json",
    }

    invalid_artifacts = dict(EXPECTED_ARTIFACTS)
    invalid_artifacts["market_data"] = (
        REPO_ROOT / "out" / "market_data.json"
    ).as_posix()
    with pytest.raises(ValueError, match="frozen A5-S1 filenames"):
        _manifest(artifacts=invalid_artifacts)


def test_manifest_rejected_quote_policy_excludes_row_details() -> None:
    market_data_payload = _market_data_payload()
    market_data_payload["rejected_quote_rows"] = [
        {
            "quote_id": "quote-001",
            "rejection_reason": "bad_mid",
            "rejection_detail": "bid was greater than ask",
        }
    ]
    market_data_payload["rejection_detail"] = "external row detail"

    manifest = _manifest(market_data_payload=market_data_payload)

    encoded = json.dumps(manifest, sort_keys=True)
    assert "rejected_quote_rows" not in encoded
    assert "rejection_detail" not in encoded
    assert "bid was greater than ask" not in encoded
    assert "external row detail" not in encoded
    assert manifest["rows"] == {
        "market_inputs": 1,
        "cleaned_quotes": 12,
        "rejected_quotes": 3,
        "heston_quotes": 12,
        "surface_inputs": 0,
    }
    assert manifest["reason_counts"] == {"bad_mid": 2, "expired": 1}


def test_warnings_json_shape_is_reserved_for_later_artifact_writing() -> None:
    warnings_payload = {
        "warnings": [],
        "data_quality": [],
        "heston_smoke": [],
    }

    assert warnings_payload == {
        "warnings": [],
        "data_quality": [],
        "heston_smoke": [],
    }
    assert all(isinstance(value, list) for value in warnings_payload.values())


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
    if name.startswith("option_pricing.marketdata.providers"):
        return True
    if name.startswith("option_pricing.diagnostics.heston"):
        return True
    if name.startswith("option_pricing.models.heston.calibration"):
        return True
    if "research" in lowered_parts:
        return True
    return False


def test_bundles_module_does_not_import_out_of_scope_dependencies() -> None:
    forbidden = [
        name for name in _imported_names(BUNDLES_FILE) if _is_disallowed_import(name)
    ]

    assert forbidden == []
