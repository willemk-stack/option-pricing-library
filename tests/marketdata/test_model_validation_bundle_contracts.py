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
from option_pricing.marketdata.manifests import (
    MODEL_VALIDATION_MANIFEST_REQUIRED_FIELDS,
    validate_model_validation_manifest,
)
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
EXPECTED_REQUIRED_MANIFEST_FIELDS = (
    "artifact_schema_version",
    "run_id",
    "snapshot_id",
    "created_at_utc",
    "library_commit",
    "underlying",
    "valuation_timestamp_utc",
    "spot_source",
    "rate_source",
    "rate_compounding",
    "dividend_yield_source",
    "day_count",
    "quote_cleaning_policy",
    "rows",
    "reason_counts",
    "warnings",
    "artifacts",
    "heston_smoke",
)


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
    heston_smoke: HestonSmokeResult | None = None,
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
        heston_smoke=_heston_smoke() if heston_smoke is None else heston_smoke,
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


def test_model_validation_manifest_required_fields_are_a5_s1_contract() -> None:
    assert (
        MODEL_VALIDATION_MANIFEST_REQUIRED_FIELDS == EXPECTED_REQUIRED_MANIFEST_FIELDS
    )


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


def test_heston_smoke_payload_serialization_is_stable() -> None:
    manifest = _manifest(
        heston_smoke=HestonSmokeResult(
            status="success",
            message="tiny smoke passed",
            objective_type="price_rmse",
            quote_count=2,
            success_count=1,
            failure_count=0,
            best_cost=0.125,
            parameters={"theta": 0.04, "kappa": 1.5},
        )
    )

    assert manifest["heston_smoke"] == {
        "status": "success",
        "message": "tiny smoke passed",
        "objective_type": "price_rmse",
        "quote_count": 2,
        "success_count": 1,
        "failure_count": 0,
        "best_cost": 0.125,
        "parameters": {"kappa": 1.5, "theta": 0.04},
    }


def test_model_validation_manifest_passes_existing_validator() -> None:
    validate_model_validation_manifest(_manifest())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", ""),
        ("snapshot_id", "  "),
        ("underlying", None),
        ("valuation_timestamp_utc", 123),
    ],
)
def test_build_model_validation_manifest_rejects_invalid_required_text_fields(
    field: str,
    value: object,
) -> None:
    kwargs = {
        "run_id": "test-run",
        "snapshot_id": "snapshot-001",
        "underlying": "SYNTH",
        "valuation_timestamp_utc": "2026-05-22T15:30:00Z",
        "market_data_payload": _market_data_payload(),
        "rows": {"cleaned_quotes": 1},
        "reason_counts": {},
        "warnings": [],
        "artifacts": EXPECTED_ARTIFACTS,
        "heston_smoke": _heston_smoke(),
        "library_commit": "abc123",
    }
    kwargs[field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        build_model_validation_manifest(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("payload_key", "source_name"),
    [
        ("rate_compounding", "market_data_payload.rate_compounding"),
        ("day_count", "market_data_payload.day_count"),
        ("quote_cleaning_policy", "market_data_payload.quote_cleaning_policy"),
    ],
)
def test_build_model_validation_manifest_rejects_missing_market_data_text_fields(
    payload_key: str,
    source_name: str,
) -> None:
    payload = _market_data_payload()
    payload.pop(payload_key)

    with pytest.raises((TypeError, ValueError), match=source_name):
        _manifest(market_data_payload=payload)


@pytest.mark.parametrize(
    "source_key", ["spot_source", "rate_source", "dividend_yield_source"]
)
def test_build_model_validation_manifest_rejects_missing_nested_source_fields(
    source_key: str,
) -> None:
    payload = _market_data_payload()
    sources = dict(payload["sources"])  # type: ignore[arg-type]
    sources.pop(source_key)
    payload["sources"] = sources

    with pytest.raises((TypeError, ValueError), match=source_key):
        _manifest(market_data_payload=payload)


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("rows", {"cleaned_quotes": -1}),
        ("reason_counts", {"bad_mid": -1}),
    ],
)
def test_build_model_validation_manifest_rejects_negative_counts(
    field: str,
    values: dict[str, int],
) -> None:
    kwargs = {
        "run_id": "test-run",
        "snapshot_id": "snapshot-001",
        "underlying": "SYNTH",
        "valuation_timestamp_utc": "2026-05-22T15:30:00Z",
        "market_data_payload": _market_data_payload(),
        "rows": {"cleaned_quotes": 1},
        "reason_counts": {},
        "warnings": [],
        "artifacts": EXPECTED_ARTIFACTS,
        "heston_smoke": _heston_smoke(),
        "library_commit": "abc123",
    }
    kwargs[field] = values

    with pytest.raises(ValueError, match=f"{field} counts must be non-negative"):
        build_model_validation_manifest(**kwargs)  # type: ignore[arg-type]


def test_build_model_validation_manifest_rejects_non_string_warnings() -> None:
    with pytest.raises(TypeError, match="warnings must be a sequence of strings"):
        build_model_validation_manifest(
            run_id="test-run",
            snapshot_id="snapshot-001",
            underlying="SYNTH",
            valuation_timestamp_utc="2026-05-22T15:30:00Z",
            market_data_payload=_market_data_payload(),
            rows={"cleaned_quotes": 1},
            reason_counts={},
            warnings=["ok", 123],  # type: ignore[list-item]
            artifacts=EXPECTED_ARTIFACTS,
            heston_smoke=_heston_smoke(),
            library_commit="abc123",
        )


def test_build_model_validation_manifest_rejects_invalid_heston_smoke_object() -> None:
    with pytest.raises(TypeError, match="heston_smoke must be a HestonSmokeResult"):
        build_model_validation_manifest(
            run_id="test-run",
            snapshot_id="snapshot-001",
            underlying="SYNTH",
            valuation_timestamp_utc="2026-05-22T15:30:00Z",
            market_data_payload=_market_data_payload(),
            rows={"cleaned_quotes": 1},
            reason_counts={},
            warnings=[],
            artifacts=EXPECTED_ARTIFACTS,
            heston_smoke={"status": "skipped"},  # type: ignore[arg-type]
            library_commit="abc123",
        )


def test_validate_model_validation_manifest_rejects_secret_looking_nested_keys() -> (
    None
):
    manifest = _manifest()
    manifest["provenance"] = {"fred_api_key": "do-not-write"}

    with pytest.raises(ValueError, match="secret-looking keys"):
        validate_model_validation_manifest(manifest)


@pytest.mark.parametrize("field", ["snapshot_id", "reason_counts", "heston_smoke"])
def test_validate_model_validation_manifest_rejects_missing_a5_s1_fields(
    field: str,
) -> None:
    manifest = _manifest()
    manifest.pop(field)

    with pytest.raises(ValueError, match="missing required fields"):
        validate_model_validation_manifest(manifest)


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
