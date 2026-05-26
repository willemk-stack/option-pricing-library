"""Manifest validation contracts for marketdata artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .schemas import MODEL_VALIDATION_BUNDLE_VERSION, DatasetName, parse_dataset_name

MODEL_VALIDATION_MANIFEST_REQUIRED_FIELDS = (
    "artifact_schema_version",
    "run_id",
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
    "warnings",
    "artifacts",
)

_SECRET_MANIFEST_KEY_TERMS = frozenset(
    {
        "api_key",
        "secret_key",
        "token",
        "password",
        "alpaca_api_key",
        "alpaca_secret_key",
        "fred_api_key",
    }
)


def _is_secret_manifest_key(key: str) -> bool:
    normalized = key.strip().lower()
    return (
        normalized in _SECRET_MANIFEST_KEY_TERMS
        or "api_key" in normalized
        or "secret_key" in normalized
        or normalized.endswith("token")
        or "password" in normalized
    )


def _find_secret_manifest_keys(value: object, path: str = "") -> list[str]:
    secret_keys: list[str] = []

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            key_path = f"{path}.{key_text}" if path else key_text

            if _is_secret_manifest_key(key_text):
                secret_keys.append(key_path)

            secret_keys.extend(_find_secret_manifest_keys(item, key_path))

    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]" if path else f"[{index}]"
            secret_keys.extend(_find_secret_manifest_keys(item, item_path))

    return secret_keys


def validate_model_validation_manifest(manifest: Mapping[str, object]) -> None:
    """Validate the minimum model-validation bundle manifest contract."""

    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")

    missing = [
        field
        for field in MODEL_VALIDATION_MANIFEST_REQUIRED_FIELDS
        if field not in manifest
    ]

    if missing:
        raise ValueError(
            "model_validation_bundle manifest is missing required fields: " f"{missing}"
        )

    artifact_schema_version = manifest["artifact_schema_version"]
    if artifact_schema_version != MODEL_VALIDATION_BUNDLE_VERSION:
        raise ValueError(
            "model_validation_bundle manifest has artifact_schema_version "
            f"{artifact_schema_version!r}; expected "
            f"{MODEL_VALIDATION_BUNDLE_VERSION!r}"
        )

    secret_keys = _find_secret_manifest_keys(manifest)
    if secret_keys:
        raise ValueError(
            "model_validation_bundle manifest contains secret-looking keys: "
            f"{secret_keys}"
        )


def validate_manifest(
    manifest: Mapping[str, object],
    dataset_name: DatasetName | str = DatasetName.MODEL_VALIDATION_BUNDLE,
) -> None:
    """Validate a dataset manifest when a manifest-level contract exists."""

    parsed_name = parse_dataset_name(dataset_name)
    if parsed_name != DatasetName.MODEL_VALIDATION_BUNDLE:
        raise ValueError(f"No manifest validator is defined for {parsed_name.value!r}")

    validate_model_validation_manifest(manifest)


__all__ = [
    "MODEL_VALIDATION_MANIFEST_REQUIRED_FIELDS",
    "validate_manifest",
    "validate_model_validation_manifest",
]
