from __future__ import annotations

from importlib import import_module

EXPECTED_MARKETDATA_IMPORTS = {
    "option_pricing.marketdata.config": (
        "AlpacaConfig",
        "FredConfig",
        "PipelineConfig",
        "StorageConfig",
    ),
    "option_pricing.marketdata.contracts": (
        "RunMetadata",
        "ResultStats",
        "PipelineResult",
        "SnapshotResult",
        "ModelValidationBundleResult",
        "ResearchBundleResult",
        "BackfillResult",
    ),
    "option_pricing.marketdata.schemas": (
        "DatasetName",
        "DATASET_COLUMNS",
        "DATASET_DTYPES",
        "MARKET_INPUTS_SCHEMA_VERSION",
        "CLEANED_QUOTES_SCHEMA_VERSION",
        "REJECTED_QUOTES_SCHEMA_VERSION",
        "HESTON_QUOTES_SCHEMA_VERSION",
        "SURFACE_INPUTS_SCHEMA_VERSION",
        "MODEL_VALIDATION_BUNDLE_VERSION",
    ),
    "option_pricing.marketdata.validation": (
        "dataset_columns",
        "dataset_dtypes",
        "validate_columns",
        "validate_dtypes",
        "order_columns",
        "coerce_frame",
    ),
    "option_pricing.marketdata.manifests": (
        "MODEL_VALIDATION_MANIFEST_REQUIRED_FIELDS",
        "validate_manifest",
        "validate_model_validation_manifest",
    ),
    "option_pricing.marketdata.storage": ("LocalStorage",),
}


def test_phase_a1_marketdata_public_import_boundaries() -> None:
    for module_name, public_symbols in EXPECTED_MARKETDATA_IMPORTS.items():
        module = import_module(module_name)

        for symbol in public_symbols:
            assert getattr(module, symbol) is not None
