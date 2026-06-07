from __future__ import annotations

import ast
import builtins
import sys
from importlib import import_module
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "option_pricing"
MARKETDATA_PROVIDER_ROOT = SOURCE_ROOT / "marketdata" / "providers"
EXPECTED_MARKETDATA_IMPORTS = {
    "option_pricing.marketdata": (
        "MarketDataPipeline",
        "ProviderSnapshotResult",
        "ProviderSnapshotDataUnavailableError",
        "LoadedModelValidationBundle",
        "LocalStorage",
        "AlpacaConfig",
        "FredConfig",
        "StorageConfig",
        "PipelineConfig",
        "load_model_validation_bundle",
        "provider_snapshot_public_summary",
    ),
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
EXPECTED_PIPELINE_IMPORTS = (
    "LocalModelValidationPipelineResult",
    "MarketDataPipeline",
    "ProviderRefreshDailyCounts",
    "ProviderRefreshDailyResult",
    "ProviderSnapshotBronzePaths",
    "ProviderSnapshotDataUnavailableError",
    "ProviderSnapshotResult",
    "ProviderSnapshotSilverPaths",
    "provider_snapshot_public_summary",
    "run_local_model_validation_pipeline",
)
ORDINARY_MARKETDATA_IMPORTS = (
    "option_pricing",
    "option_pricing.marketdata",
    "option_pricing.marketdata.config",
    "option_pricing.marketdata.schemas",
    "option_pricing.marketdata.storage",
)
OPTIONAL_MARKETDATA_IMPORT_ROOTS = {
    "alpaca",
    "alpaca_trade_api",
    "fredapi",
    "pandas",
    "pyarrow",
    "requests",
    "tenacity",
    "yfinance",
}
PROVIDER_SDK_IMPORT_ROOTS = {
    "alpaca",
    "alpaca_trade_api",
    "fredapi",
    "requests",
    "tenacity",
    "yfinance",
}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=path.as_posix())
    roots: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.split(".", maxsplit=1)[0])

    return roots


def test_phase_a1_marketdata_public_import_boundaries() -> None:
    for module_name, public_symbols in EXPECTED_MARKETDATA_IMPORTS.items():
        module = import_module(module_name)

        for symbol in public_symbols:
            assert getattr(module, symbol) is not None


def test_phase_a1_marketdata_pipeline_facade_exports() -> None:
    module = import_module("option_pricing.marketdata.pipeline")

    for symbol in EXPECTED_PIPELINE_IMPORTS:
        assert getattr(module, symbol) is not None


def test_b1_s1_ordinary_imports_do_not_require_marketdata_optional_deps(
    monkeypatch,
) -> None:
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            level == 0
            and name.split(".", maxsplit=1)[0] in OPTIONAL_MARKETDATA_IMPORT_ROOTS
        ):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return original_import(name, globals, locals, fromlist, level)

    for module_name in ORDINARY_MARKETDATA_IMPORTS:
        sys.modules.pop(module_name, None)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    for module_name in ORDINARY_MARKETDATA_IMPORTS:
        assert import_module(module_name) is not None


def test_b1_s1_non_provider_modules_do_not_import_provider_sdks() -> None:
    offenders: dict[str, list[str]] = {}

    for path in SOURCE_ROOT.rglob("*.py"):
        if MARKETDATA_PROVIDER_ROOT in path.parents:
            continue
        forbidden = sorted(_imported_roots(path) & PROVIDER_SDK_IMPORT_ROOTS)
        if forbidden:
            offenders[path.relative_to(SOURCE_ROOT).as_posix()] = forbidden

    assert offenders == {}
