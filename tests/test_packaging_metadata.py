from __future__ import annotations

import ast
import importlib
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project_metadata() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf8"))[
        "project"
    ]


def test_marketdata_dependencies_stay_optional_and_sdk_bound() -> None:
    project = project_metadata()
    core_dependencies = set(project["dependencies"])
    optional_dependencies = project["optional-dependencies"]

    assert core_dependencies == {"numpy", "scipy"}
    assert optional_dependencies["marketdata"] == [
        "pandas==2.2.3",
        "pyarrow",
        "requests",
        "tenacity",
        "alpaca-py",
    ]
    assert "pandas==2.2.3" in optional_dependencies["dev"]
    assert "pyarrow" in optional_dependencies["dev"]

    optional_only = (
        "alpaca",
        "fredapi",
        "requests",
        "tenacity",
        "yfinance",
        "duckdb",
        "pandas",
        "pyarrow",
    )
    for dependency in core_dependencies:
        assert not any(name in dependency.lower() for name in optional_only)


def test_marketdata_console_script_metadata() -> None:
    project = project_metadata()

    assert project["scripts"]["option-pricing-marketdata"] == (
        "option_pricing.marketdata.cli:main"
    )


def test_marketdata_console_script_target_imports() -> None:
    project = project_metadata()
    module_name, function_name = project["scripts"]["option-pricing-marketdata"].split(
        ":", maxsplit=1
    )

    module = importlib.import_module(module_name)

    assert callable(getattr(module, function_name))


def test_public_marketdata_imports_do_not_reference_provider_sdks() -> None:
    init_path = ROOT / "src" / "option_pricing" / "marketdata" / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf8"), filename=str(init_path))
    imported_modules: list[str] = []
    public_export_modules: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)
        elif isinstance(node, ast.Dict):
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    public_export_modules.append(value.value)

    provider_sdk_roots = ("alpaca", "alpaca_trade_api", "fredapi", "yfinance")
    assert not any(
        module_name.split(".", maxsplit=1)[0] in provider_sdk_roots
        for module_name in imported_modules
    )
    assert not any(
        ".providers." in module_name for module_name in public_export_modules
    )
