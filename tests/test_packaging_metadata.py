from __future__ import annotations

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
