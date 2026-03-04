from __future__ import annotations

import os

import pytest


def _benchmarks_enabled() -> bool:
    return os.getenv("RUN_BENCHMARKS") == "1"


def _benchmarks_requested(config: pytest.Config) -> bool:
    return any("benchmarks" in str(arg) for arg in config.args)


def _is_benchmark_item(item: pytest.Item) -> bool:
    node_path = str(item.nodeid).split("::", maxsplit=1)[0]
    return "benchmarks/" in node_path or "benchmarks\\" in node_path


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    allow = _benchmarks_enabled() and _benchmarks_requested(config)
    if allow:
        return
    skip = pytest.mark.skip(reason="Set RUN_BENCHMARKS=1 to run benchmarks.")
    for item in items:
        if _is_benchmark_item(item):
            item.add_marker(skip)
