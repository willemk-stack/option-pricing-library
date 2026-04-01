from __future__ import annotations

import argparse

import scripts.refresh_benchmark_source_manifest as refresh_manifest


def test_main_refreshes_manifest_for_benchmark_sensitive_files(
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(
        refresh_manifest,
        "_parse_args",
        lambda argv=None: argparse.Namespace(
            files=["src/option_pricing/pricers/pde_pricer.py"]
        ),
    )
    monkeypatch.setattr(
        refresh_manifest,
        "is_benchmark_freshness_input",
        lambda path, root: path.endswith("pde_pricer.py"),
    )
    monkeypatch.setattr(
        refresh_manifest.benchmark_artifacts,
        "main",
        lambda argv=None: calls.append(list(argv or [])) or 0,
    )

    assert refresh_manifest.main() == 0
    assert calls == [["--write-source-manifest"]]


def test_main_skips_manifest_refresh_for_out_of_scope_files(monkeypatch) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(
        refresh_manifest,
        "_parse_args",
        lambda argv=None: argparse.Namespace(
            files=["src/option_pricing/pricers/heston.py"]
        ),
    )
    monkeypatch.setattr(
        refresh_manifest,
        "is_benchmark_freshness_input",
        lambda path, root: False,
    )
    monkeypatch.setattr(
        refresh_manifest.benchmark_artifacts,
        "main",
        lambda argv=None: calls.append(list(argv or [])) or 0,
    )

    assert refresh_manifest.main() == 0
    assert calls == []
