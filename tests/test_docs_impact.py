from __future__ import annotations

import pytest

from scripts.docs_impact import classify_docs_impact, git_changed_files


def test_examples_only_change_requires_readme_but_not_docs_site() -> None:
    impact = classify_docs_impact(["examples/quickstart.py"])

    assert impact.docs_sensitive is True
    assert impact.readme_required is True
    assert impact.benchmark_artifacts_required is False
    assert impact.docs_site_required is False
    assert impact.d2_required is False
    assert impact.visual_assets_required is False
    assert impact.review_paths == []


def test_docs_page_change_targets_that_page() -> None:
    impact = classify_docs_impact(["docs/index.md"])

    assert impact.docs_site_required is True
    assert impact.benchmark_artifacts_required is False
    assert impact.review_paths == ["/"]
    assert impact.full_review is False
    assert impact.authoritative_tests == [
        "sentinel.spec.ts",
        "pages.spec.ts",
        "components.spec.ts",
        "embedded-panels.spec.ts",
    ]


def test_generated_benchmark_asset_change_targets_performance_page() -> None:
    impact = classify_docs_impact(
        ["docs/assets/generated/benchmarks/benchmark_overview.light.svg"]
    )

    assert impact.docs_site_required is True
    assert impact.benchmark_artifacts_required is False
    assert impact.review_paths == ["/performance/"]
    assert impact.full_review is False
    assert impact.authoritative_tests == [
        "sentinel.spec.ts",
        "pages.spec.ts",
        "components.spec.ts",
        "embedded-panels.spec.ts",
    ]


def test_src_change_forces_full_review_and_visual_asset_refresh() -> None:
    impact = classify_docs_impact(["src/option_pricing/pricers/pde_pricer.py"])

    assert impact.docs_site_required is True
    assert impact.benchmark_artifacts_required is True
    assert impact.visual_assets_required is True
    assert impact.review_paths is None
    assert impact.full_review is True
    assert impact.authoritative_tests == [
        "sentinel.spec.ts",
        "pages.spec.ts",
        "components.spec.ts",
        "embedded-panels.spec.ts",
    ]


def test_benchmark_source_change_targets_performance_page() -> None:
    impact = classify_docs_impact(["benchmarks/test_bench_iv.py"])

    assert impact.docs_sensitive is True
    assert impact.docs_site_required is False
    assert impact.benchmark_artifacts_required is True
    assert impact.review_paths == []


def test_git_changed_files_reports_missing_refs_actionably() -> None:
    with pytest.raises(RuntimeError, match="checkout is shallow|base/head commits"):
        git_changed_files("refs/heads/definitely-missing-docs-impact-ref", "HEAD")
