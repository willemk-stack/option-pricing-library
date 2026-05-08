from __future__ import annotations

from scripts.docs_audit_plan import impact_to_audit_plan
from scripts.docs_impact import classify_docs_impact


def test_docs_page_change_emits_fast_blocking_audit_plan() -> None:
    plan = impact_to_audit_plan(classify_docs_impact(["docs/performance.md"]))

    assert plan.docs_site_required is True
    assert plan.full_review is False
    assert plan.review_paths == ["/performance/"]
    assert plan.a11y_paths == ["/performance/"]
    assert plan.blocking_suites == [
        "smoke.spec.ts",
        "dom-audits.spec.ts",
        "math-audits.spec.ts",
    ]
    assert plan.a11y_suites == ["a11y.spec.ts"]
    assert plan.snapshot_suites == ["sentinel.spec.ts"]
    assert plan.blocking_projects == ["chromium-375", "chromium-1280"]
    assert plan.a11y_projects == ["chromium-1280"]
    assert plan.reason == ["docs/performance.md changed"]


def test_non_curated_page_change_skips_a11y_suite() -> None:
    plan = impact_to_audit_plan(
        classify_docs_impact(["docs/architecture/docs-pipeline.md"])
    )

    assert plan.docs_site_required is True
    assert plan.review_paths == ["/architecture/docs-pipeline/"]
    assert plan.a11y_paths == []
    assert plan.a11y_suites == []
    assert plan.a11y_projects == []


def test_full_review_emits_full_scope_a11y_and_authoritative_suites() -> None:
    plan = impact_to_audit_plan(
        classify_docs_impact(["src/option_pricing/pricers/pde_pricer.py"])
    )

    assert plan.docs_site_required is True
    assert plan.full_review is True
    assert plan.review_paths == []
    assert plan.a11y_paths == []
    assert plan.a11y_suites == ["a11y.spec.ts"]
    assert plan.authoritative_suites == [
        "sentinel.spec.ts",
        "pages.spec.ts",
        "components.spec.ts",
        "embedded-panels.spec.ts",
    ]


def test_benchmark_only_change_emits_no_browser_audit_suites() -> None:
    plan = impact_to_audit_plan(classify_docs_impact(["benchmarks/test_bench_iv.py"]))

    assert plan.docs_site_required is False
    assert plan.blocking_suites == []
    assert plan.a11y_suites == []
    assert plan.snapshot_suites == []
    assert plan.blocking_projects == []
