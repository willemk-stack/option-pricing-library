from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from .docs_impact import (
        DocsImpact,
        classify_docs_impact,
        collect_changed_files,
        git_changed_files,
    )
except ImportError:
    from docs_impact import (
        DocsImpact,
        classify_docs_impact,
        collect_changed_files,
        git_changed_files,
    )

FAST_BLOCKING_SUITES = [
    "smoke.spec.ts",
    "dom-audits.spec.ts",
    "math-audits.spec.ts",
]
FAST_BLOCKING_PROJECTS = ["chromium-375", "chromium-1280"]
A11Y_SUITES = ["a11y.spec.ts"]
A11Y_PROJECTS = ["chromium-1280"]
SNAPSHOT_SUITES = ["sentinel.spec.ts"]


@dataclass(frozen=True, slots=True)
class DocsAuditPlan:
    docs_site_required: bool
    full_review: bool
    review_paths: list[str]
    a11y_paths: list[str]
    blocking_suites: list[str]
    a11y_suites: list[str]
    snapshot_suites: list[str]
    authoritative_suites: list[str]
    blocking_projects: list[str]
    a11y_projects: list[str]
    reason: list[str]


def impact_to_audit_plan(impact: DocsImpact) -> DocsAuditPlan:
    docs_site_required = impact.docs_site_required
    review_paths = [] if impact.review_paths is None else list(impact.review_paths)
    a11y_paths = [] if impact.a11y_paths is None else list(impact.a11y_paths)
    run_a11y = docs_site_required and (impact.full_review or bool(a11y_paths))
    reasons = (
        [f"{path} changed" for path in impact.changed_files]
        if impact.changed_files
        else ["Full docs review requested"]
    )

    return DocsAuditPlan(
        docs_site_required=docs_site_required,
        full_review=impact.full_review,
        review_paths=review_paths,
        a11y_paths=a11y_paths,
        blocking_suites=list(FAST_BLOCKING_SUITES) if docs_site_required else [],
        a11y_suites=list(A11Y_SUITES) if run_a11y else [],
        snapshot_suites=list(SNAPSHOT_SUITES) if docs_site_required else [],
        authoritative_suites=(
            list(impact.authoritative_tests) if docs_site_required else []
        ),
        blocking_projects=(list(FAST_BLOCKING_PROJECTS) if docs_site_required else []),
        a11y_projects=list(A11Y_PROJECTS) if run_a11y else [],
        reason=reasons,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Emit the docs browser audit plan used by CI, local repro, and agent "
            "loops so suites, projects, and path targeting come from one source."
        )
    )
    parser.add_argument(
        "--format",
        choices=("json", "github-outputs"),
        default="json",
        help="Emit JSON to stdout or write GitHub Actions outputs.",
    )
    parser.add_argument("--base-ref", help="Base ref or commit for git diff selection.")
    parser.add_argument("--head-ref", help="Head ref or commit for git diff selection.")
    parser.add_argument(
        "--changed-file",
        action="append",
        default=[],
        help="Explicit changed file path; may be provided more than once.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Explicit file list for local dry-runs or hook invocation.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Treat the change as a full docs review regardless of changed files.",
    )
    return parser.parse_args()


def select_changed_files(args: argparse.Namespace) -> list[str]:
    explicit_files = []
    if args.files is not None:
        explicit_files.extend(args.files)
    explicit_files.extend(args.changed_file)
    if explicit_files:
        return explicit_files
    if args.base_ref and args.head_ref:
        return git_changed_files(args.base_ref, args.head_ref)
    return collect_changed_files(None)


def plan_from_args(args: argparse.Namespace) -> DocsAuditPlan:
    if args.full:
        impact = DocsImpact(
            changed_files=[],
            docs_sensitive=True,
            docs_site_required=True,
            readme_required=True,
            performance_page_required=True,
            benchmark_artifacts_required=True,
            d2_required=True,
            visual_assets_required=True,
            full_review=True,
            review_paths=None,
            a11y_paths=None,
            authoritative_tests=[
                "sentinel.spec.ts",
                "pages.spec.ts",
                "components.spec.ts",
                "embedded-panels.spec.ts",
            ],
        )
    else:
        impact = classify_docs_impact(select_changed_files(args))
    return impact_to_audit_plan(impact)


def write_github_outputs(plan: DocsAuditPlan) -> None:
    output_path = Path(os.environ["GITHUB_OUTPUT"])
    payload = json.dumps(asdict(plan), separators=(",", ":"))
    with output_path.open("a", encoding="utf8") as handle:
        handle.write(
            f"docs_site_required={'true' if plan.docs_site_required else 'false'}\n"
        )
        handle.write(f"full_review={'true' if plan.full_review else 'false'}\n")
        handle.write(f"review_paths={','.join(plan.review_paths)}\n")
        handle.write(f"a11y_paths={','.join(plan.a11y_paths)}\n")
        handle.write(f"blocking_suites={' '.join(plan.blocking_suites)}\n")
        handle.write(f"a11y_suites={' '.join(plan.a11y_suites)}\n")
        handle.write(f"snapshot_suites={' '.join(plan.snapshot_suites)}\n")
        handle.write(f"authoritative_suites={' '.join(plan.authoritative_suites)}\n")
        handle.write(f"blocking_projects={' '.join(plan.blocking_projects)}\n")
        handle.write(f"a11y_projects={' '.join(plan.a11y_projects)}\n")
        handle.write(
            "reason_json=" + json.dumps(plan.reason, separators=(",", ":")) + "\n"
        )
        handle.write("audit_plan_json=" + payload + "\n")


def main() -> int:
    args = parse_args()
    plan = plan_from_args(args)

    if args.format == "github-outputs":
        write_github_outputs(plan)
        return 0

    print(json.dumps(asdict(plan), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
