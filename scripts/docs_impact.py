from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from .benchmark_source_scope import is_benchmark_freshness_input
except ImportError:
    from benchmark_source_scope import is_benchmark_freshness_input

ROOT = Path(__file__).resolve().parents[1]
EMPTY_TREE_HASH = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

README_INPUTS = (
    "README.template.md",
    "examples/",
    "scripts/render_readme.py",
)

PERFORMANCE_PAGE_INPUTS = (
    "benchmarks/artifacts/",
    "scripts/render_performance_page.py",
    "scripts/templates/performance.md.template",
)

BENCHMARK_SNAPSHOT_INPUTS = (
    "benchmarks/",
    "docs/performance.md",
    "scripts/build_benchmark_artifacts.py",
    "scripts/render_performance_page.py",
    "scripts/templates/performance.md.template",
)

D2_INPUTS = (
    "docs/assets/diagrams/src/",
    "scripts/render_d2_diagrams.py",
)

VISUAL_ASSET_INPUTS = (
    "benchmarks/artifacts/",
    "docs/assets/generated/",
    "src/option_pricing/",
    "scripts/build_benchmark_artifacts.py",
    "scripts/build_visual_artifacts.py",
    "scripts/visual_audit/",
)

DOCS_SITE_INPUTS = (
    "benchmarks/artifacts/",
    "docs/",
    "mkdocs.yml",
    "scripts/render_performance_page.py",
    "src/",
    "tests/visual/",
    "scripts/build_benchmark_artifacts.py",
    "scripts/build_visual_artifacts.py",
    "scripts/docs_site_contract.py",
    "scripts/render_d2_diagrams.py",
    "scripts/run_ci_visual_regression.py",
    "scripts/select_docs_visual_ci.py",
    "scripts/serve_docs.py",
    "scripts/visual_audit/",
)

FULL_REVIEW_INPUTS = (
    "benchmarks/artifacts/",
    "docs/stylesheets/",
    "docs/assets/diagrams/",
    "docs/assets/javascripts/",
    "mkdocs.yml",
    "scripts/render_performance_page.py",
    "src/",
    "tests/visual/",
    "scripts/build_benchmark_artifacts.py",
    "scripts/build_visual_artifacts.py",
    "scripts/docs_site_contract.py",
    "scripts/render_d2_diagrams.py",
    "scripts/run_ci_visual_regression.py",
    "scripts/select_docs_visual_ci.py",
    "scripts/serve_docs.py",
    "scripts/visual_audit/",
)

LEGACY_PAGE_ALIASES = {
    "docs/user_guides/flagship_capstone2_page.md": "/user_guides/decision_guide/",
    "docs/user_guides/flagship_surface.md": "/user_guides/surface_workflow/",
    "docs/user_guides/flagship_essvi_bridge.md": "/user_guides/essvi_smooth_handoff/",
    "docs/user_guides/flagship_localvol_pde.md": "/user_guides/localvol_pde_validation/",
}

CURATED_A11Y_PATHS = {
    "/",
    "/architecture/",
    "/installation/",
    "/performance/",
    "/user_guides/quickstart/",
    "/user_guides/instruments/",
    "/user_guides/market_api/",
    "/user_guides/decision_guide/",
    "/user_guides/surface_workflow/",
    "/user_guides/essvi_smooth_handoff/",
    "/user_guides/localvol_pde_validation/",
    "/api/",
    "/api/public/",
    "/api/pricers/",
    "/api/vol/",
}

AUTHORITATIVE_COMPONENT_PATHS = {
    "/",
    "/performance/",
    "/user_guides/surface_workflow/",
}

AUTHORITATIVE_EMBEDDED_PANEL_PATHS = {
    "/",
    "/performance/",
}

GENERATED_ASSET_REVIEW_PATHS = {
    "benchmarks": ("/performance/",),
    "showcase": ("/",),
    "static": ("/user_guides/surface_workflow/",),
    "dupire": ("/user_guides/essvi_smooth_handoff/",),
    "numerics": ("/user_guides/localvol_pde_validation/",),
}


@dataclass(frozen=True, slots=True)
class DocsImpact:
    changed_files: list[str]
    docs_sensitive: bool
    docs_site_required: bool
    readme_required: bool
    performance_page_required: bool
    benchmark_artifacts_required: bool
    d2_required: bool
    visual_assets_required: bool
    full_review: bool
    review_paths: list[str] | None
    a11y_paths: list[str] | None
    authoritative_tests: list[str]


def normalize_path(path: str) -> str:
    return Path(path).as_posix().lstrip("./")


def _matches(path: str, patterns: tuple[str, ...]) -> bool:
    return path in patterns or any(path.startswith(pattern) for pattern in patterns)


def is_docs_sensitive(path: str) -> bool:
    return (
        _matches(path, README_INPUTS)
        or _matches(path, PERFORMANCE_PAGE_INPUTS)
        or is_benchmark_freshness_input(path, ROOT)
        or _matches(path, D2_INPUTS)
        or _matches(path, VISUAL_ASSET_INPUTS)
        or _matches(path, DOCS_SITE_INPUTS)
    )


def docs_file_to_review_path(path: str) -> str | None:
    normalized = normalize_path(path)
    if not normalized.startswith("docs/") or not normalized.endswith(".md"):
        return None

    relative = normalized.removeprefix("docs/")
    if relative == "index.md":
        return "/"

    if relative.endswith("/index.md"):
        relative = relative[: -len("index.md")]
        return f"/{relative.strip('/')}/"

    slug = relative[: -len(".md")]
    return f"/{slug.strip('/')}/"


def generated_asset_to_review_paths(path: str) -> tuple[str, ...]:
    normalized = normalize_path(path)
    marker = "docs/assets/generated/"
    if marker not in normalized:
        return ()

    relative = normalized.split(marker, 1)[1]
    family = relative.split("/", 1)[0]
    return GENERATED_ASSET_REVIEW_PATHS.get(family, ())


def determine_review_paths(changed_files: list[str]) -> list[str] | None:
    if any(_matches(path, FULL_REVIEW_INPUTS) for path in changed_files):
        return None

    review_paths: set[str] = set()
    for path in changed_files:
        alias = LEGACY_PAGE_ALIASES.get(path)
        if alias:
            review_paths.add(alias)

        review_path = docs_file_to_review_path(path)
        if review_path:
            review_paths.add(review_path)

        if _matches(path, BENCHMARK_SNAPSHOT_INPUTS):
            review_paths.add("/performance/")

        review_paths.update(generated_asset_to_review_paths(path))

    return sorted(review_paths) or None


def determine_a11y_paths(review_paths: list[str] | None) -> list[str] | None:
    if review_paths is None:
        return None

    selected = [path for path in review_paths if path in CURATED_A11Y_PATHS]
    return selected or []


def determine_authoritative_visual_tests(review_paths: list[str] | None) -> list[str]:
    tests = ["sentinel.spec.ts", "pages.spec.ts"]
    if review_paths is None:
        return [*tests, "components.spec.ts", "embedded-panels.spec.ts"]

    review_path_set = set(review_paths)
    if review_path_set & AUTHORITATIVE_COMPONENT_PATHS:
        tests.append("components.spec.ts")
    if review_path_set & AUTHORITATIVE_EMBEDDED_PANEL_PATHS:
        tests.append("embedded-panels.spec.ts")
    return tests


def classify_docs_impact(changed_files: list[str]) -> DocsImpact:
    normalized = sorted(
        {normalize_path(path) for path in changed_files if path.strip()}
    )
    docs_sensitive_files = [path for path in normalized if is_docs_sensitive(path)]
    docs_site_required = any(
        _matches(path, DOCS_SITE_INPUTS) for path in docs_sensitive_files
    )
    readme_required = any(
        _matches(path, README_INPUTS) for path in docs_sensitive_files
    )
    performance_page_required = any(
        _matches(path, PERFORMANCE_PAGE_INPUTS) for path in docs_sensitive_files
    )
    benchmark_artifacts_required = any(
        _matches(path, BENCHMARK_SNAPSHOT_INPUTS)
        or is_benchmark_freshness_input(path, ROOT)
        for path in docs_sensitive_files
    )
    d2_required = any(_matches(path, D2_INPUTS) for path in docs_sensitive_files)
    visual_assets_required = any(
        _matches(path, VISUAL_ASSET_INPUTS) for path in docs_sensitive_files
    )
    review_paths = (
        determine_review_paths(docs_sensitive_files) if docs_site_required else []
    )
    a11y_paths = determine_a11y_paths(review_paths)
    authoritative_tests = determine_authoritative_visual_tests(review_paths)
    return DocsImpact(
        changed_files=docs_sensitive_files,
        docs_sensitive=bool(docs_sensitive_files),
        docs_site_required=docs_site_required,
        readme_required=readme_required,
        performance_page_required=performance_page_required,
        benchmark_artifacts_required=benchmark_artifacts_required,
        d2_required=d2_required,
        visual_assets_required=visual_assets_required,
        full_review=review_paths is None,
        review_paths=review_paths,
        a11y_paths=a11y_paths,
        authoritative_tests=authoritative_tests,
    )


def ref_exists(ref: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def merge_base(target_ref: str, base_ref: str) -> str | None:
    if not ref_exists(base_ref):
        return None

    result = subprocess.run(
        ["git", "merge-base", target_ref, base_ref],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def guess_push_base(target_ref: str) -> str:
    for candidate in (
        "@{upstream}",
        "origin/HEAD",
        "origin/main",
        "origin/master",
        "main",
        "master",
    ):
        base = merge_base(target_ref, candidate)
        if base:
            return base

    head_parent = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{target_ref}~1"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if head_parent.returncode == 0:
        return head_parent.stdout.strip()

    return EMPTY_TREE_HASH


def git_changed_files(base_ref: str, head_ref: str) -> list[str]:
    missing_refs = [ref for ref in (base_ref, head_ref) if not ref_exists(ref)]
    if missing_refs:
        missing_list = ", ".join(missing_refs)
        raise RuntimeError(
            "Cannot determine changed files because git refs are unavailable locally: "
            f"{missing_list}. This usually means the checkout is shallow or the "
            "workflow fetched only the PR merge commit. Fetch full history or at "
            "least the base/head commits before running docs_impact selection."
        )

    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            f"{base_ref}..{head_ref}",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or "Unknown git diff failure."
        raise RuntimeError(
            "Failed to determine changed files from git diff for refs "
            f"{base_ref}..{head_ref}: {detail}"
        )
    return [path for path in result.stdout.splitlines() if path.strip()]


def collect_changed_files(explicit_files: list[str] | None) -> list[str]:
    if explicit_files is not None:
        return explicit_files

    to_ref = os.environ.get("PRE_COMMIT_TO_REF") or "HEAD"
    from_ref = os.environ.get("PRE_COMMIT_FROM_REF")

    if from_ref and set(from_ref) != {"0"}:
        base_ref = from_ref
    else:
        base_ref = guess_push_base(to_ref)

    return git_changed_files(base_ref, to_ref)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify docs-related impact for local hooks and CI so all docs refresh "
            "logic uses the same source-to-output rules."
        )
    )
    parser.add_argument(
        "mode",
        choices=("json", "github-outputs"),
        nargs="?",
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


def write_github_outputs(impact: DocsImpact) -> None:
    output_path = Path(os.environ["GITHUB_OUTPUT"])
    payload = json.dumps(asdict(impact), separators=(",", ":"))
    with output_path.open("a", encoding="utf8") as handle:
        handle.write(f"docs_sensitive={'true' if impact.docs_sensitive else 'false'}\n")
        handle.write(
            f"docs_site_required={'true' if impact.docs_site_required else 'false'}\n"
        )
        handle.write(
            f"readme_required={'true' if impact.readme_required else 'false'}\n"
        )
        handle.write(
            f"performance_page_required={'true' if impact.performance_page_required else 'false'}\n"
        )
        handle.write(
            f"benchmark_artifacts_required={'true' if impact.benchmark_artifacts_required else 'false'}\n"
        )
        handle.write(f"d2_required={'true' if impact.d2_required else 'false'}\n")
        handle.write(
            f"visual_assets_required={'true' if impact.visual_assets_required else 'false'}\n"
        )
        handle.write(f"full_review={'true' if impact.full_review else 'false'}\n")
        handle.write(f"selected_tests={' '.join(impact.authoritative_tests)}\n")
        handle.write(
            "review_paths="
            + ("" if impact.review_paths is None else ",".join(impact.review_paths))
            + "\n"
        )
        handle.write(
            "a11y_paths="
            + ("" if impact.a11y_paths is None else ",".join(impact.a11y_paths))
            + "\n"
        )
        handle.write("impact_json=" + payload + "\n")


def main() -> int:
    args = parse_args()
    if args.full:
        changed_files = ["docs/full-review"]
        impact = DocsImpact(
            changed_files=changed_files,
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
            authoritative_tests=determine_authoritative_visual_tests(None),
        )
    else:
        explicit_files = []
        if args.files is not None:
            explicit_files.extend(args.files)
        explicit_files.extend(args.changed_file)
        if explicit_files:
            changed_files = explicit_files
        elif args.base_ref and args.head_ref:
            changed_files = git_changed_files(args.base_ref, args.head_ref)
        else:
            changed_files = collect_changed_files(None)
        impact = classify_docs_impact(changed_files)

    if args.mode == "github-outputs":
        write_github_outputs(impact)
        return 0

    print(json.dumps(asdict(impact), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
