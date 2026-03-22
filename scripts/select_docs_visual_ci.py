from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from docs_impact import (
    DocsImpact,
    classify_docs_impact,
    determine_authoritative_visual_tests,
    git_changed_files,
    normalize_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the authoritative docs visual suites for CI so GitHub Actions "
            "uses the same targeting rules as the pre-push guard."
        )
    )
    parser.add_argument(
        "mode",
        choices=("json", "github-outputs"),
        nargs="?",
        default="json",
        help="Emit JSON to stdout or write GitHub Action outputs.",
    )
    parser.add_argument(
        "--base-ref",
        help="Base ref or commit for git diff selection.",
    )
    parser.add_argument(
        "--head-ref",
        help="Head ref or commit for git diff selection.",
    )
    parser.add_argument(
        "--changed-file",
        action="append",
        default=[],
        help="Explicit changed file path; may be provided more than once.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Select the full authoritative suite regardless of changed files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

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
            authoritative_tests=determine_authoritative_visual_tests(None),
        )
    else:
        explicit_files = [normalize_path(path) for path in args.changed_file if path]
        if explicit_files:
            changed_files = sorted(set(explicit_files))
        elif args.base_ref and args.head_ref:
            changed_files = git_changed_files(args.base_ref, args.head_ref)
        else:
            raise SystemExit(
                "Provide --full, at least one --changed-file, or both --base-ref and --head-ref."
            )
        impact = classify_docs_impact(changed_files)

    payload = {
        "changed_files": impact.changed_files,
        "review_paths": impact.review_paths,
        "selected_tests": impact.authoritative_tests,
    }

    if args.mode == "github-outputs":
        output_path = Path(os.environ["GITHUB_OUTPUT"])
        with output_path.open("a", encoding="utf8") as handle:
            handle.write(f"selected_tests={' '.join(impact.authoritative_tests)}\n")
            if impact.review_paths is None:
                handle.write("review_paths=\n")
            else:
                handle.write(f"review_paths={','.join(impact.review_paths)}\n")
            handle.write(
                "selection_json=" + json.dumps(payload, separators=(",", ":")) + "\n"
            )
        return 0

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
