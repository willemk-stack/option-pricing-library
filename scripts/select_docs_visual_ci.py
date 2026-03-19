from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from pre_push_docs_validation import (  # noqa: E402
    determine_authoritative_visual_tests,
    determine_review_paths,
    normalize_path,
)


def git_changed_files(base_ref: str, head_ref: str) -> list[str]:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            f"{base_ref}..{head_ref}",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(
        {normalize_path(path) for path in result.stdout.splitlines() if path.strip()}
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
        changed_files: list[str] = []
        review_paths = None
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
        review_paths = determine_review_paths(changed_files)

    selected_tests = determine_authoritative_visual_tests(review_paths)
    payload = {
        "changed_files": changed_files,
        "review_paths": review_paths,
        "selected_tests": selected_tests,
    }

    if args.mode == "github-outputs":
        output_path = Path(os.environ["GITHUB_OUTPUT"])
        with output_path.open("a", encoding="utf8") as handle:
            handle.write(f"selected_tests={' '.join(selected_tests)}\n")
            if review_paths is None:
                handle.write("review_paths=\n")
            else:
                handle.write(f"review_paths={','.join(review_paths)}\n")
            handle.write(
                "selection_json=" + json.dumps(payload, separators=(",", ":")) + "\n"
            )
        return 0

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
