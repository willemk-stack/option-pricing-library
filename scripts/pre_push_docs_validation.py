from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS_VISUAL_DIR = ROOT / "tests" / "visual"
DEFAULT_DOCS_BASE_URL = "http://127.0.0.1:8125/option-pricing-library/"
EMPTY_TREE_HASH = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

DOCS_SENSITIVE_PREFIXES = (
    "docs/",
    "tests/visual/",
    "scripts/serve_docs.py",
    "scripts/build_visual_artifacts.py",
    "scripts/render_d2_diagrams.py",
    "scripts/visual_audit/",
)

FULL_REVIEW_PREFIXES = (
    "mkdocs.yml",
    "docs/stylesheets/",
    "docs/assets/",
    "tests/visual/",
    "scripts/serve_docs.py",
    "scripts/build_visual_artifacts.py",
    "scripts/render_d2_diagrams.py",
    "scripts/visual_audit/",
)

ASSET_REBUILD_PREFIXES = (
    "docs/assets/",
    "scripts/build_visual_artifacts.py",
    "scripts/render_d2_diagrams.py",
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
    "/performance/",
    "/user_guides/decision_guide/",
    "/user_guides/surface_workflow/",
    "/user_guides/essvi_smooth_handoff/",
    "/user_guides/localvol_pde_validation/",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run targeted docs validation before push when docs-sensitive files "
            "changed."
        )
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional explicit file list for local dry-runs.",
    )
    return parser.parse_args()


def run(
    command: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> None:
    printable = " ".join(command)
    print(f"\n> {printable}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def git_stdout(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def normalize_path(path: str) -> str:
    return Path(path).as_posix().lstrip("./")


def is_zero_ref(ref: str | None) -> bool:
    return bool(ref) and set(ref) == {"0"}


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


def collect_changed_files(explicit_files: list[str] | None) -> list[str]:
    if explicit_files is not None:
        return sorted({normalize_path(path) for path in explicit_files if path})

    to_ref = os.environ.get("PRE_COMMIT_TO_REF") or "HEAD"
    from_ref = os.environ.get("PRE_COMMIT_FROM_REF")

    if from_ref and not is_zero_ref(from_ref):
        base_ref = from_ref
    else:
        base_ref = guess_push_base(to_ref)

    diff_output = git_stdout(
        ["diff", "--name-only", "--diff-filter=ACMR", f"{base_ref}..{to_ref}"]
    )
    if not diff_output:
        return []

    return sorted(
        {normalize_path(path) for path in diff_output.splitlines() if path.strip()}
    )


def is_docs_sensitive(path: str) -> bool:
    return path == "mkdocs.yml" or any(
        path.startswith(prefix) for prefix in DOCS_SENSITIVE_PREFIXES
    )


def needs_full_review(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in FULL_REVIEW_PREFIXES)


def needs_asset_rebuild(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in ASSET_REBUILD_PREFIXES)


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


def determine_review_paths(changed_files: list[str]) -> list[str] | None:
    if any(needs_full_review(path) for path in changed_files):
        return None

    review_paths: set[str] = set()
    for path in changed_files:
        alias = LEGACY_PAGE_ALIASES.get(path)
        if alias:
            review_paths.add(alias)

        review_path = docs_file_to_review_path(path)
        if review_path:
            review_paths.add(review_path)

    return sorted(review_paths) or None


def determine_a11y_paths(review_paths: list[str] | None) -> list[str] | None:
    if review_paths is None:
        return None

    selected = [path for path in review_paths if path in CURATED_A11Y_PATHS]
    return selected or []


def resolve_python_command() -> str:
    windows_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if windows_python.exists():
        return str(windows_python)

    posix_python = ROOT / ".venv" / "bin" / "python"
    if posix_python.exists():
        return str(posix_python)

    return sys.executable


def ensure_playwright_dependencies() -> None:
    if (TESTS_VISUAL_DIR / "node_modules").exists():
        return

    raise SystemExit(
        "Missing Playwright dependencies under tests/visual/node_modules.\n"
        "Run `cd tests/visual && npm ci` once, then retry the push."
    )


def resolve_npx_command() -> str:
    for candidate in ("npx.cmd", "npx"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    raise SystemExit(
        "Could not find `npx` on PATH.\n"
        "Install Node.js and ensure `npx` is available before using the docs pre-push hook."
    )


def main() -> int:
    args = parse_args()
    changed_files = collect_changed_files(args.files)
    docs_sensitive_files = [path for path in changed_files if is_docs_sensitive(path)]

    if not docs_sensitive_files:
        print(
            "No docs-sensitive changes detected in this push; skipping docs pre-push guard."
        )
        return 0

    python_command = resolve_python_command()
    npx_command = resolve_npx_command()
    ensure_playwright_dependencies()

    review_paths = determine_review_paths(docs_sensitive_files)
    should_rebuild_assets = any(
        needs_asset_rebuild(path) for path in docs_sensitive_files
    )

    print("Docs-sensitive changes detected:", flush=True)
    for path in docs_sensitive_files:
        print(f"  - {path}", flush=True)

    if review_paths is None:
        print("Running the default review-page set.", flush=True)
    else:
        print(
            "Running targeted review paths: " + ", ".join(review_paths),
            flush=True,
        )

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["DOCS_BASE_URL"] = DEFAULT_DOCS_BASE_URL

    if should_rebuild_assets:
        run([python_command, "scripts/render_d2_diagrams.py"], env=env)
        run(
            [
                python_command,
                "scripts/build_visual_artifacts.py",
                "all",
                "--profile",
                "ci",
            ],
            env=env,
        )
    else:
        print(
            "\nSkipping generated-asset rebuild; no generator inputs changed.",
            flush=True,
        )

    run([python_command, "-m", "mkdocs", "build", "--strict"], env=env)
    run([python_command, "scripts/visual_audit/check_svg_assets.py"], env=env)

    playwright_env = env.copy()
    playwright_env["SKIP_DOCS_PREBUILD"] = "1"
    if review_paths:
        playwright_env["REVIEW_PATHS"] = ",".join(review_paths)

    run(
        [
            npx_command,
            "playwright",
            "test",
            "smoke.spec.ts",
            "dom-audits.spec.ts",
        ],
        cwd=TESTS_VISUAL_DIR,
        env=playwright_env,
    )

    a11y_paths = determine_a11y_paths(review_paths)
    if a11y_paths == []:
        print(
            "\nSkipping a11y for non-curated docs pages; smoke and DOM audits already passed.",
            flush=True,
        )
        return 0

    a11y_env = env.copy()
    a11y_env["SKIP_DOCS_PREBUILD"] = "1"
    if a11y_paths:
        a11y_env["REVIEW_PATHS"] = ",".join(a11y_paths)

    run(
        [npx_command, "playwright", "test", "a11y.spec.ts"],
        cwd=TESTS_VISUAL_DIR,
        env=a11y_env,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
