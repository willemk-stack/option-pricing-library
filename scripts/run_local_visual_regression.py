from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from docs_site_contract import (  # noqa: E402
    load_docs_site_contract,
    load_docs_visual_config,
    verify_prebuilt_site,
)

VISUAL_TEST_DIR = ROOT / "tests" / "visual"
PLAYWRIGHT_CLI = VISUAL_TEST_DIR / "node_modules" / "playwright" / "cli.js"
SERIAL_TESTS = {"pages.spec.ts", "components.spec.ts"}
DEFAULT_VERIFY_TESTS = [
    "smoke.spec.ts",
    "dom-audits.spec.ts",
    "a11y.spec.ts",
    "sentinel.spec.ts",
    "repo-facts.spec.ts",
    "pages.spec.ts",
    "components.spec.ts",
    "embedded-panels.spec.ts",
]
DEFAULT_UPDATE_TESTS = [
    "sentinel.spec.ts",
    "pages.spec.ts",
    "components.spec.ts",
    "embedded-panels.spec.ts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the docs Playwright suites natively while building the docs site "
            "only once and then reusing the prebuilt output for all selected tests."
        )
    )
    parser.add_argument(
        "mode",
        choices=("verify", "update"),
        help="Verify current output or refresh Playwright snapshots natively.",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        default=None,
        help="Optional Playwright test files to run.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse the existing prebuilt site instead of rebuilding it first.",
    )
    return parser.parse_args()


def default_tests_for_mode(mode: str) -> list[str]:
    if mode == "update":
        return DEFAULT_UPDATE_TESTS
    return DEFAULT_VERIFY_TESTS


def format_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def run_stage(
    stage_name: str,
    command: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> None:
    resolved_command = command.copy()
    executable_name = command[0]
    if os.name == "nt" and executable_name in {"npm", "npx"}:
        executable_name = f"{executable_name}.cmd"

    executable = shutil.which(executable_name) or shutil.which(command[0])
    if executable:
        resolved_command[0] = executable

    print(f"\n[{stage_name}]", flush=True)
    print("> " + format_command(resolved_command), flush=True)
    subprocess.run(resolved_command, cwd=cwd, env=env, check=True)


def build_prebuilt_site(env: dict[str, str], build_profile: str) -> None:
    run_stage(
        "render-d2-diagrams",
        [sys.executable, "scripts/render_d2_diagrams.py"],
        env=env,
    )
    run_stage(
        "build-visual-artifacts",
        [
            sys.executable,
            "scripts/build_visual_artifacts.py",
            "all",
            "--profile",
            build_profile,
        ],
        env=env,
    )
    run_stage(
        "build-mkdocs",
        [sys.executable, "-m", "mkdocs", "build", "--strict"],
        env=env,
    )


def playwright_commands(
    tests: list[str], *, update_snapshots: bool
) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    node_command = shutil.which("node") or "node"
    parallel_tests = [test for test in tests if test not in SERIAL_TESTS]
    serial_tests = [test for test in tests if test in SERIAL_TESTS]

    if parallel_tests:
        command = [node_command, str(PLAYWRIGHT_CLI), "test", *parallel_tests]
        if update_snapshots:
            command.append("--update-snapshots")
        commands.append(("playwright-parallel", command))

    for test_name in serial_tests:
        command = [
            node_command,
            str(PLAYWRIGHT_CLI),
            "test",
            test_name,
            "--workers=1",
        ]
        if update_snapshots:
            command.append("--update-snapshots")
        commands.append((f"playwright-{test_name}", command))

    return commands


def main() -> int:
    args = parse_args()
    config = load_docs_visual_config()
    contract = load_docs_site_contract()
    requested_tests = list(
        dict.fromkeys(args.tests or default_tests_for_mode(args.mode))
    )
    if not requested_tests:
        raise SystemExit("No Playwright test files were selected.")

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("DOCS_BASE_URL", config["docs_base_url"])
    env.setdefault("DOCS_VISUAL_BUILD_PROFILE", config["build_profile"])

    print(f"Running native visual {args.mode}", flush=True)
    print("Selected tests: " + ", ".join(requested_tests), flush=True)
    if os.environ.get("REVIEW_PATHS"):
        print("Review paths: " + os.environ["REVIEW_PATHS"], flush=True)
    if os.environ.get("REVIEW_PAGE_KEYS"):
        print("Review page keys: " + os.environ["REVIEW_PAGE_KEYS"], flush=True)

    if args.skip_build:
        verify_prebuilt_site(contract)
        print(
            "Using existing MkDocs site build without rebuilding docs assets.",
            flush=True,
        )
    else:
        build_prebuilt_site(env, env["DOCS_VISUAL_BUILD_PROFILE"])
        verify_prebuilt_site(contract)

    env["SKIP_DOCS_PREBUILD"] = "1"
    env["SERVE_PREBUILT_SITE"] = "1"

    for stage_name, command in playwright_commands(
        requested_tests,
        update_snapshots=args.mode == "update",
    ):
        run_stage(stage_name, command, cwd=VISUAL_TEST_DIR, env=env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
