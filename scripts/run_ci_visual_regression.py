from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_VISUAL_CONFIG_PATH = ROOT / "scripts" / "visual_audit" / "docs_visual_config.json"
DEFAULT_IMAGE = "ghcr.io/catthehacker/ubuntu:runner-24.04"
DEFAULT_TESTS = [
    "sentinel.spec.ts",
    "pages.spec.ts",
    "components.spec.ts",
    "embedded-panels.spec.ts",
]
SERIAL_TESTS = {"components.spec.ts"}
FORWARDED_ENV_VARS = (
    "REVIEW_PATHS",
    "REVIEW_PAGE_KEYS",
    "DOCS_BASE_URL",
)

STAGE_METADATA = {
    "install-python-deps": (
        "environment-container-python",
        "Container Python dependency install",
        "Inspect Python dependency resolution inside the Ubuntu runner image.",
    ),
    "install-node-deps": (
        "environment-container-node",
        "Container Node dependency install",
        "Inspect npm install output inside tests/visual in the Ubuntu runner image.",
    ),
    "install-playwright": (
        "environment-container-browser",
        "Container Playwright browser install",
        "Inspect Playwright browser installation output inside the Ubuntu runner image.",
    ),
    "build-visual-artifacts": (
        "generated-asset-build",
        "SVG/PNG visual artifact generation",
        "Fix the visual artifact generator or inputs before retrying snapshot verification.",
    ),
    "build-mkdocs": (
        "build-mkdocs",
        "MkDocs strict build inside the Ubuntu runner image",
        "Fix the strict MkDocs build failure before retrying snapshot verification.",
    ),
    "playwright-sentinel-pages-embedded-panels": (
        "browser-snapshot-or-embedded-asset",
        "Ubuntu Playwright sentinel, full-page, or embedded-panel checks",
        "Inspect the snapshot or embedded-panel diff output under tests/visual/test-results.",
    ),
    "playwright-sentinel-pages": (
        "browser-snapshot-page",
        "Ubuntu Playwright sentinel or full-page snapshots",
        "Inspect the page snapshot diffs under tests/visual/test-results.",
    ),
    "playwright-components": (
        "browser-snapshot-component",
        "Ubuntu Playwright component snapshots",
        "Inspect the component snapshot diffs under tests/visual/test-results.",
    ),
    "playwright-embedded-panels": (
        "browser-snapshot-or-embedded-asset",
        "Ubuntu Playwright embedded-panel checks or generated proof assets",
        "Inspect embedded-panel failures and the linked SVG/PNG assets under docs/assets/generated.",
    ),
}


def load_docs_visual_config() -> dict[str, str]:
    return json.loads(DOCS_VISUAL_CONFIG_PATH.read_text(encoding="utf8"))


DOCS_VISUAL_CONFIG = load_docs_visual_config()
DEFAULT_BUILD_PROFILE = DOCS_VISUAL_CONFIG["build_profile"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Playwright visual suites inside a GitHub-runner-style Ubuntu "
            "container so snapshot verification and refreshes match CI."
        )
    )
    parser.add_argument(
        "mode",
        choices=("verify", "update"),
        help="Verify current snapshots or refresh them in the CI-like container.",
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help=f"Docker image to use (default: {DEFAULT_IMAGE}).",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        default=DEFAULT_TESTS,
        help="Playwright test files to run inside the container.",
    )
    return parser.parse_args()


def format_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def docker_stage_command(stage_name: str, command: list[str]) -> str:
    return f"run_stage {shlex.quote(stage_name)} {format_command(command)}"


def classify_parallel_tests(parallel_tests: list[str]) -> str:
    unique_tests = list(dict.fromkeys(parallel_tests))
    if unique_tests == ["sentinel.spec.ts", "pages.spec.ts"]:
        return "playwright-sentinel-pages"
    if "embedded-panels.spec.ts" in unique_tests:
        return "playwright-sentinel-pages-embedded-panels"
    return "playwright-sentinel-pages"


def main() -> int:
    args = parse_args()
    build_profile = os.environ.get("DOCS_VISUAL_BUILD_PROFILE", DEFAULT_BUILD_PROFILE)

    requested_tests = list(dict.fromkeys(args.tests))
    parallel_tests = [test for test in requested_tests if test not in SERIAL_TESTS]
    serial_tests = [test for test in requested_tests if test in SERIAL_TESTS]
    forwarded_env = {
        name: value for name in FORWARDED_ENV_VARS if (value := os.environ.get(name))
    }

    playwright_commands: list[list[str]] = []
    playwright_stage_names: list[str] = []
    if parallel_tests:
        playwright_commands.append(["npx", "playwright", "test", *parallel_tests])
        playwright_stage_names.append(classify_parallel_tests(parallel_tests))
    for test_name in serial_tests:
        playwright_commands.append(
            ["npx", "playwright", "test", test_name, "--workers=1"]
        )
        if test_name == "components.spec.ts":
            playwright_stage_names.append("playwright-components")
        elif test_name == "embedded-panels.spec.ts":
            playwright_stage_names.append("playwright-embedded-panels")
        else:
            playwright_stage_names.append("playwright-sentinel-pages")

    if args.mode == "update":
        for command in playwright_commands:
            command.append("--update-snapshots")

    if not playwright_commands:
        raise ValueError("No Playwright test files were selected.")

    inner_script = " && ".join(
        [
            "set -e",
            'run_stage() { stage_name="$1"; shift; printf "\\n[%s]\\n" "$stage_name"; printf "> "; printf "%q " "$@"; printf "\\n"; if ! "$@"; then python3 - <<\'PY\' "$stage_name"\nimport sys\nstage = sys.argv[1]\nmetadata = '
            + shlex.quote(json.dumps(STAGE_METADATA))
            + '\nentry = __import__("json").loads(metadata).get(stage, ("unknown", "Unknown failure layer", "Inspect the command output above."))\nprint(f"\\n{stage} failed.", file=sys.stderr)\nprint(f"Failure class: {entry[0]}", file=sys.stderr)\nprint(f"Likely layer: {entry[1]}", file=sys.stderr)\nprint(f"Next step: {entry[2]}", file=sys.stderr)\nPY\nexit 1; fi; }',
            docker_stage_command(
                "install-python-deps",
                [
                    "python3",
                    "-m",
                    "pip",
                    "install",
                    "--break-system-packages",
                    "--upgrade",
                    "pip",
                ],
            ),
            docker_stage_command(
                "install-python-deps",
                [
                    "python3",
                    "-m",
                    "pip",
                    "install",
                    "--break-system-packages",
                    "-e",
                    ".[docs,plot]",
                ],
            ),
            "cd tests/visual",
            docker_stage_command("install-node-deps", ["npm", "ci"]),
            docker_stage_command(
                "install-playwright",
                ["npx", "playwright", "install", "--with-deps", "chromium"],
            ),
            "cd /work",
            docker_stage_command(
                "build-visual-artifacts",
                [
                    "python3",
                    "scripts/build_visual_artifacts.py",
                    "all",
                    "--profile",
                    build_profile,
                ],
            ),
            docker_stage_command(
                "build-mkdocs",
                ["python3", "-m", "mkdocs", "build", "--strict"],
            ),
            "cd tests/visual",
            *(
                docker_stage_command(
                    stage_name,
                    [
                        "env",
                        "SKIP_DOCS_PREBUILD=1",
                        "SERVE_PREBUILT_SITE=1",
                        *command,
                    ],
                )
                for stage_name, command in zip(
                    playwright_stage_names, playwright_commands, strict=True
                )
            ),
        ]
    )

    docker_command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{ROOT.resolve()}:/work",
        "-w",
        "/work",
        *(argument for name in forwarded_env for argument in ("-e", name)),
        args.image,
        "bash",
        "-lc",
        inner_script,
    ]

    print(f"Running CI-like visual {args.mode} in {args.image}", flush=True)
    print("Selected tests: " + ", ".join(requested_tests), flush=True)
    if forwarded_env.get("REVIEW_PATHS"):
        print("Review paths: " + forwarded_env["REVIEW_PATHS"], flush=True)
    if forwarded_env.get("REVIEW_PAGE_KEYS"):
        print("Review page keys: " + forwarded_env["REVIEW_PAGE_KEYS"], flush=True)
    print("> " + format_command(docker_command), flush=True)
    subprocess.run(docker_command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
