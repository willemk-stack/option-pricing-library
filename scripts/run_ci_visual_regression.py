from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = "ghcr.io/catthehacker/ubuntu:runner-24.04"
DEFAULT_TESTS = [
    "sentinel.spec.ts",
    "pages.spec.ts",
    "components.spec.ts",
    "embedded-panels.spec.ts",
]


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


def main() -> int:
    args = parse_args()

    playwright_parts = ["npx", "playwright", "test", *args.tests]
    if args.mode == "update":
        playwright_parts.append("--update-snapshots")

    inner_script = " && ".join(
        [
            "set -e",
            "python3 -m pip install --break-system-packages --upgrade pip >/dev/null",
            "python3 -m pip install --break-system-packages -e '.[docs,plot]' >/dev/null",
            "cd tests/visual",
            "npm ci >/dev/null",
            "npx playwright install --with-deps chromium >/dev/null",
            "cd /work",
            "python3 scripts/build_visual_artifacts.py all --profile ci >/dev/null",
            "cd tests/visual",
            "SKIP_DOCS_PREBUILD=1 " + format_command(playwright_parts),
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
        args.image,
        "bash",
        "-lc",
        inner_script,
    ]

    print(f"Running CI-like visual {args.mode} in {args.image}", flush=True)
    print("> " + format_command(docker_command), flush=True)
    subprocess.run(docker_command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
