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
PRE_PUSH_ARTIFACTS_DIR = ROOT / "artifacts" / "pre_push"
CI_VISUAL_STAGE_STATUS_PATH = PRE_PUSH_ARTIFACTS_DIR / "docs_ci_visual_last_stage.json"
DEFAULT_TESTS = [
    "sentinel.spec.ts",
    "repo-facts.spec.ts",
    "pages.spec.ts",
    "components.spec.ts",
    "embedded-panels.spec.ts",
]
SERIAL_TESTS = {"pages.spec.ts", "components.spec.ts"}
CI_CONSTRAINTS_PATH = ROOT / "scripts" / "ci-constraints.txt"
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
    "playwright-browser-audits": (
        "browser-dom",
        "Ubuntu Playwright smoke, DOM, or math audits",
        "Inspect the uploaded browser-audit artifacts for console errors, missing routes, raw TeX, or layout overflow on the affected docs route.",
    ),
    "playwright-browser-a11y": (
        "browser-a11y",
        "Ubuntu Playwright accessibility checks",
        "Inspect the uploaded browser-audit artifacts for the reported accessibility violations on the affected curated docs route.",
    ),
    "playwright-sentinel-pages-embedded-panels": (
        "browser-snapshot-or-embedded-asset",
        "Ubuntu Playwright sentinel, full-page, or embedded-panel checks",
        "Inspect the uploaded visual regression artifact bundle (it contains tests/visual/test-results/** from the runner) and any linked docs/assets/generated files.",
    ),
    "playwright-sentinel-pages": (
        "browser-snapshot-page",
        "Ubuntu Playwright sentinel or full-page snapshots",
        "Inspect the uploaded visual regression artifact bundle; it contains tests/visual/test-results/** from the runner snapshot job.",
    ),
    "playwright-pages": (
        "browser-snapshot-page",
        "Ubuntu representative full-page snapshots",
        "Inspect the uploaded visual regression artifact bundle; it contains the representative page diffs under tests/visual/test-results/** from the runner.",
    ),
    "playwright-sentinel-repo-facts": (
        "browser-snapshot-page",
        "Ubuntu Playwright sentinel and repository-facts checks",
        "Inspect the uploaded visual regression artifact bundle; it contains the sentinel or repo-facts output under tests/visual/test-results/** from the runner.",
    ),
    "playwright-sentinel-repo-facts-embedded-panels": (
        "browser-snapshot-or-embedded-asset",
        "Ubuntu Playwright sentinel, repository-facts, or embedded-panel checks",
        "Inspect the uploaded visual regression artifact bundle (it contains tests/visual/test-results/** from the runner) and any linked docs/assets/generated files.",
    ),
    "playwright-components": (
        "browser-snapshot-component",
        "Ubuntu Playwright component snapshots",
        "Inspect the uploaded visual regression artifact bundle; it contains the component snapshot diffs under tests/visual/test-results/** from the runner.",
    ),
    "playwright-embedded-panels": (
        "browser-snapshot-or-embedded-asset",
        "Ubuntu Playwright embedded-panel checks or generated proof assets",
        "Inspect the uploaded visual regression artifact bundle for runner-side evidence and the linked SVG/PNG assets under docs/assets/generated.",
    ),
}


def load_docs_visual_config() -> dict[str, str]:
    return json.loads(DOCS_VISUAL_CONFIG_PATH.read_text(encoding="utf8"))


DOCS_VISUAL_CONFIG = load_docs_visual_config()
DEFAULT_BUILD_PROFILE = DOCS_VISUAL_CONFIG["build_profile"]


def prepare_stage_status_artifact() -> None:
    PRE_PUSH_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if CI_VISUAL_STAGE_STATUS_PATH.exists():
        CI_VISUAL_STAGE_STATUS_PATH.unlink()


def clear_stage_status_artifact() -> None:
    if CI_VISUAL_STAGE_STATUS_PATH.exists():
        CI_VISUAL_STAGE_STATUS_PATH.unlink()


def load_stage_status() -> dict[str, str] | None:
    if not CI_VISUAL_STAGE_STATUS_PATH.exists():
        return None

    try:
        data = json.loads(CI_VISUAL_STAGE_STATUS_PATH.read_text(encoding="utf8"))
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    normalized: dict[str, str] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, str):
            normalized[key] = value
    return normalized or None


def runner_failure_summary_lines(
    *,
    mode: str,
    image: str,
    stage_status: dict[str, str] | None,
) -> list[str]:
    lines = ["CI visual runner failed.", f"Mode: {mode}", f"Image: {image}"]

    if stage_status is None:
        lines.extend(
            [
                "Failure class: environment-container-startup",
                "Likely layer: Docker invocation, image pull, or container startup",
                "Next step: Run the docker command directly and inspect daemon, image-pull, or volume-mount errors.",
            ]
        )
        return lines

    stage_name = stage_status.get("stage", "unknown")
    stage_state = stage_status.get("status", "unknown")
    lines.extend(
        [
            f"Stage: {stage_name}",
            f"Stage status: {stage_state}",
            f"Failure class: {stage_status.get('failure_class', 'unknown')}",
            f"Likely layer: {stage_status.get('likely_layer', 'Unknown failure layer')}",
            f"Next step: {stage_status.get('next_step', 'Inspect the command output above.')}",
        ]
    )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run docs visual validation steps inside a GitHub-runner-style Ubuntu "
            "container so generated assets and snapshot verification match CI."
        )
    )
    parser.add_argument(
        "mode",
        choices=("verify", "update", "build-assets"),
        help=(
            "Verify current snapshots, refresh them, or rebuild generated docs assets "
            "in the CI-like container."
        ),
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
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help=(
            "Reuse an already-built MkDocs site mounted into the container instead of "
            "rebuilding generated assets and the site inside Docker."
        ),
    )
    return parser.parse_args()


def format_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def docker_stage_command(stage_name: str, command: list[str]) -> str:
    return f"run_stage {shlex.quote(stage_name)} {format_command(command)}"


def pip_install_command(*args: str) -> list[str]:
    return [
        "python3",
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--progress-bar",
        "off",
        "--break-system-packages",
        *args,
    ]


def stage_status_setup_command() -> str:
    metadata_literal = shlex.quote(json.dumps(STAGE_METADATA))
    status_path = "/work/artifacts/pre_push/docs_ci_visual_last_stage.json"
    return (
        f"STAGE_STATUS_PATH={shlex.quote(status_path)} && "
        'write_stage_status() { stage_name="$1"; stage_state="$2"; python3 - <<\'PY\' "$stage_name" "$stage_state" "$STAGE_STATUS_PATH"\n'
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "stage_name, stage_state, status_path = sys.argv[1:4]\n"
        f"metadata = {metadata_literal}\n"
        'entry = json.loads(metadata).get(stage_name, ("unknown", "Unknown failure layer", "Inspect the command output above."))\n'
        "payload = {\n"
        '    "stage": stage_name,\n'
        '    "status": stage_state,\n'
        '    "failure_class": entry[0],\n'
        '    "likely_layer": entry[1],\n'
        '    "next_step": entry[2],\n'
        "}\n"
        "path = Path(status_path)\n"
        "path.parent.mkdir(parents=True, exist_ok=True)\n"
        'path.write_text(json.dumps(payload), encoding="utf8")\n'
        "PY\n"
        "} && "
        'run_stage() { stage_name="$1"; shift; write_stage_status "$stage_name" running; printf "\\n[%s]\\n" "$stage_name"; printf "> "; printf "%q " "$@"; printf "\\n"; if ! "$@"; then write_stage_status "$stage_name" failed; python3 - <<\'PY\' "$stage_name"\n'
        "import sys\n"
        "stage = sys.argv[1]\n"
        f"metadata = {metadata_literal}\n"
        'entry = __import__("json").loads(metadata).get(stage, ("unknown", "Unknown failure layer", "Inspect the command output above."))\n'
        'print(f"\\n{stage} failed.", file=sys.stderr)\n'
        'print(f"Failure class: {entry[0]}", file=sys.stderr)\n'
        'print(f"Likely layer: {entry[1]}", file=sys.stderr)\n'
        'print(f"Next step: {entry[2]}", file=sys.stderr)\n'
        "PY\n"
        'exit 1; fi; write_stage_status "$stage_name" passed; }'
    )


def classify_parallel_tests(parallel_tests: list[str]) -> str:
    unique_tests = list(dict.fromkeys(parallel_tests))
    unique_test_set = set(unique_tests)
    browser_audit_tests = {"smoke.spec.ts", "dom-audits.spec.ts", "math-audits.spec.ts"}

    if unique_test_set == {"a11y.spec.ts"}:
        return "playwright-browser-a11y"
    if unique_test_set and unique_test_set <= browser_audit_tests:
        return "playwright-browser-audits"
    if unique_tests == ["sentinel.spec.ts", "repo-facts.spec.ts"]:
        return "playwright-sentinel-repo-facts"
    if unique_tests == ["sentinel.spec.ts"]:
        return "playwright-sentinel-pages"
    if "embedded-panels.spec.ts" in unique_tests:
        return "playwright-sentinel-repo-facts-embedded-panels"
    return "playwright-sentinel-pages"


def main() -> int:
    args = parse_args()
    prepare_stage_status_artifact()
    build_profile = os.environ.get("DOCS_VISUAL_BUILD_PROFILE", DEFAULT_BUILD_PROFILE)
    constraints_arg = []
    if CI_CONSTRAINTS_PATH.exists():
        constraints_arg = ["-c", CI_CONSTRAINTS_PATH.relative_to(ROOT).as_posix()]

    if args.mode == "build-assets" and args.skip_build:
        raise ValueError("--skip-build cannot be used with build-assets mode.")

    build_only = args.mode == "build-assets"

    requested_tests = list(dict.fromkeys(args.tests))
    parallel_tests = (
        []
        if build_only
        else [test for test in requested_tests if test not in SERIAL_TESTS]
    )
    serial_tests = (
        [] if build_only else [test for test in requested_tests if test in SERIAL_TESTS]
    )
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
        if test_name == "pages.spec.ts":
            playwright_stage_names.append("playwright-pages")
        elif test_name == "components.spec.ts":
            playwright_stage_names.append("playwright-components")
        elif test_name == "embedded-panels.spec.ts":
            playwright_stage_names.append("playwright-embedded-panels")
        else:
            playwright_stage_names.append("playwright-sentinel-pages")

    if args.mode == "update":
        for command in playwright_commands:
            command.append("--update-snapshots")

    if not playwright_commands and not build_only:
        raise ValueError("No Playwright test files were selected.")

    inner_commands = [
        "set -e",
        stage_status_setup_command(),
        docker_stage_command(
            "install-python-deps",
            pip_install_command("--upgrade", "pip"),
        ),
        docker_stage_command(
            "install-python-deps",
            pip_install_command(*constraints_arg, "-e", ".[docs,plot]"),
        ),
    ]

    if build_only:
        inner_commands.append(
            docker_stage_command(
                "build-visual-artifacts",
                [
                    "python3",
                    "scripts/build_visual_artifacts.py",
                    "all",
                    "--profile",
                    build_profile,
                ],
            )
        )
    else:
        inner_commands.extend(
            [
                "cd tests/visual",
                docker_stage_command(
                    "install-node-deps",
                    ["npm", "ci", "--no-audit", "--no-fund"],
                ),
                docker_stage_command(
                    "install-playwright",
                    ["npx", "playwright", "install", "--with-deps", "chromium"],
                ),
                "cd /work",
            ]
        )
        if not args.skip_build:
            inner_commands.extend(
                [
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
                ]
            )
        inner_commands.append("cd tests/visual")
        inner_commands.extend(
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
        )

    inner_script = " && ".join(inner_commands)

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
    if not build_only:
        print("Selected tests: " + ", ".join(requested_tests), flush=True)
    if args.skip_build and not build_only:
        print("Build mode: reuse prebuilt site", flush=True)
    if forwarded_env.get("REVIEW_PATHS"):
        print("Review paths: " + forwarded_env["REVIEW_PATHS"], flush=True)
    if forwarded_env.get("REVIEW_PAGE_KEYS"):
        print("Review page keys: " + forwarded_env["REVIEW_PAGE_KEYS"], flush=True)
    print("> " + format_command(docker_command), flush=True)
    try:
        subprocess.run(docker_command, check=True)
    except subprocess.CalledProcessError as exc:
        print("", flush=True)
        for line in runner_failure_summary_lines(
            mode=args.mode,
            image=args.image,
            stage_status=load_stage_status(),
        ):
            print(line, flush=True)
        return exc.returncode or 1

    clear_stage_status_artifact()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
