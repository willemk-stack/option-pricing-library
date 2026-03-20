from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from docs_impact import (
    classify_docs_impact,
    collect_changed_files,
    determine_a11y_paths,
)

ROOT = Path(__file__).resolve().parents[1]
TESTS_VISUAL_DIR = ROOT / "tests" / "visual"
DOCS_VISUAL_CONFIG_PATH = ROOT / "scripts" / "visual_audit" / "docs_visual_config.json"


def load_docs_visual_config() -> dict[str, str]:
    return json.loads(DOCS_VISUAL_CONFIG_PATH.read_text(encoding="utf8"))


DOCS_VISUAL_CONFIG = load_docs_visual_config()
DEFAULT_DOCS_BASE_URL = DOCS_VISUAL_CONFIG["docs_base_url"]
DEFAULT_BUILD_PROFILE = DOCS_VISUAL_CONFIG["build_profile"]

ALLOW_NO_DOCKER_ENV = "DOCS_PRE_PUSH_ALLOW_NO_DOCKER"


@dataclass(frozen=True, slots=True)
class ValidationStage:
    label: str
    failure_class: str
    likely_layer: str
    next_step: str


class HookFailure(Exception):
    """Actionable pre-push failure without a Python traceback."""

    def __init__(
        self,
        message: str,
        *,
        exit_code: int = 1,
        failure_class: str | None = None,
        likely_layer: str | None = None,
        next_step: str | None = None,
    ) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.failure_class = failure_class
        self.likely_layer = likely_layer
        self.next_step = next_step


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
    stage: ValidationStage,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> None:
    printable = " ".join(command)
    print(f"\n[{stage.label}]", flush=True)
    print(f"> {printable}", flush=True)
    try:
        subprocess.run(command, cwd=cwd, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        raise HookFailure(
            "\n".join(
                [
                    f"{stage.label} failed.",
                    f"Failure class: {stage.failure_class}",
                    f"Likely layer: {stage.likely_layer}",
                    f"Command: {printable}",
                    f"Exit code: {exc.returncode}",
                    f"Next step: {stage.next_step}",
                ]
            ),
            exit_code=exc.returncode or 1,
            failure_class=stage.failure_class,
            likely_layer=stage.likely_layer,
            next_step=stage.next_step,
        ) from None


def git_stdout(args: list[str]) -> str:
    subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def resolve_python_command() -> str:
    windows_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if windows_python.exists():
        return str(windows_python)

    posix_python = ROOT / ".venv" / "bin" / "python"
    if posix_python.exists():
        return str(posix_python)

    return sys.executable


def ensure_playwright_dependencies() -> None:
    cli_path = TESTS_VISUAL_DIR / "node_modules" / "playwright" / "cli.js"
    if (TESTS_VISUAL_DIR / "node_modules").exists() and cli_path.exists():
        return

    raise HookFailure(
        "Missing Playwright dependencies under tests/visual/node_modules.\n"
        "Run `cd tests/visual && npm ci` once, then retry the push.",
        failure_class="environment-dependency",
        likely_layer="Local Playwright dependency install",
        next_step="Install Node dependencies under tests/visual before retrying the push.",
    )


def resolve_node_command() -> str:
    for candidate in ("node.exe", "node"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    raise HookFailure(
        "Could not find `node` on PATH.\n"
        "Install Node.js and ensure `node` is available before using the docs pre-push hook.",
        failure_class="environment-dependency",
        likely_layer="Local Node.js runtime",
        next_step="Install Node.js and reopen the terminal so `node` resolves on PATH.",
    )


def resolve_playwright_command() -> list[str]:
    cli_path = TESTS_VISUAL_DIR / "node_modules" / "playwright" / "cli.js"
    if not cli_path.exists():
        raise HookFailure(
            "Could not find the local Playwright CLI under "
            "`tests/visual/node_modules/playwright/cli.js`.\n"
            "Run `cd tests/visual && npm ci` once, then retry the push.",
            failure_class="environment-dependency",
            likely_layer="Local Playwright CLI install",
            next_step="Reinstall tests/visual dependencies so the Playwright CLI exists locally.",
        )

    return [resolve_node_command(), str(cli_path)]


def playwright_command(*args: str) -> list[str]:
    return [*resolve_playwright_command(), *args]


def resolve_docker_command() -> str:
    for candidate in ("docker.exe", "docker"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    raise HookFailure(
        "Docs-sensitive pushes require Docker for authoritative Ubuntu visual checks.\n"
        "Install Docker Desktop (or another compatible Docker runtime), then retry the push.",
        failure_class="environment-docker",
        likely_layer="Local Docker installation",
        next_step="Install Docker and confirm `docker --version` succeeds in this shell.",
    )


def docker_available_detail() -> tuple[bool, str | None]:
    docker_command = resolve_docker_command()
    result = subprocess.run(
        [docker_command, "info", "--format", "{{.ServerVersion}}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True, None

    detail = (result.stderr or result.stdout).strip() or "Unknown Docker error."
    return False, detail


def allow_no_docker() -> bool:
    return os.environ.get(ALLOW_NO_DOCKER_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def docker_unavailable_failure(detail: str) -> HookFailure:
    return HookFailure(
        "Docs-sensitive pushes require Docker for authoritative Ubuntu visual checks.\n"
        f"Docker was found but is not ready: {detail}\n"
        f"Set {ALLOW_NO_DOCKER_ENV}=1 to continue with local non-Docker docs checks and defer authoritative Ubuntu visual validation to CI.",
        failure_class="environment-docker",
        likely_layer="Local Docker daemon",
        next_step=(
            "Start Docker Desktop, or opt into local degraded mode by setting "
            f"{ALLOW_NO_DOCKER_ENV}=1 before retrying the push."
        ),
    )


def ensure_clean_generated_diff(paths: list[str], *, label: str) -> None:
    if not paths:
        return

    result = subprocess.run(
        ["git", "diff", "--quiet", "--", *paths],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return

    raise HookFailure(
        "\n".join(
            [
                f"{label} updated tracked generated files.",
                "Review the regenerated outputs, stage them, and retry the push.",
                "Changed paths: " + ", ".join(paths),
            ]
        ),
        failure_class="generated-output-drift",
        likely_layer="Generated README, diagrams, or docs asset outputs",
        next_step="Review the regenerated files, stage the intended updates, then retry the push.",
    )


def stage_for_authoritative_tests(authoritative_tests: list[str]) -> ValidationStage:
    if "embedded-panels.spec.ts" in authoritative_tests:
        return ValidationStage(
            label="Authoritative Ubuntu visual verification",
            failure_class="browser-snapshot-or-embedded-asset",
            likely_layer="Ubuntu Playwright snapshots or generated proof-panel assets",
            next_step="Inspect tests/visual/test-results and the generated SVG/PNG assets for the affected review paths.",
        )
    if "components.spec.ts" in authoritative_tests:
        return ValidationStage(
            label="Authoritative Ubuntu visual verification",
            failure_class="browser-snapshot-component",
            likely_layer="Ubuntu Playwright component snapshots",
            next_step="Inspect the component snapshot diffs under tests/visual/test-results for the affected review paths.",
        )
    return ValidationStage(
        label="Authoritative Ubuntu visual verification",
        failure_class="browser-snapshot-page",
        likely_layer="Ubuntu Playwright sentinel or full-page snapshots",
        next_step="Inspect the sentinel/page snapshot diffs under tests/visual/test-results for the affected review paths.",
    )


def main() -> int:
    args = parse_args()
    changed_files = collect_changed_files(args.files)
    impact = classify_docs_impact(changed_files)

    if not impact.docs_sensitive:
        print(
            "No docs-sensitive changes detected in this push; skipping docs pre-push guard."
        )
        return 0

    python_command = resolve_python_command()
    ensure_playwright_dependencies()

    docker_ready = True
    docker_detail: str | None = None
    if impact.docs_site_required:
        docker_ready, docker_detail = docker_available_detail()
        if not docker_ready and not allow_no_docker():
            raise docker_unavailable_failure(docker_detail or "Unknown Docker error.")

    review_paths = impact.review_paths
    authoritative_tests = impact.authoritative_tests

    print("Docs-sensitive changes detected:", flush=True)
    for path in impact.changed_files:
        print(f"  - {path}", flush=True)

    if impact.docs_site_required and review_paths is None:
        print("Running the default review-page set.", flush=True)
    elif impact.docs_site_required:
        print(
            "Running targeted review paths: " + ", ".join(review_paths),
            flush=True,
        )
    else:
        print(
            "README-only docs impact; skipping site build and browser validation.",
            flush=True,
        )

    if impact.docs_site_required:
        print(
            "Authoritative Ubuntu visual suites: " + ", ".join(authoritative_tests),
            flush=True,
        )
        print(
            "Failure classes: build/link, generated-asset, browser-dom, browser-a11y, browser-snapshot",
            flush=True,
        )
        if not docker_ready:
            print(
                f"WARNING: Docker is unavailable; local authoritative Ubuntu checks are deferred to CI because {ALLOW_NO_DOCKER_ENV}=1.",
                flush=True,
            )
            if docker_detail:
                print(f"Docker detail: {docker_detail}", flush=True)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["DOCS_BASE_URL"] = DEFAULT_DOCS_BASE_URL
    env["DOCS_VISUAL_BUILD_PROFILE"] = DEFAULT_BUILD_PROFILE

    if impact.readme_required:
        run(
            [resolve_python_command(), "scripts/render_readme.py"],
            stage=ValidationStage(
                label="Refresh generated README",
                failure_class="build-readme-sync",
                likely_layer="Generated README sync from template and examples",
                next_step="Inspect README.template.md and examples/, then rerun the README generator.",
            ),
            env=env,
        )
        ensure_clean_generated_diff(["README.md"], label="Generated README refresh")

    if impact.benchmark_artifacts_required:
        run(
            [python_command, "scripts/build_benchmark_artifacts.py", "--check"],
            stage=ValidationStage(
                label="Benchmark artifact source freshness",
                failure_class="benchmark-artifact-stale",
                likely_layer="Committed benchmark snapshots versus tracked benchmark source inputs",
                next_step="Regenerate the benchmark artifacts in the authoritative benchmark workflow, review the updates, and commit the refreshed benchmarks/artifacts outputs.",
            ),
            env=env,
        )

    if impact.d2_required:
        run(
            [python_command, "scripts/render_d2_diagrams.py"],
            stage=ValidationStage(
                label="Rebuild D2 diagrams",
                failure_class="generated-asset-build",
                likely_layer="D2 diagram generation",
                next_step="Fix the diagram sources or D2 installation issue, then rerun the asset build.",
            ),
            env=env,
        )
        ensure_clean_generated_diff(
            ["docs/assets/diagrams"],
            label="D2 diagram refresh",
        )

    if not impact.docs_site_required:
        return 0

    run(
        [python_command, "scripts/check_docs_source_links.py"],
        stage=ValidationStage(
            label="Docs source links",
            failure_class="build-link",
            likely_layer="Markdown source links or relative docs references",
            next_step="Fix broken docs source links before retrying the push.",
        ),
        env=env,
    )

    if impact.visual_assets_required:
        if docker_ready:
            run(
                [
                    python_command,
                    "scripts/run_ci_visual_regression.py",
                    "build-assets",
                ],
                stage=ValidationStage(
                    label="Rebuild visual artifacts in authoritative Ubuntu container",
                    failure_class="generated-asset-build",
                    likely_layer="Authoritative Ubuntu SVG/PNG visual artifact generation",
                    next_step="Inspect the visual artifact generator inputs or Dockerized Ubuntu build output, then rerun the asset build.",
                ),
                env=env,
            )
            ensure_clean_generated_diff(
                ["docs/assets/generated"],
                label="Generated docs asset refresh",
            )
        else:
            print(
                "\nSkipping Dockerized visual-asset rebuild locally; CI remains authoritative for generated SVG/PNG docs assets.",
                flush=True,
            )
    else:
        print(
            "\nSkipping generated-asset rebuild; no generator inputs changed.",
            flush=True,
        )

    run(
        [python_command, "-m", "mkdocs", "build", "--strict"],
        stage=ValidationStage(
            label="Strict MkDocs build",
            failure_class="build-mkdocs",
            likely_layer="MkDocs config, Markdown, or generated docs content",
            next_step="Fix the strict MkDocs build failure before checking browser regressions.",
        ),
        env=env,
    )
    run(
        [python_command, "scripts/visual_audit/check_svg_assets.py"],
        stage=ValidationStage(
            label="SVG/PNG asset integrity",
            failure_class="generated-asset-integrity",
            likely_layer="Generated SVG/PNG references or embedded raster assets",
            next_step="Inspect the reported SVG/PNG asset and its linked media before rerunning browser checks.",
        ),
        env=env,
    )

    playwright_env = env.copy()
    playwright_env["SKIP_DOCS_PREBUILD"] = "1"
    playwright_env["SERVE_PREBUILT_SITE"] = "1"
    if review_paths:
        playwright_env["REVIEW_PATHS"] = ",".join(review_paths)

    run(
        playwright_command(
            "test",
            "smoke.spec.ts",
            "dom-audits.spec.ts",
            "--retries=1",
        ),
        stage=ValidationStage(
            label="Playwright smoke and DOM audits",
            failure_class="browser-dom",
            likely_layer="Rendered docs DOM, CSS layout, console errors, or missing routes",
            next_step="Inspect tests/visual/test-results for DOM overflow, console, or missing-page findings on the affected review paths.",
        ),
        cwd=TESTS_VISUAL_DIR,
        env=playwright_env,
    )

    a11y_paths = determine_a11y_paths(review_paths)
    if a11y_paths == []:
        print(
            "\nSkipping a11y for non-curated docs pages; smoke and DOM audits already passed.",
            flush=True,
        )

    else:
        a11y_env = env.copy()
        a11y_env["SKIP_DOCS_PREBUILD"] = "1"
        a11y_env["SERVE_PREBUILT_SITE"] = "1"
        if a11y_paths:
            a11y_env["REVIEW_PATHS"] = ",".join(a11y_paths)

        run(
            playwright_command("test", "a11y.spec.ts", "--retries=1"),
            stage=ValidationStage(
                label="Playwright accessibility checks",
                failure_class="browser-a11y",
                likely_layer="Rendered docs accessibility issues on curated blocking pages",
                next_step="Inspect the accessibility violations in tests/visual/test-results and fix the affected page markup or theme styling.",
            ),
            cwd=TESTS_VISUAL_DIR,
            env=a11y_env,
        )

    if docker_ready:
        docker_env = env.copy()
        if review_paths:
            docker_env["REVIEW_PATHS"] = ",".join(review_paths)
        run(
            [
                python_command,
                "scripts/run_ci_visual_regression.py",
                "verify",
                "--tests",
                *authoritative_tests,
            ],
            stage=stage_for_authoritative_tests(authoritative_tests),
            env=docker_env,
        )
    else:
        print(
            "\nSkipping authoritative Ubuntu visual verification locally; rely on CI for the final containerized snapshot gate.",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HookFailure as exc:
        print("\nDocs pre-push guard failed.", file=sys.stderr, flush=True)
        print(
            "This push was blocked by the local pre-push hook before GitHub could reject any refs.",
            file=sys.stderr,
            flush=True,
        )
        print(str(exc), file=sys.stderr, flush=True)
        raise SystemExit(exc.exit_code) from None
