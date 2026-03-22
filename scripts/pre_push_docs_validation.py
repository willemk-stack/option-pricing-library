from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from docs_impact import (
    classify_docs_impact,
    collect_changed_files,
    determine_a11y_paths,
)

ROOT = Path(__file__).resolve().parents[1]
DOCS_VISUAL_CONFIG_PATH = ROOT / "scripts" / "visual_audit" / "docs_visual_config.json"
PRE_PUSH_LOG_DIR = ROOT / "artifacts" / "pre_push"
PRE_PUSH_RUN_LOG = PRE_PUSH_LOG_DIR / "docs_pre_push_last_run.log"
PRE_PUSH_FAILURE_LOG = PRE_PUSH_LOG_DIR / "docs_pre_push_last_failure.log"
FAILURE_SUMMARY_TAIL_LINES = 25


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
        log_path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.failure_class = failure_class
        self.likely_layer = likely_layer
        self.next_step = next_step
        self.log_path = log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run targeted docs validation before push when docs-sensitive files "
            "changed."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "manual"),
        default="fast",
        help=(
            "Fast mode keeps the default pre-push hook portable and avoids browser "
            "dependencies. Manual mode adds the heavier Dockerized browser and "
            "authoritative checks."
        ),
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
    printable = subprocess.list2cmdline(command)
    print(f"\n[{stage.label}]", flush=True)
    print(f"> {printable}", flush=True)

    append_run_log(f"\n[{stage.label}]\n> {printable}\n")
    tail: deque[str] = deque(maxlen=FAILURE_SUMMARY_TAIL_LINES)

    try:
        with (
            PRE_PUSH_RUN_LOG.open("a", encoding="utf8") as run_log_file,
            PRE_PUSH_FAILURE_LOG.open("a", encoding="utf8") as failure_log_file,
        ):
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                run_log_file.write(line)
                failure_log_file.write(line)
                tail.append(line.rstrip())
            return_code = process.wait()
    except OSError as exc:
        failure_log = snapshot_failure_log()
        raise HookFailure(
            "\n".join(
                [
                    f"{stage.label} could not start.",
                    f"Failure class: {stage.failure_class}",
                    f"Likely layer: {stage.likely_layer}",
                    f"Command: {printable}",
                    f"OS error: {exc}",
                    f"Next step: {stage.next_step}",
                ]
            ),
            exit_code=1,
            failure_class=stage.failure_class,
            likely_layer=stage.likely_layer,
            next_step=stage.next_step,
            log_path=failure_log,
        ) from None

    if return_code != 0:
        failure_log = snapshot_failure_log()
        summary_lines = [
            f"{stage.label} failed.",
            f"Failure class: {stage.failure_class}",
            f"Likely layer: {stage.likely_layer}",
            f"Command: {printable}",
            f"Exit code: {return_code}",
            f"Next step: {stage.next_step}",
        ]
        output_tail = [line for line in tail if line]
        if output_tail:
            summary_lines.append("Output tail:")
            summary_lines.extend(f"  {line}" for line in output_tail)
        raise HookFailure(
            "\n".join(summary_lines),
            exit_code=return_code,
            failure_class=stage.failure_class,
            likely_layer=stage.likely_layer,
            next_step=stage.next_step,
            log_path=failure_log,
        ) from None


def prepare_run_log() -> None:
    PRE_PUSH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_header = "Docs pre-push guard log\n"
    PRE_PUSH_RUN_LOG.write_text(log_header, encoding="utf8")
    PRE_PUSH_FAILURE_LOG.write_text(log_header, encoding="utf8")


def append_run_log(message: str) -> None:
    PRE_PUSH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with PRE_PUSH_RUN_LOG.open("a", encoding="utf8") as log_file:
        log_file.write(message)
    with PRE_PUSH_FAILURE_LOG.open("a", encoding="utf8") as log_file:
        log_file.write(message)


def snapshot_failure_log() -> Path:
    PRE_PUSH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not PRE_PUSH_FAILURE_LOG.exists():
        PRE_PUSH_FAILURE_LOG.write_text(
            "Docs pre-push guard failed before any stage output was captured.\n",
            encoding="utf8",
        )
    return PRE_PUSH_FAILURE_LOG


def clear_failure_log() -> None:
    if PRE_PUSH_FAILURE_LOG.exists():
        PRE_PUSH_FAILURE_LOG.unlink()


def git_stdout(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def resolve_python_command() -> str:
    windows_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if windows_python.exists():
        return str(windows_python)

    posix_python = ROOT / ".venv" / "bin" / "python"
    if posix_python.exists():
        return str(posix_python)

    return sys.executable


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
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *paths,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (
            result.stderr or result.stdout
        ).strip() or "Unknown git status failure."
        raise HookFailure(
            f"{label} could not inspect git status.\nDetail: {detail}",
            failure_class="generated-output-drift",
            likely_layer="Generated docs asset status inspection",
            next_step="Run `git status --short --untracked-files=all` locally and resolve the status inspection failure.",
        )

    changed_entries = [
        line.strip() for line in result.stdout.splitlines() if line.strip()
    ]
    if not changed_entries:
        return

    raise HookFailure(
        "\n".join(
            [
                f"{label} detected tracked or untracked generated-file drift.",
                "Review the generated outputs, stage the intended updates, and retry the push.",
                "Changed entries:",
                *(f"  {entry}" for entry in changed_entries),
            ]
        ),
        failure_class="generated-output-drift",
        likely_layer="Generated README, diagrams, or docs asset outputs",
        next_step="Review the generated files, stage the intended updates, then retry the push.",
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
    prepare_run_log()
    args = parse_args()
    changed_files = collect_changed_files(args.files)
    impact = classify_docs_impact(changed_files)

    if not impact.docs_sensitive:
        print(
            "No docs-sensitive changes detected in this push; skipping docs pre-push guard."
        )
        return 0

    python_command = resolve_python_command()
    manual_mode = args.mode == "manual"

    docker_ready = False
    docker_detail: str | None = None
    if manual_mode and impact.docs_site_required:
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
        if manual_mode:
            print(
                "Authoritative Ubuntu visual suites: " + ", ".join(authoritative_tests),
                flush=True,
            )
            print(
                "Failure classes: build/link, generated-asset, browser-dom, browser-math, browser-a11y, browser-snapshot",
                flush=True,
            )
        else:
            print(
                "Fast mode skips browser and Docker validation locally; PR CI remains authoritative for: "
                + ", ".join(authoritative_tests),
                flush=True,
            )

        if manual_mode and not docker_ready:
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
            [python_command, "scripts/render_readme.py", "--check"],
            stage=ValidationStage(
                label="Check generated README",
                failure_class="build-readme-sync",
                likely_layer="Generated README sync from template and examples",
                next_step="Inspect README.template.md and examples/, then rerun the README generator in write mode if the committed README is stale.",
            ),
            env=env,
        )

    if impact.performance_page_required:
        run(
            [python_command, "scripts/render_performance_page.py", "--check"],
            stage=ValidationStage(
                label="Check generated performance page",
                failure_class="build-performance-sync",
                likely_layer="Generated performance page sync from committed benchmark artifacts",
                next_step="Inspect the benchmark artifacts or performance page template, then rerun the renderer in write mode if the committed page is stale.",
            ),
            env=env,
        )

    if impact.benchmark_artifacts_required:
        run(
            [python_command, "scripts/build_benchmark_artifacts.py", "--check"],
            stage=ValidationStage(
                label="Performance snapshot freshness",
                failure_class="benchmark-artifact-stale",
                likely_layer="Committed performance snapshots versus tracked benchmark source inputs",
                next_step="Refresh the benchmark/performance snapshot bundle in the dedicated performance workflow, review the updates, and commit the refreshed benchmarks/artifacts outputs.",
            ),
            env=env,
        )

    if impact.d2_required:
        run(
            [python_command, "scripts/render_d2_diagrams.py", "--check"],
            stage=ValidationStage(
                label="Check D2 diagrams",
                failure_class="generated-asset-build",
                likely_layer="D2 diagram generation",
                next_step="Fix the diagram sources or D2 installation issue, then rerun the D2 refresh in write mode if the committed diagrams are stale.",
            ),
            env=env,
        )
        ensure_clean_generated_diff(
            ["docs/assets/diagrams"],
            label="D2 diagram tree",
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
        if manual_mode:
            run(
                [
                    python_command,
                    "scripts/build_visual_artifacts.py",
                    "all",
                    "--profile",
                    DEFAULT_BUILD_PROFILE,
                    "--check",
                ],
                stage=ValidationStage(
                    label="Check generated visual assets",
                    failure_class="generated-asset-build",
                    likely_layer="SVG/PNG visual artifact generation",
                    next_step="Inspect the visual artifact generator inputs, then rerun the asset refresh workflow or an explicit local refresh if the committed assets are stale.",
                ),
                env=env,
            )
        else:
            print(
                "\nFast mode skips local generated-asset rendering; CI and the manual docs guard remain authoritative for semantic visual-asset drift.",
                flush=True,
            )

        ensure_clean_generated_diff(
            ["docs/assets/generated"],
            label="Generated docs asset tree",
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

    if not manual_mode:
        clear_failure_log()
        return 0

    docker_env = env.copy()
    if review_paths:
        docker_env["REVIEW_PATHS"] = ",".join(review_paths)

    run(
        [
            python_command,
            "scripts/run_ci_visual_regression.py",
            "verify",
            "--skip-build",
            "--tests",
            "smoke.spec.ts",
            "dom-audits.spec.ts",
        ],
        stage=ValidationStage(
            label="Dockerized smoke and DOM audits",
            failure_class="browser-dom",
            likely_layer="Ubuntu Playwright DOM, CSS layout, console, or missing-route checks inside the CI-like container",
            next_step="Inspect tests/visual/test-results for DOM overflow, console, or missing-page findings on the affected review paths in the Dockerized Ubuntu run.",
        ),
        env=docker_env,
    )

    run(
        [
            python_command,
            "scripts/run_ci_visual_regression.py",
            "verify",
            "--skip-build",
            "--tests",
            "math-audits.spec.ts",
        ],
        stage=ValidationStage(
            label="Dockerized math audits",
            failure_class="browser-math",
            likely_layer="Ubuntu Playwright math typesetting, MathJax console failures, or SVG math layout inside the CI-like container",
            next_step="Inspect tests/visual/test-results for the affected math route and fix the TeX source, MathJax configuration, or math-specific CSS overflow in the Dockerized Ubuntu run.",
        ),
        env=docker_env,
    )

    a11y_paths = determine_a11y_paths(review_paths)
    if a11y_paths == []:
        print(
            "\nSkipping a11y for non-curated docs pages; smoke and DOM audits already passed.",
            flush=True,
        )

    else:
        a11y_env = env.copy()
        if a11y_paths:
            a11y_env["REVIEW_PATHS"] = ",".join(a11y_paths)

        run(
            [
                python_command,
                "scripts/run_ci_visual_regression.py",
                "verify",
                "--skip-build",
                "--tests",
                "a11y.spec.ts",
            ],
            stage=ValidationStage(
                label="Dockerized accessibility checks",
                failure_class="browser-a11y",
                likely_layer="Ubuntu Playwright accessibility issues on curated blocking pages inside the CI-like container",
                next_step="Inspect the accessibility violations in tests/visual/test-results and fix the affected page markup or theme styling in the Dockerized Ubuntu run.",
            ),
            env=a11y_env,
        )

    if docker_ready:
        run(
            [
                python_command,
                "scripts/run_ci_visual_regression.py",
                "verify",
                "--skip-build",
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

    clear_failure_log()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HookFailure as exc:
        failure_log = exc.log_path
        if failure_log is None:
            failure_log = snapshot_failure_log()
        print("\nDocs pre-push guard failed.", file=sys.stderr, flush=True)
        print(
            "This push was blocked by the local pre-push hook before GitHub could reject any refs.",
            file=sys.stderr,
            flush=True,
        )
        print(str(exc), file=sys.stderr, flush=True)
        print(f"Full hook log: {failure_log}", file=sys.stderr, flush=True)
        print(
            "Manual reproduction: pre-commit run docs-pre-push-guard --hook-stage pre-push -v or pre-commit run docs-manual-guard --hook-stage manual -v",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(exc.exit_code) from None
