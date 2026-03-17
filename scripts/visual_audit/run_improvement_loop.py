from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
VISUAL_DIR = ROOT / "tests" / "visual"
OUT_ROOT = ROOT / "artifacts" / "visual-state" / "improvement-runs"
SPECS_DIR = ROOT / "design" / "page_specs"


@dataclass(frozen=True, slots=True)
class LoopStageResult:
    score: int | None
    suggestions: list[str]
    validation_ok: bool
    validation_steps: list[dict[str, str | int | bool]]
    capture_dir: str
    score_report_json: str
    score_report_md: str


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf8")


def load_spec(page_key: str) -> dict:
    for path in sorted(SPECS_DIR.glob("*.yml")):
        payload = yaml.safe_load(read_text(path))
        if payload.get("page_key") == page_key:
            return payload
    raise ValueError(f"Could not find page spec for page_key={page_key!r}")


def npm_command() -> str:
    return "npm.cmd" if os.name == "nt" else "npm"


def run(
    command: list[str], *, cwd: Path, env: dict[str, str], check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=check,
        text=True,
        capture_output=True,
    )


def build_docs(env: dict[str, str]) -> None:
    run([sys.executable, "scripts/render_d2_diagrams.py"], cwd=ROOT, env=env)
    run(
        [sys.executable, "scripts/build_visual_artifacts.py", "all", "--profile", "ci"],
        cwd=ROOT,
        env=env,
    )
    run([sys.executable, "-m", "mkdocs", "build", "--strict"], cwd=ROOT, env=env)


def generate_score_reports(env: dict[str, str]) -> tuple[Path, Path]:
    run([sys.executable, "scripts/visual_audit/score_pages.py"], cwd=ROOT, env=env)
    return (
        ROOT / "artifacts" / "visual-state" / "improvement-report.json",
        ROOT / "artifacts" / "visual-state" / "improvement-report.md",
    )


def extract_page_score(
    score_json_path: Path, page_key: str
) -> tuple[int | None, list[str]]:
    payload = json.loads(read_text(score_json_path))
    for score in payload.get("scores", []):
        if score.get("page_key") == page_key:
            return int(score.get("total_score", 0)), list(score.get("suggestions", []))
    return None, []


def stage_dir(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf8")


def run_targeted_validation(
    *,
    page_key: str,
    page_path: str,
    stage_root: Path,
    include_embedded_panels: bool,
    projects: list[str],
    env: dict[str, str],
) -> tuple[bool, list[dict[str, str | int | bool]]]:
    capture_dir = stage_dir(stage_root, "captures")
    local_env = {
        **env,
        "SKIP_DOCS_PREBUILD": "1",
        "REVIEW_PAGE_KEYS": page_key,
        "REVIEW_PATHS": page_path,
        "IMPROVEMENT_CAPTURE_DIR": str(capture_dir),
    }

    project_args = [
        argument for project in projects for argument in ("--project", project)
    ]

    run_component_snapshots = (
        os.environ.get("RUN_COMPONENT_SNAPSHOT_VALIDATION") == "1" or os.name == "nt"
    )

    commands: list[tuple[str, list[str]]] = [
        (
            "review-capture",
            [
                npm_command(),
                "exec",
                "playwright",
                "test",
                "review-capture.spec.ts",
                *project_args,
            ],
        ),
        (
            "smoke",
            [
                npm_command(),
                "exec",
                "playwright",
                "test",
                "smoke.spec.ts",
                *project_args,
            ],
        ),
        (
            "dom-audits",
            [
                npm_command(),
                "exec",
                "playwright",
                "test",
                "dom-audits.spec.ts",
                *project_args,
            ],
        ),
        (
            "a11y",
            [
                npm_command(),
                "exec",
                "playwright",
                "test",
                "a11y.spec.ts",
                *project_args,
            ],
        ),
    ]
    if run_component_snapshots:
        commands.append(
            (
                "components",
                [
                    npm_command(),
                    "exec",
                    "playwright",
                    "test",
                    "components.spec.ts",
                    *project_args,
                ],
            )
        )
    if include_embedded_panels:
        commands.append(
            (
                "embedded-panels",
                [
                    npm_command(),
                    "exec",
                    "playwright",
                    "test",
                    "embedded-panels.spec.ts",
                    "--project",
                    projects[-1],
                ],
            )
        )

    all_ok = True
    step_results: list[dict[str, str | int | bool]] = []
    for index, (label, command) in enumerate(commands, start=1):
        log_name = f"{index:02d}-{label}.log"
        log_path = stage_root / "logs" / log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf8") as handle:
            result = subprocess.run(
                command,
                cwd=VISUAL_DIR,
                env=local_env,
                check=False,
                text=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        step_results.append(
            {
                "label": label,
                "ok": result.returncode == 0,
                "exit_code": int(result.returncode),
                "log": str(log_path),
            }
        )
        if result.returncode != 0:
            all_ok = False

    return all_ok, step_results


def copy_report(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def run_edit_hook(edit_command: str, env: dict[str, str], stage_root: Path) -> int:
    result = subprocess.run(
        edit_command,
        cwd=ROOT,
        env=env,
        shell=True,
        text=True,
        capture_output=True,
    )
    write_text(
        stage_root / "logs" / "edit-command.log", result.stdout + "\n" + result.stderr
    )
    return int(result.returncode)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded before/after improvement loop for a single docs page."
    )
    parser.add_argument(
        "--page-key", required=True, help="Page spec key under design/page_specs/*.yml"
    )
    parser.add_argument(
        "--run-name",
        default="manual",
        help="Run folder name under artifacts/visual-state/improvement-runs/",
    )
    parser.add_argument(
        "--edit-command",
        default="",
        help="Optional shell command that applies the bounded edit between before/after validation.",
    )
    parser.add_argument(
        "--project",
        dest="projects",
        action="append",
        default=[],
        help="Playwright project to use for targeted validation. Repeat to widen the slice. Defaults to chromium-375 and chromium-1280.",
    )
    return parser.parse_args(argv)


def render_summary(
    page_key: str,
    page_path: str,
    before: LoopStageResult,
    after: LoopStageResult,
    *,
    edit_command: str,
    edit_exit_code: int | None,
) -> str:
    lines = [
        "# Improvement loop summary",
        "",
        f"- page key: `{page_key}`",
        f"- page path: `{page_path}`",
        f"- edit command: `{edit_command or 'none'}`",
    ]
    if edit_exit_code is not None:
        lines.append(f"- edit command exit code: `{edit_exit_code}`")
    lines.extend(
        [
            "",
            "## Before",
            f"- score: `{before.score}`",
            f"- validation ok: `{before.validation_ok}`",
            f"- capture dir: `{before.capture_dir}`",
            f"- top suggestions: `{', '.join(before.suggestions) if before.suggestions else 'none'}`",
            f"- validation steps: `{before.validation_steps}`",
            "",
            "## After",
            f"- score: `{after.score}`",
            f"- validation ok: `{after.validation_ok}`",
            f"- capture dir: `{after.capture_dir}`",
            f"- top suggestions: `{', '.join(after.suggestions) if after.suggestions else 'none'}`",
            f"- validation steps: `{after.validation_steps}`",
            "",
            "## Delta",
            f"- score delta: `{(after.score or 0) - (before.score or 0)}`",
            f"- validation delta: `{before.validation_ok}` -> `{after.validation_ok}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = load_spec(args.page_key)
    page_path = str(spec["page_path"])
    review_path = (
        "/"
        if page_path == "docs/index.md"
        else f"/{page_path.removeprefix('docs/').removesuffix('.md')}/"
    )
    run_root = stage_dir(OUT_ROOT, args.run_name) / args.page_key
    run_root.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "DOCS_BASE_URL": os.environ.get(
            "DOCS_BASE_URL", "http://127.0.0.1:8000/option-pricing-library/"
        ),
    }
    projects = args.projects or ["chromium-375", "chromium-1280"]

    include_embedded_panels = args.page_key in {"homepage", "visual_report"}

    before_root = stage_dir(run_root, "before")
    build_docs(env)
    before_score_json, before_score_md = generate_score_reports(env)
    before_validation_ok, before_steps = run_targeted_validation(
        page_key=args.page_key,
        page_path=review_path,
        stage_root=before_root,
        include_embedded_panels=include_embedded_panels,
        projects=projects,
        env=env,
    )
    before_score = extract_page_score(before_score_json, args.page_key)
    before_result = LoopStageResult(
        score=before_score[0],
        suggestions=before_score[1],
        validation_ok=before_validation_ok,
        validation_steps=before_steps,
        capture_dir=str(before_root / "captures"),
        score_report_json=str(
            copy_report(before_score_json, before_root / "improvement-report.json")
        ),
        score_report_md=str(
            copy_report(before_score_md, before_root / "improvement-report.md")
        ),
    )

    edit_exit_code: int | None = None
    if args.edit_command:
        edit_env = {
            **env,
            "IMPROVEMENT_PAGE_KEY": args.page_key,
            "IMPROVEMENT_PAGE_PATH": page_path,
            "IMPROVEMENT_REVIEW_PATH": review_path,
            "IMPROVEMENT_SPEC_PATH": str(SPECS_DIR / f"{args.page_key}.yml"),
            "IMPROVEMENT_RUN_ROOT": str(run_root),
            "IMPROVEMENT_BEFORE_ROOT": str(before_root),
        }
        edit_exit_code = run_edit_hook(args.edit_command, edit_env, run_root)

    after_root = run_root / "after"
    if not args.edit_command:
        copy_tree(before_root, after_root)
        after_result = LoopStageResult(
            score=before_result.score,
            suggestions=before_result.suggestions,
            validation_ok=before_result.validation_ok,
            validation_steps=before_result.validation_steps,
            capture_dir=str(after_root / "captures"),
            score_report_json=str(after_root / "improvement-report.json"),
            score_report_md=str(after_root / "improvement-report.md"),
        )
    else:
        after_root.mkdir(parents=True, exist_ok=True)
        build_docs(env)
        after_score_json, after_score_md = generate_score_reports(env)
        after_validation_ok, after_steps = run_targeted_validation(
            page_key=args.page_key,
            page_path=review_path,
            stage_root=after_root,
            include_embedded_panels=include_embedded_panels,
            projects=projects,
            env=env,
        )
        after_score = extract_page_score(after_score_json, args.page_key)
        after_result = LoopStageResult(
            score=after_score[0],
            suggestions=after_score[1],
            validation_ok=after_validation_ok,
            validation_steps=after_steps,
            capture_dir=str(after_root / "captures"),
            score_report_json=str(
                copy_report(after_score_json, after_root / "improvement-report.json")
            ),
            score_report_md=str(
                copy_report(after_score_md, after_root / "improvement-report.md")
            ),
        )

    summary = {
        "page_key": args.page_key,
        "page_path": page_path,
        "review_path": review_path,
        "edit_command": args.edit_command,
        "edit_exit_code": edit_exit_code,
        "before": asdict(before_result),
        "after": asdict(after_result),
    }
    write_text(run_root / "summary.json", json.dumps(summary, indent=2))
    write_text(
        run_root / "summary.md",
        render_summary(
            args.page_key,
            page_path,
            before_result,
            after_result,
            edit_command=args.edit_command,
            edit_exit_code=edit_exit_code,
        ),
    )

    print(f"Wrote {run_root / 'summary.md'} and {run_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
