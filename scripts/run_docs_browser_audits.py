from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections import Counter
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
    "math-audits.spec.ts",
    "a11y.spec.ts",
]
DEFAULT_UPDATE_TESTS = [
    "sentinel.spec.ts",
    "pages.spec.ts",
    "components.spec.ts",
    "embedded-panels.spec.ts",
]
DEFAULT_FAST_PROJECTS = ["chromium-375", "chromium-1280"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run docs Playwright audits natively against a prebuilt site. This is "
            "the normal GitHub Actions path; use run_ci_visual_regression.py when "
            "you specifically need Dockerized reproduction."
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
        "--project",
        action="append",
        default=[],
        help="Playwright project to run. Repeat to select multiple projects.",
    )
    parser.add_argument(
        "--review-paths",
        nargs="+",
        default=None,
        help="Optional docs paths to forward as REVIEW_PATHS.",
    )
    parser.add_argument(
        "--review-page-keys",
        nargs="+",
        default=None,
        help="Optional page keys to forward as REVIEW_PAGE_KEYS.",
    )
    parser.add_argument(
        "--findings-json",
        type=Path,
        help="Write aggregated agent-readable findings to this JSON file.",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        help="Optional markdown summary path. Defaults next to --findings-json.",
    )
    parser.add_argument(
        "--update-snapshots",
        action="store_true",
        help="Refresh snapshots even when mode=verify.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build visual assets and MkDocs before running Playwright.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Explicitly reuse the existing prebuilt site and verify its contract.",
    )
    return parser.parse_args()


def default_tests_for_mode(mode: str) -> list[str]:
    if mode == "update":
        return DEFAULT_UPDATE_TESTS
    return DEFAULT_VERIFY_TESTS


def format_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def unique_values(values: list[str] | None) -> list[str]:
    if not values:
        return []
    return list(dict.fromkeys(value for value in values if value))


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


def build_runner_env(
    *,
    review_paths: list[str] | None = None,
    review_page_keys: list[str] | None = None,
    findings_parts_dir: Path | None = None,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    config = load_docs_visual_config()
    env = dict(base_env or os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("DOCS_BASE_URL", config["docs_base_url"])
    env.setdefault("DOCS_VISUAL_BUILD_PROFILE", config["build_profile"])
    env["SKIP_DOCS_PREBUILD"] = "1"
    env["SERVE_PREBUILT_SITE"] = "1"

    normalized_review_paths = unique_values(review_paths)
    normalized_review_page_keys = unique_values(review_page_keys)
    if normalized_review_paths:
        env["REVIEW_PATHS"] = ",".join(normalized_review_paths)
    else:
        env.pop("REVIEW_PATHS", None)
    if normalized_review_page_keys:
        env["REVIEW_PAGE_KEYS"] = ",".join(normalized_review_page_keys)
    else:
        env.pop("REVIEW_PAGE_KEYS", None)
    if findings_parts_dir is not None:
        env["DOCS_AUDIT_FINDINGS_DIR"] = findings_parts_dir.as_posix()
    else:
        env.pop("DOCS_AUDIT_FINDINGS_DIR", None)

    return env


def playwright_stage_commands(
    tests: list[str],
    *,
    projects: list[str],
    update_snapshots: bool,
) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    node_command = shutil.which("node") or "node"
    project_args = [
        argument for project in projects for argument in ("--project", project)
    ]
    parallel_tests = [test for test in tests if test not in SERIAL_TESTS]
    serial_tests = [test for test in tests if test in SERIAL_TESTS]

    if parallel_tests:
        command = [node_command, str(PLAYWRIGHT_CLI), "test", *parallel_tests]
        command.extend(project_args)
        if update_snapshots:
            command.append("--update-snapshots")
        commands.append(("playwright-parallel", command))

    for test_name in serial_tests:
        command = [node_command, str(PLAYWRIGHT_CLI), "test", test_name]
        command.extend(project_args)
        command.append("--workers=1")
        if update_snapshots:
            command.append("--update-snapshots")
        commands.append((f"playwright-{test_name}", command))

    return commands


def prepare_findings_output(
    findings_json: Path | None,
    summary_md: Path | None,
) -> tuple[Path | None, Path | None]:
    if findings_json is None:
        return None, None

    findings_json.parent.mkdir(parents=True, exist_ok=True)
    findings_parts_dir = findings_json.parent / f"{findings_json.stem}.parts"
    if findings_parts_dir.exists():
        shutil.rmtree(findings_parts_dir)
    findings_parts_dir.mkdir(parents=True, exist_ok=True)
    if summary_md is None:
        summary_md = findings_json.parent / "summary.md"
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    return findings_parts_dir, summary_md


def aggregate_findings_records(findings_parts_dir: Path) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    if not findings_parts_dir.exists():
        return findings

    for file_path in sorted(findings_parts_dir.glob("*.json")):
        payload = json.loads(file_path.read_text(encoding="utf8"))
        if not isinstance(payload, list):
            continue
        for entry in payload:
            if isinstance(entry, dict):
                findings.append(entry)
    return findings


def write_findings_summary(
    summary_path: Path,
    findings: list[dict[str, object]],
) -> None:
    severity_counts = Counter(
        str(entry.get("severity", "unknown")) for entry in findings
    )
    suite_counts = Counter(str(entry.get("suite", "unknown")) for entry in findings)
    lines = ["# Docs Audit Summary", ""]
    lines.append(f"Total findings: {len(findings)}")
    if severity_counts:
        lines.append(
            "By severity: "
            + ", ".join(
                f"{severity}={severity_counts[severity]}"
                for severity in sorted(severity_counts)
            )
        )
    if suite_counts:
        lines.append(
            "By suite: "
            + ", ".join(
                f"{suite}={suite_counts[suite]}" for suite in sorted(suite_counts)
            )
        )
    lines.append("")

    if not findings:
        lines.append("No findings were recorded.")
    else:
        lines.append("## Findings")
        lines.append("")
        for entry in findings[:50]:
            lines.append(
                "- "
                + " | ".join(
                    [
                        str(entry.get("severity", "unknown")),
                        str(entry.get("suite", "unknown")),
                        str(entry.get("route", "unknown")),
                        str(entry.get("project", "unknown")),
                        str(entry.get("rule", "unknown")),
                        str(entry.get("message", "")),
                    ]
                )
            )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf8")


def main() -> int:
    args = parse_args()
    if args.build and args.skip_build:
        raise SystemExit("Use either --build or --skip-build, not both.")

    requested_tests = unique_values(args.tests or default_tests_for_mode(args.mode))
    if not requested_tests:
        raise SystemExit("No Playwright test files were selected.")
    requested_projects = unique_values(args.project) or list(DEFAULT_FAST_PROJECTS)
    update_snapshots = args.mode == "update" or args.update_snapshots
    findings_parts_dir, summary_path = prepare_findings_output(
        args.findings_json,
        args.summary_md,
    )

    env = build_runner_env(
        review_paths=args.review_paths,
        review_page_keys=args.review_page_keys,
        findings_parts_dir=findings_parts_dir,
    )
    contract = load_docs_site_contract()

    print(f"Running native docs browser audits in {args.mode} mode", flush=True)
    print("Selected tests: " + ", ".join(requested_tests), flush=True)
    print("Selected projects: " + ", ".join(requested_projects), flush=True)
    if env.get("REVIEW_PATHS"):
        print("Review paths: " + env["REVIEW_PATHS"], flush=True)
    if env.get("REVIEW_PAGE_KEYS"):
        print("Review page keys: " + env["REVIEW_PAGE_KEYS"], flush=True)

    if args.build:
        build_prebuilt_site(env, env["DOCS_VISUAL_BUILD_PROFILE"])
    verify_prebuilt_site(contract)

    for stage_name, command in playwright_stage_commands(
        requested_tests,
        projects=requested_projects,
        update_snapshots=update_snapshots,
    ):
        run_stage(stage_name, command, cwd=VISUAL_TEST_DIR, env=env)

    if args.findings_json is not None:
        findings = aggregate_findings_records(findings_parts_dir or args.findings_json)
        args.findings_json.write_text(json.dumps(findings, indent=2), encoding="utf8")
        if summary_path is not None:
            write_findings_summary(summary_path, findings)
        print(f"Wrote findings JSON to {args.findings_json}", flush=True)
        if summary_path is not None:
            print(f"Wrote findings summary to {summary_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
