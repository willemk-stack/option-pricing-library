from __future__ import annotations

import json
from pathlib import Path

from scripts import run_docs_browser_audits as runner


def test_playwright_stage_commands_forward_projects_and_split_serial_tests() -> None:
    commands = runner.playwright_stage_commands(
        ["smoke.spec.ts", "dom-audits.spec.ts", "pages.spec.ts"],
        projects=["chromium-375", "chromium-1280"],
        update_snapshots=False,
    )

    assert [stage_name for stage_name, _ in commands] == [
        "playwright-parallel",
        "playwright-pages.spec.ts",
    ]
    parallel_command = commands[0][1]
    serial_command = commands[1][1]
    assert parallel_command[2:5] == ["test", "smoke.spec.ts", "dom-audits.spec.ts"]
    assert parallel_command.count("--project") == 2
    assert "--workers=1" not in parallel_command
    assert serial_command[-1] == "--workers=1"
    assert serial_command.count("--project") == 2


def test_build_runner_env_sets_prebuilt_and_filter_variables(tmp_path: Path) -> None:
    env = runner.build_runner_env(
        review_paths=["/performance/"],
        review_page_keys=["visual_report"],
        findings_parts_dir=tmp_path,
        base_env={"DOCS_BASE_URL": "http://example.test/docs/"},
    )

    assert env["SKIP_DOCS_PREBUILD"] == "1"
    assert env["SERVE_PREBUILT_SITE"] == "1"
    assert env["REVIEW_PATHS"] == "/performance/"
    assert env["REVIEW_PAGE_KEYS"] == "visual_report"
    assert env["DOCS_AUDIT_FINDINGS_DIR"] == tmp_path.as_posix()
    assert env["DOCS_BASE_URL"] == "http://example.test/docs/"


def test_prepare_findings_output_resets_parts_directory(tmp_path: Path) -> None:
    findings_json = tmp_path / "docs-audit" / "findings.json"
    stale_parts_dir = findings_json.parent / "findings.parts"
    stale_parts_dir.mkdir(parents=True)
    (stale_parts_dir / "stale.json").write_text("[]", encoding="utf8")

    parts_dir, summary_path = runner.prepare_findings_output(findings_json, None)

    assert parts_dir == stale_parts_dir
    assert parts_dir.exists()
    assert list(parts_dir.iterdir()) == []
    assert summary_path == findings_json.parent / "summary.md"


def test_aggregate_findings_records_merges_all_json_payloads(tmp_path: Path) -> None:
    parts_dir = tmp_path / "findings.parts"
    parts_dir.mkdir()
    (parts_dir / "one.json").write_text(
        json.dumps(
            [
                {
                    "suite": "dom-audits.spec.ts",
                    "route": "/performance/",
                    "severity": "major",
                    "project": "chromium-1280",
                    "rule": "text-overflow",
                    "message": "overflow",
                }
            ]
        ),
        encoding="utf8",
    )
    (parts_dir / "two.json").write_text(
        json.dumps(
            [
                {
                    "suite": "math-audits.spec.ts",
                    "route": "/performance/",
                    "severity": "critical",
                    "project": "chromium-375",
                    "rule": "raw-tex-leftover",
                    "message": "raw tex",
                }
            ]
        ),
        encoding="utf8",
    )

    findings = runner.aggregate_findings_records(parts_dir)

    assert findings == [
        {
            "suite": "dom-audits.spec.ts",
            "route": "/performance/",
            "severity": "major",
            "project": "chromium-1280",
            "rule": "text-overflow",
            "message": "overflow",
        },
        {
            "suite": "math-audits.spec.ts",
            "route": "/performance/",
            "severity": "critical",
            "project": "chromium-375",
            "rule": "raw-tex-leftover",
            "message": "raw tex",
        },
    ]


def test_write_findings_summary_reports_counts(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.md"

    runner.write_findings_summary(
        summary_path,
        [
            {
                "severity": "critical",
                "suite": "math-audits.spec.ts",
                "route": "/performance/",
                "project": "chromium-375",
                "rule": "raw-tex-leftover",
                "message": "raw tex",
            },
            {
                "severity": "major",
                "suite": "dom-audits.spec.ts",
                "route": "/performance/",
                "project": "chromium-1280",
                "rule": "text-overflow",
                "message": "overflow",
            },
        ],
    )

    summary = summary_path.read_text(encoding="utf8")
    assert "Total findings: 2" in summary
    assert "critical=1" in summary
    assert "math-audits.spec.ts=1" in summary
    assert "raw tex" in summary
