from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "visual_audit"))

loop = importlib.import_module("run_improvement_loop")


def test_parse_args_supports_repeated_project_and_suite_filters() -> None:
    args = loop.parse_args(
        [
            "--page-key",
            "homepage",
            "--project",
            "chromium-375",
            "--project",
            "chromium-1280",
            "--suite",
            "smoke",
            "--suite",
            "dom-audits",
        ]
    )

    assert args.projects == ["chromium-375", "chromium-1280"]
    assert args.suites == ["smoke", "dom-audits"]


def test_unknown_suite_is_rejected_during_targeted_validation(tmp_path: Path) -> None:
    try:
        loop.run_targeted_validation(
            page_key="homepage",
            page_path="/",
            stage_root=tmp_path,
            include_embedded_panels=False,
            projects=["chromium-375", "chromium-1280"],
            suites=["unknown-suite"],
            env={"DOCS_BASE_URL": "http://127.0.0.1:8000/option-pricing-library/"},
        )
    except ValueError as exc:
        assert "Unknown improvement-loop suites requested" in str(exc)
    else:
        raise AssertionError(
            "Expected run_targeted_validation to reject an unknown suite."
        )
