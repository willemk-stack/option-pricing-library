from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_ci_visual_regression as runner


def test_prepare_stage_status_artifact_clears_stale_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    stage_status_path = tmp_path / "docs_ci_visual_last_stage.json"

    monkeypatch.setattr(runner, "PRE_PUSH_ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(runner, "CI_VISUAL_STAGE_STATUS_PATH", stage_status_path)

    stage_status_path.write_text("stale", encoding="utf8")

    runner.prepare_stage_status_artifact()

    assert not stage_status_path.exists()


def test_runner_failure_summary_lines_use_stage_metadata() -> None:
    lines = runner.runner_failure_summary_lines(
        mode="build-assets",
        image="example/image",
        stage_status={
            "stage": "install-python-deps",
            "status": "failed",
            "failure_class": "environment-container-python",
            "likely_layer": "Container Python dependency install",
            "next_step": "Inspect Python dependency resolution inside the Ubuntu runner image.",
        },
    )

    assert lines == [
        "CI visual runner failed.",
        "Mode: build-assets",
        "Image: example/image",
        "Stage: install-python-deps",
        "Stage status: failed",
        "Failure class: environment-container-python",
        "Likely layer: Container Python dependency install",
        "Next step: Inspect Python dependency resolution inside the Ubuntu runner image.",
    ]


def test_load_stage_status_round_trips_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    stage_status_path = tmp_path / "docs_ci_visual_last_stage.json"

    monkeypatch.setattr(runner, "PRE_PUSH_ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(runner, "CI_VISUAL_STAGE_STATUS_PATH", stage_status_path)

    payload = {
        "stage": "build-visual-artifacts",
        "status": "failed",
        "failure_class": "generated-asset-build",
        "likely_layer": "SVG/PNG visual artifact generation",
        "next_step": "Fix the visual artifact generator or inputs before retrying snapshot verification.",
    }
    stage_status_path.write_text(json.dumps(payload), encoding="utf8")

    assert runner.load_stage_status() == payload


def test_pip_install_command_disables_progress_noise() -> None:
    command = runner.pip_install_command("--upgrade", "pip")

    assert command[:8] == [
        "python3",
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--progress-bar",
        "off",
        "--break-system-packages",
    ]


def test_browser_snapshot_stage_metadata_points_to_uploaded_artifact_bundle() -> None:
    failure_class, likely_layer, next_step = runner.STAGE_METADATA["playwright-pages"]

    assert failure_class == "browser-snapshot-page"
    assert likely_layer == "Ubuntu representative full-page snapshots"
    assert "visual regression artifact bundle" in next_step
    assert "tests/visual/test-results/**" in next_step
