from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = ROOT / ".github" / "workflows"


def workflow_text(name: str) -> str:
    return (WORKFLOWS_DIR / name).read_text(encoding="utf8")


def test_docs_ci_checks_generated_assets_without_rewriting_them() -> None:
    workflow = workflow_text("docs-ci.yml")

    assert "run: python scripts/render_d2_diagrams.py --check" in workflow
    assert (
        "run: python scripts/build_visual_artifacts.py all --profile ci --check"
        in workflow
    )


def test_docs_assets_refresh_remains_the_write_mode_generator() -> None:
    workflow = workflow_text("docs-assets-refresh.yml")

    assert "run: python scripts/render_d2_diagrams.py\n" in workflow
    assert (
        "run: python scripts/build_visual_artifacts.py all --profile ci\n" in workflow
    )


def test_docs_ci_runs_docs_impact_selected_authoritative_visual_suites() -> None:
    workflow = workflow_text("docs-ci.yml")

    assert "Run docs-impact-selected authoritative visual suites" in workflow
    assert "python scripts/run_ci_visual_regression.py verify" in workflow
    assert "--skip-build" in workflow
    assert "--tests ${{ needs.build.outputs.selected_tests }}" in workflow


def test_docs_ci_owns_deploy_and_deploy_docs_workflow_is_gone() -> None:
    workflow = workflow_text("docs-ci.yml")

    assert "uses: actions/upload-pages-artifact@" in workflow
    assert "uses: actions/deploy-pages@" in workflow
    assert not (WORKFLOWS_DIR / "deploy-docs.yml").exists()
