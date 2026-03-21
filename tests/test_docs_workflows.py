from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = ROOT / ".github" / "workflows"


def workflow_text(name: str) -> str:
    return (WORKFLOWS_DIR / name).read_text(encoding="utf8")


def pre_commit_hooks() -> list[dict[str, object]]:
    config = yaml.safe_load(
        (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf8")
    )
    local_repo = next(repo for repo in config["repos"] if repo["repo"] == "local")
    return list(local_repo["hooks"])


def dev_dependencies() -> list[str]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf8"))
    return list(pyproject["project"]["optional-dependencies"]["dev"])


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


def test_pre_commit_refreshes_benchmark_source_manifest_for_benchmark_inputs() -> None:
    hook = next(
        hook
        for hook in pre_commit_hooks()
        if hook["id"] == "benchmark-source-manifest-refresh"
    )

    assert hook["entry"] == "python"
    assert hook["args"] == [
        "scripts/build_benchmark_artifacts.py",
        "--write-source-manifest",
    ]
    assert hook["pass_filenames"] is False
    assert hook["stages"] == ["pre-commit"]
    assert "test_bench_" in str(hook["files"])
    assert "src/option_pricing/" in str(hook["files"])
    assert "scripts/build_benchmark_artifacts\\.py" in str(hook["files"])


def test_dev_extras_include_pre_commit_for_local_docs_guards() -> None:
    assert "pre-commit==4.5.1" in dev_dependencies()
