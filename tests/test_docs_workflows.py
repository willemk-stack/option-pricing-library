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
    assert (
        "run: python scripts/build_benchmark_artifacts.py --write-source-manifest"
        in workflow
    )
    assert (
        "run: python scripts/run_ci_visual_regression.py update --skip-build"
        in workflow
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


def test_docs_visual_assets_auto_refresh_runs_on_push_and_guards_against_loops() -> (
    None
):
    workflow = yaml.safe_load(workflow_text("docs-visual-assets-auto-refresh.yml"))

    # In PyYAML, the YAML keyword `on` is parsed as boolean True.
    on_push = workflow[True]["push"]

    # Must trigger on push to non-main branches when source files change.
    assert "main" in on_push["branches-ignore"]
    watched_paths = on_push["paths"]
    assert ".github/workflows/docs-ci.yml" in watched_paths
    assert ".github/workflows/docs-visual-assets-auto-refresh.yml" in watched_paths
    assert "benchmarks/artifacts/**" in watched_paths
    assert any(p.startswith("src/option_pricing") for p in watched_paths)
    assert "scripts/build_benchmark_artifacts.py" in watched_paths
    assert "scripts/build_visual_artifacts.py" in watched_paths
    # Must cover CSS/JS/theme overrides and the MkDocs config — the only docs/
    # subdirectories that can cause pixel-level changes in sentinel renders.
    assert "docs/stylesheets/**" in watched_paths
    assert "docs/overrides/**" in watched_paths
    assert "docs/assets/**" in watched_paths
    assert "mkdocs.yml" in watched_paths
    # Must NOT use the broad docs/** glob — pure markdown text edits do not
    # affect pixel output and should not fire the full expensive pipeline.
    assert "docs/**" not in watched_paths

    # Must have write permission to push back refreshed assets.
    assert workflow["permissions"]["contents"] == "write"

    job = next(iter(workflow["jobs"].values()))

    # Must guard against infinite push loops triggered by the bot's own commit.
    assert "github-actions[bot]" in job["if"]

    # Must run on the same Ubuntu version as docs-ci to produce identical pixels.
    assert job["runs-on"] == "ubuntu-24.04"

    # Must regenerate visual assets and commit them.
    step_runs = [s.get("run", "") for s in job["steps"]]
    assert any("build_visual_artifacts.py all --profile ci" in r for r in step_runs)
    assert any("git commit" in r and "git push" in r for r in step_runs)

    # Must also refresh authoritative Playwright sentinel snapshots so that
    # docs-ci visual checks pass after any asset or docs content change.
    assert any(
        "playwright" in r and "sentinel.spec.ts" in r and "update-snapshots" in r
        for r in step_runs
    )
    # The commit must stage both generated assets and snapshots together.
    commit_run = next(r for r in step_runs if "git commit" in r and "git push" in r)
    assert "sentinel.spec.ts-snapshots" in commit_run

    # RISK MITIGATION: after --update-snapshots a verify run (without the flag)
    # must confirm the new snapshots are stable.  If the verify fails, the
    # workflow exits non-zero and nothing is committed, surfacing the regression.
    update_idx = next(
        i
        for i, r in enumerate(step_runs)
        if "sentinel.spec.ts" in r and "--update-snapshots" in r
    )
    verify_runs_after_update = [
        r
        for r in step_runs[update_idx + 1 :]
        if "playwright" in r and "sentinel.spec.ts" in r and "--update-snapshots" not in r
    ]
    assert verify_runs_after_update, (
        "A verify step (npx playwright test sentinel.spec.ts without "
        "--update-snapshots) must follow the update step to guard against "
        "committing regressions."
    )
