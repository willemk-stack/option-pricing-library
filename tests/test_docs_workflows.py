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
    parsed = yaml.safe_load(workflow)

    assert "run: python scripts/render_d2_diagrams.py --check" in workflow
    assert (
        "run: python scripts/build_visual_artifacts.py all --profile ci --check"
        in workflow
    )
    assert parsed[True]["workflow_run"]["workflows"] == [
        "docs-visual-assets-auto-refresh"
    ]
    assert parsed[True]["workflow_run"]["types"] == ["completed"]


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


def test_docs_ci_runs_browser_audits_via_ci_like_container() -> None:
    workflow = workflow_text("docs-ci.yml")

    assert "Run CI-like smoke and DOM audits" in workflow
    assert (
        "run_ci_visual_regression.py verify \\\n            --skip-build \\\n            --tests smoke.spec.ts dom-audits.spec.ts"
        in workflow
    )
    assert "Run CI-like math audits" in workflow
    assert (
        "run_ci_visual_regression.py verify \\\n            --skip-build \\\n            --tests math-audits.spec.ts"
        in workflow
    )
    assert "Run CI-like accessibility checks" in workflow
    assert (
        "run_ci_visual_regression.py verify \\\n            --skip-build \\\n            --tests a11y.spec.ts"
        in workflow
    )
    assert "npx playwright test smoke.spec.ts dom-audits.spec.ts" not in workflow
    assert "npx playwright test math-audits.spec.ts" not in workflow


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
    on_pull_request = workflow[True]["pull_request"]

    # Must trigger on push to same-repo branches, including main, when source
    # files change so stale generated assets are auto-refreshed before docs-ci
    # revalidates the refreshed branch head.
    assert "branches-ignore" not in on_push
    watched_paths = on_push["paths"]
    assert ".github/workflows/docs-ci.yml" in watched_paths
    assert ".github/workflows/docs-visual-assets-auto-refresh.yml" in watched_paths
    assert "benchmarks/artifacts/**" in watched_paths
    assert any(p.startswith("src/option_pricing") for p in watched_paths)
    assert "scripts/build_benchmark_artifacts.py" in watched_paths
    assert "scripts/build_visual_artifacts.py" in watched_paths
    assert "scripts/install_d2.sh" in watched_paths
    assert "scripts/render_d2_diagrams.py" in watched_paths
    # Must cover CSS/JS/theme overrides and the MkDocs config — the only docs/
    # subdirectories that can cause pixel-level changes in sentinel renders.
    assert "docs/stylesheets/**" in watched_paths
    assert "docs/overrides/**" in watched_paths
    assert "docs/assets/**" in watched_paths
    assert "mkdocs.yml" in watched_paths
    # Must NOT use the broad docs/** glob — pure markdown text edits do not
    # affect pixel output and should not fire the full expensive pipeline.
    assert "docs/**" not in watched_paths
    assert on_pull_request["paths"] == watched_paths

    # Must have write permission to push back refreshed assets.
    assert workflow["permissions"]["contents"] == "write"
    assert (
        workflow["concurrency"]["group"]
        == "docs-visual-assets-auto-refresh-${{ github.event.pull_request.head.ref || github.ref_name }}"
    )

    job = next(iter(workflow["jobs"].values()))

    # Must guard against infinite push loops triggered by the bot's own commit.
    assert "github-actions[bot]" in job["if"]
    # Must only attempt branch updates from pull requests whose head branch
    # lives in the same repository; forks do not have safe write access.
    assert (
        "github.event.pull_request.head.repo.full_name == github.repository"
        in job["if"]
    )

    # Must run on the same Ubuntu version and D2 version as docs-ci to
    # produce identical pixels.
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["env"]["D2_VERSION"] == "v0.7.1"
    # On pull_request the workflow must operate on the head branch, not the
    # synthetic merge ref, so refreshed assets push back to the PR branch.
    checkout_step = next(s for s in job["steps"] if s.get("name") == "Checkout")
    assert checkout_step["with"]["ref"] == "${{ github.head_ref || github.ref }}"

    # Must regenerate D2 diagrams, visual assets, and commit them.
    step_runs = [s.get("run", "") for s in job["steps"]]
    assert any('bash scripts/install_d2.sh "${D2_VERSION}"' in r for r in step_runs)
    assert any("render_d2_diagrams.py" in r for r in step_runs)
    assert any("build_visual_artifacts.py all --profile ci" in r for r in step_runs)
    assert any("git commit" in r and "git push" in r for r in step_runs)

    # Must refresh and verify snapshots through the same runner-container wrapper
    # that docs-ci uses. Running raw Playwright on the host runner can produce
    # font/rasterization drift that breaks the container-based authoritative job.
    assert any(
        "run_ci_visual_regression.py update --skip-build" in r for r in step_runs
    )
    assert any(
        "run_ci_visual_regression.py verify --skip-build" in r for r in step_runs
    )
    assert not any("npx playwright test" in r for r in step_runs)
    # The commit must stage generated assets plus all committed authoritative
    # snapshot suites that the wrapper can refresh.
    commit_run = next(r for r in step_runs if "git commit" in r and "git push" in r)
    assert "docs/assets/diagrams/" in commit_run
    assert "sentinel.spec.ts-snapshots" in commit_run
    assert "pages.spec.ts-snapshots" in commit_run
    assert "components.spec.ts-snapshots" in commit_run

    # RISK MITIGATION: after the update wrapper runs, a verify wrapper run
    # must confirm the refreshed snapshots are stable. If the verify fails, the
    # workflow exits non-zero and nothing is committed, surfacing the regression.
    last_update_idx = max(
        i
        for i, r in enumerate(step_runs)
        if "run_ci_visual_regression.py update --skip-build" in r
    )
    verify_runs_after_update = [
        r
        for r in step_runs[last_update_idx + 1 :]
        if "run_ci_visual_regression.py verify --skip-build" in r
    ]
    assert verify_runs_after_update, (
        "A verify step must follow the update step to guard against committing "
        "regressions."
    )


def test_docs_ci_reruns_after_auto_refresh_on_latest_branch_head() -> None:
    workflow = yaml.safe_load(workflow_text("docs-ci.yml"))

    assert (
        workflow["concurrency"]["group"]
        == "docs-ci-${{ github.event.pull_request.number || github.event.workflow_run.pull_requests[0].number || github.event.workflow_run.head_branch || github.ref_name }}"
    )

    build_job = workflow["jobs"]["build"]
    assert build_job["if"] == (
        "github.event_name != 'workflow_run' || "
        "github.event.workflow_run.conclusion == 'success'"
    )

    checkout_step = next(s for s in build_job["steps"] if s.get("name") == "Checkout")
    assert checkout_step["with"]["ref"] == (
        "${{ github.event.workflow_run.head_branch || github.ref }}"
    )
