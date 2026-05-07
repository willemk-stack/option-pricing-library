from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
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


def test_docs_ci_stays_a_deterministic_blocking_docs_gate() -> None:
    workflow = workflow_text("docs-ci.yml")
    parsed = yaml.safe_load(workflow)

    assert "python scripts/render_d2_diagrams.py --check" in workflow
    assert (
        "python scripts/build_visual_artifacts.py all --profile ci --check"
        not in workflow
    )
    assert "workflow_run" not in parsed[True]


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
    assert not (WORKFLOWS_DIR / "docs-visual-assets-auto-refresh.yml").exists()


def test_docs_advisory_owns_visual_drift_checks() -> None:
    workflow = workflow_text("docs-advisory.yml")

    assert (
        "run: python scripts/build_visual_artifacts.py all --profile ci --check"
        in workflow
    )
    assert "Run full authoritative visual regression" in workflow
    assert "python scripts/run_ci_visual_regression.py verify" in workflow
    assert "--skip-build" in workflow
    assert "--tests ${{ needs.build.outputs.selected_tests }}" not in workflow


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
    assert hook["args"] == ["scripts/refresh_benchmark_source_manifest.py"]
    assert hook["pass_filenames"] is True
    assert hook["stages"] == ["pre-commit"]
    assert "benchmarks/" in str(hook["files"])
    assert "scripts/" in str(hook["files"])
    assert "src/option_pricing/" in str(hook["files"])


def test_dev_extras_include_pre_commit_for_local_docs_guards() -> None:
    assert "pre-commit==4.5.1" in dev_dependencies()


def test_docs_ci_no_longer_depends_on_workflow_run_or_authoritative_snapshots() -> None:
    workflow = yaml.safe_load(workflow_text("docs-ci.yml"))

    assert (
        workflow["concurrency"]["group"]
        == "docs-ci-${{ github.event.pull_request.number || github.ref_name }}"
    )

    build_job = workflow["jobs"]["build"]
    assert "if" not in build_job

    checkout_step = next(s for s in build_job["steps"] if s.get("name") == "Checkout")
    assert checkout_step["with"]["ref"] == "${{ github.ref }}"
    assert "authoritative-visual" not in workflow["jobs"]
    assert workflow["jobs"]["stage-pages-artifact"]["needs"] == [
        "build",
        "browser-audits",
    ]
