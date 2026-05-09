# Docs Workflow Architecture

!!! note "Status: repository policy"
    This note documents the documentation workflow for generated artifacts and
    review pages. It is repository process policy, not model validation
    evidence.

## Validation workflow

`.github/workflows/docs-ci.yml` is the authoritative blocking workflow for docs changes.

- It uses `scripts/docs_impact.py` as the single selector for docs scope, review paths, and accessibility scope.
- It validates generated outputs in check-only mode for README, performance docs, D2 diagrams, and benchmark artifacts.
- It builds the MkDocs site once, publishes the built-site artifact, and reuses that artifact for browser audits and deployment.

## Deployment workflow

`docs-ci` also owns deployment on `push` to `main`.

- The deploy path only runs when `docs_site_required == true`.
- Pages deployment consumes the artifact produced by the same validated workflow run.
- Post-deploy verification checks curated live docs routes via `scripts/docs_site_contract.py verify-public`.

## Write-mode workflows

The only workflows that intentionally rewrite committed generated docs assets are explicit refresh/update workflows.

- `.github/workflows/docs-assets-refresh.yml` is the write-mode workflow for README, D2 diagrams, generated visual assets, and the performance page.
- `.github/workflows/docs-advisory.yml` owns the heavier generated-asset and full-page snapshot drift checks.

## Local vs CI

Local hooks and CI intentionally have different weight.

- `benchmark-source-manifest-refresh` runs in pre-commit for benchmark-sensitive source edits and refreshes `benchmarks/artifacts/benchmark_source_manifest.json` before the change is committed.
- `docs-pre-push-guard` runs `scripts/pre_push_docs_validation.py --mode fast` and keeps the default push path portable: no Playwright, no Docker, and no browser gate.
- `docs-manual-guard` runs `scripts/pre_push_docs_validation.py --mode manual` for contributors who want the heavier local browser and Dockerized snapshot checks.
- PR CI stays focused on sync/build plus smoke, DOM, math, and curated accessibility checks; advisory/manual workflows own the full-page snapshot drift checks.
