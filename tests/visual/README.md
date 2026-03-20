# Visual tests

These tests run browser-based visual checks against the local MkDocs site.

## Install

```bash
cd tests/visual
npm install
npx playwright install
```

## Run

Playwright starts and reuses the local docs server automatically. For a quick baseline check:

```bash
cd tests/visual
npm run test:baseline
```

For the full browser suite, including accessibility:

```bash
cd tests/visual
npm test
```

For the CI-shaped split stages:

```bash
cd tests/visual
npm run test:local:quick
npm run test:local:full
npm run test:local:update
npm run test:smoke
npm run test:audits
npm run test:a11y
npm run test:sentinel
npm run test:repo-facts
npm run test:baseline
npm run test:components
npm run test:artifacts
```

The `test:local:*` scripts are the preferred native entrypoints. They build the docs
site once, then reuse the prebuilt output for all selected suites instead of letting
each standalone Playwright command trigger its own rebuild.

The lower-level `npm run test:*` suite commands and `npm run test:raw` remain useful
for one-off debugging, but each fresh invocation can rebuild diagrams, generated assets,
and MkDocs before Playwright starts the server.

`npm run test:baseline` and `npm run test:components` intentionally use one worker.
The blocking full-page suite is now a representative subset, and the mobile
`/performance/` snapshot table can still flip between two raster states under heavier
Ubuntu parallelism, so the authoritative page suite keeps that route out of the
blocking full-page subset and leaves it to narrower component or embedded-panel checks.

In CI, the snapshot-based suites (`sentinel`, `baseline`, `components`, and `artifacts`) are authoritative on Ubuntu. Windows still runs the non-snapshot browser checks (`smoke`, `audits`, and `a11y`) to catch cross-platform issues without forcing a second snapshot baseline set.
The shared repo-facts shell widget is checked separately with mocked data so full-page baselines stay focused on stable docs content instead of async page chrome.
The two composite proof panels now inline their child thumbnails, and MathJax is served
from a vendored local asset so visual CI no longer depends on nested browser fetches.

For the closest local match to CI, run the visual suites inside the GitHub-runner-style
Ubuntu container. The GitHub visual-regression job now uses this same container wrapper
and selects suites with the same path-aware rules as the docs pre-push guard:

```bash
python ../../scripts/run_ci_visual_regression.py verify
```

Native Windows page-snapshot runs are useful for debugging, but they are not the
authoritative baseline source. When a snapshot diff matters, verify or refresh it
through the CI-like Ubuntu runner instead of relying on native Windows output.

For the fast blocking docs contract without the heavier advisory audits, run:

```bash
python ../../scripts/docs_doctor.py
```

When you do want a native run, prefer the cross-platform wrapper:

```bash
python ../../scripts/run_local_visual_regression.py verify
python ../../scripts/run_local_visual_regression.py verify --skip-build --tests smoke.spec.ts repo-facts.spec.ts
```

The docs pre-push hook now requires Docker for docs-sensitive pushes. It still runs the
fast local strict-build, smoke, DOM, and targeted accessibility checks first, then runs
the authoritative Ubuntu snapshot suites in Docker against the same filtered review paths.
When the hook fails, it also writes the full output to
`artifacts/pre_push/docs_pre_push_last_failure.log` so the failing stage can be
inspected without digging through Git's generic push error.
It also refreshes linked generated outputs before validating: `README.md` from
`README.template.md` and `examples/`, D2 diagrams from `docs/assets/diagrams/src/`,
and docs figures under `docs/assets/generated/` when their linked sources change.
Those docs figures now rebuild through the same CI-like Ubuntu container used for
authoritative snapshot verification, so native Windows rendering differences no
longer cause false stale-asset failures in the pre-push hook.
If any of those refreshes modify tracked files, the hook now blocks the push so you can
review and stage the updated generated outputs instead of pushing stale snapshots.
When the touched change can invalidate the published performance evidence, the hook also
checks the committed benchmark source manifest and blocks until the benchmark snapshot is
refreshed in the authoritative benchmark workflow.
For targeted page edits, the Docker step always includes `sentinel.spec.ts` and the
representative `pages.spec.ts` subset, and adds component or embedded-panel suites only
when the changed paths touch pages that own those snapshots.
When one of those stages fails, the scripts now print a failure class so you can tell
whether the regression came from environment setup, MkDocs/build output, generated assets,
DOM/a11y checks, or Ubuntu-authoritative snapshots.
In GitHub Actions, the PR job summary can also show warning-level entries for
non-blocking visual-state advisories, so cleanup items can be surfaced without failing
the docs job.

For bounded improvement-loop capture runs, set a page filter and capture directory:

```bash
cd tests/visual
set REVIEW_PAGE_KEYS=homepage
set IMPROVEMENT_CAPTURE_DIR=..\..\artifacts\visual-state\improvement-runs\manual\homepage\before\captures
npm run test:capture
```

## What these tests check

- configured proof-path and generated-asset pages render in light and dark themes
- smoke, DOM, accessibility, and review-capture coverage use the broad page and width matrix defined in `scripts/visual_audit/review_targets.json`
- blocking full-page baselines use the smaller `page_snapshot_pages` and `page_snapshot_widths` subset from `scripts/visual_audit/review_targets.json`
- no obvious DOM overflow in main content
- structured DOM/CSS findings with `critical`, `major`, and `minor` severity
- console errors and uncaught page errors during navigation
- images are loaded
- theme-swapped light/dark figure pairs resolve to a single visible asset
- audited routes do not silently fall through to MkDocs 404 pages
- screenshots stay within baseline tolerance
- the repository facts shell widget renders deterministic mocked data outside full-page baselines
- component-level screenshots for key proof and metric blocks
- the reviewer proof panel and benchmark overview no longer depend on browser-resolved linked PNG thumbnails
- a small sentinel component snapshot subset can fail fast before the full pixel suite

## Snapshot update

Only update snapshots intentionally after reviewing the visual diff:

```bash
cd tests/visual
npm run test:update
```

Use the narrower commands when you only intend to refresh one part of the snapshot suite:

```bash
cd tests/visual
npm run test:update:sentinel
npm run test:update:pages
npm run test:update:components
npm run test:update:artifacts
```

`npm run test:update` refreshes all committed snapshot suites in one pass through the
shared wrapper, so serialized page and component suites keep the same ordering as CI.

When you need authoritative snapshot updates that match CI, prefer:

```bash
python ../../scripts/run_ci_visual_regression.py update
```

For a fast local freshness check of generated docs assets, you can run:

```bash
python ../../scripts/build_visual_artifacts.py all --profile ci --check
```

That direct check uses the local renderer, so on Windows it is useful for iteration but
not authoritative. For the authoritative Ubuntu refresh path that matches CI, use:

```bash
python ../../scripts/run_ci_visual_regression.py build-assets
```

For committed performance evidence, you can check whether the tracked benchmark-source
inputs still match the current benchmark snapshot without rerunning the full benchmark
suite:

```bash
python ../../scripts/build_benchmark_artifacts.py --check
```

When that check fails, the authoritative refresh path is the `benchmarks` GitHub
workflow, which now rebuilds the benchmark bundle and opens a draft PR with the
refreshed artifacts.

That Ubuntu container path is also the preferred way to investigate snapshot mismatches.
Native Windows page snapshots can legitimately differ from the shared Ubuntu baselines.
For native iteration, the `run_local_visual_regression.py` wrapper is more stable than
running several standalone Playwright commands back-to-back because it reuses one
prebuilt site across the whole selected suite set.

The authoritative audit and snapshot targets live in `scripts/visual_audit/review_targets.json`.
Use `pages`, `themes`, and `widths` for the broad browser audit surface, and
`page_snapshot_pages` plus `page_snapshot_widths` for the blocking full-page baseline subset.
The browser suite also supports `REVIEW_PAGE_KEYS` and `REVIEW_PATHS` to narrow runs
to one page during bounded improvement passes.
