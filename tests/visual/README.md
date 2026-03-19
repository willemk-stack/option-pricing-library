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
npm run test:smoke
npm run test:audits
npm run test:a11y
npm run test:sentinel
npm run test:repo-facts
npm run test:baseline
npm run test:components
npm run test:artifacts
```

`npm run test:components` intentionally uses one worker. The mobile `/performance/`
snapshot table can flip between two raster states under heavier Ubuntu parallelism,
so component snapshots are serialized to keep the authoritative CI baseline stable.

In CI, the snapshot-based suites (`sentinel`, `baseline`, `components`, and `artifacts`) are authoritative on Ubuntu. Windows still runs the non-snapshot browser checks (`smoke`, `audits`, and `a11y`) to catch cross-platform issues without forcing a second snapshot baseline set.
The shared repo-facts shell widget is checked separately with mocked data so full-page baselines stay focused on stable docs content instead of async page chrome.

For the closest local match to CI, run the visual suites inside the GitHub-runner-style
Ubuntu container:

```bash
python ../../scripts/run_ci_visual_regression.py verify
```

Native Windows page-snapshot runs are useful for debugging, but they are not the
authoritative baseline source. When a snapshot diff matters, verify or refresh it
through the CI-like Ubuntu runner instead of relying on native Windows output.

The docs pre-push hook now requires Docker for docs-sensitive pushes. It still runs the
fast local strict-build, smoke, DOM, and targeted accessibility checks first, then runs
the authoritative Ubuntu snapshot suites in Docker against the same filtered review paths.
For targeted page edits, the Docker step always includes `sentinel.spec.ts` and
`pages.spec.ts`, and adds component or embedded-panel suites only when the changed paths
touch pages that own those snapshots.
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
- configured pages render at the widths defined in `scripts/visual_audit/review_targets.json`
- no obvious DOM overflow in main content
- structured DOM/CSS findings with `critical`, `major`, and `minor` severity
- console errors and uncaught page errors during navigation
- images are loaded
- theme-swapped light/dark figure pairs resolve to a single visible asset
- audited routes do not silently fall through to MkDocs 404 pages
- screenshots stay within baseline tolerance
- the repository facts shell widget renders deterministic mocked data outside full-page baselines
- component-level screenshots for key proof and metric blocks
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

`npm run test:update` refreshes all committed snapshot suites in one pass.

When you need authoritative snapshot updates that match CI, prefer:

```bash
python ../../scripts/run_ci_visual_regression.py update
```

That Ubuntu container path is also the preferred way to investigate snapshot mismatches.
Native Windows page snapshots can legitimately differ from the shared Ubuntu baselines.

The authoritative route, theme, and width matrix lives in `scripts/visual_audit/review_targets.json`.
The browser suite also supports `REVIEW_PAGE_KEYS` and `REVIEW_PATHS` to narrow runs to one page during bounded improvement passes.
