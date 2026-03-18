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
npm run test:baseline
npm run test:components
npm run test:artifacts
```

In CI, the snapshot-based suites (`sentinel`, `baseline`, `components`, and `artifacts`) are authoritative on Ubuntu. Windows still runs the non-snapshot browser checks (`smoke`, `audits`, and `a11y`) to catch cross-platform issues without forcing a second snapshot baseline set.

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
- component-level screenshots for key proof and metric blocks
- a small sentinel snapshot subset can fail fast before the full pixel suite

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

The authoritative route, theme, and width matrix lives in `scripts/visual_audit/review_targets.json`.
The browser suite also supports `REVIEW_PAGE_KEYS` and `REVIEW_PATHS` to narrow runs to one page during bounded improvement passes.
