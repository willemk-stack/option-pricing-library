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

## What these tests check

- configured proof-path and generated-asset pages render in light and dark themes
- configured pages render at the widths defined in `scripts/visual_audit/review_targets.json`
- no obvious DOM overflow in main content
- images are loaded
- theme-swapped light/dark figure pairs resolve to a single visible asset
- audited routes do not silently fall through to MkDocs 404 pages
- screenshots stay within baseline tolerance

## Snapshot update

Only update snapshots intentionally after reviewing the visual diff:

```bash
cd tests/visual
npm run test:update
```

The authoritative route, theme, and width matrix lives in `scripts/visual_audit/review_targets.json`.
