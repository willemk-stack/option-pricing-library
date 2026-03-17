# Visual tests

These tests run browser-based visual checks against the local MkDocs site.

## Install

```bash
cd tests/visual
npm install
npx playwright install
```

## Run

Start the docs server in another shell:

```bash
mkdocs serve -a 127.0.0.1:8000
```

Then run:

```bash
cd tests/visual
npm test
```

## What these tests check

- homepage renders in light and dark themes
- configured pages render at representative widths
- no obvious DOM overflow in main content
- images are loaded
- screenshots stay within baseline tolerance

## Snapshot update

Only update snapshots intentionally after reviewing the visual diff:

```bash
cd tests/visual
npm run test:update
```
