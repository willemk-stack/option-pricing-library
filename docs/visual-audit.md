# Visual audit workflow

This document defines the visual QA workflow for the docs site.

## Scope

The audit covers:

- text overflow or clipping
- overlapping or misaligned content
- blank or missing-looking media panels
- light/dark theme inconsistencies
- obvious color/contrast regressions
- generated SVG/PNG proof assets embedded in docs pages
- responsive regressions at representative widths

## Why this exists

This docs site is not only regular MkDocs DOM/CSS. Some pages embed generated visual assets.
A visual issue may therefore live in:

1. page DOM/CSS/layout
2. a generated SVG or PNG asset
3. theme/color choices

Fix the root cause in the correct layer.

## Primary pages to audit

Start with the most important user-facing pages:

- `/`
- any page that embeds generated figures, proof panels, or diagrams
- any page changed by the current PR

Expand only if needed.

## Standard widths

Use these viewport widths unless a task requires others:

- 375
- 768
- 1280
- 1536

## Standard themes

Validate both:

- light
- dark

## Completion standard

A visual fix is not complete unless:

- the issue was reproduced in the rendered site
- the relevant docs/assets were rebuilt
- the affected page was rechecked after the change
- the nearest targeted visual checks passed

## Local setup

Install docs dependencies and build the docs artifacts first:

```bash
python -m pip install -e "[docs,plot]"
python scripts/render_d2_diagrams.py
MPLBACKEND=Agg python scripts/make_docs_figures.py
mkdocs build --strict
```

For local browser-based audit:

```bash
mkdocs serve -a 127.0.0.1:8000
```

## Visual test workflow

Run the visual audit in this order:

1. rebuild the relevant docs/assets
2. serve the docs locally
3. run the scripted visual audit
4. inspect any failing screenshots or reports
5. patch the smallest correct root cause
6. rerun the nearest targeted checks

## Root-cause buckets

Classify each issue as one of:

### 1. DOM/CSS/layout issue

Examples:

- text escaping a card
- excessive whitespace
- overlapping elements
- layout collapse at a breakpoint

Typical fix area:

- `docs/`
- `docs/stylesheets/extra.css`
- docs templates or theme overrides

### 2. Generated asset issue

Examples:

- overflow inside SVG
- blank-looking subpanel inside a generated figure
- theme mismatch in a generated image
- missing linked media inside exported assets

Typical fix area:

- generator script
- figure source logic
- exported asset generation path

Do not hand-edit built/generated output unless explicitly requested.

### 3. Theme/color issue

Examples:

- inconsistent panel fills across themes
- unreadable or low-contrast text
- dark-mode asset mismatch

Typical fix area:

- theme CSS variables
- docs styles
- asset export palette

## Reporting format

When filing or fixing a visual issue, record:

- page
- theme
- width
- symptom
- root-cause bucket
- likely source file(s)
- validation run
- before/after screenshot path if available

## Manual color and contrast review

Automated tests do not replace manual color review.

For any docs visual task that changes theme styling, cards, diagrams, or generated visuals:

1. Open the affected page in Chrome.
2. Open DevTools.
3. Run **CSS Overview**.
4. Record:
   - page color inventory
   - low-contrast issues
   - suspicious dark/light inconsistencies
5. Inspect any flagged elements in the Elements panel.
6. If a contrast issue appears inside an embedded SVG/PNG-backed visual, treat it as a generated asset issue, not only a page CSS issue.

Minimum manual review pages:
- homepage
- any changed docs page
- any page with generated proof visuals