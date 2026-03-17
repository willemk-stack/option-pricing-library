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

## Modes

The docs quality system runs in three modes:

- Validation mode: blocking PR checks for build, links, Playwright, DOM and CSS audits, accessibility, and visual regression.
- Improvement mode: manual or nightly quality passes guided by specs under `design/` and bounded by explicit allowed edits.
- Artifact mode: focused validation of generated SVG, PNG, and composite proof assets independent of the surrounding page shell.

Progress against the rollout:

- Phase 1 is in place: deterministic build, blocking docs validation workflow, baseline visual regression, design specs, and agent prompts.
- Phase 2 is in progress: artifact-specific CI and severity-tagged browser audits are now part of the implementation surface.
- Phase 3 has started in report-only form: `.github/workflows/improve-docs.yml` generates non-blocking improvement suggestions from the current page specs.
- Phase 3 also now has a bounded page-loop scaffold: `scripts/visual_audit/run_improvement_loop.py` and `.github/workflows/improve-page-loop.yml` capture before/after state for one page and replay targeted validation.
- Phase 4 remains future work: automated improvement PRs.

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

## PR definition of done

A docs-facing PR is not done until:

- the deterministic docs build passes
- generated artifact validation passes for affected assets
- no critical or major findings remain on the required review pages
- accessibility checks pass in the required scope
- screenshot baselines only change when the rendered result is intentionally correct

## Local setup

Install docs dependencies and build the docs artifacts first:

```bash
python -m pip install -e "[docs,plot]"
python scripts/render_d2_diagrams.py
python scripts/build_visual_artifacts.py all --profile ci
mkdocs build --strict
```

For a quick browser baseline check, Playwright can start the docs server itself:

```bash
cd tests/visual
npm run test:baseline
```

To block pushes when docs-sensitive changes would fail local docs validation, install the
pre-push hook once:

```bash
pre-commit install --hook-type pre-commit --hook-type pre-push
```

That hook runs on pushes touching docs-sensitive files such as `docs/**`, `mkdocs.yml`,
`docs/stylesheets/**`, `tests/visual/**`, and the docs visual-audit scripts. It performs a
strict MkDocs build, SVG asset integrity checks, targeted Playwright smoke and DOM checks
against the affected docs paths, and targeted accessibility checks on the curated blocking
review pages.

For the full repeatable audit flow from the repo root:

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_visual_audit.ps1
```

For pull requests, the blocking workflow lives in `.github/workflows/validate-docs.yml`.
Generated-asset validation also has a dedicated workflow in `.github/workflows/validate-docs-artifacts.yml`.
The non-blocking quality-report workflow lives in `.github/workflows/improve-docs.yml`.
The single-page bounded loop workflow lives in `.github/workflows/improve-page-loop.yml`.
The manual draft-PR bridge lives in `.github/workflows/improve-page-pr.yml`.

## Visual test workflow

Run the visual audit in this order:

1. rebuild the relevant docs/assets
2. run strict docs build and asset integrity checks
3. run smoke navigation and DOM/CSS audits
4. run accessibility and artifact-panel checks
5. inspect any failing screenshots or reports
6. patch the smallest correct root cause
7. rerun the nearest targeted checks

Refresh screenshot baselines only when the rendered result is intentionally correct:

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_visual_audit.ps1 -UpdateSnapshots
```

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

## Severity expectations

- `critical`: broken images, missing required content, clipped primary heading, overlapping interactive elements, missing required media on visual-evidence pages.
- `major`: text overflow, empty visual containers, unexpected horizontal scroll, console errors, responsive collapse, theme-variant mismatches.
- `minor`: weak grouping, uneven spacing, odd wrapping, and other non-blocking polish issues.

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

## Design spec inputs

Improvement work should read the specs under `design/` before proposing changes.
Those specs define archetype goals, allowed edits, forbidden edits, and quality targets that go beyond "no bug".

The report-only improvement pass currently writes `artifacts/visual-state/improvement-report.md` and `artifacts/visual-state/improvement-report.json`.

The bounded loop currently writes before/after captures, logs, score reports, and a summary under `artifacts/visual-state/improvement-runs/<run>/<page_key>/`.
The draft-PR workflow consumes that summary and only opens a PR when the bounded after-state still validates cleanly.

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
