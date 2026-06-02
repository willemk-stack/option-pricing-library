# Docs visual QA

This guide defines the visual quality checks for the MkDocs site. It is aimed at
contributors who need to verify that documentation changes render correctly
across the supported themes, widths, and generated proof assets.

## Scope

The visual QA pass covers:

- text overflow or clipping
- overlapping or misaligned content
- blank or missing-looking media panels
- light/dark theme inconsistencies
- obvious color or contrast regressions
- generated SVG/PNG proof assets embedded in docs pages
- responsive regressions at representative widths

## Why this exists

This docs site is not only regular MkDocs DOM/CSS. Some pages embed generated
visual assets. A visual issue may therefore live in:

1. page DOM/CSS/layout
2. a generated SVG or PNG asset
3. theme or color choices

Fix the root cause in the correct layer.

## Primary pages to audit

Start with the most important user-facing pages:

- `/`
- proof-path pages
- `performance/`
- any page that embeds generated figures, proof panels, or diagrams
- any page changed by the current pull request

Expand only if the rendered result or audit output points to a wider issue.

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

A visual fix is complete when:

- the issue was reproduced in the rendered site
- the relevant docs or assets were rebuilt
- the affected page was rechecked after the change
- the nearest targeted visual checks passed

## Pull request checks

A docs-facing pull request should pass:

- strict MkDocs build
- generated artifact validation for affected assets
- smoke navigation checks for affected routes
- DOM/CSS audits at the relevant widths
- accessibility checks in the required scope
- visual snapshot checks when the rendered output intentionally changes

Screenshot baselines should change only when the new rendered result is correct.

## Local setup

Install docs dependencies and build the docs artifacts first:

```bash
python -m pip install -e ".[docs,plot]"
python scripts/render_d2_diagrams.py
python scripts/build_visual_artifacts.py all --profile ci
mkdocs build --strict
```

For a quick browser baseline check, Playwright can start the docs server itself:

```bash
cd tests/visual
npm run test:baseline
```

To block pushes when docs-sensitive changes would fail local docs validation,
install the hooks once:

```bash
pre-commit install --hook-type pre-commit --hook-type pre-push
```

The local guard runs the same core checks used for docs-sensitive changes:
strict MkDocs build, SVG asset integrity checks, targeted Playwright smoke and
DOM checks, targeted accessibility checks, and the Ubuntu-backed snapshot suites
when Docker is available. If Docker Desktop is installed but its Linux engine is
temporarily unavailable, set `DOCS_PRE_PUSH_ALLOW_NO_DOCKER=1` to keep the
non-Docker checks running locally and defer the Ubuntu-only snapshot contract to
CI.

For the full repeatable audit flow from the repo root:

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_visual_audit.ps1
```

## Visual test workflow

Run the visual audit in this order:

1. rebuild the relevant docs and assets
2. run strict docs build and asset integrity checks
3. run smoke navigation and DOM/CSS audits
4. run accessibility and artifact-panel checks
5. inspect any failing screenshots or reports
6. patch the smallest correct root cause
7. rerun the nearest targeted checks

Refresh screenshot baselines only when the rendered result is intentionally
correct:

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_visual_audit.ps1 -UpdateSnapshots
```

For authoritative Playwright snapshot refreshes that match the Ubuntu CI runner,
prefer:

```bash
python scripts/run_ci_visual_regression.py update
```

For the fast native check against a local build artifact, prefer:

```bash
python scripts/docs_audit_plan.py --base-ref main --head-ref HEAD --format json
python scripts/run_docs_browser_audits.py verify --build --tests smoke.spec.ts dom-audits.spec.ts math-audits.spec.ts --project chromium-375 --project chromium-1280 --findings-json artifacts/docs-audit/findings.json
python scripts/run_docs_browser_audits.py verify --skip-build --tests a11y.spec.ts --project chromium-1280 --review-paths /performance/ --findings-json artifacts/docs-audit/a11y-findings.json
```

For generated docs figures that should also match the Ubuntu runner exactly,
prefer:

```bash
python scripts/run_ci_visual_regression.py build-assets
```

Use the CI-like Ubuntu runner for authoritative snapshot verification when a
page diff is under investigation. Native Windows page-snapshot runs are helpful
for debugging, but the shared baselines are owned by the Ubuntu visual-regression
workflow.

For native local iteration, prefer:

```bash
python scripts/run_docs_browser_audits.py verify --build --tests smoke.spec.ts dom-audits.spec.ts math-audits.spec.ts --project chromium-375 --project chromium-1280 --findings-json artifacts/docs-audit/findings.json
python scripts/run_docs_browser_audits.py verify --skip-build --tests a11y.spec.ts --project chromium-1280 --findings-json artifacts/docs-audit/a11y-findings.json
python scripts/run_local_visual_regression.py verify --skip-build --tests smoke.spec.ts repo-facts.spec.ts
```

The native runner verifies the prebuilt-site contract, reuses one site build
across selected suites, supports explicit `--project`, `--review-paths`, and
`--review-page-keys` filters, and emits automation-readable findings under
`artifacts/docs-audit/`. The older `run_local_visual_regression.py` wrapper
remains useful for broad native replay.

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
- `docs/stylesheets/`
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

- `critical`: broken images, missing required content, clipped primary heading,
  overlapping interactive elements, missing required media on visual-evidence
  pages.
- `major`: text overflow, empty visual containers, unexpected horizontal
  scroll, console errors, responsive collapse, theme-variant mismatches.
- `minor`: weak grouping, uneven spacing, odd wrapping, and other non-blocking
  polish issues.

## Reporting format

When filing or fixing a visual issue, record:

- page
- theme
- width
- symptom
- root-cause bucket
- likely source file or files
- validation run
- before/after screenshot path if available

## Design spec inputs

When a visual task is about design quality rather than a concrete rendering
bug, read the specs under `design/` before proposing changes. Those specs define
archetype goals, allowed edits, forbidden edits, and quality targets that go
beyond "no bug".

## Manual color and contrast review

Automated tests do not replace manual color review.

For any docs visual task that changes theme styling, cards, diagrams, or
generated visuals:

1. Open the affected page in Chrome.
2. Open DevTools.
3. Run **CSS Overview**.
4. Record the page color inventory, low-contrast issues, and suspicious
   dark/light inconsistencies.
5. Inspect any flagged elements in the Elements panel.
6. If a contrast issue appears inside an embedded SVG/PNG-backed visual, treat
   it as a generated asset issue, not only a page CSS issue.

Minimum manual review pages:

- homepage
- any changed docs page
- any page with generated proof visuals
