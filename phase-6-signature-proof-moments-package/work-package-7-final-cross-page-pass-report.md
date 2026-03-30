# Stage Report

## Stage

Name: Work Package 7 - Final Cross-Page Pass

## Summary

This pass tightened the proof-page system without reopening the earlier design decisions. The main change is a small shared proof-route strip that now links the homepage, surface repair, eSSVI smooth handoff, local-vol/PDE validation, and performance evidence as one readable sequence. The performance page also received a cleaner opening treatment so its benchmark overview reads more like the other flagship proof objects.

## Goals addressed

- Make proof routing between homepage, eSSVI, surface repair, local-vol/PDE, and performance easier to follow.
- Ensure the flagship pages feel related without flattening them into one visual template.
- Align figure title/subtitle treatment where that improves continuity.
- Check that shared CSS stayed scoped and did not spill onto quiet pages.

## Files changed

- `docs/index.md`
  Added the homepage proof-route strip and tightened one proof-path label so performance reads as a follow-up rather than a parallel hero.
- `docs/user_guides/surface_workflow.md`
  Added the shared proof-route strip with Step 1 state.
- `docs/user_guides/essvi_smooth_handoff.md`
  Added the shared proof-route strip with Step 2 state.
- `docs/user_guides/localvol_pde_validation.md`
  Added the shared proof-route strip with Step 3 state.
- `scripts/templates/performance.md.template`
  Added the shared proof-route strip with follow-up state and a clearer `Benchmark overview` opening section.
- `docs/performance.md`
  Regenerated from the updated template.
- `docs/stylesheets/extra.css`
  Added fully scoped styling for the shared proof-route strip, including dark-theme and responsive behavior.
- `tests/visual/pages.spec.ts-snapshots/*`
  Refreshed the affected full-page snapshots after the routing pass.

## Visual changes

- Added one restrained proof-route strip near the top of each flagship page so the pages read as a sequence rather than as isolated moments.
- Kept the route strip quiet: low-height cards, no extra hero treatment, no large metrics, no decorative escalation.
- Gave the performance page a clearer opening hierarchy by explicitly naming the benchmark overview section before the existing overview figure.
- Preserved each page's existing flagship asset and composition rather than normalizing the pages into one repeated layout.

## Content changes

- intros / section leads
  The performance page now opens the benchmark overview with a lead that explains why that figure is the page's first review object.
- framing text
  Homepage proof-path framing now labels performance as a follow-up instead of a competing proof path step.
- anything beyond readability cleanup
  The shared route strip makes the intended reading order explicit instead of leaving cross-page sequencing implied.

## Screenshots

Before/after capture directories:

- `tests/visual/artifacts/phase-6-final-cross-page-pass/before`
- `tests/visual/artifacts/phase-6-final-cross-page-pass/after`

Representative full-page captures:

- Homepage
  - `tests/visual/artifacts/phase-6-final-cross-page-pass/before/home-light-chromium-1280.png`
  - `tests/visual/artifacts/phase-6-final-cross-page-pass/after/home-light-chromium-1280.png`
- Surface repair
  - `tests/visual/artifacts/phase-6-final-cross-page-pass/before/user-guides-surface-workflow-light-chromium-1280.png`
  - `tests/visual/artifacts/phase-6-final-cross-page-pass/after/user-guides-surface-workflow-light-chromium-1280.png`
- eSSVI smooth handoff
  - `tests/visual/artifacts/phase-6-final-cross-page-pass/before/user-guides-essvi-smooth-handoff-light-chromium-1280.png`
  - `tests/visual/artifacts/phase-6-final-cross-page-pass/after/user-guides-essvi-smooth-handoff-light-chromium-1280.png`
- Local-vol / PDE validation
  - `tests/visual/artifacts/phase-6-final-cross-page-pass/before/user-guides-localvol-pde-validation-light-chromium-1280.png`
  - `tests/visual/artifacts/phase-6-final-cross-page-pass/after/user-guides-localvol-pde-validation-light-chromium-1280.png`
- Performance
  - `tests/visual/artifacts/phase-6-final-cross-page-pass/before/performance-light-chromium-1280.png`
  - `tests/visual/artifacts/phase-6-final-cross-page-pass/after/performance-light-chromium-1280.png`

## Why these changes were made

The earlier work packages established page-level proof moments well, but the system still relied too much on user inference to connect those moments into one route. This pass adds just enough routing language and shared framing to make the sequence legible without restyling the pages into sameness. The performance opening adjustment uses the same logic: align the figure hierarchy where helpful, but do not create a second flagship treatment or inflate supporting chrome.

## What was intentionally kept restrained

- No new flagship asset was introduced.
- No approved flagship figure was redesigned.
- No new metrics wall or dashboard strip was added.
- Quiet pages were not given the proof-route component.
- The proof-route strip stays small and functional instead of becoming a decorative banner.

## Anti-regression check

- Did any wrapper become louder than the proof?
  No. The new route strip is visibly subordinate to the main figure and section lead on every touched page.
- Did a second competing hero appear?
  No. The existing flagship proof objects remain the clear anchors.
- Did the page become more premium without becoming more informative?
  No. The main value is sequencing and framing clarity, not surface polish alone.
- Did quiet pages get louder as a side effect?
  No route-strip spillover was observed on the quiet-page spot-check. The shared CSS stayed component-scoped.

## Risks / what still feels off

- The route strip is intentionally quiet, but it is still another wrapper near the top of each flagship page; if later passes add more top-of-page scaffolding, that area could become crowded.
- Quiet-page spot-checking surfaced one pre-existing issue unrelated to this pass: `/api/` still has a dark-theme inline-code contrast miss on `.doc-link-card__copy > code` such as `PricingContext` at `4.09:1` instead of `4.5:1`.

## Validation

- `python -m mkdocs build --strict`
- Focused Playwright verification on `/`, `/user_guides/surface_workflow/`, `/user_guides/essvi_smooth_handoff/`, `/user_guides/localvol_pde_validation/`, and `/performance/`
  - `dom-audits.spec.ts`
  - `a11y.spec.ts`
  - `pages.spec.ts --update-snapshots`
  - Result: `120 passed`
- Review capture refresh for the same five routes
  - `review-capture.spec.ts`
  - Result: `128 passed`
- Quiet-page spillover spot-check on `/installation/` and `/api/`
  - `/installation/` passed
  - `/api/` only failed the pre-existing dark-theme inline-code contrast issue noted above

## Approval checkpoint

Do not continue to the next work package until this pass is reviewed.
