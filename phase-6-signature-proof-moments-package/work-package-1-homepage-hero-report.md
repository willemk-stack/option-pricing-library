# Stage

Name: Work Package 1 — Homepage hero

## Summary

Replaced the homepage’s reviewer-panel/dashboard treatment with one surface-centered proof moment.
The opening now lands as:
- premium intro
- compact why-this-matters framing
- one dominant 3D eSSVI surface
- two smaller supporting diagnostics
- tighter proof-page routing

## Goals addressed

- give the homepage one unmistakable signature visual early
- replace the summary-dashboard feel with a more memorable first proof object
- keep supporting proof signals secondary
- keep routes to surface repair, eSSVI, local-vol/PDE validation, and performance obvious
- preserve mobile reading order and theme parity

## Files changed

- `docs/index.md`
  - rewrote the homepage opening sequence around one main proof object and calmer routing
- `docs/stylesheets/extra.css`
  - demoted the text hero wrapper, added responsive layout for the signature figure/support diagnostics, and tightened the route grid
- `src/option_pricing/demos/publishing/plots.py`
  - added a generated 3D `homepage_essvi_surface_3d` asset in the existing `showcase` preset
- `docs/assets/generated/showcase/homepage_essvi_surface_3d.light.png`
  - generated light-theme flagship homepage surface
- `docs/assets/generated/showcase/homepage_essvi_surface_3d.dark.png`
  - generated dark-theme flagship homepage surface
- `docs/assets/generated/showcase/homepage_essvi_surface_3d.png`
  - light-copy canonical generated asset
- `design/page_specs/homepage.yml`
  - updated the homepage contract to match the new Phase 6 signature-proof hierarchy
- `tests/visual/embedded-panels.spec.ts`
  - removed the homepage from the embedded-SVG panel audit because the page no longer uses the old reviewer proof panel
- `tests/visual/pages.spec.ts-snapshots/home-*.png`
  - refreshed homepage full-page baselines for the new hero at `375`, `768`, `1280`, and `1536`
- `tests/visual/components.spec.ts-snapshots/home-home-proof-panel-*.png`
  - refreshed the main homepage proof-figure baseline
- `tests/visual/components.spec.ts-snapshots/home-home-snapshot-grid-*.png`
  - refreshed the homepage supporting-diagnostics baseline
- `tests/visual/sentinel.spec.ts-snapshots/home-home-snapshot-grid-light-*.png`
  - refreshed the light-theme sentinel baseline for the homepage support grid
- `tests/visual/artifacts/phase-6-homepage-hero/after/captures/*`
  - captured after screenshots for the report

## Visual changes

- The intro card stays first, but it is visually calmer so it does not compete with the proof object.
- The dominant homepage figure is now a generated 3D smoothed-eSSVI surface instead of a wide multi-card proof dashboard.
- The old proof strip became two smaller support diagnostics:
  - quoted-vs-repaired comparison
  - PDE round-trip validation
- Proof routing is still explicit, but the four routes now sit in a compact 2x2 quiet grid instead of sprawling underneath a dashboard wall.
- Mobile now reads in the intended order:
  - intro
  - why it matters
  - main surface
  - support diagnostics
  - proof-page routes

## Content changes

Describe any wording changes.
Separate these into:

- intros / section leads
  - tightened the lead copy so it says the repo is strongest as one workflow from noisy quotes through local-vol/PDE validation
  - rewrote the quiet framing copy to emphasize reviewer-visible proof rather than generic pricing capability
  - added a short section lead for the signature proof moment to explain why the smooth surface is the right homepage object
- framing text
  - changed the main figure caption to make the eSSVI handoff the centerpiece of the argument
  - changed the support captions so they explain what remains visible before and after the flagship surface
  - tightened the proof-page route copy so each card says exactly what that page proves
- anything beyond readability cleanup
  - no technical claims were silently expanded; the edits are framing and emphasis changes around already-published proof objects and metrics

## Screenshots

- Full page before, light, `1280`:
  - [home-light-chromium-1280.png](../tests/visual/artifacts/phase-3-pilot-pages/after-final-preview/captures/home-light-chromium-1280.png)
- Full page after, light, `1280`:
  - [home-light-chromium-1280.png](../tests/visual/artifacts/phase-6-homepage-hero/after/captures/home-light-chromium-1280.png)
- Main proof object before, light, `1280`:
  - [home-home-proof-panel-light-chromium-1280.png](../tests/visual/artifacts/phase-3-pilot-pages/after-final-preview/captures/home-home-proof-panel-light-chromium-1280.png)
- Main proof object after, light, `1280`:
  - [home-home-proof-panel-light-chromium-1280.png](../tests/visual/artifacts/phase-6-homepage-hero/after/captures/home-home-proof-panel-light-chromium-1280.png)
- Supporting proof strip before, light, `1280`:
  - [home-home-snapshot-grid-light-chromium-1280.png](../tests/visual/artifacts/phase-3-pilot-pages/after-final-preview/captures/home-home-snapshot-grid-light-chromium-1280.png)
- Supporting diagnostics after, light, `1280`:
  - [home-home-snapshot-grid-light-chromium-1280.png](../tests/visual/artifacts/phase-6-homepage-hero/after/captures/home-home-snapshot-grid-light-chromium-1280.png)
- Mobile after, light, `375`:
  - [home-light-chromium-375.png](../tests/visual/artifacts/phase-6-homepage-hero/after/captures/home-light-chromium-375.png)
- Desktop after, dark, `1280`:
  - [home-dark-chromium-1280.png](../tests/visual/artifacts/phase-6-homepage-hero/after/captures/home-dark-chromium-1280.png)

## Why these changes were made

The old homepage proof moment was informative, but it read like a compact dashboard: workflow strip, metrics, and thumbnails all competing inside one frame. Phase 6 asked for a faster, more memorable landing for hiring managers.

The new structure spends the emphasis budget on one real proof object: the smooth surface that makes the Dupire handoff meaningful. That is more technically honest than adding decorative chrome, because it promotes the actual surface geometry while still keeping the repair evidence and PDE validation visible in smaller supporting roles.

## What was intentionally kept restrained

- The text hero stayed compact and quieter than the figure.
- The support diagnostics remained two small panels instead of another metric wall.
- The route block stayed quiet and 2x2 instead of turning into a large card showcase.
- Performance stayed a route, not a second visual hero.
- Shared CSS changes were kept homepage-scoped so API and Quickstart did not get louder.

## Anti-regression check

- Did any wrapper become louder than the proof?
  - No. The intro wrapper was toned down, and the main figure carries the emphasis.
- Did a second competing hero appear?
  - No. The 3D surface is the only dominant proof object.
- Did the page become more premium without becoming more informative?
  - No. The stronger treatment is tied directly to the smooth-surface handoff, and the supporting diagnostics still show repair and validation evidence.
- Did quiet pages get louder as a side effect?
  - No. The CSS additions are homepage-specific, and the build/DOM checks did not surface spillover on quiet pages.

## Risks / what still feels off

- The 3D surface is stronger than the old panel, but it still depends on a static raster. A later pass could consider a slightly more authored surface annotation treatment if it remains disciplined.
- The homepage now routes well into the proof pages, but the surface-repair and eSSVI pages themselves still need their own Phase 6 upgrades to match the stronger landing.
- The repo already had unrelated local changes and untracked package folders outside this pass; they were left untouched.

## Validation

- Rebuilt generated visuals:
  - `& 'C:\Users\ouwez\AppData\Local\Programs\Python\Python312\python.exe' scripts/build_visual_artifacts.py all --profile ci`
- Verified publishing pipeline:
  - `& 'C:\Users\ouwez\AppData\Local\Programs\Python\Python312\python.exe' -m pytest tests/test_visual_publishing_pipeline.py -q`
- Rebuilt docs:
  - `& 'C:\Users\ouwez\AppData\Local\Programs\Python\Python312\python.exe' -m mkdocs build --strict`
- Ran homepage DOM audits across light/dark and `375`, `768`, `1280`, `1536`:
  - `$env:PYTHON_EXECUTABLE='C:\Users\ouwez\AppData\Local\Programs\Python\Python312\python.exe'; $env:SERVE_PREBUILT_SITE='1'; $env:REVIEW_PATHS='/'; npx.cmd playwright test dom-audits.spec.ts`
- Captured homepage review images across light/dark and `375`, `768`, `1280`, `1536`:
  - `$env:PYTHON_EXECUTABLE='C:\Users\ouwez\AppData\Local\Programs\Python\Python312\python.exe'; $env:SERVE_PREBUILT_SITE='1'; $env:REVIEW_PATHS='/'; $env:IMPROVEMENT_CAPTURE_DIR='C:\Users\ouwez\Documents\Quant\option-pricing-library-agent-docs\tests\visual\artifacts\phase-6-homepage-hero\after\captures'; npx.cmd playwright test review-capture.spec.ts`
- Updated homepage page/component/sentinel baselines:
  - `$env:PYTHON_EXECUTABLE='C:\Users\ouwez\AppData\Local\Programs\Python\Python312\python.exe'; $env:SERVE_PREBUILT_SITE='1'; $env:REVIEW_PATHS='/'; npx.cmd playwright test pages.spec.ts components.spec.ts sentinel.spec.ts --update-snapshots`

## Approval checkpoint

Do not continue to the next work package until this pass is reviewed.
