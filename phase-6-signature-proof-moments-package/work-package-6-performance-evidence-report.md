# Stage

Name: Work Package 6 - Performance evidence refinement

## Summary

Reworked the performance page from a competent benchmark dump into a more authored benchmark case study.
The page now lands as:
- a benchmark overview that stays the main proof object
- a stronger reading line that tells the reader what to care about before they start scanning plots
- one editorial flagship moment built around runtime priorities and stage-budget attribution
- sharper interpretation of where runtime is really spent, what the bundle justifies, and where optimization effort is not yet worth spending
- a more bounded closing argument about what the committed benchmark bundle proves and what it does not

## Goals addressed

- keep the benchmark overview as a major proof object
- promote one figure family into a cleaner editorial flagship rather than trying to hero every plot
- strengthen interpretation around what actually matters, what dominates runtime, and where optimization effort is worthwhile
- make the benchmark bundle read as a justification set rather than a timing leaderboard
- reduce the artifact-dump feeling without turning the page into another metric wall
- stay restrained: no 3D detour, no dashboard explosion, no second competing hero

## Files changed

- `scripts/render_performance_page.py`
  - rewrote the performance-page narrative assembly so the generated page now emphasizes reading order, optimization judgment, and bundle scope rather than fast-signal cards
- `scripts/templates/performance.md.template`
  - restructured the page, rewrote the section leads, removed the dashboard-like signal block, and added the new flagship runtime-priority section plus stronger closing claim framing
- `scripts/build_benchmark_artifacts.py`
  - upgraded the macro pipeline figure into an authored stage-budget composition and sharpened the benchmark overview callouts so the visual interpretation matches the page argument
- `scripts/build_visual_artifacts.py`
  - ensured the refreshed macro pipeline summary and benchmark overview can be rebuilt directly from committed snapshot artifacts
- `docs/stylesheets/extra.css`
  - added route-specific performance-page styling for the reading panel, flagship shell, justification panels, and scoped caption contrast fixes
- `docs/performance.md`
  - regenerated the page from the updated template and summary logic
- `docs/assets/generated/benchmarks/macro_pipeline_summary.*`
  - regenerated the upgraded flagship figure in canonical, light, and dark variants
- `docs/assets/generated/benchmarks/benchmark_overview.*`
  - regenerated the overview figure so its callouts align with the stronger runtime-priority interpretation
- `tests/visual/targets.ts`
  - added a dedicated component target for the new flagship figure and narrowed the overview selector to the page-specific benchmark overview panel
- `tests/test_visual_publishing_pipeline.py`
  - added a publishing test that renders the new macro pipeline summary from committed snapshots and checks the themed assets
- `tests/visual/pages.spec.ts-snapshots/performance-*.png`
  - refreshed the full-page performance baselines at `375`, `768`, `1280`, and `1536`
- `tests/visual/components.spec.ts-snapshots/performance-*.png`
  - refreshed component baselines for the overview panel, snapshot table, and new flagship figure
- `tests/visual/artifacts/phase-6-performance-evidence/before/*`
  - captured before screenshots for the report
- `tests/visual/artifacts/phase-6-performance-evidence/after/*`
  - captured after screenshots for the report

## Visual changes

- The benchmark overview stays high on the page and remains the main proof object rather than being demoted behind a new hero stack.
- The page now has one clearly authored flagship moment: the macro pipeline stage-budget figure, presented as a premium but restrained editorial shell under `Runtime priorities`.
- The flagship figure itself now reads less like a raw benchmark artifact and more like an interpretation surface:
  - stage-budget bars across small, medium, and large scenarios
  - concise right-hand callout cards
  - explicit emphasis on optimization order rather than just total elapsed time
- The old `Fast signals` block is gone, which removes the metrics-strip feeling and makes the benchmark family flow cleaner.
- The snapshot table remains present, but it now sits inside a tighter argument about what the committed bundle establishes instead of behaving like another dashboard surface.
- Route-specific panel styling gives the performance page a more deliberate editorial rhythm without making the supporting benchmark families visually louder than the benchmark overview or flagship figure.

## Content changes

Describe any wording changes.
Separate these into:

- intros / section leads
  - rewrote the page intro so it opens on the review question the benchmark bundle is meant to answer, not on generic performance framing
  - added a `What actually matters` reading panel that tells the reader how to interpret the page before the family figures begin
  - introduced `Runtime priorities` as the editorial flagship section and rewrote the lead so it clearly answers the optimization-order question
  - reframed the snapshot table as `What the committed snapshot establishes` instead of as a passive benchmark index
  - rewrote the family section leads so each one states the judgment being made, not just the figure topic
- framing text
  - strengthened the explanation that absolute latency numbers are not the main claim; relative slope, cost-versus-accuracy tradeoff, and stage-budget dominance are
  - made the macro-family prose explicit that surface fitting is the first optimization target, PDE repricing is the second, and the handoff stages are already cheap
  - sharpened the PDE prose so it says what extra runtime buys and why medium grids remain the practical starting point
  - strengthened the closing justification so the page ends on defaults and optimization order rather than on timings in isolation
  - added an explicit `What this bundle does not prove` panel so the limits of the evidence are stated rather than implied
- anything beyond readability cleanup
  - replaced the old `Fast signals` card block with authored interpretation derived from committed snapshot data
  - promoted the macro pipeline family into a purpose-built editorial flagship because it is the best figure family for the required questions:
    - what dominates runtime
    - where optimization effort is worthwhile
    - what the benchmark bundle really justifies
  - carried committed bundle numbers directly into the interpretation:
    - surface fitting plus PDE repricing consume `97.6%` to `98.8%` of the integrated runtime budget
    - quote mesh + handoff + local-vol stay between `7.71 ms` and `14.13 ms` combined
    - surface fitting rises from `250.25 ms` to `794.53 ms`
    - PDE repricing rises from `78.65 ms` to `328.23 ms`
  - clarified the published local-vol PDE tradeoff statement:
    - the committed curve runs through `401x401` against a `601x601` local-vol reference solve

## Screenshots

Full page before, light, `1280`:
![Performance before, full page, light 1280](../tests/visual/artifacts/phase-6-performance-evidence/before/performance-light-chromium-1280.png)

Full page after, light, `1280`:
![Performance after, full page, light 1280](../tests/visual/artifacts/phase-6-performance-evidence/after/performance-light-chromium-1280.png)

Overview panel after, light, `1280`:
![Performance after, overview panel, light 1280](../tests/visual/artifacts/phase-6-performance-evidence/after/performance-performance-overview-panel-light-chromium-1280.png)

Flagship figure after, light, `1280`:
![Performance after, flagship figure, light 1280](../tests/visual/artifacts/phase-6-performance-evidence/after/performance-performance-flagship-figure-light-chromium-1280.png)

Mobile before, light, `375`:
![Performance before, mobile, light 375](../tests/visual/artifacts/phase-6-performance-evidence/before/performance-light-chromium-375.png)

Mobile after, light, `375`:
![Performance after, mobile, light 375](../tests/visual/artifacts/phase-6-performance-evidence/after/performance-light-chromium-375.png)

Desktop after, dark, `1280`:
![Performance after, desktop, dark 1280](../tests/visual/artifacts/phase-6-performance-evidence/after/performance-dark-chromium-1280.png)

## Why these changes were made

Phase 6 asked for a refinement in authorship and interpretation, not a louder or more decorative performance page. The previous page already had strong underlying evidence, especially in the benchmark overview and macro pipeline artifacts, but it still read too much like a clean artifact shelf. That made the numbers available without making the editorial judgment clear enough.

This pass keeps the benchmark overview as a major proof object because it is still the fastest way to scan the committed evidence set. The page then promotes one figure family, not all of them. The macro pipeline summary was the right flagship candidate because it answers the most consequential review questions directly: what actually dominates runtime, what does not, and therefore where optimization effort is worth spending. That is a better use of emphasis than trying to make every benchmark family feel equally heroic.

The prose changes are equally important. The page now states that the benchmark bundle should be read as a bounded justification set rather than as machine-independent latency advertising. It tells the reader which default choices the evidence supports, which runtime lines are already low leverage, and where the evidence stops. That is the distinction Phase 6 asked for: more authorship, sharper interpretation, and more explicit engineering judgment without hiding the numerical caveats.

## What was intentionally kept restrained

- The benchmark overview remained a major proof object instead of being displaced by a new full-page hero.
- Only the macro pipeline family was promoted into a flagship treatment; the other families stayed cleaner and quieter.
- No extra metric strip, KPI wall, or dashboard layer was added to replace the removed signal cards.
- No 3D figure was introduced; the page stays in 2D benchmark and tradeoff space.
- The new page styling is scoped to the performance route so other pages do not become louder as a side effect.

## Anti-regression check

- Did any wrapper become louder than the proof?
  - No. The new shell around the flagship figure is restrained and only exists to clarify the runtime-priority reading, not to outperform the figure itself.
- Did a second competing hero appear?
  - No. The page has two emphasized proof moments, but they serve different jobs: the benchmark overview remains the broad proof object, and the macro stage-budget figure is the one editorial flagship.
- Did the page become more premium without becoming more informative?
  - No. The upgraded figure treatment is tied directly to stronger interpretation of budget dominance, optimization order, and evidence scope.
- Did quiet pages get louder as a side effect?
  - No. The CSS changes are route-specific to the performance page.

## Risks / what still feels off

- The benchmark overview and flagship figure now work together well, but they still rely on the reader understanding that the page is a bounded publication bundle rather than an exhaustive benchmark taxonomy.
- The performance page is generated from committed artifacts, which is correct for stability, but future benchmark-refresh passes will still require narrative review so the interpretation stays aligned with the published numbers.
- The worktree already contains unrelated modified and untracked files outside this pass; they were left untouched.

## Validation

- Rebuilt the generated performance page:
  - `& 'C:\Users\ouwez\AppData\Local\Programs\Python\Python312\python.exe' scripts\render_performance_page.py`
- Rebuilt the performance figures from committed snapshot artifacts:
  - `& 'C:\Users\ouwez\AppData\Local\Programs\Python\Python312\python.exe' -c "from pathlib import Path; from scripts.build_benchmark_artifacts import build_macro_pipeline_summary_asset, build_benchmark_overview_asset; root = Path(r'C:/Users/ouwez/Documents/Quant/option-pricing-library-agent-docs'); artifacts = root / 'benchmarks' / 'artifacts'; plot_dir = root / 'docs' / 'assets' / 'generated' / 'benchmarks'; build_macro_pipeline_summary_asset(artifacts_dir=artifacts, plot_dir=plot_dir); build_benchmark_overview_asset(artifacts_dir=artifacts, plot_dir=plot_dir)"`
- Verified the changed Python sources compile:
  - `& 'C:\Users\ouwez\AppData\Local\Programs\Python\Python312\python.exe' -m py_compile scripts\build_benchmark_artifacts.py scripts\build_visual_artifacts.py scripts\render_performance_page.py`
- Rebuilt the docs site:
  - `& 'C:\Users\ouwez\AppData\Local\Programs\Python\Python312\python.exe' -m mkdocs build --strict`
- Verified the visual publishing pipeline:
  - `py -3.12 -m pytest tests/test_visual_publishing_pipeline.py -q`
- Ran focused DOM, accessibility, page-snapshot, and component-snapshot checks for `/performance/` across light/dark and `375`, `768`, `1280`, `1536`:
  - `$env:PYTHON_EXECUTABLE='C:\Users\ouwez\AppData\Local\Programs\Python\Python312\python.exe'; $env:SERVE_PREBUILT_SITE='1'; $env:REVIEW_PATHS='/performance/'; .\node_modules\.bin\playwright.cmd test dom-audits.spec.ts a11y.spec.ts pages.spec.ts components.spec.ts --update-snapshots`
- Ran targeted pre-commit checks on the touched performance files:
  - `& 'C:\Users\ouwez\Documents\Quant\option-pricing-library\.venv\Scripts\python.exe' -m pre_commit run --files scripts\render_performance_page.py scripts\build_benchmark_artifacts.py scripts\build_visual_artifacts.py scripts\templates\performance.md.template docs\stylesheets\extra.css docs\performance.md tests\visual\targets.ts tests\test_visual_publishing_pipeline.py`
- Captured before screenshots for the report:
  - `$env:PYTHON_EXECUTABLE='C:\Users\ouwez\AppData\Local\Programs\Python\Python312\python.exe'; $env:SERVE_PREBUILT_SITE='1'; $env:REVIEW_PATHS='/performance/'; $env:IMPROVEMENT_CAPTURE_DIR='C:\Users\ouwez\Documents\Quant\option-pricing-library-agent-docs\tests\visual\artifacts\phase-6-performance-evidence\before'; .\node_modules\.bin\playwright.cmd test review-capture.spec.ts`
- Captured after screenshots for the report:
  - `$env:PYTHON_EXECUTABLE='C:\Users\ouwez\AppData\Local\Programs\Python\Python312\python.exe'; $env:SERVE_PREBUILT_SITE='1'; $env:REVIEW_PATHS='/performance/'; $env:IMPROVEMENT_CAPTURE_DIR='C:\Users\ouwez\Documents\Quant\option-pricing-library-agent-docs\tests\visual\artifacts\phase-6-performance-evidence\after'; .\node_modules\.bin\playwright.cmd test review-capture.spec.ts`

## Approval checkpoint

Do not continue to the next work package until this pass is reviewed.
