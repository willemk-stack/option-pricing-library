# Plan 2 — Net-New Asset Creation Plan

## Goal

Create a very small number of high-leverage, interviewer-facing assets that improve skim impact **after** all usable existing assets from `assets.zip` have been adopted.

## Preconditions

This plan assumes Plan 1 is fully complete:

- existing generated assets from `assets.zip` have been promoted into tracked docs assets
- broken asset references have been fixed
- the proof pages are already using the strongest existing visuals

This plan should also assume Plan 3 will produce real benchmark artifacts that can be reused.

## Principle

Do not create a large visual family. Create only a minimal number of assets whose value is clearly higher than the cost.

## Recommendation

Create **only two** new assets.

## Asset A — Executive proof panel

### Purpose

A recruiter, hiring manager, or interviewer should be able to glance at one visual and understand:

- this is a real quant engineering project
- it has validation, not just implementation
- it connects surfaces, smoothing, local vol, and PDE repricing

### Proposed output

- `docs/assets/generated/showcase/reviewer_proof_panel.(png|svg)`

### Content requirements

The asset should combine:

- a small workflow strip: static surface → eSSVI smoothing → local vol → PDE repricing
- 3–4 proven metrics only, such as:
  - seam jump reduction
  - zero invalid Dupire projection count
  - repriced option count
  - mean/max error metric
- one supporting visual panel composed from existing tracked figures, not newly simulated content
- very short labels rather than paragraph text

### Design constraints

- no decorative fluff
- legible at README width
- derived only from already-supported evidence
- optimized for fast skim, not exhaustive explanation

### Recommended usage

Use this asset near the top of:

- `README.md`
- optionally `docs/index.md`

## Asset B — Benchmark overview figure

### Purpose

After Plan 3 is complete, turn benchmark results into a reviewer-friendly summary artifact instead of forcing the reader through raw tables and multiple plots.

### Proposed output

- `docs/assets/generated/benchmarks/benchmark_overview.(png|svg)`

### Content requirements

Use measured benchmark outputs only. Likely components:

- one scaling panel
- one throughput or latency panel
- one accuracy-vs-cost panel or concise result callout

### Design constraints

- must be driven only by real benchmark outputs
- must summarize, not replace, the detailed benchmark page
- must be understandable without reading surrounding text

### Recommended usage

Use this asset in:

- `docs/performance.md`
- optionally `README.md`
- optionally `docs/index.md`

## Non-goals

Do **not** create:

- extra poster variants
- decorative hero art
- a large family of social-card graphics
- cosmetic plot restyles with no information gain
- replacement versions of assets that already exist and are adequate

## Acceptance criteria

- No more than 2 new assets are created.
- Each new asset has a clear interviewer-facing purpose.
- Each new asset is referenced in README/docs and improves first-scan comprehension.
- Neither asset duplicates something already available in `assets.zip`.
