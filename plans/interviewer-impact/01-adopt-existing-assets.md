# Plan 1 — Adopt and Track Existing Assets from `assets.zip`

## Goal

Use the already-generated assets in `assets.zip` to improve the docs/site immediately, instead of regenerating or redesigning replacements.

## Why this matters

The asset bundle already contains useful, interviewer-relevant figures that strengthen the surface, eSSVI, and local-vol/PDE story. Some are currently missing from the tracked docs assets, and at least some existing pages reference files that are not committed. That creates avoidable credibility loss.

## Repo-grounded findings

The following assets exist in `assets.zip` and should be promoted into tracked docs assets:

### Static surface assets

- `generated/static/quote_surface_compare.png`
- `generated/static/svi_repaired_surface_heatmap.png`
- `generated/static/svi_smile_slices.png`

### Dupire / eSSVI assets

- `generated/dupire/essvi_smoothed_surface_heatmap.png`
- `generated/dupire/localvol_gatheral_heatmap.png`
- `generated/dupire/gatheral_vs_dupire_diff_heatmap.png`

### Numerics / PDE assets

- `generated/numerics/pde_roundtrip_scatter.png`
- `generated/numerics/pde_convergence.png`
- `generated/numerics/pde_price_error_heatmap.png`

## Known broken-or-missing reference to fix

At minimum, the following page currently references numerics assets that should be promoted into the repo:

- `docs/user_guides/flagship_localvol_pde.md`

Referenced files:

- `assets/generated/numerics/pde_roundtrip_scatter.png`
- `assets/generated/numerics/pde_convergence.png`

## Requirements

### 1. Promote the existing generated assets into the repo

Copy the missing assets from the unpacked bundle into tracked locations under:

- `docs/assets/generated/static/`
- `docs/assets/generated/dupire/`
- `docs/assets/generated/numerics/`

Prefer keeping the existing filenames unless a naming conflict or consistency problem makes a rename necessary.

### 2. Fix all missing references

Ensure all pages that already reference these generated files render correctly after the asset promotion step.

This includes, at minimum:

- `docs/user_guides/flagship_localvol_pde.md`

### 3. Strengthen the relevant proof pages with the promoted assets

Use the promoted assets to improve the pages that already tell the strongest engineering story.

#### Surface page

Add or upgrade figure usage using:

- `generated/static/quote_surface_compare.png`
- `generated/static/svi_repaired_surface_heatmap.png`
- `generated/static/svi_smile_slices.png`

#### eSSVI bridge page

Add or upgrade figure usage using:

- `generated/dupire/essvi_smoothed_surface_heatmap.png`
- `generated/dupire/localvol_gatheral_heatmap.png`
- `generated/dupire/gatheral_vs_dupire_diff_heatmap.png`

#### Local vol + PDE page

Add or upgrade figure usage using:

- `generated/numerics/pde_roundtrip_scatter.png`
- `generated/numerics/pde_convergence.png`
- `generated/numerics/pde_price_error_heatmap.png`

### 4. Standardize captions and alt text

For every inserted figure:

- the caption must explain the **engineering takeaway**, not merely restate the filename
- alt text must be specific and descriptive
- figure layout should favor **one strong primary figure and one supporting figure** rather than cluttered galleries

### 5. Reuse existing tracked diagrams more deliberately

The repo already contains strong diagram assets that should be surfaced more prominently where useful:

- `diagrams/architecture_layers.*`
- `diagrams/validation_stack.*`
- `diagrams/workflow_surface_to_pde.*`

Prefer promoting these rather than drawing replacements.

## Suggested files to edit

- `docs/user_guides/flagship_surface.md`
- `docs/user_guides/flagship_essvi_bridge.md`
- `docs/user_guides/flagship_localvol_pde.md`
- `docs/architecture.md`
- optionally `docs/index.md`
- optionally `README.md`

## Non-goals

- Do not generate new benchmark figures here.
- Do not redesign visual identity.
- Do not create decorative hero art.
- Do not duplicate assets that already exist in usable form.

## Acceptance criteria

- All existing broken/missing generated-asset references are fixed.
- All 9 missing generated assets from `assets.zip` are tracked in the repo.
- The surface, eSSVI, and local-vol/PDE pages each gain stronger visuals using those assets.
- Captions explain reviewer-relevant takeaways.
- No unnecessary asset duplication is introduced.
