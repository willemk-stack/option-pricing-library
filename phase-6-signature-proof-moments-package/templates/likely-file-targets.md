# Likely File Targets

These are likely source files or ownership areas based on the prior package, stage reports, and the built-site structure.

Verify exact paths in the repo before editing.

## Homepage
- `docs/index.md`
- homepage hero/proof asset source(s)
- minimal shared CSS only if needed

## Surface repair workflow
- `docs/user_guides/surface_workflow.md`
- source for repair/workflow figures

## eSSVI smooth handoff
- `docs/user_guides/essvi_smooth_handoff.md`
- source for smoothed surface / handoff visuals

## Architecture
- `docs/architecture.md`
- diagram sources under something like:
  - `docs/assets/diagrams/src/architecture_layers.d2`
  - `docs/assets/diagrams/src/workflow_surface_to_pde.d2`
  - `docs/assets/diagrams/src/validation_stack.d2`

## Local-vol / PDE validation
- `docs/user_guides/localvol_pde_validation.md`
- source for validation figures

## Performance evidence
- `scripts/templates/performance.md.template`
- `scripts/render_performance_page.py`
- regenerated page output such as `docs/performance.md`
- benchmark figure-generation sources if they exist in-repo

## Shared styling or utilities if needed
- `docs/stylesheets/extra.css`
- small reusable classes/components only if justified and stable

## Principle

Prefer editing source markdown, figure source, and generation code rather than touching generated site artifacts directly.
