# Refresh README proof card with Heston evidence

## Summary

Refresh the generated README proof card so it reflects the reviewer-facing
Heston/model-comparison evidence added in the docs restructure.

## Motivation

The current README proof card is still weighted toward the surface and
local-vol/PDE proof path. The branch now has a clearer Capstone 3 reviewer path,
but the generated README card does not yet make that evidence visible.

## Proposed scope

- Update the proof-card source or generator inputs rather than editing SVGs by hand.
- Decide whether Heston evidence belongs in the existing card, a companion card,
  or a revised caption block.
- Regenerate the light and dark README proof-card assets.
- Re-check the README layout and card legibility in both themes.

## Non-goals

- No manual edits to generated SVG assets.
- No new benchmark numbers unless they come from committed artifacts.
- No broader homepage redesign.

## Acceptance criteria

- The generated README proof card accurately mentions reviewer-facing Heston
  evidence.
- README rendering stays synchronized with the card source.
- Light and dark variants remain visually legible.

## Validation plan

- Run `python scripts/build_visual_artifacts.py all --profile publish`.
- Run `python scripts/render_readme.py`.
- Run `mkdocs build --strict`.