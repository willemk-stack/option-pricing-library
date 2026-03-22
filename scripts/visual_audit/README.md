# Visual audit scripts

These scripts provide repo-specific helpers for docs visual QA.

`review_targets.json` is the single source of truth for the Playwright audit surface.
It now carries both the broad audit matrix (`pages`, `themes`, `widths`) and the
smaller blocking full-page baseline subset (`page_snapshot_pages`,
`page_snapshot_widths`).

## Scripts

- `collect_docs_routes.py`:
  collect candidate docs routes for browser-based audit
- `check_svg_assets.py`:
  inspect generated SVG and PNG assets for missing pairs, missing links, or suspiciously tiny outputs
- `review_targets.json`:
  curated pages/assets to prioritize during visual review

## Example usage

```bash
python scripts/visual_audit/collect_docs_routes.py
python scripts/visual_audit/check_svg_assets.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_visual_audit.ps1
```
