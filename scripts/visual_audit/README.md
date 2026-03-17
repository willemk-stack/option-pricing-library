# Visual audit scripts

These scripts provide repo-specific helpers for docs visual QA.

## Scripts

- `collect_docs_routes.py`:
  collect candidate docs routes for browser-based audit
- `check_svg_assets.py`:
  inspect generated SVG assets for suspicious linked media or missing references
- `review_targets.json`:
  curated pages/assets to prioritize during visual review

## Example usage

```bash
python scripts/visual_audit/collect_docs_routes.py
python scripts/visual_audit/check_svg_assets.py
```
