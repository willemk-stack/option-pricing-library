# Visual state report

## Summary

- **pages**: `7`
- **themes**: `2`
- **widths**: `4`
- **expected_snapshots**: `56`
- **actual_snapshots**: `56`
- **missing_snapshots**: `0`
- **stale_snapshots**: `0`
- **site_url**: `https://willemk-stack.github.io/option-pricing-library/`
- **playwright_base_url**: `http://127.0.0.1:8000/option-pricing-library/`
- **serve_docs_base_url**: `http://127.0.0.1:8000/option-pricing-library/`
- **run_visual_audit_base_url**: `http://127.0.0.1:8000/option-pricing-library/`
- **visual_ci_present**: `False`

## High findings

- None

## Medium findings

- **assets**: SVG contains linked raster/image subpanels that need browser validation: ['../static/quote_surface_compare.png', '../dupire/essvi_smoothed_surface_heatmap.png', '../poster/poster_essvi_localvol_pde.png'] (`C:\Users\ouwez\Documents\Quant\option-pricing-library\docs\assets\generated\showcase\reviewer_proof_panel.light.svg`)
- **assets**: SVG contains linked raster/image subpanels that need browser validation: ['../static/quote_surface_compare.dark.png', '../dupire/essvi_smoothed_surface_heatmap.dark.png', '../poster/poster_essvi_localvol_pde.dark.png'] (`C:\Users\ouwez\Documents\Quant\option-pricing-library\docs\assets\generated\showcase\reviewer_proof_panel.dark.svg`)
- **assets**: SVG contains linked raster/image subpanels that need browser validation: ['./iv_scaling.png', './pde_runtime_error_tradeoff.png', './macro_pipeline_summary.png'] (`C:\Users\ouwez\Documents\Quant\option-pricing-library\docs\assets\generated\benchmarks\benchmark_overview.light.svg`)
- **assets**: SVG contains linked raster/image subpanels that need browser validation: ['./iv_scaling.dark.png', './pde_runtime_error_tradeoff.dark.png', './macro_pipeline_summary.dark.png'] (`C:\Users\ouwez\Documents\Quant\option-pricing-library\docs\assets\generated\benchmarks\benchmark_overview.dark.svg`)
- **ci**: No obvious CI workflow currently runs Playwright or the visual audit scripts

## Info findings

- **targets**: pages.spec.ts and a11y.spec.ts both read targets from tests/visual/targets.ts
- **snapshots**: Snapshot matrix is complete: 56 files (`C:\Users\ouwez\Documents\Quant\option-pricing-library\tests\visual\pages.spec.ts-snapshots`)
- **snapshots**: No stale snapshot filenames detected (`C:\Users\ouwez\Documents\Quant\option-pricing-library\tests\visual\pages.spec.ts-snapshots`)
- **docs**: Found 4 references to build_visual_artifacts.py
