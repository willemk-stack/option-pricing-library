# Add Heston performance evidence

## Summary

Add committed, reviewer-usable performance evidence for the Heston pricing,
calibration, and comparison workflows.

## Motivation

The branch now explains what the Heston workflow proves, but it still lacks a
performance evidence page that is comparable in discipline to the existing
surface and PDE benchmark story.

## Proposed scope

- Choose the smallest useful benchmark set for Heston Fourier pricing,
  calibration diagnostics, Monte Carlo sweeps, and model-comparison overhead.
- Define fixture scope, environment notes, and artifact schema for committed
  benchmark outputs.
- Publish the resulting evidence through docs pages or proof tables without
  overstating generality.

## Non-goals

- No ad hoc timing claims copied into markdown without committed artifacts.
- No full benchmark-suite expansion unless a smaller scoped subset is
  insufficient.
- No change to pricing or calibration behavior in the same issue.

## Acceptance criteria

- Heston performance evidence is backed by committed benchmark artifacts.
- The docs explain fixture scope and environment dependence.
- Reviewer-facing pages can link to the new evidence without hand-wavy numbers.

## Validation plan

- Run the selected Heston benchmark builders.
- Regenerate any benchmark manifests or docs artifacts they feed.
- Run `pytest -q tests/diagnostics/heston/test_heston_calibration_benchmark.py`.
- Run `mkdocs build --strict`.