# Add Heston performance evidence

## Summary

Keep the committed, reviewer-usable Heston performance and diagnostics evidence
discoverable from the public docs.

## Motivation

The branch now exposes Heston docs artifacts, the model-comparison CSV bundle,
and calibration benchmark artifacts. The remaining tracking value is to keep
fixture scope, environment dependence, smoke-vs-release intent, and artifact
links aligned as the bundle is regenerated.

## Proposed scope

- Keep `docs/performance.md`, `docs/user_guides/heston_model_comparison.md`,
  and `docs/validation_matrix.md` linked to the Heston artifact manifest,
  builder scripts, generated CSVs, and nearest tests.
- Preserve the distinction between smoke diagnostics and stronger release/review
  artifacts.
- Keep issue #93's matched direct-PDE subset CSV visible when bucket summaries
  are discussed.

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
