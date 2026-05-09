# Complete Heston citation audit

## Summary

Finish the citation and evidence audit across the Heston notes, replacing
temporary TODO markers with citations, benchmark links, or softer wording.

## Motivation

The P0 cleanup pass replaced the generic `TODO(evidence)` markers with
repository-policy language, evidence links, or softer wording. This follow-up
is now for owner review of any remaining policy calls that should be upgraded
to stronger citations, benchmark artifacts, or issue-specific evidence.

## Proposed scope

- Review the Heston notes one by one for strong language about defaults,
  stability, weak identification, and benchmark roles.
- Replace any remaining policy-only wording with a citation, a link to
  committed evidence, or toned-down wording when owner review finds the current
  label too strong.
- Keep mathematical references separate from repository-policy claims.

## Non-goals

- No rewrite of the Heston implementation itself.
- No fabricated citations or benchmark claims.
- No automatic issue creation or PR generation.

## Acceptance criteria

- Each Heston note labels theory, repository policy, diagnostics evidence, and
  provisional heuristics clearly enough for public review.
- The notes remain readable and reviewer-facing.
- Repository-policy language is clearly separated from literature-backed claims.

## Validation plan

- Re-run `mkdocs build --strict`.
- Re-run any nearest targeted Heston docs or diagnostics tests when wording now
  points to new evidence locations.
- Manually review the affected notes for citation consistency.
