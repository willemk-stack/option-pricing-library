# Complete Heston citation audit

## Summary

Finish the citation and evidence audit across the Heston notes, replacing
temporary TODO markers with citations, benchmark links, or softer wording.

## Motivation

The current cleanup pass adds targeted `TODO(evidence)` markers near strong
repository-policy claims. Those markers need a follow-up pass by someone who can
verify the supporting papers, tests, and benchmark artifacts carefully.

## Proposed scope

- Review the Heston notes one by one for strong language about defaults,
  stability, weak identification, and benchmark roles.
- Replace each TODO marker with either a citation, a link to committed evidence,
  or toned-down wording.
- Keep mathematical references separate from repository-policy claims.

## Non-goals

- No rewrite of the Heston implementation itself.
- No fabricated citations or benchmark claims.
- No automatic issue creation or PR generation.

## Acceptance criteria

- Each Heston note with a TODO marker is either cited, linked to evidence, or
  softened.
- The notes remain readable and reviewer-facing.
- Repository-policy language is clearly separated from literature-backed claims.

## Validation plan

- Re-run `mkdocs build --strict`.
- Re-run any nearest targeted Heston docs or diagnostics tests when wording now
  points to new evidence locations.
- Manually review the affected notes for citation consistency.