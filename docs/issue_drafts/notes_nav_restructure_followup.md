# Notes navigation restructure follow-up

## Summary

Review whether the notes section needs a broader navigation restructure beyond
the new "Notes by purpose" routing block.

## Motivation

The current cleanup improves the landing page, but the notes corpus is large
enough that future growth may warrant stronger routing than a single index page
section.

## Proposed scope

- Audit how readers enter the notes from the decision guide, user guides, and
  search.
- Evaluate whether the notes should expose separate theory, implementation,
  validation, and provisional-policy landing pages.
- Decide whether any MkDocs nav changes are justified after that audit.

## Non-goals

- No unnecessary nav churn during the current docs cleanup pass.
- No renaming of note files just to change ordering.
- No change to generated artifacts unless routing decisions require it.

## Acceptance criteria

- There is a clear recommendation for whether the current index is sufficient or
  whether deeper nav changes are justified.
- Any proposed nav changes preserve the existing reading-order story.
- Follow-up work distinguishes routing improvements from content rewrites.

## Validation plan

- Review the rendered notes index and nearby entry pages.
- If nav changes are proposed, run `mkdocs build --strict`.
- Spot-check representative note routes in the rendered docs.