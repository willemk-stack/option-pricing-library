# Docs Quality Principles

This directory defines what high-quality docs should look like in this repo beyond simple build correctness.

## Objectives

- Preserve the current documentation information architecture while improving scanability, hierarchy, readability, and presentation quality.
- Treat generated visual artifacts as first-class docs assets, not decorative attachments.
- Keep light and dark themes equivalent in clarity, contrast, and perceived completeness.
- Prefer bounded improvements that measurably improve a page over broad restyling.

## Quality dimensions

- Structural clarity: headings, section rhythm, and CTA placement make the next action obvious.
- Visual stability: no overflow, clipping, overlap, horizontal scroll, or broken responsive collapse.
- Evidence quality: featured figures, proof panels, and charts look intentional and non-empty at supported widths.
- Theme parity: the same page communicates equally well in light and dark mode.
- Accessibility: primary content remains legible, navigable, and audit-clean in the required scope.

## Supported breakpoints

- 375
- 768
- 1280
- 1536

## Required themes

- light
- dark

## Allowed edits

- Markdown structure and section ordering when the page meaning stays intact.
- MkDocs overrides and template structure.
- CSS, theme tokens, spacing, and layout rules.
- Generated visual source code, asset references, and artifact export configuration.

## Forbidden edits

- Silent changes to factual claims, benchmark conclusions, API behavior, or mathematical statements.
- Hand-editing generated assets when the source or generator can be fixed instead.
- Broad aesthetic rewrites that are not justified by the page spec or the failing audits.

## Severity model

- `critical`: broken images, missing required content, clipped primary heading, overlapping interactive elements, broken required artifact blocks.
- `major`: text overflow, hidden content, responsive collapse, empty visual panels, major contrast failures, missing theme variants.
- `minor`: uneven spacing, odd wrapping, weak grouping, inconsistent card heights, low-signal polish issues.