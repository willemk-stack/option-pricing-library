# Interviewer Impact Plan Pack

This plan pack is split into separate markdown files so an AI agent can execute them cleanly and in order.

## Recommended execution order

1. `01-adopt-existing-assets.md`
2. `03-benchmarks-plan.md`
3. `02-new-asset-creation-plan.md`
4. `04-general-docs-readme-pages-overhaul.md`

## Why this split

- **Existing assets first**: there are already valuable generated visuals in `assets.zip` that should be promoted into tracked docs assets before anyone creates new ones.
- **Benchmarks second**: benchmark evidence should be real and committed before the docs rewrite leans on it.
- **Net-new assets third**: only create new visuals after existing assets and benchmark outputs have been fully exploited.
- **Main docs/readme overhaul last**: the public-facing rewrite should assume the asset and benchmark work is already done.

## Shared guardrails for all plans

Apply these rules across all plan files:

- Do **not invent** metrics, benchmark outcomes, or validation claims.
- Prefer **reusing existing generated assets** over generating replacements.
- Remove placeholders; do not leave "to add later" sections in user-facing docs.
- Favor **capability + proof** over meta framing.
- Reduce repeated use of words like **"flagship"** and remove public-facing **"capstone2"** naming.
- Preserve external links where practical. If renaming pages, add compatibility stubs or redirects where supported.
- Every strong public claim should map to a visible artifact, table, benchmark, test, or notebook.
- Optimize for a reviewer understanding the project’s strongest proof in **~20 seconds**.
- Keep all strong claims reviewer-defensible.

## File list

- `01-adopt-existing-assets.md`
- `02-new-asset-creation-plan.md`
- `03-benchmarks-plan.md`
- `04-general-docs-readme-pages-overhaul.md`

## Notes for agent execution

- Each plan is self-contained, but they are meant to compose.
- The general docs/readme/pages overhaul assumes the other plans are fully implemented.
- The new-asset plan is intentionally narrow so the agent does not create unnecessary visuals.
