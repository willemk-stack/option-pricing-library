# Phase 6 — Signature Proof Moments Package

This package is meant to be dropped into the repo root or a worktree root and used with a coding agent such as Codex CLI.

It is intentionally shaped like the earlier docs-overhaul package so you can run it the same way.

## What this phase is for

Phase 6 is **not** a broad redesign.

It is a targeted emphasis rebalance:
- keep the calm baseline
- keep quiet pages quiet
- increase ambition only on the flagship proof moments that should convince hiring managers

The main goal is to move the site from **cleaned up and credible** to **memorable and technically authoritative**, without regressing into clutter.

## Recommended usage

Suggested order:

1. Read `01-phase-6-master-brief.md`
2. Read `02-work-package-order-and-rules.md`
3. Optionally read `03-paste-ready-master-prompt.md`
4. Run one prompt from `prompts/` at a time
5. Review the stage report after every work package before continuing

## Package contents

- `01-phase-6-master-brief.md` — main context and success criteria
- `02-work-package-order-and-rules.md` — execution sequence, guardrails, anti-regression rules
- `03-paste-ready-master-prompt.md` — single prompt version if you want one top-level instruction block
- `prompts/phase-6-1-homepage-hero.md`
- `prompts/phase-6-2-surface-repair.md`
- `prompts/phase-6-3-essvi-handoff.md`
- `prompts/phase-6-4-architecture.md`
- `prompts/phase-6-5-localvol-pde.md`
- `prompts/phase-6-6-performance.md`
- `prompts/phase-6-7-final-cross-page-pass.md`
- `templates/stage-report-template.md`
- `templates/dont-regress-into-clutter-checklist.md`
- `templates/likely-file-targets.md`

## Operating model

- One work package per run
- Review after every work package
- Prefer source changes over generated output changes
- Rewrite intros and section leads freely
- Rewrite framing text carefully
- Do not rewrite technical explanations beyond readability cleanup unless you clearly report it
- Do not modify Notes markdown content

## Why this structure

A small multi-file package is better than one giant prompt because it:
- keeps the direction stable across runs
- reduces prompt drift
- makes review checkpoints easy
- lets you rerun a single pass without re-explaining the project
- matches the staged package already used earlier
