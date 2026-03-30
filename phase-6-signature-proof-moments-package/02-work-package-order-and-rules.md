# Work Package Order and Rules

## Execution order

Run these in order:

1. homepage hero
2. surface repair workflow
3. eSSVI smooth handoff
4. architecture
5. local-vol / PDE validation
6. performance evidence
7. final cross-page proof-routing pass, only if still needed

This order is intentional:
- it fixes the highest hiring-manager-impact page first
- it strengthens the two strongest proof pages next
- it upgrades architecture before lower-priority refinement
- it delays final consistency work until the main proof objects are in place

## Review cadence

Stop after every work package.
Do not continue until the output has been reviewed.

## General rules

1. Prefer source changes over editing generated output directly.
2. Preserve the distinction between flagship pages and quiet pages.
3. Use the smallest number of new component variants possible.
4. Keep dark mode, mobile hierarchy, and maintainability in view.
5. If exact source paths are unclear, find them first and report what was chosen.

## Anti-regression rules

Before finishing any work package, explicitly check:

- Is there more than one dominant hero competing for attention?
- Did a wrapper become louder than the evidence it contains?
- Did route cards or callouts multiply beyond what the page needs?
- Did pills/badges start serving buzzwords instead of navigation or metadata?
- Did repeated stacked cards create a dashboard feel?
- Did the page become more premium without becoming more informative?
- Did a supporting figure become as loud as the main proof object?
- Did architecture regain framing noise?
- Did API or Quickstart get louder because of shared CSS or component spillover?
- Did any new component become fragile or hard to stabilize?
- Did wording become more promotional than technical?

If any answer is yes, scale the page back.

## Page-specific restraint rules

### Homepage
- one dominant proof object only
- no dashboard wall
- no extra route-card sprawl

### Surface repair
- do not replace slice-level evidence with a glamour image alone
- do not bury the proof under preamble chrome

### eSSVI
- do not drift into generic math explainer mode
- keep diagnostics visible

### Architecture
- let the figure carry authority
- do not solve weakness with more surrounding boxes

### Local-vol / PDE
- keep validation logic primary
- do not let a pretty local-vol surface become the whole page

### Performance
- improve authorship and interpretation
- do not force 3D or a pseudo-product dashboard feel

## 3D rules

A 3D asset is justified only if it:
- clarifies geometry or continuity
- meaningfully improves first-impression seriousness
- remains readable on the page
- does not displace the diagnostics that actually prove the case

## Reporting requirement

Use `templates/stage-report-template.md`.
For each pass include:
- files changed
- visual changes
- wording changes
- screenshots
- what was intentionally kept restrained
- risks / what still feels off
- validation checks
