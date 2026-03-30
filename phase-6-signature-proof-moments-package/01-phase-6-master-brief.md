# Phase 6 Master Brief — Signature Proof Moments

## Mission

Take the documentation from **cleaned up and credible** to **memorable and hiring-manager-convincing** without regressing into clutter.

This phase should preserve the calm baseline and quiet-page discipline from the earlier overhaul while spending more of the emphasis budget on the strongest proof moments.

## Core interpretation of the project goals

The site is primarily for hiring managers.
A new visitor should first get a polished welcome, then a strong proof moment, then clear routing into the pages that best demonstrate mathematical depth, robustness, and engineering judgment.

The redesign should prove:
- the author is truly a quant, not just someone who can code
- the project reflects deep awareness of assumptions, failure modes, safeguards, and diagnostics
- the site is premium and memorable where it matters, but not generically flashy

## Success criteria for this phase

After Phase 6, a hiring manager should get:

1. a polished first impression
2. one unmistakable signature visual early
3. stronger proof that the author understands tradeoffs, assumptions, and safeguards
4. more memorable flagship pages
5. no sense that the site has turned into a noisy dashboard or decorative portfolio

## The main design rule

If a treatment is flashy, it must communicate something real.

No decorative hype.
No premium framing without substantive evidence underneath.

## What this phase should improve

### 1. Homepage
The homepage should have one unmistakable signature proof object.
It should feel less like a strong summary panel and more like a memorable first impression.

### 2. Surface repair workflow
This should become the strongest production-minded proof page.
It should better communicate engineering judgment, reviewability, and what is preserved versus regularized.

### 3. eSSVI smooth handoff
This should become the strongest quant-proof page.
It should say more, clearly and compactly, about assumptions, parameterization, continuity needs, and why naive handoff fails.

### 4. Local-vol / PDE validation
This should feel more authored as a validation argument and less like a cleaned-up notebook.
The page should make its numerical judgment and tradeoffs easier to see.

### 5. Architecture
The page should move one level beyond “cleaned up” and feel more like a premium portfolio systems graphic.
The visual should carry more authority and make checkpoints/safeguards easier to see.

### 6. Performance evidence
Improve authorship and interpretation, not sheer visual intensity.
This page should feel like a benchmark case study, not a report dump.

## 3D usage policy

3D is appropriate only where it materially improves surface/proof understanding.

Use 3D for:
- homepage hero
- surface repair hero
- eSSVI hero
- maybe one supporting local-vol context figure

Do not use 3D as the primary device for:
- performance
- architecture
- API or utility pages

A 3D figure only passes if:
- the viewing angle reveals real structure
- labels remain readable
- it supports, rather than replaces, diagnostics
- it increases first-impression seriousness rather than just spectacle

## Scope

Primary pages in scope:
- homepage
- surface repair workflow
- eSSVI smooth handoff
- architecture
- local-vol / PDE validation
- performance evidence

Secondary scope:
- a final cross-page pass for proof routing and consistency, only if needed

## Non-negotiables

1. Do not make the site louder overall.
2. Keep API and Quickstart restrained.
3. Do not silently rewrite technical content.
4. Do not touch Notes markdown content.
5. Keep the emphasis budget disciplined.
6. Reuse existing system patterns where that helps stability, but new components/classes are allowed if they are robust and maintainable.

## Content boundaries

Allowed:
- rewrite intros
- rewrite section leads
- strengthen why-this-matters framing
- strengthen tradeoff/decision/safeguard framing
- improve transitions between problem, method, evidence, and tradeoffs

Not allowed without explicit callout:
- silent technical-content rewrites
- broad mathematical exposition drift
- Notes content edits

## Deliverables expected after every work package

For every sub-pass, require:
- changed files list
- short rationale for each change
- before/after screenshots
- notes on risks or what still feels off
- clear callout of any wording changes beyond readability cleanup

## Final acceptance gate

Do not call Phase 6 complete unless all of the following are true:

- homepage has one unmistakable signature visual
- surface repair is the clearest production-minded proof page
- eSSVI is the clearest quant-proof page
- architecture feels premium and intentional, not merely calmer
- local-vol / PDE reads as a real validation argument
- performance reads as an authored case study
- API and Quickstart did not get louder as a side effect
- the site still feels restrained by default, with stronger emphasis only where earned
