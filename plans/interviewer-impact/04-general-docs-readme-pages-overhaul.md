# Plan 4 — General README / Docs / Pages Overhaul

## Goal

Maximize interviewer impact by making the repo’s strongest evidence visible faster, with clearer information architecture and less meta-language.

## Assumption

This plan assumes Plans 1–3 are fully implemented:

- existing useful assets from `assets.zip` are tracked and integrated
- benchmark artifacts and benchmark documentation are real and committed
- any net-new visuals were created only where they add clear value

## Core problem to solve

The project has strong substance, but the current presentation still makes a reviewer learn the repo’s framing and taxonomy before they learn its strongest achievement.

This overhaul should make the README/docs/site feel:

- more professional
- less “capstone/portfolio taxonomy”
- more outcome- and evidence-driven

## Workstream A — Messaging rewrite

### Objective

Rewrite top-level copy so the first impression is:

> this is a typed, validated quant engineering library with credible surface / eSSVI / local-vol / PDE workflows

and not:

> this is a documentation system explaining its own structure

### Requirements

Across the README and docs landing pages:

- lead with capability + proof
- reduce phrases such as:
  - “the repo now presents”
  - “the goal of this front door”
  - “this page is the repo’s main X story”
  - “positioning”
  - “what it does not try to prove”
- keep “flagship” only where it materially helps navigation

### Preferred content pattern

For each top-level page:

1. **What this is**
2. **Why it is hard**
3. **What evidence proves it**
4. **What to click next**

## Workstream B — Rename and simplify information architecture

### Objective

Remove low-signal naming and reduce “capstone/flagship” fatigue.

### Requirements

#### 1. Replace `flagship_capstone2_page` with a more professional decision-guide slug

Create a new page such as:

- `docs/user_guides/decision_guide.md`

Then either:

- keep `flagship_capstone2_page.md` as a short compatibility stub, or
- add an equivalent redirect mechanism if the docs system supports it

#### 2. Rename visible nav labels

Recommended direction:

- `Flagship demos` → `Proof path` or `Showcase`
- `Decision guide` can remain if implemented cleanly
- `Surface flagship` → `Surface workflow`
- `Local vol + PDE flagship` → `Local vol + PDE validation`

Exact wording can vary, but make it less branded and more descriptive.

#### 3. Ensure nav, H1, and button consistency

Fix title mismatches where page purpose, nav label, H1, and CTA wording currently disagree.

### Acceptance criteria

- no visible `capstone2` naming remains in public-facing UX
- nav labels and H1s are consistent
- page titles describe user value, not internal taxonomy

## Workstream C — README overhaul

### Objective

Make the README strong enough that many interviewers never need to leave it to recognize the project’s quality.

### Requirements

#### 1. Strengthen the opening block

The first screen should clearly state:

- what the library does
- what is unusually strong about it
- where the evidence is

#### 2. Add a top-level proof block

Pull the strongest proof metrics up into the README near the top.

Use already-supported evidence such as:

- seam jump reduction
- zero invalid Dupire projection count
- repriced option count
- mean/max error figure

Do not bury these below navigation copy.

#### 3. Replace “Best places to start” with a more interviewer-oriented section

Examples:

- “Start here for the strongest engineering proof”
- “Start here for the public API”
- “Start here for architecture”

#### 4. Surface architecture earlier

The architecture page is one of the repo’s strongest trust-builders. Link it prominently near the top.

#### 5. Reduce repetition

Cut repeated usage of:

- “flagship”
- “split demo suite”
- “positioning”
- repeated link lists that say nearly the same thing

#### 6. Move installation lower

Installation is important, but not the strongest first-scan differentiator. Keep it, but after proof and orientation.

### Suggested README structure

1. Title + one-line value proposition
2. Hero asset
3. **Proof block**
4. “What this library is strongest at”
5. “Best first clicks”
6. Architecture link / trust signals
7. Installation
8. API styles
9. Examples
10. Deeper docs links

### Acceptance criteria

- A reviewer can understand the project’s strongest claim in ~20 seconds.
- The strongest metrics appear above the fold or very near it.
- The README no longer relies on repeated “flagship” language to create importance.

## Workstream D — Docs homepage rewrite

### Objective

Make `docs/index.md` concrete and fast.

### Requirements

#### 1. Rewrite the hero copy

Replace self-referential copy with a concrete statement of capability and validation.

#### 2. Keep and refine the validation snapshot

The current snapshot is one of the best parts of the site. Keep it, but ensure the surrounding copy is equally concrete.

#### 3. Improve CTA wording

Sharpen the first clicks. Examples:

- “Read the proof path”
- “See the recommended API”
- “Browse the public API”
- “Review benchmark evidence”

#### 4. Organize first clicks by audience or intent

Examples:

- strongest proof
- best API entry point
- architecture/system design
- performance evidence

### Acceptance criteria

- The homepage hero states what the library does without meta-language.
- The validation snapshot remains visible and central.
- The first clicks reflect actual reviewer priorities.

## Workstream E — Rewrite the decision / surface / eSSVI / local-vol pages

### Objective

Make each page feel like an interviewer-grade proof page rather than a page about the repo’s own presentation system.

### Requirements

#### Decision guide page

- make it functional and quick
- clarify “if you want to prove X, click Y”
- remove mismatch between H1 and page purpose
- reduce narrative about the guide system itself

#### Surface page

Lead with:

- the hard problem: noisy quote surface, static arbitrage, repair
- the method
- the result
- the supporting artifacts

Use the tracked static assets from Plan 1.

#### eSSVI page

Lead with:

- why slice-wise SVI is not the final Dupire handoff
- what eSSVI fixes
- what evidence supports the smoothing choice

Use the tracked dupire assets from Plan 1.

#### Local vol + PDE page

Lead with:

- validated numerical workflow
- repricing accuracy
- convergence and diagnostics

Use the tracked numerics assets from Plan 1 and benchmark outputs where helpful.

### Acceptance criteria

- Every page begins with a concrete engineering claim.
- Each page has at least one strong figure and one strong result block.
- The wording emphasizes validated outcomes over repo taxonomy.

## Workstream F — Fix the performance page using the benchmark plan

### Objective

Turn `docs/performance.md` from a credibility hole into a strength.

### Requirements

- remove the placeholder block entirely
- insert benchmark plots and summary tables from Plan 3
- add a short interpretation section covering:
  - what is fast
  - what scales
  - what tradeoffs are worth it
  - what the preferred practical paths are

### Acceptance criteria

- the performance page is no longer placeholder-like
- it becomes a page worth linking from the README top section

## Workstream G — Installation page polish

### Objective

Remove generic/template feel from the install docs.

### Requirements

- make extras concrete by reading them from `pyproject.toml`
- present them as actual supported install modes, not hypothetical examples
- explain the Python version requirement succinctly if needed
- keep the page crisp and practical

### Acceptance criteria

- installation instructions read like they belong to this repo specifically
- no generic “if your pyproject defines extras” language remains

## Workstream H — Consistency pass

### Objective

Make the public docs/readme feel intentionally designed rather than accreted.

### Requirements

- standardize terminology
- standardize CTA button style and labels
- standardize figure captions
- remove repeated link sections that say the same thing
- verify all paths, assets, and page links

### Acceptance criteria

- no broken links
- no broken images
- no placeholder sections
- no obvious copy mismatch between nav, H1, and CTA labels
- the public docs feel coherent across README and site

## Compact agent brief

Use this as a top-level instruction if needed:

```text
Overhaul this repo’s public-facing README/docs/site for interviewer impact without weakening technical accuracy. Prefer reusing existing generated assets from assets.zip before creating anything new. Remove placeholders, reduce meta-language and “flagship/capstone” branding, and make every landing page lead with capability + evidence. Build a real benchmark story with committed plots and reproducible artifacts, then rewrite README/docs/pages so a reviewer can understand the strongest engineering proof in ~20 seconds. Preserve link compatibility where practical, and do not invent claims or metrics.
```
