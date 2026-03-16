# AGENT_BRIEF.md

This repository contains a staged plan pack for improving interviewer impact across the README, docs, pages, assets, and benchmarks.

## Where to start

Read these files first, in this order:

1. `plans/interviewer-impact/00-index.md`
2. The specific requirement-set file you are asked to execute next

## Canonical execution order

Unless explicitly told otherwise, execute the plan pack in this order:

1. `plans/interviewer-impact/01-adopt-existing-assets.md`
2. `plans/interviewer-impact/03-benchmarks-plan.md`
3. `plans/interviewer-impact/02-new-asset-creation-plan.md`
4. `plans/interviewer-impact/04-general-docs-readme-pages-overhaul.md`

Do **not** jump ahead across workstreams in a single run unless explicitly instructed.

## Global rules

- Work directly in this repository.
- Prefer **existing assets** over creating new ones.
- Do **not invent** metrics, benchmark results, validation claims, or performance conclusions.
- Remove placeholders instead of leaving “to do later” text in public-facing pages.
- Use **outcome-first** wording:
  - what the project does
  - what is technically hard
  - what evidence proves it
  - what the reader should click next
- Reduce meta-language and branding repetition such as “flagship”, “capstone”, “positioning”, and explanations of the documentation system itself.
- Preserve link compatibility where practical. If renaming pages, add redirects, aliases, or compatibility stubs where the docs system supports them.
- Keep claims tied to visible evidence: tests, notebooks, benchmark artifacts, figures, or committed docs pages.

## Operating mode

For each run, execute **only one requirement set** unless told otherwise.

At the end of the run, stop and report:

1. Files changed
2. Commands run
3. Validation performed
4. What was intentionally not touched
5. Any blockers or follow-up items

## Quality bar

Every landing page should be understandable to a strong SWE/interviewer in about **20 seconds**.

A reviewer should be able to quickly answer:

- What is this project?
- What is technically impressive here?
- What evidence says it works?
- Where should I look next?

## Validation expectations

Before stopping, validate relevant changes.

Examples:

- docs build succeeds
- no broken links introduced
- no broken image references introduced
- benchmark artifacts referenced by docs actually exist
- renamed pages still have working navigation/compatibility paths if needed

## Preferred response format after each run

Return a concise execution summary with these headings:

- `Completed`
- `Files changed`
- `Validation`
- `Not done`
- `Follow-ups`

## Example invocation

If asked to execute the first phase, do this:

- read `plans/interviewer-impact/00-index.md`
- execute only `plans/interviewer-impact/01-adopt-existing-assets.md`
- stop after completing that requirement set

## Non-goals unless explicitly requested

- broad refactors unrelated to interviewer impact
- rewriting core library logic without plan support
- adding speculative benchmark claims
- generating decorative assets with no reviewer value
