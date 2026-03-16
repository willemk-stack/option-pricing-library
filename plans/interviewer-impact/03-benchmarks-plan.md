# Plan 3 — Truly Impressive Benchmark Plan

## Goal

Replace the current placeholder performance story with a benchmark suite and benchmark documentation that are:

- real
- reproducible
- reviewer-impressive
- and honest

## Current issue to fix

`docs/performance.md` currently contains a visible placeholder benchmark section instead of committed benchmark artifacts and a defensible benchmark narrative.

That weakens credibility. For a performance page, placeholder content is worse than omission.

## Benchmark philosophy

The benchmark work should answer more than "how fast is this?"

It should tell a reviewer:

- what scales well
- what accuracy costs
- what the preferred pathways are
- where the engineering tradeoffs are
- and what the repo can defend in an interview

## Existing benchmark base to extend

Keep and deepen the existing benchmark suite. Do **not** throw it away.

Existing files:

- `benchmarks/test_bench_iv.py`
- `benchmarks/test_bench_localvol.py`
- `benchmarks/test_bench_macro.py`
- `benchmarks/test_bench_pde.py`
- `benchmarks/test_bench_tree.py`

## Requirements

### 1. Add a benchmark result publishing pipeline

Create a reproducible benchmark artifact flow.

#### Inputs

- `pytest-benchmark` JSON output

#### Outputs

- machine-readable benchmark snapshots:
  - `benchmarks/artifacts/*.json`
  - optionally `benchmarks/artifacts/*.csv`
- generated plots:
  - `docs/assets/generated/benchmarks/*.png`
- generated markdown or data summary consumed by:
  - `docs/performance.md`

### 2. Benchmark families to include

#### A. Implied-vol inversion throughput and scaling

Show:

- scalar inversion latency across representative scenarios
- vectorized slice inversion scaling across strike counts
- throughput advantage of vectorized workflows where applicable

Why it matters:

- this shows practical numerics rather than only formula availability

#### B. PDE runtime + error scaling

For vanilla Black–Scholes PDE and local-vol PDE, show:

- runtime vs grid size
- error vs analytic or trusted reference target
- runtime/error tradeoff for different methods where meaningful

Why it matters:

- this is much stronger than raw runtime alone

#### C. Digital PDE remedies benchmark

Measure the cost and benefit of remedies for discontinuous payoffs, including:

- runtime impact
- error reduction
- stability improvement

Why it matters:

- this is interviewer-impressive because it demonstrates domain-aware numerical engineering

#### D. Local-vol extraction scaling and validity

Measure:

- runtime vs `(nT, nK)` grid size
- invalid-rate or diagnostics summary where meaningful
- best practical operating region

Why it matters:

- it proves the local-vol path is not just implemented but characterized

#### E. End-to-end macro pipeline benchmark

Benchmark something close to the actual showcase story, including:

- implied surface or fitted surface stage
- smooth handoff stage
- local-vol extraction stage
- PDE repricing of a representative option set

Why it matters:

- this is the most valuable interviewer benchmark because it speaks to integrated-system behavior

#### F. Tree benchmark with stronger framing

Retain the tree benchmark but reframe it around:

- scaling with `n_steps`
- where it sits relative to intended use
- when analytic or PDE methods are the better practical choice

### 3. Add at least one comparative baseline

A benchmark becomes more persuasive when something is compared rather than timed in isolation.

Use one or more of:

- analytic Black–Scholes baseline for error comparisons
- QuantLib comparisons where the repo already has meaningful parity tests
- method-vs-method tradeoff charts within the library

Rules:

- only compare apples to apples
- do not overclaim “faster than X” unless measured carefully and reproducibly

### 4. Generate benchmark visuals

Create a minimum visual set:

- `iv_scaling.png`
- `pde_runtime_error_tradeoff.png`
- `digital_remedies_tradeoff.png`
- `localvol_scaling.png`
- `macro_pipeline_summary.png`

These should feed both:

- `docs/performance.md`
- the optional benchmark summary asset from Plan 2

### 5. Upgrade CI and reproducibility

Extend the benchmark workflow so results are easy to regenerate and review.

Possible additions:

- commit a publish-grade benchmark snapshot produced under a controlled config
- upload raw JSON and derived plots as workflow artifacts
- include software/runtime metadata in the published summary

## Requirements for `docs/performance.md`

The final page should include:

- one opening paragraph explaining what the benchmarks prove
- 2–4 real figures, not placeholders
- concise tables with scenario sizes and conditions
- a hardware/environment disclaimer
- a short “how to reproduce” section
- a short “what to conclude” section

## Non-goals

- Do not pad the page with raw benchmark dumps.
- Do not treat a single machine’s numbers as universal truth.
- Do not optimize for absolute speed at the expense of interpretability.
- Do not make unsupported claims against external libraries.

## Strong acceptance criteria

- `docs/performance.md` contains no placeholder content.
- The page includes real, committed plots generated from actual benchmark results.
- At least one benchmark family shows **accuracy vs cost**, not just runtime.
- At least one benchmark family shows **end-to-end pipeline** behavior.
- The benchmark story is strong enough to be referenced in the README/docs as evidence of engineering maturity.
