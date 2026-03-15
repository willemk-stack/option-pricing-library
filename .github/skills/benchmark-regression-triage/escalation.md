# Escalation rules

Use this file to decide when to recommend a full benchmark run and when to stop at cheap local evidence.
The default posture is conservative: do not spend large local benchmark budgets unless the change or request justifies it.

## Default policy

Do **not** recommend the full `benchmarks/` suite for routine local validation.
Start with correctness tests and the smallest relevant benchmark or smoke proxy.
Escalate only when one of the conditions below applies.

## Escalate to a full local benchmark when

Recommend a focused benchmark file, and only then consider broader coverage, when at least one of these is true:

1. **The user explicitly asked for performance measurement.**
   - Example: “measure whether this refactor sped up the PDE solver.”

2. **The change is clearly in a hot path.**
   - tight Python loops
   - vectorization/de-vectorization
   - array allocation patterns
   - tridiagonal solver internals
   - PDE time stepping or operator assembly
   - tree recursion/backward induction

3. **The change plausibly alters algorithmic complexity or constant factors in a benchmarked path.**
   - new nested loops
   - extra interpolation passes
   - repeated object construction in inner loops
   - changed grid sizes or default solver settings

4. **Correctness passes, but performance risk remains high.**
   - the code is now obviously doing more work
   - a smoke check suggests a slowdown
   - the change moved work from setup into the priced path

5. **You need before/after evidence for a PR, release, or regression investigation.**

## Prefer CI or a dedicated perf environment instead of local when

Recommend CI or a dedicated machine instead of a local run when:

- the benchmark is known to be rough or very expensive
- the result would be too noisy to trust on the developer workstation
- the benchmark needs repeated runs for confidence
- the benchmark includes very large parameter grids or tree depths
- you want comparable numbers across branches or commits

This is especially true for:
- `benchmarks/test_bench_tree.py` because the CRR benchmark currently reaches `n_steps=20000`
- broad runs across both PDE and macro benchmarks

## Tree / CRR-specific guidance

Because the current CRR implementation is slow and the tree benchmark is costly:

- do **not** recommend `benchmarks/test_bench_tree.py` as the default local validation path
- use correctness tests and reduced-step manual smoke checks first
- recommend the full tree benchmark only if:
  - the change explicitly targets tree performance, or
  - the user explicitly asked for a timing comparison, or
  - a release/perf investigation requires measured scaling evidence

Suggested wording:

> The tree benchmark is currently too expensive for routine local use. I recommend correctness tests plus a reduced-step smoke check locally, and deferring full CRR scaling measurements to CI or a dedicated perf run unless you specifically want local timing numbers.

## Escalation ladder

Use this order unless the user asks for something else:

1. correctness tests
2. targeted smoke check
3. single relevant benchmark file
4. multiple benchmark files in the same subsystem
5. integration or full-suite benchmark run

## How to explain the decision

Always explain:

- why the benchmark is or is not justified
- what cheaper proxy was used first
- what remains unmeasured
- whether the conclusion is measured or inferred

Good example:

> I did not recommend the full local benchmark suite because the change only touched tree correctness and the current CRR benchmark is high-cost. I used targeted tree tests first. Full tree scaling should be run only if you want explicit performance evidence.

Bad example:

> I skipped benchmarks.

## When to suggest improving the benchmark suite itself

Call out benchmark-maintenance work when you repeatedly hit one of these problems:

- a benchmark is too expensive for routine local use
- a benchmark mixes too many scenarios into one file
- there is no cheap representative case for a hot path
- benchmark coverage exists but is hard to map from changed files

For this repo, likely future improvements include:
- adding a cheaper CRR smoke benchmark with smaller `n_steps`
- separating tree correctness smoke from deep scaling measurement
- adding explicit “fast local” vs “deep perf” benchmark tiers
