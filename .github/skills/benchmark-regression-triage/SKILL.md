---
name: benchmark-regression-triage
description: Use when refactoring performance-sensitive numerics or asking whether a change risks slowing down local-vol, PDE, tree, or implied-vol workflows. Map the change to the right benchmarks and summarize likely hot paths.
---
This skill helps separate correctness validation from performance validation.

Use it when the task mentions:
- benchmark, regression, slowdown, hot path, scaling
- PDE solver performance
- local-vol extraction cost
- implied-vol throughput
- tree scaling or macro benchmark changes

## Goal
Choose the right benchmark subset for the touched code, pair it with the right correctness checks, and report whether the observed change is likely meaningful.

## First steps
1. Identify the touched files or subsystem.
2. Read `benchmark-map.md`.
3. Pick the smallest benchmark file that exercises the suspected hot path.
4. Pair the benchmark with at least one representative correctness test before interpreting any speedup or slowdown.

## Targeted benchmark commands
### PDE changes
```bash
pytest benchmarks/test_bench_pde.py --benchmark-only --benchmark-verbose
pytest -q tests/test_pde_pricer.py tests/test_localvol_pde_vanilla_vs_bs.py
```

### Local-vol or surface extraction changes
```bash
pytest benchmarks/test_bench_localvol.py --benchmark-only --benchmark-verbose
pytest -q tests/test_dupire_constant_vol.py tests/test_surface_svi_and_localvol.py
```

### Implied-vol changes
```bash
pytest benchmarks/test_bench_iv.py --benchmark-only --benchmark-verbose
pytest -q tests/test_iv.py tests/test_iv_edge_cases.py
```

### Tree changes
```bash
pytest benchmarks/test_bench_tree.py --benchmark-only --benchmark-verbose
pytest -q tests/test_binomial.py tests/test_tree_pricer_branches.py
```

### Broad workflow changes
```bash
pytest benchmarks/test_bench_macro.py --benchmark-only --benchmark-verbose
```

### Actions-equivalent full benchmark run
Use this only when the change is broad enough to justify it:
```bash
pytest benchmarks --benchmark-only --benchmark-json benchmarks.json --benchmark-verbose
```

## Interpretation rules

When asked to assess performance impact:
- Do not assume the full benchmark suite should be run locally.
- First identify whether the change likely affects a performance-critical path.
- Prefer the smallest relevant benchmark or reduced-cost proxy.
- If no safe cheap local benchmark exists, explain what should be validated in CI or a dedicated perf run.
- Clearly separate measured results from inferred risk.

- Never report a speedup without confirming the related correctness tests still pass.
- Prefer targeted benchmarks over a full benchmark sweep when the edit is narrow.
- If runtime changes vary by size, mention that the change may affect scaling rather than fixed overhead.
- Treat single noisy observations carefully; look for a repeatable trend before calling it a regression.
- If the change improves stability or correctness at a performance cost, say so explicitly instead of labeling it a simple regression.

## Expected output
End with a short triage summary:
1. **Hot path suspected**
2. **Benchmarks run**
3. **Correctness checks paired with them**
4. **Observed result**
5. **Recommended next step**

## Examples
- "Use benchmark regression triage before I refactor the PDE solver."
- "Use benchmark regression triage to assess whether this Dupire change risks a slowdown."
- "Use benchmark regression triage after changing implied-vol root finding."
