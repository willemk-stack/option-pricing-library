# Local smoke checks

Use these when the full benchmark suite is too slow or too expensive for routine local work.
The goal is not to produce publication-quality benchmark numbers.
The goal is to cheaply detect obvious regressions, broken hot paths, and accidental complexity increases.

## General rules

- Prefer correctness and convergence tests before performance work.
- Prefer a single benchmark file over the whole suite.
- Prefer one representative scenario over the full parameter grid.
- If a benchmark is known to be rough or expensive, explicitly say so.
- Separate measured evidence from inferred performance risk.

## Cheap commands by area

### Implied vol
Use correctness-focused tests first:

```bash
pytest tests/test_iv.py tests/test_iv_edge_cases.py tests/test_implied_vol_slice_branches.py -q
```

If you need a quick benchmark sample, run only the IV benchmark file:

```bash
RUN_BENCHMARKS=1 pytest benchmarks/test_bench_iv.py -q --benchmark-only
```

Use this for:
- vectorization changes
- seed/root-finding changes
- slice inversion changes

---

### PDE core
Use correctness and branch coverage first:

```bash
pytest \
  tests/test_pde_pricer.py \
  tests/test_pricers_finite_diff.py \
  tests/test_pde_digital_wrappers.py \
  tests/test_convergence_remedies_digital.py \
  -q
```

If you need runtime evidence, run only the PDE benchmark file:

```bash
RUN_BENCHMARKS=1 pytest benchmarks/test_bench_pde.py -q --benchmark-only
```

Use this for:
- solver loop changes
- method selection changes
- IC remedy changes
- tridiagonal solver changes

---

### Local vol and SVI
Use correctness tests first:

```bash
pytest \
  tests/test_dupire_constant_vol.py \
  tests/test_surface_svi_and_localvol.py \
  tests/test_svi.py \
  tests/test_svi_calibrate_smoke_paths.py \
  -q
```

If the change is eSSVI-specific, add:

```bash
pytest tests/test_essvi_surface.py tests/test_essvi_calibrate.py tests/test_essvi_objective.py -q
```

If you need runtime evidence, run only the local-vol benchmark file:

```bash
RUN_BENCHMARKS=1 pytest benchmarks/test_bench_localvol.py -q --benchmark-only
```

Use this for:
- Dupire changes
- local-vol surface changes
- SVI fitting changes
- no-arb/interpolation changes that may affect critical paths

---

### Macro pipeline
Default to a correctness proxy instead of the macro benchmark:

```bash
pytest \
  tests/test_localvol_pde_vanilla_vs_bs.py \
  tests/test_localvol_digital_vs_bs.py \
  tests/test_surface_time_diagnosis.py \
  tests/test_diagnostics_vol_surface_report.py \
  -q
```

Run the macro benchmark only when you need integration-level runtime evidence:

```bash
RUN_BENCHMARKS=1 pytest benchmarks/test_bench_macro.py -q --benchmark-only
```

Use this for:
- end-to-end surface -> local-vol -> PDE changes
- performance changes that span more than one subsystem

---

### Tree / CRR
Because the benchmark includes very large depths, default to correctness-only checks first:

```bash
pytest tests/test_binomial.py tests/test_tree_pricer_branches.py -q
```

If you need a manual local smoke check for a tree change, prefer a small one-off script or REPL run using a reduced step count such as 200, 500, or 1000 rather than the benchmark file. Example:

```python
from option_pricing.pricers.tree import binom_price_from_ctx
from option_pricing.types import MarketData, OptionType

ctx = MarketData(spot=100.0, rate=0.02, dividend_yield=0.0).to_context()
for n_steps in (200, 500, 1000):
    price = binom_price_from_ctx(
        kind=OptionType.CALL,
        strike=100.0,
        sigma=0.2,
        tau=1.0,
        ctx=ctx,
        n_steps=n_steps,
        american=False,
        method="tree",
    )
    print(n_steps, price)
```

Only run the tree benchmark file locally when the change is explicitly about tree performance and you accept the cost:

```bash
RUN_BENCHMARKS=1 pytest benchmarks/test_bench_tree.py -q --benchmark-only
```

## What to report back

When using smoke checks instead of a full benchmark, report:

- what changed
- which cheap checks were run
- whether they passed
- whether any runtime evidence was actually measured
- what still needs CI or a dedicated perf run

Example:

> I did not run the full benchmark suite locally because the tree benchmark is currently too expensive for routine use. I ran the tree correctness tests and a reduced-step manual smoke check instead. Those passed, so there is no obvious local evidence of a severe regression, but full CRR scaling still needs confirmation in CI or a dedicated perf run.
