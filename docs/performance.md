# Performance evidence

This page is the committed benchmark snapshot for the workflows that matter most in review: implied-vol inversion, PDE runtime versus error, digital-payoff remedies, local-vol extraction, the end-to-end surface-to-PDE path, and CRR tree scaling. The goal is not to claim universal absolute speed on one machine; it is to show which workflows scale cleanly, what extra accuracy costs, and which implementation choices are the defensible defaults.

![Benchmark overview: committed IV scaling, PDE runtime-versus-error, macro stage budget, and benchmark callouts derived from published artifacts.](../assets/generated/benchmarks/benchmark_overview.svg)

## Snapshot

| Family | Conditions | Reference | What the committed snapshot shows |
| --- | --- | --- | --- |
| Implied-vol inversion | Scalar BS inversion and Black-76 slice inversion on 21-801 strikes | Scalar loop baseline | The vectorized slice path reaches `446x` speedup at `801` strikes while preserving zero measured vol difference versus the scalar loop. Scalar single-option inversions stay around `0.61-0.62 ms` across the representative scenarios. |
| Vanilla PDE | Black-Scholes European call, `Nx=Nt` from `101` to `801` | Analytic Black-Scholes price | Absolute error falls from about `1.74e-2` at `101x101` to about `2.28e-4` at `801x801`. Runtime rises from about `41.5 ms` to about `985.6 ms` across the same refinement path. |
| Digital PDE remedies | Digital call, `Nx=Nt` in `{101, 201, 401}` | Analytic digital price | The untreated path keeps an `~1.18e-2` refinement spread, while `Rannacher + cell average` cuts it to `3.00e-5` and `Rannacher + L2 projection` cuts it to `1.43e-5`. |
| End-to-end macro pipeline | Synthetic SVI quote mesh -> fitted surface -> handoff probe -> local-vol surface -> representative local-vol PDE | Stage-level timing only | Surface fitting dominates the measured stage budget (`259.6 ms`, `369.5 ms`, `782.5 ms` for small, medium, large), with PDE repricing second (`75.7 ms`, `177.6 ms`, `305.6 ms`). |

## Implied-vol inversion

![IV scaling figure: vectorized slice inversion versus scalar-loop inversion, plus scalar single-option latency scenarios.](../assets/generated/benchmarks/iv_scaling.png)

Use the vectorized slice inverter whenever the workload is a smile or surface rather than an isolated option. At `801` strikes the committed snapshot records about `1.12 ms` for the vectorized slice path versus about `498.04 ms` for the scalar-loop baseline.

## PDE runtime versus error

![PDE runtime versus error tradeoff for Black-Scholes vanilla PDE and local-vol PDE refinement.](../assets/generated/benchmarks/pde_runtime_error_tradeoff.png)

The vanilla PDE curve shows the expected refinement tradeoff against an analytic benchmark. The local-vol curve uses a finer-grid local-vol PDE solve as its reference because there is no closed-form target for that path. That is the right comparison for review: the figure shows what extra grid density buys rather than just how long one solve took.

## Digital-payoff remedies

![Digital payoff remedy tradeoff: runtime versus error, plus grid-refinement price spread across remedy choices.](../assets/generated/benchmarks/digital_remedies_tradeoff.png)

This is the clearest benchmark family for domain-aware numerical engineering. Leaving the discontinuity untreated keeps runtime modest, but it also leaves materially larger error and much wider grid-to-grid drift. The cell-average and L2-projection remedies both stabilize the refinement path without changing the runtime order of magnitude.

## End-to-end pipeline

![Macro pipeline benchmark: stacked stage timing for the synthetic surface-to-local-vol-to-PDE workflow.](../assets/generated/benchmarks/macro_pipeline_summary.png)

For the integrated benchmark, the key observation is where time actually goes. In this snapshot the handoff itself stays cheap, the local-vol extraction step stays around `0.46-1.01 ms`, and the expensive parts are fitting the surface and running the PDE. That is useful when discussing optimization priorities because it argues against spending engineering effort on the wrong stage.

## Additional families

### Local-vol extraction

The published local-vol extraction run compares `strike_coordinate="K"` and `strike_coordinate="logK"` on the same smooth synthetic SVI-driven call grids. Both stayed at `0%` interior invalid share through the largest tested grid, while runtime stayed close to `1.3 ms`.

| Coordinate | Largest tested grid | Median runtime | Interior invalid share |
| --- | --- | --- | --- |
| `K` | `121 x 241` | `1.327 ms` | `0.0%` |
| `logK` | `121 x 241` | `1.312 ms` | `0.0%` |

The full figure is available at [localvol_scaling.png](../assets/generated/benchmarks/localvol_scaling.png).

### Tree scaling

The CRR benchmark remains useful for convergence discussion, but the published numbers make its placement clear: it is an interpretable reference path, not the preferred engine for repeated European-vanilla work at large step counts.

| `n_steps` | Median runtime | Absolute error vs Black-Scholes |
| --- | --- | --- |
| `200` | `46.8 ms` | `9.90e-3` |
| `1000` | `900.7 ms` | `1.98e-3` |
| `5000` | `19.7 s` | `3.96e-4` |

The full figure is available at [tree_scaling.png](../assets/generated/benchmarks/tree_scaling.png).

## Environment and reproducibility

- Snapshot environment: Windows 11, Python `3.12.7`, NumPy `1.26.4`, SciPy `1.17.1`, pandas `2.0.3`, `16` logical CPUs.
- Treat the absolute timings as machine-specific. The defensible signal is the relative slope of each curve, the runtime/error tradeoff, and which workflow dominates the stage budget.
- The machine-readable snapshot used for this page is committed under `benchmarks/artifacts/`, with exact runtime metadata in `benchmarks/artifacts/performance_summary.json`.

To reproduce the published bundle from a standard project environment:

```bash
python -m pip install -e ".[dev]"
RUN_BENCHMARKS=1 pytest benchmarks -q --benchmark-only --benchmark-json benchmarks/artifacts/pytest-benchmark.json --benchmark-verbose
python scripts/build_benchmark_artifacts.py --pytest-benchmark-json benchmarks/artifacts/pytest-benchmark.json
```

The `pytest-benchmark` JSON is kept as the raw timing record. The publishing script turns that raw run plus direct error and reference computations into committed CSV, JSON, and figure artifacts for docs use.

## What to conclude

- Prefer the vectorized implied-vol slice API for smiles and surfaces; the scalar API is fine for isolated options but not for surface-scale work.
- For vanilla PDE work, refinement behaves as expected: the `801x801` grid buys another step down in error, but it does so at a materially higher runtime than `201x201` or `401x401`.
- For digital payoffs, do not discuss the PDE path without the remedy choice. `cell_avg` and `L2 projection` materially improve refinement behavior.
- In the integrated workflow, surface fitting dominates the stage budget. Local-vol construction is cheap relative to fitting and PDE repricing.
- CRR remains useful as a convergence reference, but the benchmark data makes the intended method placement clear: analytic or PDE methods are the more practical choice for repeated European-vanilla valuation.
