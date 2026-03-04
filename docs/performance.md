# Performance

This page describes the v1 benchmark suite built on `pytest-benchmark`. The goal is to track scaling and identify regressions in the most frequently used numerics.

## What is benchmarked

- Implied vol: vectorized slice inversion and scalar inversion scenarios.
- PDE solvers: Black-Scholes PDE (vanilla) and digital PDE with IC remedies.
- Local vol extraction: Dupire inversion on call-price grids.
- SVI fitting: single-slice calibration micro-benchmark.
- Macro pipeline: implied surface -> local vol surface -> digital PDE price.
- Binomial CRR: European call scaling with tree depth.

## Run locally

```bash
RUN_BENCHMARKS=1 pytest benchmarks -q --benchmark-only
```

To emit JSON for comparison or dashboards:

```bash
RUN_BENCHMARKS=1 pytest benchmarks -q --benchmark-only --benchmark-json benchmarks.json
```

## How to interpret scaling

- IV slice inversion should scale roughly $O(n_K)$ with strike count.
- PDE solve time should scale roughly $O(N_x N_t)$ for fixed method and coefficients.
- Dupire local vol inversion should scale roughly $O(n_T n_K)$ on the call grid.

## Recommended usage notes

- Prefer the vectorized slice implied vol function for surfaces or grids.
- Use `store="final"` in PDE solves unless you need the full time history.
- For digitals, Rannacher + IC remedies reduces oscillations at the strike.
