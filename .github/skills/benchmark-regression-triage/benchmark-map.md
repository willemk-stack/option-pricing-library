# Benchmark map

Use this file to map changed code to the smallest relevant benchmark scope.
Prefer the narrowest benchmark or smoke check that can answer the question.
Do not assume the whole `benchmarks/` suite should run locally.

## Benchmark inventory

### `benchmarks/test_bench_iv.py`
Purpose:
- implied-vol inversion performance
- vectorized slice inversion scaling
- scalar inversion scenarios

Primary code paths:
- `src/option_pricing/vol/implied_vol_scalar.py`
- `src/option_pricing/vol/implied_vol_slice.py`
- `src/option_pricing/vol/implied_vol_common.py`
- `src/option_pricing/vol/implied_vol_seed.py`
- `src/option_pricing/vol/implied_vol_types.py`
- `src/option_pricing/pricers/black_scholes.py`
- `src/option_pricing/models/black_scholes/bs.py`
- `src/option_pricing/types.py`

Relevant correctness tests first:
- `tests/test_iv.py`
- `tests/test_iv_edge_cases.py`
- `tests/test_implied_vol_slice_branches.py`
- `tests/test_black_scholes_bs_module_branches.py`

Run when:
- inversion algorithms change
- vectorization changes
- root-finding or seed logic changes
- Black-Scholes price plumbing changes in a way that may affect IV paths

---

### `benchmarks/test_bench_pde.py`
Purpose:
- Black-Scholes PDE runtime scaling
- digital PDE runtime and IC-remedy comparisons

Primary code paths:
- `src/option_pricing/pricers/pde_pricer.py`
- `src/option_pricing/pricers/pde/`
- `src/option_pricing/numerics/pde/solver.py`
- `src/option_pricing/numerics/pde/methods.py`
- `src/option_pricing/numerics/pde/time_steppers.py`
- `src/option_pricing/numerics/pde/operators.py`
- `src/option_pricing/numerics/pde/ic_remedies.py`
- `src/option_pricing/pricers/pde/domain.py`
- `src/option_pricing/models/black_scholes/pde.py`
- `src/option_pricing/numerics/grids.py`
- `src/option_pricing/numerics/tridiag.py`

Relevant correctness tests first:
- `tests/test_pde_pricer.py`
- `tests/test_pricers_finite_diff.py`
- `tests/test_finite_diff_pricer_branches.py`
- `tests/test_pde_digital_wrappers.py`
- `tests/test_pde_domain_branches.py`
- `tests/test_convergence_remedies_digital.py`
- `tests/test_tridiag_solvers.py`

Run when:
- solver loops or stencil assembly change
- grid/domain policy changes
- IC remedy or method selection changes
- tridiagonal solver changes

---

### `benchmarks/test_bench_localvol.py`
Purpose:
- Dupire local-vol extraction scaling on call-price grids
- SVI single-slice fit micro-benchmark

Primary code paths:
- `src/option_pricing/vol/local_vol_dupire.py`
- `src/option_pricing/vol/local_vol_surface.py`
- `src/option_pricing/vol/local_vol_fd.py`
- `src/option_pricing/vol/local_vol_gatheral.py`
- `src/option_pricing/vol/svi/`
- `src/option_pricing/vol/surface_core.py`
- `src/option_pricing/diagnostics/vol_surface/`

Relevant correctness tests first:
- `tests/test_dupire_constant_vol.py`
- `tests/test_surface_svi_and_localvol.py`
- `tests/test_svi.py`
- `tests/test_svi_calibrate_smoke_paths.py`
- `tests/test_svi_calibrate_additional_paths.py`
- `tests/test_svi_repair.py`
- `tests/test_local_vol_fd_branches.py`
- `tests/test_localvol_pde_vanilla_vs_bs.py`

Also consider these if the change is in eSSVI rather than classic SVI:
- `tests/test_essvi_surface.py`
- `tests/test_essvi_calibrate.py`
- `tests/test_essvi_objective.py`
- `tests/test_essvi_mingone_projection.py`

Run when:
- local-vol extraction logic changes
- SVI/eSSVI calibration logic changes
- smile/surface interpolation changes
- vol-surface diagnostics or no-arb logic changes in a way that may affect runtime

---

### `benchmarks/test_bench_macro.py`
Purpose:
- end-to-end synthetic pipeline benchmark
- implied surface -> local-vol surface -> digital PDE repricing

Primary code paths:
- `src/option_pricing/vol/surface_core.py`
- `src/option_pricing/vol/local_vol_surface.py`
- `src/option_pricing/vol/svi/`
- `src/option_pricing/pricers/pde/digital_local_vol.py`
- `src/option_pricing/pricers/pde/domain.py`
- `src/option_pricing/numerics/pde/`
- `src/option_pricing/diagnostics/vol_surface/`

Relevant correctness tests first:
- `tests/test_localvol_pde_vanilla_vs_bs.py`
- `tests/test_localvol_digital_vs_bs.py`
- `tests/test_localvol_digital_vs_call_strike_derivative.py`
- `tests/test_localvol_digital_convergence_sweep.py`
- `tests/test_diagnostics_vol_surface_report.py`
- `tests/test_surface_time_diagnosis.py`
- `tests/test_surface_svi_and_localvol.py`

Run when:
- changes cross surface calibration, local-vol extraction, and PDE repricing
- a single subsystem benchmark is not enough to estimate impact
- you need integration-level perf evidence rather than component-level evidence

---

### `benchmarks/test_bench_tree.py`
Purpose:
- CRR/binomial scaling with tree depth

Primary code paths:
- `src/option_pricing/pricers/tree.py`
- `src/option_pricing/models/binomial_crr.py`
- `src/option_pricing/instruments/`
- `src/option_pricing/market/curves.py`
- `src/option_pricing/types.py`

Relevant correctness tests first:
- `tests/test_binomial.py`
- `tests/test_tree_pricer_branches.py`
- `tests/test_api_smoke.py`

Run when:
- CRR recursion, backward induction, or branching logic changes
- closed-form/tree method dispatch changes
- exercise-style handling changes

Important note:
- this benchmark currently includes very expensive depths up to `n_steps=20000`
- treat it as opt-in and high-cost
- do not recommend it for routine local validation unless the change is explicitly about tree performance

## Quick change-to-benchmark mapping

Use this section as a first-pass routing guide.

- `src/option_pricing/vol/implied_vol_*` -> `test_bench_iv.py`
- `src/option_pricing/vol/svi/` -> `test_bench_localvol.py` and possibly `test_bench_macro.py`
- `src/option_pricing/vol/ssvi/` -> usually no direct benchmark; use correctness tests first, then consider `test_bench_localvol.py` or `test_bench_macro.py` only if the change affects the same downstream local-vol/PDE path
- `src/option_pricing/vol/local_vol_*` -> `test_bench_localvol.py`, maybe `test_bench_macro.py`
- `src/option_pricing/numerics/pde/` -> `test_bench_pde.py`, maybe `test_bench_macro.py`
- `src/option_pricing/pricers/pde/` -> `test_bench_pde.py`, maybe `test_bench_macro.py`
- `src/option_pricing/pricers/tree.py` or `src/option_pricing/models/binomial_crr.py` -> `test_bench_tree.py`
- `src/option_pricing/numerics/tridiag.py` -> `test_bench_pde.py`
- `src/option_pricing/numerics/grids.py` -> `test_bench_pde.py`, `test_bench_macro.py`, and sometimes `test_bench_localvol.py`
- `src/option_pricing/diagnostics/vol_surface/` -> usually correctness/docs first; benchmark only if diagnostics code is on the critical path of benchmarked workflows

## Recommended default behavior

1. Start from the changed files.
2. Pick the smallest relevant benchmark file.
3. Prefer a smoke proxy from `local-smoke.md` before a full benchmark.
4. Escalate to full local benchmarks only when justified by `escalation.md`.
5. Report whether performance conclusions are measured or inferred.
