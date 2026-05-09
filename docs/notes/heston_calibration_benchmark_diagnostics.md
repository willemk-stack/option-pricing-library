# Heston Calibration Benchmark Diagnostics

`run_heston_calibration_benchmark_diagnostics` compares the current Heston
calibration objective under two SciPy least-squares Jacobian modes:
finite-difference (`jac="2-point"`) and the repo's analytic
`HestonObjective.jac`.

The diagnostic reports wall time, optimizer evaluation counts, final residual
statistics, optimizer status, fitted parameters, and synthetic parameter
recovery when a truth set is available. It returns the standard Heston
diagnostics topology:

- `meta`: scenario, optimizer, backend, and provenance fields
- `tables`: `runs`, `summary`, `parameter_recovery`, and `residuals`
- `arrays`: quote vectors, fitted parameters, and residual vectors

Run the default smoke diagnostic from Python:

```python
from option_pricing.diagnostics.heston import (
    run_heston_calibration_benchmark_diagnostics,
)

diagnostics = run_heston_calibration_benchmark_diagnostics()
diagnostics.tables["summary"]
```

Build CSV and JSON artifacts on demand:

```bash
python scripts/build_heston_calibration_benchmark_artifacts.py
```

The artifact builder now writes two clearly separated profiles under
`benchmarks/artifacts/heston_calibration/` and indexes them in
`heston_calibration_jacobian_artifacts.json`:

- `smoke`: the default lightweight regression artifact, using the small
    synthetic grid, `repeat=1`, `warmup=0`, and cheap quadrature so wiring and
    analytics coverage stay fast.
- `release`: a stronger deterministic review artifact for this repository
    scenario, using a larger quote grid, `repeat=5`, `warmup=1`, and denser
    Gauss-Legendre quadrature. Its scope is recorded in
    [`heston_calibration_jacobian_artifacts.json`](https://github.com/willemk-stack/option-pricing-library/blob/main/benchmarks/artifacts/heston_calibration/heston_calibration_jacobian_artifacts.json);
    it is not a machine-independent performance claim.

Build one profile explicitly when needed:

```bash
python scripts/build_heston_calibration_benchmark_artifacts.py --profile smoke
python scripts/build_heston_calibration_benchmark_artifacts.py --profile release
```

NOTE: The default synthetic quote grid and quadrature settings are smoke
benchmark choices, not a production calibration benchmark.

NOTE: The release artifact is stronger than the smoke artifact for this
configured scenario, but both timing sets remain environment-dependent.
Treat speedups as directional for the configured scenario, not universal
guarantees.

NOTE: This diagnostic mirrors the paper's analytic-vs-numerical-gradient
motivation, but it uses this repo's current vega-scaled price residual and
optimizer defaults unless those inputs are explicitly changed.

NOTE: The benchmark objective remains the repo's current vega-scaled price
residual. Reported IV-style fit metrics stay diagnostic; the benchmark does not
prove universal analytic-Jacobian performance and does not change the default
calibration objective.

## References

- Heston, S. L. (1993). A Closed-Form Solution for Options with Stochastic
    Volatility with Applications to Bond and Currency Options. *Review of
    Financial Studies*, 6(2), 327-343.
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*.
    Wiley.
- Cui, Y., del Baño Rollin, S., & Germano, G. (2017). Full and fast
    calibration of the Heston stochastic volatility model. *European Journal of
    Operational Research*, 263(2), 625-638.
