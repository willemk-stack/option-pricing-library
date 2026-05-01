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

REVIEW: The default synthetic quote grid and quadrature settings are smoke
benchmark choices, not a production calibration benchmark.

REVIEW: Timings are environment-dependent. Treat speedups as directional for
the configured scenario, not universal guarantees.

REVIEW: This diagnostic mirrors the paper's analytic-vs-numerical-gradient
motivation, but it uses this repo's current vega-scaled price residual and
optimizer defaults unless those inputs are explicitly changed.
