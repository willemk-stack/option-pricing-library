# Heston quadrature policy

This note defines the intended quadrature policy for the Heston Fourier pricer.
It is meant to sit beside the implementation and diagnostics docs, not replace
source-code docstrings.

## Scope

The policy applies to European vanilla Heston prices computed through the
semi-analytic probability-integral path:

- `heston_price_call_from_ctx`
- `heston_price_put_from_ctx`
- `heston_price_from_ctx`
- calibration objectives that call the same pricing path
- diagnostics that compare fixed-rule integration and SciPy `quad`

Monte Carlo, local-vol PDE, and implied-vol inversion are outside the scope of
this note.

## Production default

The production default should be the fixed Gauss-Legendre backend:

```python
backend = "gauss_legendre"
```

Rationale:

- it is deterministic;
- it supports vectorized strike slices;
- it is much faster than scalar adaptive `quad` inside calibration loops;
- it can be configured and diagnosed panel-by-panel;
- it supports stable benchmarking once a rule is frozen.

`quad` remains valuable as a cross-check backend, but it should not be the
default calibration backend because it evaluates scalar integrals and is much
more expensive across strikes, maturities, and optimizer iterations.

REVIEW: This default is a performance/numerical-policy decision. Revisit it
after the reference-price and branch-stability test suites are frozen.

## Configuration tiers

Use four practical tiers. The exact numeric values can live in code, but the
policy intent should stay stable.

| Tier | Intended use | Behavior |
|---|---|---|
| `fast` | exploratory plots and quick smoke checks | smaller `u_max`, fewer panels, fewer nodes |
| `balanced` | ordinary pricing and examples | default accuracy/runtime trade-off |
| `robust` | calibration runs, published notebooks, reference tests | larger `u_max`, more panels/nodes |
| `diagnostics` | investigating warnings and stress regimes | densest rule, preferably with richer panel diagnostics |

The public recommender should map a strike/maturity/parameter regime to one of
these tiers. Tests should not assume that all regimes use the same raw rule.

## Suggested baseline values

These values are intentionally conservative starting points, not final law.

```python
fast        = QuadratureConfig(u_max=125.0, n_panels=20, nodes_per_panel=12)
balanced    = QuadratureConfig(u_max=150.0, n_panels=24, nodes_per_panel=16)
robust      = QuadratureConfig(u_max=220.0, n_panels=44, nodes_per_panel=24)
diagnostics = QuadratureConfig(u_max=260.0, n_panels=56, nodes_per_panel=32)
```

REVIEW: Bless these values only after running backend-agreement, reference-price,
and branch-stability tests across the frozen case grid.

## When to use each tier

Use `fast` for:

- interactive notebook exploration;
- coarse smile-shape checks;
- examples where exact numerical values are not reported.

Use `balanced` for:

- default user-facing pricing;
- small demos;
- API examples;
- ordinary non-stress parameter regimes.

Use `robust` for:

- calibration objective evaluation;
- final capstone notebooks;
- reference-price regression tests;
- comparing Heston against local volatility;
- any plot or table whose numbers are interpreted in the writeup.

Use `diagnostics` for:

- high vol-of-vol;
- `rho` close to `-1` or `1`;
- short expiries;
- far-wing strikes;
- near-deterministic variance limits;
- any case producing panel warnings, jagged smiles, or backend disagreement.

## Backend cross-check policy

For frozen reference cases, compare:

1. fixed Gauss-Legendre using a robust or diagnostic rule;
2. SciPy `quad` as an independent adaptive integration check;
3. static no-arbitrage constraints such as put-call parity and monotonicity.

Expected use:

```python
price_gl = heston_price_from_ctx(..., backend="gauss_legendre", quad_cfg=robust_cfg)
price_quad = heston_price_from_ctx(..., backend="quad")
```

Acceptance tolerance should be tighter for ordinary regimes and looser for
short-expiry, high-vol-of-vol, or far-wing regimes. Tolerance should be reported
with the case, not hidden in a single global magic number.

## Warning interpretation

Diagnostics should distinguish between warnings that require rerunning with a
denser rule and warnings that should cause the result to be rejected.

### Rerun with denser quadrature

Rerun with `robust` or `diagnostics` if diagnostics show:

- large tail contribution;
- under-resolved tail panels;
- excessive cancellation;
- isolated panel spikes;
- backend disagreement slightly above tolerance.

### Reject or quarantine the result

Reject, quarantine, or clearly label the result if diagnostics show:

- non-finite integrands;
- non-finite total integral;
- probabilities materially outside `[0, 1]`;
- persistent backend disagreement after using the diagnostics tier;
- jagged implied-vol smiles that survive denser quadrature.

REVIEW: The exact warning thresholds should be empirical and tied to the frozen
reference grid. Do not tune them to make one demo pass.

## Calibration policy

Calibration should not use the cheapest quadrature rule by default. Optimizers
can exploit numerical noise, especially in weakly identified Heston regimes.

Recommended policy:

- use `robust` for final calibration runs;
- use `balanced` only for exploratory initialization or smoke tests;
- rerun final fitted parameters with `diagnostics` for verification;
- report backend agreement for final fits;
- treat large backend disagreement as a calibration-quality issue, not merely a
  pricing issue.

REVIEW: The final capstone should state whether calibration residuals are driven
by model misspecification or by numerical integration error. The quadrature
policy exists to make that distinction visible.

## Reference tests expected from this policy

At minimum, keep tests for:

- frozen Heston reference prices;
- fixed-rule vs `quad` backend agreement;
- put-call parity;
- call monotonicity in strike;
- put monotonicity in strike;
- dense-slice smoothness;
- small-parameter-perturbation continuity;
- implied-vol smile smoothness for representative regimes.

These tests should cover:

- mild negative skew;
- high vol-of-vol;
- short maturity;
- long maturity;
- near-deterministic variance;
- positive `rho`;
- `rho` close to `-1` but still admissible.

## Documentation expectation

Whenever a notebook reports Heston prices, calibration residuals, or model
comparison metrics, it should state:

- backend used;
- quadrature configuration or tier;
- whether diagnostics were clean;
- whether `quad` cross-checks were run;
- any known warnings or rejected cases.

A final capstone table should not quote Heston calibration quality without also
stating the quadrature tier used to produce it.
