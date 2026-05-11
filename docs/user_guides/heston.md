# Heston

## Overview

Heston adds stochastic variance to the vanilla-option workflow. In this library
that means:

- Fourier-based European vanilla pricing under a five-parameter
    stochastic-volatility model;
- Monte Carlo pricing with both full-truncation Euler and Andersen QE schemes;
- least-squares calibration from vanilla quote sets;
- notebook-friendly diagnostics for pricing stability, calibration fit, Monte
    Carlo convergence, and model comparison;
- a comparison layer against eSSVI plus a direct local-vol PDE validation grid.

Compared with Black-Scholes, Heston introduces mean reversion, long-run
variance, vol-of-vol, and spot/variance correlation. Compared with the
local-vol and eSSVI tooling in this repository, Heston gives a stochastic
variance story rather than a pure vanilla-surface fit.

This library currently keeps the Heston stack namespaced. At present, the
package root `option_pricing` does not re-export the Heston pricers or
calibrators, so expect imports from:

- `option_pricing.models.heston`
- `option_pricing.pricers.heston`
- `option_pricing.pricers.heston_mc`
- `option_pricing.models.heston.calibration`
- `option_pricing.diagnostics.heston`

The practical workflow is split into pricing, implied-vol smile generation,
quadrature policy, Monte Carlo, calibration, diagnostics, and model
comparison.

For generated signatures and exact import paths across that whole stack, use
the [Heston API reference](../api/heston.md). It covers the namespaced
parameter object, Fourier pricers, Monte Carlo pricers, simulation primitives,
calibration helpers, quote-set container, diagnostics, and model-comparison
entrypoints without adding root-level exports.

## Parameters

The implementation parameter object is
`option_pricing.models.heston.HestonParams`. Its exact field names are:

| Literature notation | Repo name | Meaning |
|---|---|---|
| $\kappa$ | `kappa` | mean-reversion speed of variance |
| $\theta$ | `vbar` | long-run variance |
| $\xi$ or $\sigma_v$ | `eta` | volatility of variance / vol-of-vol |
| $\rho$ | `rho` | spot/variance Brownian correlation |
| $v_0$ | `v` | initial variance |

The repo uses `vbar`, `eta`, and `v`, while many texts write `theta`, `xi`,
and `v0`. Those literature names are notation only here; use the repo names
when calling the library.

`HestonParams` validates the basic admissibility conditions:

- `kappa > 0`
- `vbar >= 0`
- `eta >= 0`
- `v >= 0`
- `rho in [-1, 1]`

The class does not enforce the Feller condition. Calibration is therefore
allowed to explore outside that region. Fit diagnostics report the Feller value,
margin, and status by default, and calibration regularization can add an
optional soft Feller penalty without hard-blocking violations.

For routine calibration, use `HestonCalibrationBounds` to keep the optimizer in
a practical box. Those bounds are optimizer safeguards, not a proof of
identifiability, no-arbitrage, or model correctness. Multistart calibration
defaults to the bounded parameter transform for exactly that reason, and
single-start `calibrate_heston(...)` now uses the same bounded public default.

## Pricing

The vanilla pricing entrypoints live in
`option_pricing.pricers.heston`:

- `heston_price_call_from_ctx(...)`
- `heston_price_put_from_ctx(...)`
- `heston_price_from_ctx(...)`
- `heston_price_instrument_from_ctx(...)`
- `heston_price_instrument(...)`

`heston_price_from_ctx(...)` dispatches by `OptionType`; the explicit call and
put helpers are there when you want the formula-specific surface.
`heston_price_instrument(...)` and `heston_price_instrument_from_ctx(...)` are
the instrument-based wrappers for `VanillaOption`.

Implementation convention: the pricer separates the strike/cash leg from the
forward/asset leg. Probability index `0` is `P_K`, probability index `1` is
`P_F`, and the call helper prices vanilla options as
`df * (forward * P_F - strike * P_K)`.

The default backend is fixed-rule Gauss-Legendre integration. That is the
production-oriented path for ordinary pricing and calibration because it is
deterministic and vectorized across strike slices. The `quad` backend is still
available as an adaptive scalar cross-check.

Analytic parameter-Jacobian helpers currently support only the guarded fixed
Gauss-Legendre production domain: `backend="gauss_legendre"`, nonzero fixed-rule
quadrature nodes, `eta >= HESTON_ANALYTIC_JAC_ETA_MIN` (`1e-6`), and parameters
inside the documented calibration bounds. They return columns in the order
`[kappa, vbar, eta, rho, v]`. Scalar strikes return Jacobian shape `(5,)`;
array-valued strikes return `strike.shape + (5,)`. Ordinary price-only helpers
continue to support both `gauss_legendre` and `quad`, including the
deterministic-variance pricing limit near `eta=0`.

```python
import numpy as np

from option_pricing import MarketData, OptionType
from option_pricing.models.heston import HestonParams
from option_pricing.pricers.heston import heston_price_from_ctx

ctx = MarketData(spot=100.0, rate=0.02, dividend_yield=0.0).to_context()
params = HestonParams(kappa=1.5, vbar=0.04, eta=0.50, rho=-0.70, v=0.04)

call_slice = heston_price_from_ctx(
    kind=OptionType.CALL,
    strike=np.array([90.0, 100.0, 110.0]),
    tau=1.0,
    ctx=ctx,
    params=params,
)
```

Call and put pricing are both supported. For array-valued strikes, the
Gauss-Legendre backend batches the slice directly; the `quad` backend accepts
arrays for API consistency but still loops over scalar integrals under the
hood.

See the [pricing convention note](../notes/heston/heston_pricing_conventions.md) for
the exact `log(F / K)` convention, semantic `P_K`/`P_F` probability-index
mapping, eta-near-zero deterministic fallback, and call/put formulas.

## Implied-vol smile workflow

The library does not expose a special Heston-only implied-vol wrapper. The
normal workflow is:

1. price a strike slice with `heston_price_from_ctx(...)`;
2. build one `OptionSpec` per quote;
3. invert each price with `implied_vol_bs(...)`.

```python
from option_pricing import OptionSpec, OptionType, implied_vol_bs

strikes = np.array([90.0, 100.0, 110.0], dtype=float)
prices = heston_price_from_ctx(
    kind=OptionType.CALL,
    strike=strikes,
    tau=1.0,
    ctx=ctx,
    params=params,
)

smile = [
    implied_vol_bs(
    mkt_price=float(price),
    spec=OptionSpec(kind=OptionType.CALL, strike=float(strike), expiry=1.0),
    market=ctx,
    )
    for strike, price in zip(strikes, prices, strict=True)
]
```

That inversion is scalar by design, so a smile grid is usually built with a
small loop or list comprehension. For calibration or diagnostics, keep the
pricing backend and quadrature settings fixed while you generate the smile so
the inversion reflects model behavior rather than mixed numerical settings.

## Quadrature and numerical policy

The numerical-policy note for this stack is the
[Heston quadrature policy](../notes/heston/heston_quadrature_policy.md). The high-level
policy in this library is:

- fixed Gauss-Legendre is the default production backend;
- `quad` is a cross-check backend, not the default calibration engine;
- quality tiers are `fast`, `balanced`, `robust`, and `diagnostics`.

Use `balanced` for ordinary user-facing pricing, `robust` for final calibration
tables and plots, and at least one `diagnostics` rerun for headline fitted
parameters or warning review. Lighter settings are for quick interactive
exploration only. The low-level recommender is
`recommend_heston_quadrature_config(...)` in `option_pricing.models.heston`.

At the user level, the hard red flags are non-finite integrals, non-finite
probabilities, probabilities materially outside `[0, 1]`, or persistent backend
disagreement after robust/diagnostics reruns. Tail, cancellation, oscillation,
and near-origin warnings require review. Calibration summaries report whether
quotes were blocked, quarantined, reviewed, or retained.

## Monte Carlo

The public Monte Carlo entrypoints live in `option_pricing.pricers.heston_mc`:

- `heston_mc_price_from_ctx(...)`
- `heston_mc_price(...)`
- `heston_mc_price_instrument(...)`
- `heston_mc_price_instrument_from_ctx(...)`
- `heston_mc_price_path_payoff_from_ctx(...)`
- `heston_mc_price_with_vanilla_control_from_ctx(...)`

This library currently uses full-truncation Euler as the baseline scheme and
Andersen QE (`scheme="quadratic_exponential"`) as the production-oriented
default. Use Euler when you want a simple baseline or an educational
comparison; use QE when the goal is a better production Monte Carlo workflow at
practical step counts.

```python
from option_pricing import MCConfig, OptionType
from option_pricing.pricers.heston_mc import heston_mc_price_from_ctx

mc_result = heston_mc_price_from_ctx(
    ctx=ctx,
    kind=OptionType.CALL,
    strike=100.0,
    tau=1.0,
    params=params,
    n_steps=64,
    scheme="quadratic_exponential",
    cfg=MCConfig(n_paths=20_000, antithetic=True),
)

ci95 = mc_result.confidence_interval()
```

The return type is `MonteCarloResult`, so standard errors are available through
`mc_result.stderr` and normal-approximation confidence intervals through
`mc_result.confidence_interval(...)`. Antithetic sampling is supported through
`MCConfig(antithetic=True)`. A semi-analytic vanilla control variate is also
available, but the diagnostics workflow intentionally keeps that control off by
default when the point is to compare raw Monte Carlo behavior against the
Fourier price.

See the [Monte Carlo note](../notes/heston/heston_monte_carlo.md) and the
[Andersen QE note](../notes/heston/heston_qe_notes.md) for the scheme-level details
and validation rationale.

## Calibration

Calibration helpers are namespaced under `option_pricing.models.heston.calibration`.
The calibrators themselves are re-exported from that package, but
`HestonQuoteSet` currently lives in
`option_pricing.models.heston.calibration.heston_types`.

A `HestonQuoteSet` stores:

- required market context and quote arrays: `ctx`, `strike`, `expiry`,
    `is_call`, `mid`;
- optional calibration fields: `bs_vega`, `sqrt_weights`, `bid`, `ask`,
    `iv_mid`, `labels`.

`discount` and `forward` slices are derived from `ctx`, so you do not pass them
separately. At the user level:

- provide `bs_vega` if you use `objective_type="vega_scaled_price"`;
- provide `bid` and `ask` if you use
    `objective_type="bid_ask_normalized"`;
- provide `iv_mid` if you want the built-in seed heuristics and IV diagnostics
    to work from actual quoted IVs rather than only prices.

The current calibration/reporting split is:

| Quantity | Status | Interpretation |
|---|---|---|
| `"price_rmse"` | available calibration objective | direct price residual objective |
| `"relative_price_rmse"` | available calibration objective | price residual scaled by the target price |
| `"vega_scaled_price"` | available calibration objective and the current default | price residual scaled by Black-Scholes vega to approximate IV-error behavior without repeated IV inversion |
| `"bid_ask_normalized"` | available calibration objective | residual scaled by bid-ask width |
| direct IV-RMSE optimization | not implemented in this library | do not claim calibration directly minimizes IV RMSE |
| reported IV residuals / IV RMSE | reporting metric | evaluated after repricing and Black inversion when diagnostics are available |

When describing the project, say that calibration uses vega-scaled price
residuals and reports implied-volatility residual diagnostics; do not describe
this as direct IV-RMSE optimization unless such an objective is explicitly
implemented.

There is no direct IV-RMSE calibration objective in this library.
`"vega_scaled_price"` stays in price space and uses Black-Scholes vega as the
residual scale, while IV residuals and IV RMSE are reported later by the fit
diagnostics after repricing and Black inversion.

For starting points, use `default_heston_seed(...)` when you want one
market-aware seed and `heston_seed_grid(...)` when you want the deterministic
multistart spoke set. `calibrate_heston(...)` now defaults to
`parameter_transform="bounded"` because vanilla-only Heston calibration is
weakly identifiable and the safer public single-start path should stay inside a
practical box. `parameter_transform="unconstrained"` remains available for
low-level experiments. For production-style workflows, prefer
`calibrate_heston_multistart(...)`, which also defaults to
`parameter_transform="bounded"` and `max_seeds=12`.

Invalid calibration inputs raise before optimization. In multistart mode,
failed seeds are retained when at least one seed succeeds because they are
useful diagnostics about initialization sensitivity. If every seed fails,
`calibrate_heston_multistart(...)` raises `NoConvergenceError` instead of
returning a result without a usable `best_run` and `best_params`.

```python
from option_pricing.models.heston.calibration import (
    HestonCalibrationBounds,
    calibrate_heston_multistart,
)

result = calibrate_heston_multistart(
    quotes=quotes,
    objective_type="vega_scaled_price",
    bounds=HestonCalibrationBounds(),
    max_seeds=8,
    parameter_transform="bounded",
)
```

Use the [calibration evidence note](../notes/heston/heston_calibration.md) for fit
interpretation and the [seed-design note](../notes/heston/heston_calibration_seeds.md)
for the default seed and multistart heuristics.

## Diagnostics

The notebook-facing diagnostics live in `option_pricing.diagnostics.heston`.
The main report families are:

- pricing stability and backend review:
    `run_heston_pricing_diagnostics(...)`,
    `run_heston_slice_diagnostics(...)`, `compare_backend_slice(...)`,
    `price_slice_with_diagnostics(...)`, and
    `probability_slice_with_diagnostics(...)`;
- calibration fit evidence: `run_heston_calibration_fit_diagnostics(...)` and
    `run_heston_calibration_diagnostics(...)`;
- Monte Carlo comparison: `run_heston_mc_comparison_sweep(...)`,
    `summarize_bias_vs_timestep(...)`, `summarize_runtime_vs_error(...)`, and
    `compare_heston_mc_schemes(...)`;
- synthetic benchmark and Jacobian smoke or release coverage:
    `build_synthetic_heston_quote_set(...)` and
    `run_heston_calibration_benchmark_diagnostics(...)`;
- deterministic model-comparison fixture:
    `build_market_like_heston_quote_set(...)`;
- model comparison: `run_heston_vs_local_vol_comparison(...)`.

Use benchmark diagnostics as smoke, regression, or synthetic repricing-recovery
evidence.
Do not present them as the same thing as final calibration quality on live
quotes. The artifact builder emits separate smoke and release manifests under
`benchmarks/artifacts/heston_calibration/` so the lightweight wiring benchmark
and the stronger release benchmark stay distinguishable.

Synthetic calibration checks in this repo are primarily repricing-recovery
diagnostics. They validate pricing/calibration wiring and residual behavior on
Heston-generated quotes; they do not prove unique recovery of the generating
parameters. Similar vanilla repricing errors can arise from different
parameter vectors, and multistart diagnostics expose that sensitivity rather
than making the fit unique.

The detailed notebook-oriented walkthrough is in the
[Heston diagnostics guide](heston_diagnostics.md).

## Heston versus local volatility / eSSVI comparison

`run_heston_vs_local_vol_comparison(...)` compares one fitted Heston model
against eSSVI implied-surface repricing and a direct local-vol PDE repricing
audit on the same `HestonQuoteSet` target. The final capstone target should use
`build_market_like_heston_quote_set(...)`, which returns a deterministic
market-like synthetic fixture rather than market data or a Heston-generated
quote target. The report packages quote-level price and IV residuals, direct
PDE rows, ATM and wing buckets, optional held-out splits, and a concise
trade-off summary.

[Open the comparison notebook](https://github.com/willemk-stack/option-pricing-library/blob/main/demos/13_heston_calibration_vs_localvol.ipynb){ .md-button .md-button--primary }

The deeper written discussion is in the
[comparison note](../notes/heston/heston_vs_local_vol.md).

Read the trade-off honestly:

- Heston is the interpretable stochastic-variance model: mean reversion,
    long-run variance, vol-of-vol, correlation, and initial variance all have
    direct model meaning.
- eSSVI/local-vol-facing tools are the flexible vanilla-fit side: they can
    often match quoted smile nodes more closely, but they do not tell the same
    stochastic-variance story.
- Direct local-vol PDE rows audit a small validation grid. They are useful
    numerical evidence, not a global proof of local-vol accuracy across every
    boundary, extrapolation, and grid regime.

## Limitations and when to distrust results

- Heston calibration from vanilla quotes can be weakly identifiable.
- Multistart can expose sensitivity to initialization, but it does not make the
    parameters unique.
- A good vanilla fit does not prove the model's dynamics are correct.
- Numerical integration warnings and backend disagreement should be
    investigated, not waved away.
- Monte Carlo results still contain discretization error and sampling error,
    even when the confidence interval looks tight.
- The direct local-vol PDE comparison is intentionally small; distinguish model
    residuals from PDE grid, boundary, and projection error.
- Nothing in this guide should be read as a production-trading claim.

For bounded interview/CV wording, see the
[Heston Capstone interview framing](../notes/heston/heston_interview_framing.md)
note.

## Minimal workflow checklist

1. Price a vanilla call/put slice with the Gauss-Legendre Fourier pricer.
2. Invert those prices to Black implied vols if you want a smile view.
3. Build a `HestonQuoteSet` and run `calibrate_heston_multistart(...)`.
4. Run `run_heston_calibration_fit_diagnostics(...)` on the fitted result.
5. Review held-out errors separately when a held-out mask exists.
6. Compare against eSSVI and the direct local-vol PDE validation grid with
     `run_heston_vs_local_vol_comparison(...)`.
7. Re-read the limitations before drawing capstone conclusions.
