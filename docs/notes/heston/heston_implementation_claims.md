# Heston Implementation Claims and Limitations

This note summarizes what the Heston implementation demonstrates and where the
claim boundary should stay. Read it as public limitation discipline for the
Heston proof path, not as production market-data evidence.

## Supported implementation claims

The repository includes a Heston stochastic-volatility module with:

- stable Fourier vanilla pricing with reference-price tests
- Andersen QE Monte Carlo cross-checks with confidence intervals
- bounded and multistart calibration tooling
- diagnostic comparison against an eSSVI/local-vol workflow on deterministic
  synthetic fixtures
- generated artifacts that expose calibration residuals, stability checks, and
  model-comparison tradeoffs

These claims are strongest when tied to the committed synthetic fixtures,
published diagnostics, and regression tests.

## Claim boundaries

Avoid reading the current implementation as evidence for:

- a production Heston calibration engine
- live-market validation of Heston parameters
- direct IV-RMSE calibration unless that objective is implemented and validated
  separately
- universal Heston outperformance versus local volatility or eSSVI workflows
- trading performance, strategy quality, or empirical market conclusions

The comparison against eSSVI and local volatility is model-choice evidence on
deterministic synthetic fixtures. It is not a claim that one model always wins.

## Diagnostic interpretation

Heston adds an interpretable stochastic-volatility model on top of the existing
vanilla, eSSVI, and local-vol stack. The implementation focuses on stable
semi-analytic Fourier pricing for European vanillas, then uses Andersen QE Monte
Carlo as a dynamic cross-check instead of treating one pricing method as enough
on its own.

Calibration is bounded and multistart because vanilla-only fits can be weakly
identified. The diagnostics therefore emphasize residual review, parameter
stability, and held-out behavior where available rather than claiming that a
single optimizer run proves the model.

## What the implementation demonstrates

- Stable Fourier pricing conventions with documented probability-leg mapping.
- Reference-price regression discipline plus deterministic synthetic fixtures.
- Andersen QE Monte Carlo cross-checks with confidence-interval reporting.
- Calibration tooling that keeps bounds, multistart behavior, and weak
  identifiability visible.
- Careful comparison between an interpretable stochastic-volatility model and a
  flexible vanilla-surface workflow.

## Future validation directions

- Add broader held-out and stress-test protocols around calibration stability.
- Add a separate market-data validation track before making any live-market
  calibration claims.
- Consider a direct IV-space objective only if it is implemented and validated
  as a distinct optimization path.
- Expand the comparison grid so local-vol PDE error and model error stay easier
  to separate.
