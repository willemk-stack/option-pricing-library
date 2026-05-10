# Heston Capstone Interview Framing

## Safe CV wording

Implemented and validated a Heston stochastic-volatility module: stable
Fourier vanilla pricing with reference-price tests, Andersen QE Monte Carlo
cross-checks with confidence intervals, bounded/multistart calibration
tooling, and diagnostic comparison against an eSSVI/local-vol workflow on
deterministic synthetic fixtures.

## Claims to avoid

- Production Heston calibration engine.
- Market-data validated Heston model.
- Direct IV-RMSE calibration, unless it is implemented later.
- Heston universally outperforms local volatility.

## Sixty-second explanation

I used Heston to add a stochastic-volatility model on top of the existing
vanilla, eSSVI, and local-vol stack. The implementation focuses on stable
semi-analytic Fourier pricing for European vanillas, then uses Andersen QE
Monte Carlo as a dynamic cross-check instead of treating one pricing method as
enough on its own. Calibration is bounded and multistart because vanilla-only
fits can be weakly identified, so the project emphasizes diagnostics and
residual review rather than claiming a single optimizer run proves the model.
The comparison against eSSVI and local volatility is framed as model-choice
evidence on deterministic synthetic fixtures, not as a claim that one model
always wins.

## What the implementation demonstrates

- Stable Fourier pricing conventions with documented probability-leg mapping.
- Reference-price regression discipline plus deterministic synthetic fixtures.
- Andersen QE Monte Carlo cross-checks with confidence-interval reporting.
- Calibration tooling that keeps bounds, multistart behavior, and weak
    identifiability visible.
- Careful comparison between an interpretable stochastic-volatility model and a
    flexible vanilla-surface workflow.

## What I would improve next

- Add broader held-out and stress-test protocols around calibration stability.
- Add a separate market-data validation track before making any live-market
    calibration claims.
- Consider a direct IV-space objective only if it is implemented and validated
    as a distinct optimization path.
- Expand the comparison grid so local-vol PDE error and model error stay easier
    to separate.