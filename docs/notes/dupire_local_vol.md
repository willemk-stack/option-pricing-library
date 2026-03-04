# Dupire local vol

**goals**
- describe the "problem"/aims

## Context

Black-Scholes is a popular instrument in the world of option-pricing, partly due to it's simplicity in pricing vanilla options. However, it's simplicity is also it's limiting factor. The assumption of the underlying with a constant volatility can hardly ever be true. One attempt at a more realistic moedl is described by dupire. His approach assumes the underlying asset follows the following stochastic-process:

$$
\frac{dS}{S} = \mu _t,\dt + \sigma,\(S_t, t; S_0),\dZ
$$

Where $\mu _t = r(t) - D(t)$ is the risk-neutral drift with $r(t)$, $D(t)$ the risk-free and dividend rate.

We try to find a function for the local volatility so that our risk-neutral price process generates (maybe be more specific) all observed market prices.


## Core dupire local vol math

### Explain getting neutral density func from (euro) option prices
Given a range of call prices $C(K, T)$, for a fixed maturity T, we can calculate the risk-neutral density. following from the definition

- Why the risk-neutral density must satisfy the Fokker-Planck equation
- Marginal distributions vs term-structure / time evolution
    - Different diffusion processes can generate the same distribution (See Dupire), however the joint distrubions at different times of the processes can vary greatly => affects hedging ($\frac{dC}{dS}$) => affects pricing
- Local vol from call prices formula (bs form, black form, $x = \log(K)$ log strike)

### Engeneering notes / Dupire local vol approximation from a call grid

## Model choices
- Why interpolation in log strike
    - Provide formula
    - Convexity ()
    - No explicit K in denumerator (can amplify num noise)
- Masking boundaries?


## 
### Check where to fit in

1.  Dupire formula + what’s being differentiated (call PV vs forward price)
1. SVI, SSVI parametrization.
    - Fitting unconstrained (softplus ect ect)

## Assumptions when pricing under local vol
When pricing under local vol we assume a deterministic volatility a function of only spot price and time to expiry given a fixed strike prices. We also assume a complete market.

## SVI Calibration

1.  Your guardrails (trim, denom floor, curvature threshold, NaN masking) — what each prevents
1.  Why derivatives are fragile (piecewise linear surfaces, wings, calendar kinks)

### Diagnostics
1.  “How to read the diagnostics” — where local vol is unreliable


## Failure modes / what can go wrong
A short “known failure modes” section (noise → spikes, convexity violations, extrapolation artifacts)

If you must do finite differences, use smoothing that penalizes curvature “wiggles,” e.g.


### Interesting further reading
Tikhonov / ridge-type penalties on second derivatives,
smoothing splines,
filtering (carefully, because filtering can violate arbitrage constraints unless imposed jointly).


### References:
PRICING AND HEDGING WITH SMILES, Bruno Dupire
Jim Gatheral - The Volatility Surface
