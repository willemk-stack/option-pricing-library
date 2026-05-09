# Dupire local vol

!!! note "Status: mathematical background and repository policy"
    Dupire identities are mathematical background. Statements about smooth
    eSSVI handoff, masking, trimming, and downstream diagnostics are repository
    implementation policy unless they are linked to validation evidence.

Dupire local volatility replaces the constant-volatility assumption with a deterministic surface
\(\sigma_{\mathrm{loc}}(S,t)\) chosen so that the model reproduces a continuum of European vanilla prices.

## Role in the library

This note is the mathematical handoff between the volatility-surface workflow
and the PDE validation path. Raw SVI slices may be useful calibration evidence,
but Dupire differentiation needs a smoother cross-maturity surface. The
[eSSVI calibration design](../volatility/essvi_calibration_design.md),
[smooth handoff guide](../../user_guides/essvi_smooth_handoff.md), and
[local-vol/PDE validation guide](../../user_guides/localvol_pde_validation.md)
document that reviewed path.

## Context

Under the risk-neutral measure, the local-vol model writes

\[
dS_t = \bigl(r(t)-q(t)\bigr) S_t\,dt + \sigma_{\mathrm{loc}}(S_t,t)\,S_t\,dW_t^{\mathbb{Q}},
\qquad S_0 > 0.
\]

Here \(r(t)\) is the risk-free rate and \(q(t)\) is the dividend yield. The objective is not to forecast future
realized volatility. It is to find a deterministic diffusion that matches the market's vanilla option prices across
strike and maturity.

## Core Dupire math

### From call prices to density

Let \(C(K,T)\) be the discounted price of a European call with strike \(K\) and maturity \(T\). In the absence of
static arbitrage:

- \(C(K,T)\) is decreasing in \(K\),
- \(C(K,T)\) is convex in \(K\),
- and the curvature \(C_{KK}(K,T)\) is proportional to the risk-neutral density.

That convexity matters because Dupire uses second strike derivatives. A surface that is visually smooth but not
convex enough can produce negative or unstable local variance.

### Dupire formula in strike space

For discounted call prices, the codebase uses

\[
\sigma_{\mathrm{loc}}^2(K,T)
=
\frac{2\left(C_T + \bigl(r(T)-q(T)\bigr)K C_K + q(T) C\right)}
{K^2 C_{KK}}.
\]

If you work with forward or undiscounted call prices \(c(K,T)\), the numerator changes to

\[
\sigma_{\mathrm{loc}}^2(K,T)
=
\frac{2\left(c_T + \bigl(r(T)-q(T)\bigr)\bigl(K c_K - c\bigr)\right)}
{K^2 c_{KK}}.
\]

### Why the code often works in log strike

With \(x = \log K\), the derivatives satisfy

\[
K C_K = C_x,
\qquad
K^2 C_{KK} = C_{xx} - C_x.
\]

So the discounted-call formula becomes

\[
\sigma_{\mathrm{loc}}^2(x,T)
=
\frac{2\left(C_T + \bigl(r(T)-q(T)\bigr) C_x + q(T) C\right)}
{C_{xx} - C_x}.
\]

This is often numerically preferable because it removes the explicit \(K^2\) factor from the denominator. In noisy
wing data, that can reduce avoidable amplification.

## Why matching marginals is not the whole story

Breeden-Litzenberger tells you about the marginal distribution at one maturity. Dupire is stronger: it specifies a
full time-inhomogeneous diffusion. Two models can agree on a single maturity distribution and still produce very
different dynamics between maturities. That difference matters for hedging, path-dependent pricing, and PDE behavior.

## Engineering notes

- Local vol is derivative-hungry. You need a surface that is stable in maturity and sufficiently smooth in strike.
- Piecewise-linear behavior in total variance can make \(w_T\) piecewise constant, which shows up as visible banding
  in local vol.
- That is why the docs recommend a smooth eSSVI projection for Dupire-oriented workflows rather than a raw stack of
  repaired slices.
- Boundary points are the least trustworthy part of the surface. Trimming and masking are a feature, not a failure.

## Assumptions behind local-vol pricing

- The market is modeled as complete enough for a deterministic local volatility surface to be meaningful.
- Vanilla option prices are arbitrage-consistent across strike and maturity.
- The required derivatives exist in a stable numerical sense after interpolation or smoothing.

## Diagnostics and failure modes

The main warning signs are:

- noisy or sign-flipping curvature \(C_{KK}\),
- denominator values that are too small or non-positive,
- negative local variance after differentiation,
- boundary artifacts from sparse wings or short maturity spacing,
- and seam or banding artifacts when \(w_T\) is not time-smooth.

When a local-vol report masks points as invalid, that usually means the differentiation problem is ill-conditioned at
those coordinates, not that the diagnostics are being overly conservative.

## Known limitations

Dupire local volatility is a vanilla-surface model. It reproduces a continuum of
vanilla prices under its assumptions, but it does not by itself validate
path-dependent dynamics, extrapolated wings, sparse maturities, or hedging
behavior outside the calibrated surface. Numerical derivatives remain sensitive
to smoothing, boundary trimming, and quote quality.

## References

- Dupire, B. (1993). Pricing and hedging with smiles.
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley.
