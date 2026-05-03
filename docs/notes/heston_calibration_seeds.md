# Heston calibration seed design

This note explains the default Heston calibration seed used by
`default_heston_seed` and the compact diagnostic seed grid exposed by
`heston_seed_grid`.

The seed is not a parameter estimator. It is a deterministic, market-aware
starting point for nonlinear least-squares calibration. The final fitted
parameters should be judged with calibration diagnostics, sensitivity checks,
and multi-start calibration.

## Inputs

The seed expects a `HestonQuoteSet` with mid implied volatilities:

- `mid`: option mid prices, usually `(bid + ask) / 2`
- `iv_mid`: Black-76 / Black-Scholes implied volatilities corresponding to those
  mid prices
- `strike`
- `expiry`
- forward and discount information through the pricing context

The seed uses implied volatilities rather than raw option prices because IVs are
more comparable across strikes and maturities.

## ATM variance extraction

For each expiry, the seed estimates an ATM implied volatility at

```text
log_moneyness = log(K / F) = 0
```

The procedure is:

1. group quotes by expiry,
2. collapse duplicate log-moneyness points, for example call/put duplicates,
3. if an exact ATM quote exists, use its median IV,
4. if the slice brackets ATM, linearly interpolate IV at `log_moneyness = 0`,
5. otherwise use the closest few quotes as a fallback,
6. square the ATM IV to obtain ATM implied variance.

This gives a rough ATM variance term structure.

## Initial variance `v`

The Heston initial variance seed is:

```text
v = shortest-expiry ATM implied variance
```

This is intuitive because short-dated ATM variance is the most direct market
proxy for the current variance level.

## Long-run variance `vbar`

The long-run variance seed is based on the longer-dated ATM variance:

```text
vbar = 0.70 * long_expiry_ATM_variance + 0.30 * median_ATM_variance
```

The blend makes the seed less sensitive to one noisy far-expiry quote. If only
one expiry is available, the seed uses the median ATM variance.

This is a heuristic proxy, not an estimate of the true long-run variance.

## Mean reversion `kappa`

The default seed uses a moderate fixed value:

```text
kappa = 1.50
```

The seed deliberately does not infer `kappa` aggressively from the ATM variance
term structure.

Reason: `kappa` is often weakly identified from vanilla option prices alone.
A seemingly clever formula can be misleading. For example, a large short-long
variance difference does not automatically imply fast mean reversion; in many
interpretations, fast mean reversion would make short- and long-horizon variance
levels more similar.

The intended workflow is:

1. use a moderate deterministic default seed,
2. run multi-start calibration with slow, medium, and fast `kappa` values,
3. report parameter sensitivity.

The compact seed grid therefore includes:

```text
kappa in [0.50, 1.50, 3.00, 6.00]
```

## Correlation `rho`

The seed estimates near-ATM implied-volatility skew:

```text
skew = dIV / dlog_moneyness
```

A negative skew means IV is higher for lower strikes. In equity-like markets,
that usually corresponds to negative spot/variance correlation, so the seed maps
skew to `rho` with:

```text
rho = tanh(rho_skew_scale * skew)
```

The hyperbolic tangent keeps the seed inside a valid correlation range and avoids
starting too close to impossible values.

If skew cannot be estimated, the default fallback is neutral:

```text
fallback_rho = 0.0
```

For equity or index calibration, a caller may choose:

```text
fallback_rho = -0.30
```

This keeps the default generic while allowing an explicit equity-style prior.

## Vol-of-vol `eta`

The seed uses a moderate base level and increases it for stronger near-ATM skew:

```text
eta = 0.50 + min(1.00, 2.00 * abs(skew))
```

This is intentionally rough. In Heston, `rho` and `eta` jointly affect skew and
smile curvature, so the seed should not try to identify them separately. The
optimizer and multi-start diagnostics should resolve the fit.

## Bounds

After computing the raw seed, all parameters are clipped into the practical
`HestonCalibrationBounds`.

These bounds are optimizer safeguards, not no-arbitrage guarantees. The Feller
condition is handled separately through diagnostics or optional soft
regularization.

## Compact multi-start grid

`heston_seed_grid` is a deterministic extension of the default seed. It is not a
large Cartesian grid. A full cross-product over `kappa`, `vbar`, `eta`, and
`rho` would create many starts that mostly repeat the same weakly identified
tradeoffs while increasing calibration time.

The grid keeps the default seed first and then adds one-dimensional spokes:

- `kappa` values for slow, moderate, fast, and very fast mean reversion,
- `vbar` multipliers around the market-implied long-run variance proxy,
- `eta` multipliers around the skew-sensitive vol-of-vol proxy,
- `rho` offsets around the skew-implied correlation,
- a few skew-heavy combined starts that pair stronger `eta` with directional
  `rho` values.

This shape is deliberate. Heston vanilla calibration often has shallow
directions where different combinations of mean reversion, long-run variance,
vol-of-vol, and correlation explain similar smile features. The seed grid is
meant to expose those optimizer sensitivities without pretending the initial
grid itself identifies the model.

Every proposed start is clipped into `HestonCalibrationBounds` and deduplicated
after clipping. That matters because tight user bounds can collapse several
diagnostic spokes to the same feasible point. Returning unique bounded seeds
keeps multi-start runs compact and makes downstream sensitivity reports easier
to read.

The default cap is small:

```text
max_seeds = 12
```

This keeps routine calibration affordable while still covering the main weakly
identified directions. Callers that are doing a dedicated sensitivity study can
set `max_seeds=None` to inspect every unique proposed start.

## Defensible explanation

The default seed is intentionally simple:

- `v` comes from short-dated ATM implied variance,
- `vbar` comes from longer-dated ATM implied variance,
- `kappa` is a moderate fixed value because it is weakly identified,
- `rho` uses the sign of near-ATM skew,
- `eta` starts moderate and increases with skew strength.

The seed and seed grid are designed to make calibration runnable without
hand-picking parameters. They are not designed to prove that the resulting
parameters are unique or economically meaningful. That responsibility belongs to
calibration diagnostics: residual plots, synthetic recovery, objective slices,
and multi-start sensitivity.
