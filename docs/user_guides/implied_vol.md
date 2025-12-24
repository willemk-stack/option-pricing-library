# Implied Volatility user guide (`implied_vol`)

This guide explains how to compute **Black–Scholes implied volatility (IV)** in this library, and how to use the diagnostics that come with **Option A** (root-finders return `RootResult`).

---

## What “implied volatility” means here

Given:

- an option specification (call/put, strike, expiry),
- market data (spot, risk-free rate, dividend yield),
- a **market option price**,

the implied volatility is the value of `sigma` such that the **Black–Scholes** model price equals the market price.

This module assumes the standard Black–Scholes setup with:

- continuous risk-free rate `r`,
- continuous dividend yield `q`,
- European options.

Time is measured in **year fractions** (e.g. 0.5 = 6 months).

---

## Key types

```python
from option_pricing import MarketData, OptionSpec, OptionType
```

- `MarketData(spot, rate, dividend_yield=0.0)`
- `OptionSpec(kind, strike, expiry)`
  - `kind`: `OptionType.CALL` or `OptionType.PUT`
  - `expiry`: absolute expiry time (same “clock” as `t` below)

The IV functions accept a `t` parameter (default `0.0`) and compute:

- `tau = expiry - t`

If `tau <= 0`, a `ValueError` is raised.

---

## Root-finder contract (Option A)

Under Option A, **all root-finders return `RootResult`**:

```python
from option_pricing.numerics.root_finding import RootResult, bracketed_newton
```

A `RootResult` contains (at minimum):

- `root`: the solution
- `converged`: did it converge?
- `iterations`: number of iterations used
- `method`: name of the algorithm
- `f_at_root`: the residual at the returned root
- `bracket`: final bracket (if the method maintains one)

This lets you keep iteration counts and convergence diagnostics all the way through IV.

---

## Main APIs

```python
from option_pricing.vol.implied_vol import implied_vol_bs, implied_vol_bs_result
```

### `implied_vol_bs_result(...) -> ImpliedVolResult` (recommended)

Returns a rich result that includes the root-finder diagnostics.

**Signature (conceptually):**

```python
implied_vol_bs_result(
    mkt_price: float,
    spec: OptionSpec,
    market: MarketData,
    root_method: Callable[..., RootResult],
    *,
    t: float = 0.0,
    sigma0: float | None = None,
    sigma_lo: float = 1e-8,
    sigma_hi: float = 5.0,
    tol_f: float = 1e-10,
    tol_x: float = 1e-10,
    max_iter: int | None = None,
) -> ImpliedVolResult
```

Where `ImpliedVolResult` contains:

- `vol`: implied volatility (float)
- `root_result`: the `RootResult` from `root_method`
- `mkt_price`: the input market price
- `bounds`: theoretical no-arbitrage bounds used for validation
- `tau`: time to expiry

### `implied_vol_bs(...) -> float` (convenience)

Same inputs as `implied_vol_bs_result`, but returns only the volatility float.

Internally it calls `implied_vol_bs_result(...).vol`.

---

## Minimal example

```python
from dataclasses import replace

from option_pricing import MarketData, OptionSpec, OptionType, PricingInputs, bs_price
from option_pricing.numerics.root_finding import bracketed_newton
from option_pricing.vol.implied_vol import implied_vol_bs_result

market = MarketData(spot=100.0, rate=0.05, dividend_yield=0.0)
spec = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)

sigma_true = 0.20
p_true = PricingInputs(spec=spec, market=market, sigma=sigma_true, t=0.0)
mkt_price = bs_price(p_true)

ivres = implied_vol_bs_result(
    mkt_price=float(mkt_price),
    spec=spec,
    market=market,
    root_method=bracketed_newton,
    sigma0=0.30,
)

print("IV:", ivres.vol)
print("Converged:", ivres.root_result.converged)
print("Iterations:", ivres.root_result.iterations)
print("Residual:", ivres.root_result.f_at_root)
print("Bracket:", ivres.root_result.bracket)
```

---

## No-arbitrage bounds and validation

Before any root-finding starts, the solver checks whether the input price is feasible.

With continuous dividend yield `q`, the solver uses **prepaid-forward bounds**:

Let:

- `tau = expiry - t`
- `df = exp(-r*tau)`
- `Fp = S * exp(-q*tau)`  (prepaid forward)

Then:

- Call bounds: `max(Fp - K*df, 0) <= C <= Fp`
- Put bounds:  `max(K*df - Fp, 0) <= P <= K*df`

If the price is outside bounds (with a tiny numeric slack), the solver raises:

- `InvalidOptionPriceError`

---

## Choosing solver settings

### Bracket: `sigma_lo`, `sigma_hi`

- Defaults are typically fine: `sigma_lo=1e-8`, `sigma_hi=5.0`.
- If you see “not bracketed” errors, your interval may be too small **or** the price might be inconsistent.

If your numerics module provides `ensure_bracket`, you can expand `sigma_hi` automatically before solving.

### Initial guess: `sigma0`

- For `bracketed_newton`, an initial guess helps speed up convergence.
- Common defaults: `0.2` or `0.3`.

### Tolerances: `tol_f`, `tol_x`

- `tol_f`: absolute tolerance on `|Fn(sigma)|` (price residual).
- `tol_x`: tolerance on the sigma step / bracket width.

For most use-cases: `1e-10` is plenty.

---

## Error handling

Typical errors you may see:

- `InvalidOptionPriceError`: market price is outside no-arbitrage bounds.
- `ValueError("Need expiry > t")`: `expiry - t <= 0`.
- `NotBracketedError`: the root wasn’t bracketed in `[sigma_lo, sigma_hi]`.
- `NoConvergenceError` / `DerivativeTooSmallError`: numerical failure inside Newton-like methods.

In diagnostics/benchmark code, it’s common to catch exceptions and record the error message per case.

---

## Best practice

- Use **`implied_vol_bs_result`** in notebooks/benchmarks/tests to keep diagnostics.
- Use **`implied_vol_bs`** in production code if you only need the volatility number.

---

## Quick reference

- **Float one-liner**
  ```python
  iv = implied_vol_bs(price, spec, market, root_method=bracketed_newton)
  ```

- **Diagnostics**
  ```python
  ivres = implied_vol_bs_result(price, spec, market, root_method=bracketed_newton)
  # ivres.vol, ivres.root_result.iterations, ...
  ```
