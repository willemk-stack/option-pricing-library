````md
# Implied Volatility user guide (`implied_vol`)

This guide explains how to compute **Black–Scholes implied volatility (IV)** in this library using the **config-driven (`cfg`) API**, and how to keep solver diagnostics via the returned `RootResult`.

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

Time is measured in **year fractions** (e.g. `0.5 = 6 months`).

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

## Configuration (`ImpliedVolConfig`)

Implied-vol inversion is **config-driven**: solver selection, bracketing interval, tolerances, and seeding are controlled by `ImpliedVolConfig`.

```python
from option_pricing.config import ImpliedVolConfig
from option_pricing.numerics.root_finding import RootMethod
```

Common knobs you may want to adjust:

- `root_method: RootMethod`  
  Which solver to use (default is typically `RootMethod.BRACKETED_NEWTON`).
- `sigma_lo, sigma_hi: float`  
  Volatility search interval.
- `bounds_eps: float`  
  Slack used when validating no-arbitrage bounds.
- `seed_strategy`  
  How to choose an initial guess when `sigma0` is not provided.
- `numerics`  
  Tolerances and iteration limits: `abs_tol`, `rel_tol`, `max_iter`, plus safety clamps (e.g. `min_vega`).

You pass config via the `cfg=` keyword argument to the IV functions.

---

## Root-finder diagnostics (`RootResult`)

All solvers in `option_pricing.numerics.root_finding` return a `RootResult`:

```python
from option_pricing.numerics.root_finding import RootResult
```

A `RootResult` contains (at minimum):

- `root`: the solution
- `converged`: did it converge?
- `iterations`: number of iterations used
- `method`: name of the algorithm
- `f_at_root`: the residual at the returned root
- `bracket`: final bracket (if the method maintains one)

When you call `implied_vol_bs_result`, the returned `ImpliedVolResult` includes the solver’s `RootResult` so you can keep iteration counts and convergence diagnostics.

---

## Main APIs

```python
from option_pricing import implied_vol_bs, implied_vol_bs_result
```

### `implied_vol_bs_result(...) -> ImpliedVolResult` (recommended)

Returns a rich result that includes root-finder diagnostics.

```python
implied_vol_bs_result(
    mkt_price: float,
    spec: OptionSpec,
    market: MarketData,
    *,
    cfg: ImpliedVolConfig | None = None,
    t: float = 0.0,
    sigma0: float | None = None,
) -> ImpliedVolResult
```

Where `ImpliedVolResult` contains:

- `vol`: implied volatility (float)
- `root_result`: the `RootResult` produced by the configured solver
- `mkt_price`: the input market price
- `bounds`: theoretical no-arbitrage bounds used for validation
- `tau`: time to expiry

### `implied_vol_bs(...) -> float` (convenience)

Same inputs as `implied_vol_bs_result`, but returns only the volatility float:

```python
iv = implied_vol_bs(...)
```

Internally it calls `implied_vol_bs_result(...).vol`.

---

## Minimal example

```python
from option_pricing import (
    MarketData, OptionSpec, OptionType,
    PricingInputs, bs_price,
    implied_vol_bs_result,
)
from option_pricing.config import ImpliedVolConfig
from option_pricing.numerics.root_finding import RootMethod

market = MarketData(spot=100.0, rate=0.05, dividend_yield=0.0)
spec = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)

sigma_true = 0.20
p_true = PricingInputs(spec=spec, market=market, sigma=sigma_true, t=0.0)
mkt_price = float(bs_price(p_true))

cfg = ImpliedVolConfig(root_method=RootMethod.BRACKETED_NEWTON)

ivres = implied_vol_bs_result(
    mkt_price=mkt_price,
    spec=spec,
    market=market,
    cfg=cfg,
    sigma0=0.30,  # optional (helps convergence)
)

print("IV:", ivres.vol)
print("Converged:", ivres.root_result.converged)
print("Iterations:", ivres.root_result.iterations)
print("Residual:", ivres.root_result.f_at_root)
print("Bracket:", ivres.root_result.bracket)
```

---

## Handling invalid market prices

When computing implied volatility, `implied_vol_bs` / `implied_vol_bs_result` validate that the market price lies within the **European no-arbitrage bounds** implied by `S`, `K`, `r`, `q`, and `T`. If the price is outside these bounds (within a small tolerance), the functions raise `InvalidOptionPriceError`.

### Bounds (European)

Let `τ = expiry - t`, `df = exp(-rτ)`, `Fp = S * exp(-qτ)` (prepaid forward), and `Kdf = K * df`.

- Call: `max(Fp - Kdf, 0) ≤ C ≤ Fp`
- Put:  `max(Kdf - Fp, 0) ≤ P ≤ Kdf`

### Example: skip/flag bad quotes

```python
from option_pricing import implied_vol_bs
from option_pricing.exceptions import InvalidOptionPriceError

try:
    iv = implied_vol_bs(mkt_price, spec, market, cfg=cfg)
except InvalidOptionPriceError:
    # e.g. drop the quote, log it, or mark missing
    iv = None
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

If the price is outside bounds (with numeric slack `cfg.bounds_eps`), the solver raises:

- `InvalidOptionPriceError`

```python
from option_pricing.exceptions import InvalidOptionPriceError
```

---

## Choosing solver settings

### Root method: `cfg.root_method`

Pick the solver via the enum:

```python
from option_pricing.config import ImpliedVolConfig
from option_pricing.numerics.root_finding import RootMethod

cfg = ImpliedVolConfig(root_method=RootMethod.BRACKETED_NEWTON)
```

`BRACKETED_NEWTON` is a good default: robust and usually fast. Other available methods include:

- `RootMethod.BISECTION`
- `RootMethod.NEWTON`
- `RootMethod.BRACKETED_NEWTON`

### Bracket: `cfg.sigma_lo`, `cfg.sigma_hi`

- Defaults are typically fine: `sigma_lo=1e-8`, `sigma_hi=5.0`.
- If you see bracketing-related errors, widen the interval **or** re-check the input price.

Note: the bracketed solver will attempt to **auto-expand** a bracket internally; if that fails you may see `NoBracketError`.

### Initial guess: `sigma0` and `cfg.seed_strategy`

- If you pass `sigma0`, it will be clamped into `[sigma_lo, sigma_hi]` and used as the initial guess.
- If you don’t pass `sigma0`, the seed is chosen according to `cfg.seed_strategy`:
  - `HEURISTIC`: compute a robust seed from time value (default).
  - `USE_GUESS` or `LAST_SOLUTION`: **requires** that you provide `sigma0`.

### Tolerances and iterations: `cfg.numerics`

The config’s `numerics` controls:

- `abs_tol`: tolerance on the price residual (`|model_price - mkt_price|`)
- `rel_tol`: tolerance on sigma step / bracket width
- `max_iter`: max iterations
- `min_vega`: safety clamp for vega inside Newton-like methods

Defaults are usually sufficient; tune for very tight tolerances or very noisy prices.

---

## Error handling

Typical errors you may see:

- `InvalidOptionPriceError`: market price is outside no-arbitrage bounds.
- `ValueError("Need expiry > t")`: `expiry - t <= 0`.
- Root-finding failures:
  - `NotBracketedError` / `NoBracketError`
  - `NoConvergenceError`
  - `DerivativeTooSmallError`

```python
from option_pricing.exceptions import (
    InvalidOptionPriceError,
    NotBracketedError,
    NoBracketError,
    NoConvergenceError,
    DerivativeTooSmallError,
)
```

In diagnostics/benchmark code, it’s common to catch exceptions and record the error message per case.

---

## Best practice

- Use **`implied_vol_bs_result`** in notebooks/benchmarks/tests to keep diagnostics.
- Use **`implied_vol_bs`** in production code if you only need the volatility number.

---

## Quick reference

- **Float one-liner (default config)**
  ```python
  from option_pricing import implied_vol_bs
  iv = implied_vol_bs(price, spec, market)
  ```

- **Diagnostics (default config)**
  ```python
  from option_pricing import implied_vol_bs_result
  ivres = implied_vol_bs_result(price, spec, market)
  # ivres.vol, ivres.root_result.iterations, ...
  ```

- **Custom config**
  ```python
  from option_pricing import implied_vol_bs_result
  from option_pricing.config import ImpliedVolConfig
  from option_pricing.numerics.root_finding import RootMethod

  cfg = ImpliedVolConfig(root_method=RootMethod.BRACKETED_NEWTON, sigma_hi=8.0)
  ivres = implied_vol_bs_result(price, spec, market, cfg=cfg, sigma0=0.25)
  ```
````
