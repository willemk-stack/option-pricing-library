# Public API

This page documents the *supported*, top-level imports from `option_pricing`.

```python
from option_pricing import (
    # Types
    OptionType,
    OptionSpec,
    MarketData,
    PricingInputs,

    # Pricers
    bs_price_call,
    bs_call_greeks,
    binom_price_call,
    binom_price_put,
    mc_price_call,
    mc_price_put,
)
```

## Core types

### `OptionType`

```python
class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"
```

### `MarketData`

```python
@dataclass(frozen=True, slots=True)
class MarketData:
    spot: float
    rate: float
    dividend_yield: float = 0.0
```

Notes:
- `dividend_yield` exists for future extension; current pricers assume **no dividends**.

### `OptionSpec`

```python
@dataclass(frozen=True, slots=True)
class OptionSpec:
    kind: OptionType
    strike: float
    expiry: float
```

Notes:
- `expiry` is an **absolute** time `T` (typically measured in years).

### `PricingInputs`

```python
@dataclass(frozen=True, slots=True)
class PricingInputs:
    spec: OptionSpec
    market: MarketData
    sigma: float
    t: float = 0.0

    # convenience properties
    @property
    def S(self): ...
    @property
    def K(self): ...
    @property
    def r(self): ...
    @property
    def T(self): ...
```

Notes:
- Time-to-maturity is `tau = T - t`.
- Rates are assumed **continuously compounded**.

## Pricers

All pricers take a `PricingInputs` bundle.

### `bs_price_call(p)`

```python
def bs_price_call(p: PricingInputs) -> float: ...
```

Black–Scholes price of a **European call** (no dividends).

Related (not top-level):

```python
from option_pricing.models.bs import bs_put_from_inputs
```

### `bs_call_greeks(p)`

```python
def bs_call_greeks(p: PricingInputs) -> dict[str, float]: ...
```

Analytic call Greeks under Black–Scholes.

Returned keys:
- `price`, `delta`, `gamma`, `vega`, `theta`

### `binom_price_call(p, n_steps)` / `binom_price_put(p, n_steps)`

```python
def binom_price_call(p: PricingInputs, n_steps: int) -> float: ...

def binom_price_put(p: PricingInputs, n_steps: int) -> float: ...
```

Cox–Ross–Rubinstein (CRR) binomial model for **European** options.

### `mc_price_call(p, n_paths, seed=None, rng=None)`

```python
def mc_price_call(
    p: PricingInputs,
    n_paths: int,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]: ...
```

Monte Carlo under risk-neutral GBM. Returns `(price, standard_error)`.

### `mc_price_put(p, n_paths)`

```python
def mc_price_put(p: PricingInputs, n_paths: int) -> tuple[float, float]: ...
```

Monte Carlo put pricer. Returns `(price, standard_error)`.

## Optional / advanced modules

These modules are useful in notebooks, but may require extra dependencies such as `matplotlib` and/or `pandas`:

- `option_pricing.diagnostics.*`
- `option_pricing.plotting.*`
