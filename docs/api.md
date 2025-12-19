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
    bs_price,
    bs_greeks,
    mc_price,
    binom_price,

    # Advanced Monte Carlo building blocks
    ControlVariate,
    McGBMModel,
)
```

## Types

### `OptionType`

```python
class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"
```

### `OptionSpec`

```python
@dataclass(frozen=True, slots=True)
class OptionSpec:
    kind: OptionType
    strike: float
    expiry: float
```

`expiry` is an **absolute time** `T` (typically expressed in years). The current time is `t` in `PricingInputs`, and time-to-maturity is `tau = T - t`.

### `MarketData`

```python
@dataclass(frozen=True, slots=True)
class MarketData:
    spot: float
    rate: float
    dividend_yield: float = 0.0
```

### `PricingInputs`

```python
@dataclass(frozen=True, slots=True)
class PricingInputs:
    spec: OptionSpec
    market: MarketData
    sigma: float
    t: float = 0.0
```

`PricingInputs` also exposes convenient properties (`S`, `K`, `r`, `q`, `tau`) derived from the bundle.

## Pricers

### `bs_price(p)`

```python
def bs_price(p: PricingInputs) -> float: ...
```

Black–Scholes price for a European option. Dispatches on `p.spec.kind` (`CALL` / `PUT`).

### `bs_greeks(p)`

```python
def bs_greeks(p: PricingInputs) -> dict[str, float]: ...
```

Analytic Black–Scholes Greeks. Dispatches on `p.spec.kind`.

Returned keys:
- `price`, `delta`, `gamma`, `vega`, `theta`

### `mc_price(p, *, n_paths, ...)`

```python
def mc_price(
    p: PricingInputs,
    *,
    n_paths: int,
    antithetic: bool = False,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]: ...
```

Monte Carlo price under risk-neutral GBM. Returns `(price, standard_error)`.

Dispatches on `p.spec.kind` (`CALL` / `PUT`).

### `binom_price(p, n_steps, *, american=False, method="tree")`

```python
def binom_price(
    p: PricingInputs,
    n_steps: int,
    *,
    american: bool = False,
    method: Literal["tree", "closed_form"] = "tree",
) -> float: ...
```

CRR binomial tree pricing. Dispatches on `p.spec.kind`.

- `american=True` enables early exercise via backward induction (tree method only).
- `method="closed_form"` uses a fast European-only binomial sum (no early exercise).

## Advanced Monte Carlo building blocks

### `ControlVariate`

```python
@dataclass(frozen=True, slots=True)
class ControlVariate:
    values: Callable[[np.ndarray], np.ndarray]
    mean: float
```

Used with `McGBMModel.price_european(..., control=...)` to reduce estimator variance.

### `McGBMModel`

```python
@dataclass(frozen=True, slots=True)
class McGBMModel:
    S0: float
    r: float
    q: float
    sigma: float
    tau: float
    n_paths: int
    antithetic: bool = False
    rng: np.random.Generator = ...
```

Provides:
- `simulate_terminal()` — simulate terminal prices `S_T`
- `price_european(payoff, *, control=None)` — price a payoff and return `(price, standard_error)`
