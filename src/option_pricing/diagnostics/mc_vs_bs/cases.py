from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

# For diagnostics modules we want *runtime* flexibility but also clean mypy.
# Import the real type only during type-checking; at runtime treat as Any.
if TYPE_CHECKING:  # pragma: no cover
    from option_pricing.types import PricingInputs
else:  # pragma: no cover
    PricingInputs = Any  # type: ignore


def _with_params(
    base: PricingInputs,
    *,
    S: float | None = None,
    K: float | None = None,
    r: float | None = None,
    q: float | None = None,
    T: float | None = None,
    sigma: float | None = None,
) -> PricingInputs:
    """Return a modified copy of a PricingInputs-like object.

    Best-effort supports the common pattern:
      - base.market has fields: spot, rate, dividend_yield
      - base.spec has fields: strike, expiry, kind
      - base has field: sigma

    If your PricingInputs differs, you can swap this helper out with something
    specific to your types.
    """
    # Market (only if base has a market dataclass that exposes these fields)
    market = getattr(base, "market", None)
    market2 = market
    if market is not None and hasattr(market, "__dataclass_fields__"):
        market_changes: dict[str, Any] = {}
        if S is not None and hasattr(market, "spot"):
            market_changes["spot"] = float(S)
        if r is not None and hasattr(market, "rate"):
            market_changes["rate"] = float(r)
        if q is not None and hasattr(market, "dividend_yield"):
            market_changes["dividend_yield"] = float(q)
        if market_changes:
            market2 = replace(cast(Any, market), **market_changes)

    # Spec (only if base has a spec dataclass that exposes strike/expiry)
    spec = getattr(base, "spec", None)
    spec2 = spec
    if spec is not None and hasattr(spec, "__dataclass_fields__"):
        spec_changes: dict[str, Any] = {}
        if K is not None and hasattr(spec, "strike"):
            spec_changes["strike"] = float(K)
        if T is not None and hasattr(spec, "expiry"):
            spec_changes["expiry"] = float(T)
        if spec_changes:
            spec2 = replace(cast(Any, spec), **spec_changes)

    out = base
    if hasattr(base, "__dataclass_fields__"):
        base_changes: dict[str, Any] = {}
        if market2 is not None and hasattr(base, "market"):
            base_changes["market"] = market2
        if spec2 is not None and hasattr(base, "spec"):
            base_changes["spec"] = spec2
        if sigma is not None and hasattr(base, "sigma"):
            base_changes["sigma"] = float(sigma)
        if base_changes:
            out = replace(cast(Any, base), **base_changes)

    return out


def default_cases(base: PricingInputs) -> list[tuple[str, PricingInputs]]:
    """Curated regimes used in MC-vs-BS demo notebooks."""
    # Some projects store tau/expiry differently; this is only used to
    # produce a "short" and "long" maturity variant.
    t0 = float(getattr(base, "t", getattr(base, "tau", 0.0)))
    return [
        ("ATM base", base),
        ("ITM (S=120)", _with_params(base, S=120.0)),
        ("OTM (S=80)", _with_params(base, S=80.0)),
        ("Deep ITM (S=150)", _with_params(base, S=150.0)),
        ("Deep OTM (S=50)", _with_params(base, S=50.0)),
        ("Short tau (1w)", _with_params(base, T=t0 + 1.0 / 52.0)),
        ("Long tau (5y)", _with_params(base, T=t0 + 5.0)),
        ("Low vol (5%)", _with_params(base, sigma=0.05)),
        ("High vol (80%)", _with_params(base, sigma=0.80)),
        ("High rate (10%)", _with_params(base, r=0.10)),
        ("Dividend (q=2%)", _with_params(base, q=0.02)),
    ]


__all__ = ["default_cases"]
