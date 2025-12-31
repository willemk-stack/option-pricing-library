from __future__ import annotations

from dataclasses import replace

from option_pricing.types import PricingInputs


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
    # Market
    market = getattr(base, "market", None)
    if market is None:
        market = base  # best-effort fallback

    if hasattr(market, "__dataclass_fields__"):
        market2 = replace(
            market,
            spot=float(S) if S is not None else market.spot,
            rate=float(r) if r is not None else market.rate,
            dividend_yield=float(q) if q is not None else market.dividend_yield,
        )
    else:  # pragma: no cover
        market2 = market

    # Spec
    spec = getattr(base, "spec", None)
    if spec is None:
        spec = base  # best-effort fallback

    if hasattr(spec, "__dataclass_fields__"):
        # not all specs expose strike/expiry; best-effort
        kwargs: dict[str, object] = {}
        if K is not None and hasattr(spec, "strike"):
            kwargs["strike"] = float(K)
        if T is not None and hasattr(spec, "expiry"):
            kwargs["expiry"] = float(T)
        spec2 = replace(spec, **kwargs) if kwargs else spec
    else:  # pragma: no cover
        spec2 = spec

    out = base
    if hasattr(base, "__dataclass_fields__"):
        # only set if these fields exist
        kwargs: dict[str, object] = {}
        if hasattr(base, "market"):
            kwargs["market"] = market2
        if hasattr(base, "spec"):
            kwargs["spec"] = spec2
        out = replace(base, **kwargs) if kwargs else base

    if (
        sigma is not None
        and hasattr(out, "__dataclass_fields__")
        and hasattr(out, "sigma")
    ):
        out = replace(out, sigma=float(sigma))

    return out


def default_cases(base: PricingInputs) -> list[tuple[str, PricingInputs]]:
    """Curated regimes used in MC-vs-BS demo notebooks."""
    t0 = float(getattr(base, "t", 0.0))
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
