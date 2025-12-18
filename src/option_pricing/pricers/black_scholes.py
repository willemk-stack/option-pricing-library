from __future__ import annotations

from ..models import bs as bs_model
from ..types import OptionType, PricingInputs


def bs_price_call(p: PricingInputs) -> float:
    return bs_model.call_price(
        spot=p.S,
        strike=p.K,
        r=p.r,
        q=p.market.dividend_yield,
        sigma=p.sigma,
        tau=p.tau,
    )


def bs_price_put(p: PricingInputs) -> float:
    return bs_model.put_price(
        spot=p.S,
        strike=p.K,
        r=p.r,
        q=p.market.dividend_yield,
        sigma=p.sigma,
        tau=p.tau,
    )


def bs_price(p: PricingInputs) -> float:
    """
    Generic Black–Scholes price that dispatches on p.spec.kind.
    """
    if p.spec.kind == OptionType.CALL:
        return bs_price_call(p)
    if p.spec.kind == OptionType.PUT:
        return bs_price_put(p)
    raise ValueError(f"Unsupported option kind: {p.spec.kind}")


def bs_call_greeks(p: PricingInputs) -> dict[str, float]:
    return bs_model.call_greeks(
        spot=p.S,
        strike=p.K,
        r=p.r,
        q=p.market.dividend_yield,
        sigma=p.sigma,
        tau=p.tau,
    )


def bs_put_greeks(p: PricingInputs) -> dict[str, float]:
    return bs_model.put_greeks(
        spot=p.S,
        strike=p.K,
        r=p.r,
        q=p.market.dividend_yield,
        sigma=p.sigma,
        tau=p.tau,
    )


def bs_greeks(p: PricingInputs) -> dict[str, float]:
    """
    Generic Black–Scholes greeks that dispatches on p.spec.kind.
    Returns dict with keys: price, delta, gamma, vega, theta.
    """
    if p.spec.kind == OptionType.CALL:
        return bs_call_greeks(p)
    if p.spec.kind == OptionType.PUT:
        return bs_put_greeks(p)
    raise ValueError(f"Unsupported option kind: {p.spec.kind}")


# Backwards-compatible aliases (if you used these names in notebooks)
def bs_call_from_inputs(p: PricingInputs) -> float:
    return bs_price_call(p)


def bs_put_from_inputs(p: PricingInputs) -> float:
    return bs_price_put(p)


def bs_call_greeks_analytic_from_inputs(p: PricingInputs) -> dict[str, float]:
    return bs_call_greeks(p)


def bs_put_greeks_analytic_from_inputs(p: PricingInputs) -> dict[str, float]:
    return bs_put_greeks(p)
