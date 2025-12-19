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
    Compute the Black-Scholes(-Merton) price of a European vanilla option.

    This convenience wrapper dispatches to the call/put implementation based on
    ``p.spec.kind`` using the inputs in ``p`` (spot ``S``, strike ``K``, risk-free
    rate ``r``, dividend yield ``q``, volatility ``sigma``, and time to maturity
    ``tau``).

    Parameters
    ----------
    p : PricingInputs
        Pricing inputs for a European vanilla option, including ``spec.kind`` to
        indicate call or put.

    Returns
    -------
    float
        Present value (price) of the option under the Black-Scholes-Merton model.

    Raises
    ------
    ValueError
        If ``p.spec.kind`` is not a supported vanilla type (call or put).

    Notes
    -----
    Assumes European exercise and a continuous dividend yield ``q``.
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
    Compute Black–Scholes(-Merton) price and primary greeks for a vanilla European option.

    This is a thin convenience wrapper that dispatches to the call/put implementation
    based on ``p.spec.kind`` and returns a standardized dictionary of results.

    Parameters
    ----------
    p : PricingInputs
        Pricing inputs for a European vanilla option. Expected to provide (at least)
        spot ``S``, strike ``K``, risk-free rate ``r``, dividend yield ``q``,
        volatility ``sigma``, time to maturity ``tau``, and ``spec.kind`` indicating
        call or put.

    Returns
    -------
    greeks : dict[str, float]
        Dictionary with the following keys:

        - ``"price"`` : option present value
        - ``"delta"`` : ∂price/∂S
        - ``"gamma"`` : ∂²price/∂S²
        - ``"vega"``  : ∂price/∂sigma (per 1.0 change in volatility, not per 1%)
        - ``"theta"`` : ∂price/∂t (time decay; sign convention follows the underlying
          call/put implementation)

    Raises
    ------
    ValueError
        If ``p.spec.kind`` is not a supported vanilla type (call or put).

    Notes
    -----
    Uses the Black–Scholes-Merton framework (continuous dividend yield ``q``) and
    assumes European exercise.
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
