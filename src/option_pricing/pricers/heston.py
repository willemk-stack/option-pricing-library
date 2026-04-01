import numpy as np

from ..instruments import VanillaOption
from ..models.heston import HestonParams, P_j
from ..types import MarketData, OptionType, PricingContext


def _to_ctx(market: MarketData | PricingContext) -> PricingContext:
    """Normalize flat MarketData or an already-curves-first PricingContext."""
    if isinstance(market, PricingContext):
        return market
    return market.to_context()


def _validate_inputs_Heston(spot: float, strike: float, tau: float):
    if tau < 0.0:
        raise ValueError("Time to expiry 'tau' must be non-negative")
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if spot <= 0.0:
        raise ValueError("spot must be positive")


def heston_price_call_from_ctx(
    *,
    strike: float,
    ctx: PricingContext,
    tau: float,
    params: HestonParams,
    P_0: float | None = None,
    P_1: float | None = None,
) -> float:
    _validate_inputs_Heston(spot=ctx.spot, strike=strike, tau=tau)

    forward = ctx.fwd(tau=tau)
    df = ctx.df(tau=tau)

    x = np.log(forward / strike)

    if P_0 is None:
        P_0 = P_j(x=x, tau=tau, params=params, j=0)
    if P_1 is None:
        P_1 = P_j(x=x, tau=tau, params=params, j=1)

    # Return var
    Call = df * (forward * P_1 - strike * P_0)

    return Call


def heston_price_put_from_ctx(
    *,
    strike: float,
    tau: float,
    ctx: PricingContext,
    params: HestonParams,
    P_0: float | None = None,
    P_1: float | None = None,
) -> float:

    forward = ctx.fwd(tau=tau)
    df = ctx.df(tau=tau)
    x = np.log(forward / strike)

    if P_0 is None:
        P_0 = P_j(x=x, tau=tau, params=params, j=0)
    if P_1 is None:
        P_1 = P_j(x=x, tau=tau, params=params, j=1)

    Put = df * (strike * (1 - P_0) - forward * (1 - P_1))

    return Put


def heston_price_from_ctx(
    *,
    kind: OptionType,
    strike: float,
    tau: float,
    ctx: PricingContext,
    params: HestonParams,
) -> float:
    forward = ctx.fwd(tau=tau)
    x = np.log(forward / strike)

    P_0 = P_j(x=x, tau=tau, params=params, j=0)
    P_1 = P_j(x=x, tau=tau, params=params, j=1)
    if kind == OptionType.CALL:
        return heston_price_call_from_ctx(
            strike=strike, ctx=ctx, tau=tau, params=params, P_0=P_0, P_1=P_1
        )
    elif kind == OptionType.PUT:
        return heston_price_put_from_ctx(
            strike=strike, ctx=ctx, tau=tau, params=params, P_0=P_0, P_1=P_1
        )

    raise ValueError(f"kind should be an OptionType enum, here: {kind}")


def heston_price_instrument_from_ctx(
    *, inst: VanillaOption, ctx: PricingContext, params: HestonParams
) -> float:
    return heston_price_from_ctx(
        kind=inst.kind,
        strike=inst.strike,
        tau=float(inst.expiry),
        ctx=ctx,
        params=params,
    )


def heston_price_instrument(
    inst: VanillaOption, *, market: MarketData | PricingContext, params: HestonParams
) -> float:
    """Convenience wrapper accepting flat `MarketData`."""
    return heston_price_instrument_from_ctx(
        inst=inst, ctx=_to_ctx(market), params=params
    )
