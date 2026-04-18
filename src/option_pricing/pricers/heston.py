from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray

from ..instruments import VanillaOption
from ..models.heston import HestonParams, P_j_Scalar
from ..numerics.quadrature import CompositeRule, QuadratureConfig
from ..types import MarketData, OptionType, PricingContext
from ..typing import FloatArray

type RealArray = NDArray[np.float64]
type HestonBackend = Literal["gauss_legendre", "quad"]
type HestonProbabilityIndex = Literal[0, 1]


def _to_ctx(market: MarketData | PricingContext) -> PricingContext:
    """Normalize flat MarketData or an already-curves-first PricingContext."""
    if isinstance(market, PricingContext):
        return market
    return market.to_context()


def _validate_inputs_Heston(
    spot: float, strike: float | FloatArray, tau: float
) -> None:
    if tau < 0.0:
        raise ValueError("Time to expiry 'tau' must be non-negative")
    if np.any(strike <= 0.0):
        raise ValueError("strike(s) must be positive")
    if spot <= 0.0:
        raise ValueError("spot must be positive")


def _normalize_strike(
    strike: float | FloatArray,
) -> tuple[RealArray, bool, tuple[int, ...]]:
    strike_arr = np.asarray(strike, dtype=np.float64)
    scalar_input = strike_arr.ndim == 0
    original_shape = strike_arr.shape
    strike_arr = np.asarray(np.atleast_1d(strike_arr), dtype=np.float64)

    if not np.all(np.isfinite(strike_arr)):
        raise ValueError("strike(s) must be finite")
    if np.any(strike_arr <= 0.0):
        raise ValueError("strike(s) must be positive")

    return strike_arr, scalar_input, original_shape


def _restore_output(
    value: RealArray,
    scalar_input: bool,
    original_shape: tuple[int, ...],
) -> float | RealArray:
    value = np.asarray(value, dtype=np.float64)
    if scalar_input:
        return float(value[0])
    return np.asarray(value.reshape(original_shape), dtype=np.float64)


def _intrinsic_value(spot: float, strike: RealArray, kind: OptionType) -> RealArray:
    if kind == OptionType.CALL:
        return np.asarray(np.maximum(spot - strike, 0.0), dtype=np.float64)
    if kind == OptionType.PUT:
        return np.asarray(np.maximum(strike - spot, 0.0), dtype=np.float64)

    raise ValueError(f"Invalid kind passed: {kind}, should be an `OptionType` Enum")


def _probability_array(
    *,
    probability: float | None,
    x: RealArray,
    tau: float,
    params: HestonParams,
    j: HestonProbabilityIndex,
    backend: HestonBackend,
    quad_cfg: QuadratureConfig | None,
    rule: CompositeRule | None,
) -> RealArray:
    if probability is not None:
        return np.full(x.shape, probability, dtype=np.float64)

    return np.asarray(
        [
            P_j_Scalar(
                x=float(log_moneyness),
                tau=tau,
                params=params,
                j=j,
                backend=backend,
                quad_cfg=quad_cfg,
                rule=rule,
            )
            for log_moneyness in x
        ],
        dtype=np.float64,
    )


@overload
def heston_price_call_from_ctx(
    *,
    strike: float,
    ctx: PricingContext,
    tau: float,
    params: HestonParams,
    P_0: float | None = None,
    P_1: float | None = None,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> float: ...


@overload
def heston_price_call_from_ctx(
    *,
    strike: FloatArray,
    ctx: PricingContext,
    tau: float,
    params: HestonParams,
    P_0: float | None = None,
    P_1: float | None = None,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> FloatArray: ...


def heston_price_call_from_ctx(
    *,
    strike: float | FloatArray,
    ctx: PricingContext,
    tau: float,
    params: HestonParams,
    P_0: float | None = None,
    P_1: float | None = None,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> float | FloatArray:
    strike_arr, scalar_input, original_shape = _normalize_strike(strike)
    _validate_inputs_Heston(spot=ctx.spot, strike=strike_arr, tau=tau)

    if tau == 0:
        intrinsic = _intrinsic_value(
            spot=ctx.spot,
            strike=strike_arr,
            kind=OptionType.CALL,
        )
        return _restore_output(intrinsic, scalar_input, original_shape)

    forward = float(ctx.fwd(tau=tau))
    df = float(ctx.df(tau=tau))

    x = np.asarray(np.log(forward / strike_arr), dtype=np.float64)
    p_0_arr = _probability_array(
        probability=P_0,
        x=x,
        tau=tau,
        params=params,
        j=0,
        backend=backend,
        quad_cfg=quad_cfg,
        rule=rule,
    )
    p_1_arr = _probability_array(
        probability=P_1,
        x=x,
        tau=tau,
        params=params,
        j=1,
        backend=backend,
        quad_cfg=quad_cfg,
        rule=rule,
    )

    call = np.asarray(
        df * (forward * p_1_arr - strike_arr * p_0_arr),
        dtype=np.float64,
    )

    return _restore_output(call, scalar_input, original_shape)


@overload
def heston_price_put_from_ctx(
    *,
    strike: float,
    tau: float,
    ctx: PricingContext,
    params: HestonParams,
    P_0: float | None = None,
    P_1: float | None = None,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> float: ...


@overload
def heston_price_put_from_ctx(
    *,
    strike: FloatArray,
    tau: float,
    ctx: PricingContext,
    params: HestonParams,
    P_0: float | None = None,
    P_1: float | None = None,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> FloatArray: ...


def heston_price_put_from_ctx(
    *,
    strike: float | FloatArray,
    tau: float,
    ctx: PricingContext,
    params: HestonParams,
    P_0: float | None = None,
    P_1: float | None = None,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> float | FloatArray:
    strike_arr, scalar_input, original_shape = _normalize_strike(strike)
    _validate_inputs_Heston(spot=ctx.spot, strike=strike_arr, tau=tau)

    if tau == 0:
        intrinsic = _intrinsic_value(
            spot=ctx.spot,
            strike=strike_arr,
            kind=OptionType.PUT,
        )
        return _restore_output(intrinsic, scalar_input, original_shape)

    forward = float(ctx.fwd(tau=tau))
    df = float(ctx.df(tau=tau))
    x = np.asarray(np.log(forward / strike_arr), dtype=np.float64)

    p_0_arr = _probability_array(
        probability=P_0,
        x=x,
        tau=tau,
        params=params,
        j=0,
        backend=backend,
        quad_cfg=quad_cfg,
        rule=rule,
    )
    p_1_arr = _probability_array(
        probability=P_1,
        x=x,
        tau=tau,
        params=params,
        j=1,
        backend=backend,
        quad_cfg=quad_cfg,
        rule=rule,
    )

    put = np.asarray(
        df * (strike_arr * (1.0 - p_0_arr) - forward * (1.0 - p_1_arr)),
        dtype=np.float64,
    )
    return _restore_output(put, scalar_input, original_shape)


@overload
def heston_price_from_ctx(
    *,
    kind: OptionType,
    strike: float,
    tau: float,
    ctx: PricingContext,
    params: HestonParams,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> float: ...


@overload
def heston_price_from_ctx(
    *,
    kind: OptionType,
    strike: FloatArray,
    tau: float,
    ctx: PricingContext,
    params: HestonParams,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> FloatArray: ...


def heston_price_from_ctx(
    *,
    kind: OptionType,
    strike: float | FloatArray,
    tau: float,
    ctx: PricingContext,
    params: HestonParams,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> float | FloatArray:
    if kind == OptionType.CALL:
        return heston_price_call_from_ctx(
            strike=strike,
            ctx=ctx,
            tau=tau,
            params=params,
            backend=backend,
            quad_cfg=quad_cfg,
            rule=rule,
        )
    if kind == OptionType.PUT:
        return heston_price_put_from_ctx(
            strike=strike,
            tau=tau,
            ctx=ctx,
            params=params,
            backend=backend,
            quad_cfg=quad_cfg,
            rule=rule,
        )

    raise ValueError(f"kind should be an OptionType enum, here: {kind}")


def heston_price_instrument_from_ctx(
    *,
    inst: VanillaOption,
    ctx: PricingContext,
    params: HestonParams,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> float:
    return heston_price_from_ctx(
        kind=inst.kind,
        strike=inst.strike,
        tau=float(inst.expiry),
        ctx=ctx,
        params=params,
        backend=backend,
        quad_cfg=quad_cfg,
        rule=rule,
    )


def heston_price_instrument(
    inst: VanillaOption,
    *,
    market: MarketData | PricingContext,
    params: HestonParams,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> float:
    """Convenience wrapper accepting flat `MarketData`."""
    return heston_price_instrument_from_ctx(
        inst=inst,
        ctx=_to_ctx(market),
        params=params,
        backend=backend,
        quad_cfg=quad_cfg,
        rule=rule,
    )
