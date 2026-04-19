"""Public Heston pricing helpers built on the Fourier probability routines."""

from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray

from ..instruments import VanillaOption
from ..models.heston import HestonParams, P_j
from ..models.heston.fourier import (
    HestonIntegralBatchDiagnostics,
    HestonIntegralDiagnostics,
)
from ..numerics.quadrature import CompositeRule, QuadratureConfig
from ..types import MarketData, OptionType, PricingContext
from ..typing import ArrayLike, FloatArray

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
    probability_override: float | ArrayLike | None,
    x: RealArray,
    tau: float,
    params: HestonParams,
    j: HestonProbabilityIndex,
    backend: HestonBackend,
    quad_cfg: QuadratureConfig | None,
    rule: CompositeRule | None,
) -> RealArray:
    """Resolve a probability slice from overrides or from ``P_j``.

    Parameters
    ----------
    probability_override : float or array-like, optional
        Precomputed probability override. Must be scalar or broadcastable to
        ``x.shape``.
    x : ndarray
        Log-forward moneyness slice.
    tau : float
        Time to expiry in years.
    params : HestonParams
        Heston parameter set.
    j : {0, 1}
        Probability index.
    backend : {"gauss_legendre", "quad"}
        Probability integration backend.
    quad_cfg : QuadratureConfig, optional
        Fixed-rule configuration for ``backend="gauss_legendre"``.
    rule : CompositeRule, optional
        Prebuilt fixed rule for ``backend="gauss_legendre"``.

    Returns
    -------
    ndarray
        Probability array with shape ``x.shape``.

    Notes
    -----
    When ``backend="quad"``, array-valued ``x`` is handled by the public
    :func:`option_pricing.models.heston.P_j` API through a scalar loop over the
    batch. Only the fixed-rule Gauss-Legendre backend is vectorized across the
    slice.
    """
    if probability_override is not None:
        override = np.asarray(probability_override, dtype=np.float64)

        try:
            return np.asarray(np.broadcast_to(override, x.shape), dtype=np.float64)
        except ValueError as exc:
            raise ValueError(
                "probability_override must be scalar or broadcastable to x.shape. "
                f"Got override shape {override.shape} and x.shape {x.shape}."
            ) from exc

    return np.asarray(
        P_j(
            x=x,
            tau=tau,
            params=params,
            j=j,
            backend=backend,
            quad_cfg=quad_cfg,
            rule=rule,
        ),
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class HestonPriceDiagnostics:
    """Scalar pricing diagnostics assembled from ``P_0`` and ``P_1``."""

    strike: float
    tau: float
    forward: float
    df: float
    p0: HestonIntegralDiagnostics
    p1: HestonIntegralDiagnostics
    price: float


@dataclass(frozen=True, slots=True)
class HestonPriceBatchDiagnostics:
    """Batch pricing diagnostics container for a strike slice.

    Notes
    -----
    This shape is primarily suited to vectorized fixed-rule evaluation. The
    current public pricing helpers do not yet expose a batch diagnostics API.
    """

    strike: FloatArray  # batch_shape
    tau: float
    forward: float
    df: float
    p0: HestonIntegralBatchDiagnostics
    p1: HestonIntegralBatchDiagnostics
    price: FloatArray  # batch_shape


@overload
def heston_price_call_from_ctx(
    *,
    strike: float,
    ctx: PricingContext,
    tau: float,
    params: HestonParams,
    P_0: float | ArrayLike | None = None,
    P_1: float | ArrayLike | None = None,
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
    P_0: float | ArrayLike | None = None,
    P_1: float | ArrayLike | None = None,
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
    P_0: float | ArrayLike | None = None,
    P_1: float | ArrayLike | None = None,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> float | FloatArray:
    """Price a European call under Heston from a pricing context.

    Parameters
    ----------
    strike : float or ndarray
        Positive strike or strike slice.
    ctx : PricingContext
        Market context providing spot, forwards, and discount factors.
    tau : float
        Time to expiry in years. Must be nonnegative.
    params : HestonParams
        Heston parameter set.
    P_0, P_1 : float or array-like, optional
        Optional probability overrides. Each override must be scalar or
        broadcastable to ``strike.shape``.
    backend : {"gauss_legendre", "quad"}, default "gauss_legendre"
        Probability integration backend.
    quad_cfg : QuadratureConfig, optional
        Fixed-rule configuration for ``backend="gauss_legendre"``.
    rule : CompositeRule, optional
        Prebuilt fixed rule for ``backend="gauss_legendre"``.

    Returns
    -------
    float or ndarray
        Call price or call-price slice.

    Notes
    -----
    For array-valued ``strike``, the function computes ``x = log(F / K)``
    elementwise and then evaluates ``P_0`` and ``P_1`` on that slice.

    The Gauss-Legendre backend is vectorized across the slice. The ``quad``
    backend accepts batched strikes for API consistency, but under the hood it
    still evaluates one scalar SciPy ``quad`` integral per strike and per
    probability index, so batch ``quad`` pricing is not vectorized.

    When ``tau == 0``, the function short-circuits to intrinsic value.
    """
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
        probability_override=P_0,
        x=x,
        tau=tau,
        params=params,
        j=0,
        backend=backend,
        quad_cfg=quad_cfg,
        rule=rule,
    )
    p_1_arr = _probability_array(
        probability_override=P_1,
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
    P_0: float | ArrayLike | None = None,
    P_1: float | ArrayLike | None = None,
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
    P_0: float | ArrayLike | None = None,
    P_1: float | ArrayLike | None = None,
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
    P_0: float | ArrayLike | None = None,
    P_1: float | ArrayLike | None = None,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> float | FloatArray:
    """Price a European put under Heston from a pricing context.

    Parameters
    ----------
    strike : float or ndarray
        Positive strike or strike slice.
    tau : float
        Time to expiry in years. Must be nonnegative.
    ctx : PricingContext
        Market context providing spot, forwards, and discount factors.
    params : HestonParams
        Heston parameter set.
    P_0, P_1 : float or array-like, optional
        Optional probability overrides. Each override must be scalar or
        broadcastable to ``strike.shape``.
    backend : {"gauss_legendre", "quad"}, default "gauss_legendre"
        Probability integration backend.
    quad_cfg : QuadratureConfig, optional
        Fixed-rule configuration for ``backend="gauss_legendre"``.
    rule : CompositeRule, optional
        Prebuilt fixed rule for ``backend="gauss_legendre"``.

    Returns
    -------
    float or ndarray
        Put price or put-price slice.

    Notes
    -----
    The same backend caveat as the call pricer applies here: the
    Gauss-Legendre path batches across array inputs, while the ``quad`` path
    loops over strikes and is therefore not vectorized.

    When ``tau == 0``, the function short-circuits to intrinsic value.
    """
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
        probability_override=P_0,
        x=x,
        tau=tau,
        params=params,
        j=0,
        backend=backend,
        quad_cfg=quad_cfg,
        rule=rule,
    )
    p_1_arr = _probability_array(
        probability_override=P_1,
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
    """Price a European option under Heston from a pricing context.

    Parameters
    ----------
    kind : OptionType
        Option kind, either call or put.
    strike : float or ndarray
        Positive strike or strike slice.
    tau : float
        Time to expiry in years. Must be nonnegative.
    ctx : PricingContext
        Market context providing spot, forwards, and discount factors.
    params : HestonParams
        Heston parameter set.
    backend : {"gauss_legendre", "quad"}, default "gauss_legendre"
        Probability integration backend.
    quad_cfg : QuadratureConfig, optional
        Fixed-rule configuration for ``backend="gauss_legendre"``.
    rule : CompositeRule, optional
        Prebuilt fixed rule for ``backend="gauss_legendre"``.

    Returns
    -------
    float or ndarray
        Option price or option-price slice.

    Notes
    -----
    This function dispatches to the call or put implementation based on
    ``kind``. The same vectorization caveat applies: array-valued pricing with
    ``backend="quad"`` is supported for convenience but is evaluated through
    scalar quadrature calls rather than true batch integration.
    """
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
    """Price a vanilla option instrument under Heston from a pricing context.

    Parameters
    ----------
    inst : VanillaOption
        Instrument carrying strike, expiry, and option kind.
    ctx : PricingContext
        Market context providing spot, forwards, and discount factors.
    params : HestonParams
        Heston parameter set.
    backend : {"gauss_legendre", "quad"}, default "gauss_legendre"
        Probability integration backend.
    quad_cfg : QuadratureConfig, optional
        Fixed-rule configuration for ``backend="gauss_legendre"``.
    rule : CompositeRule, optional
        Prebuilt fixed rule for ``backend="gauss_legendre"``.

    Returns
    -------
    float
        Instrument price.
    """
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
    """Price a vanilla option under Heston from flat market data.

    Parameters
    ----------
    inst : VanillaOption
        Instrument carrying strike, expiry, and option kind.
    market : MarketData or PricingContext
        Flat market inputs or an already-constructed pricing context.
    params : HestonParams
        Heston parameter set.
    backend : {"gauss_legendre", "quad"}, default "gauss_legendre"
        Probability integration backend.
    quad_cfg : QuadratureConfig, optional
        Fixed-rule configuration for ``backend="gauss_legendre"``.
    rule : CompositeRule, optional
        Prebuilt fixed rule for ``backend="gauss_legendre"``.

    Returns
    -------
    float
        Instrument price.
    """
    return heston_price_instrument_from_ctx(
        inst=inst,
        ctx=_to_ctx(market),
        params=params,
        backend=backend,
        quad_cfg=quad_cfg,
        rule=rule,
    )
