from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from ...config import ImpliedVolConfig
from ...market.curves import PricingContext
from ...types import MarketData, OptionSpec, OptionType
from ...typing import ArrayLike
from ...vol.implied_vol_scalar import implied_vol_bs
from ..svi.transforms import logit, softplus_inv
from .math import _black76_price_broadcast
from .mingone import (
    compute_p_sequence,
    reconstruct_nodes_from_global_params,
)
from .models import (
    DEFAULT_NUMERICAL_TOL,
    ESSVINodeSet,
    MingoneGlobalParams,
    ThetaTermStructure,
)
from .validation import ESSVINodeConstraintReport, validate_essvi_nodes


def _as_float_vector(
    name: str, value: NDArray[np.float64] | np.ndarray | float
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    return arr


def _to_ctx(market: MarketData | PricingContext) -> PricingContext:
    if isinstance(market, PricingContext):
        return market
    return market.to_context()


def _resolve_sqrt_weights(
    y: np.ndarray,
    *,
    sqrt_weights: NDArray[np.float64] | None,
    weights: NDArray[np.float64] | None,
) -> np.ndarray:
    if sqrt_weights is not None and weights is not None:
        raise ValueError("Pass either sqrt_weights or weights, not both.")
    if sqrt_weights is None and weights is None:
        return np.ones_like(y, dtype=np.float64)
    if sqrt_weights is not None:
        out = _as_float_vector("sqrt_weights", sqrt_weights)
    else:
        assert weights is not None
        warnings.warn(
            "'weights' is deprecated for calibrate_essvi; use 'sqrt_weights' "
            "because the residual is sqrt_weights * price_error.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        out = _as_float_vector("weights", weights)
    if np.any(out < 0.0):
        raise ValueError("sqrt_weights must be >= 0.")
    if out.size != y.size:
        raise ValueError("sqrt_weights must have the same size as y.")
    return out


def _call_mask(y: np.ndarray, is_call: NDArray[np.bool_] | None) -> np.ndarray:
    if is_call is None:
        return np.asarray(y >= 0.0, dtype=np.bool_)
    out = np.asarray(is_call, dtype=np.bool_).reshape(-1)
    if out.size != y.size:
        raise ValueError("is_call must have the same size as y.")
    return out


@dataclass(frozen=True, slots=True)
class ATMThetaDiagnostics:
    expiries: np.ndarray
    theta_raw: np.ndarray
    theta_isotonic: np.ndarray
    extraction_methods: tuple[str, ...]

    @property
    def used_fallback(self) -> np.ndarray:
        return np.asarray(
            [method != "exact" for method in self.extraction_methods], dtype=np.bool_
        )


class SampledThetaTermStructure(ThetaTermStructure):
    __slots__ = (
        "sample_expiries",
        "theta_raw_nodes",
        "theta_isotonic_nodes",
        "extraction_methods",
    )

    def __init__(
        self,
        *,
        value,
        first_derivative=None,
        second_derivative=None,
        input_variable="T",
        sample_expiries: ArrayLike,
        theta_raw_nodes: ArrayLike,
        theta_isotonic_nodes: ArrayLike,
        extraction_methods: tuple[str, ...],
    ) -> None:
        super().__init__(
            value=value,
            first_derivative=first_derivative,
            second_derivative=second_derivative,
            input_variable=input_variable,
        )
        object.__setattr__(
            self,
            "sample_expiries",
            np.asarray(sample_expiries, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "theta_raw_nodes",
            np.asarray(theta_raw_nodes, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "theta_isotonic_nodes",
            np.asarray(theta_isotonic_nodes, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "extraction_methods",
            tuple(str(method) for method in extraction_methods),
        )


@dataclass(frozen=True, slots=True)
class ESSVIGlobalCalibrationConfig:
    objective: Literal["price"] = "price"
    max_nfev: int = 4_000
    atm_y_tol: float = 1e-8
    constraint_tol: float = 1e-10
    invalid_residual_value: float = 1e6
    rho_cap: float = 0.95
    strict_validation: bool = False
    iv_cfg: ImpliedVolConfig | None = None

    def __post_init__(self) -> None:
        if self.objective != "price":
            raise ValueError(f"Unsupported objective={self.objective!r}.")
        if self.max_nfev <= 0:
            raise ValueError("max_nfev must be > 0.")
        if self.atm_y_tol < 0.0:
            raise ValueError("atm_y_tol must be >= 0.")
        if self.constraint_tol < 0.0:
            raise ValueError("constraint_tol must be >= 0.")
        if self.invalid_residual_value <= 0.0:
            raise ValueError("invalid_residual_value must be > 0.")
        if not (0.0 < self.rho_cap < 1.0):
            raise ValueError("rho_cap must lie in (0, 1).")

    @property
    def model_eps(self) -> float:
        return float(max(DEFAULT_NUMERICAL_TOL, self.constraint_tol))


ESSVICalibrationConfig = ESSVIGlobalCalibrationConfig


@dataclass(frozen=True, slots=True)
class ESSVIFitDiagnostics:
    success: bool
    status: int
    message: str
    nfev: int
    cost: float
    x0: np.ndarray
    x_opt: np.ndarray
    theta: ATMThetaDiagnostics
    node_validation: ESSVINodeConstraintReport
    price_rmse: float
    max_abs_price_error: float
    invalid_candidate_count: int
    last_invalid_reason: str | None


@dataclass(frozen=True, slots=True)
class ESSVIFitResult:
    nodes: ESSVINodeSet
    diag: ESSVIFitDiagnostics


@dataclass(frozen=True, slots=True)
class _MarketSnapshot:
    y: np.ndarray
    T: np.ndarray
    price_mkt: np.ndarray
    sqrt_weights: np.ndarray
    is_call: np.ndarray
    strike: np.ndarray
    forward: np.ndarray
    df: np.ndarray
    implied_vol: np.ndarray
    total_variance: np.ndarray


def _market_snapshot(
    *,
    y: NDArray[np.float64],
    T: NDArray[np.float64],
    price_mkt: NDArray[np.float64],
    market: MarketData | PricingContext,
    sqrt_weights: NDArray[np.float64] | None,
    weights: NDArray[np.float64] | None,
    is_call: NDArray[np.bool_] | None,
    iv_cfg: ImpliedVolConfig | None,
) -> _MarketSnapshot:
    y_arr = _as_float_vector("y", y)
    T_arr = _as_float_vector("T", T)
    price_arr = _as_float_vector("price_mkt", price_mkt)
    if not (y_arr.size == T_arr.size == price_arr.size):
        raise ValueError("y, T, and price_mkt must have the same size.")
    if np.any(T_arr <= 0.0):
        raise ValueError("T must be > 0.")
    if np.any(price_arr < 0.0):
        raise ValueError("price_mkt must be >= 0.")

    sqrt_w = _resolve_sqrt_weights(y_arr, sqrt_weights=sqrt_weights, weights=weights)
    call_mask = _call_mask(y_arr, is_call)
    ctx = _to_ctx(market)
    forward = np.asarray([ctx.fwd(float(tau)) for tau in T_arr], dtype=np.float64)
    df = np.asarray([ctx.df(float(tau)) for tau in T_arr], dtype=np.float64)
    strike = np.asarray(forward * np.exp(y_arr), dtype=np.float64)

    implied_vol = np.empty_like(T_arr)
    for i, (tau, strike_i, is_call_i, price_i) in enumerate(
        zip(T_arr, strike, call_mask, price_arr, strict=True)
    ):
        spec = OptionSpec(
            kind=OptionType.CALL if bool(is_call_i) else OptionType.PUT,
            strike=float(strike_i),
            expiry=float(tau),
        )
        implied_vol[i] = float(implied_vol_bs(float(price_i), spec, market, cfg=iv_cfg))

    total_variance = np.asarray(T_arr * implied_vol * implied_vol, dtype=np.float64)
    return _MarketSnapshot(
        y=y_arr,
        T=T_arr,
        price_mkt=price_arr,
        sqrt_weights=sqrt_w,
        is_call=call_mask,
        strike=strike,
        forward=forward,
        df=df,
        implied_vol=implied_vol,
        total_variance=total_variance,
    )


def _theta_interp_from_nodes(
    expiries: np.ndarray,
    theta: np.ndarray,
) -> tuple[Callable[[ArrayLike], np.ndarray], Callable[[ArrayLike], np.ndarray]]:
    from ...numerics.interpolation import (
        FritschCarlson,
        linear_interp_derivative_factory,
        linear_interp_factory,
    )

    if expiries.size < 2:
        theta0 = float(theta[0])

        def theta_const(xq: ArrayLike) -> np.ndarray:
            xq_arr = np.asarray(xq, dtype=np.float64)
            return np.full_like(xq_arr, theta0, dtype=np.float64)

        def theta_const_deriv(xq: ArrayLike) -> np.ndarray:
            xq_arr = np.asarray(xq, dtype=np.float64)
            return np.zeros_like(xq_arr, dtype=np.float64)

        return theta_const, theta_const_deriv

    if expiries.size >= 3:
        interp_value, interp_deriv = FritschCarlson(expiries, theta)
    else:
        interp_value = linear_interp_factory(expiries, theta)
        interp_deriv = linear_interp_derivative_factory(expiries, theta)

    def theta_interp(xq: ArrayLike) -> np.ndarray:
        return np.asarray(
            interp_value(np.asarray(xq, dtype=np.float64)), dtype=np.float64
        )

    def theta_interp_deriv(xq: ArrayLike) -> np.ndarray:
        return np.asarray(
            interp_deriv(np.asarray(xq, dtype=np.float64)), dtype=np.float64
        )

    return theta_interp, theta_interp_deriv


def _theta_term_from_total_variance(
    *,
    y: np.ndarray,
    T: np.ndarray,
    total_variance: np.ndarray,
    atm_y_tol: float,
) -> tuple[SampledThetaTermStructure, ATMThetaDiagnostics]:
    from ...numerics.regression import isotonic_regression

    expiries = np.unique(np.asarray(T, dtype=np.float64))
    theta_raw = np.empty_like(expiries)
    methods: list[str] = []

    for i, tau in enumerate(expiries):
        mask = np.isclose(T, tau, rtol=0.0, atol=1e-12)
        y_slice = np.asarray(y[mask], dtype=np.float64)
        w_slice = np.asarray(total_variance[mask], dtype=np.float64)
        order = np.argsort(y_slice)
        y_slice = y_slice[order]
        w_slice = w_slice[order]

        exact = np.flatnonzero(np.abs(y_slice) <= float(atm_y_tol))
        if exact.size:
            theta_raw[i] = float(np.mean(w_slice[exact]))
            methods.append("exact")
            continue

        left = np.flatnonzero(y_slice < 0.0)
        right = np.flatnonzero(y_slice > 0.0)
        if left.size and right.size:
            i0 = int(left[-1])
            i1 = int(right[0])
            y0 = float(y_slice[i0])
            y1 = float(y_slice[i1])
            w0 = float(w_slice[i0])
            w1 = float(w_slice[i1])
            theta_raw[i] = float(w0 + (-y0) * (w1 - w0) / (y1 - y0))
            methods.append("bracket")
            continue

        idx = int(np.argmin(np.abs(y_slice)))
        theta_raw[i] = float(w_slice[idx])
        methods.append("nearest")

    theta_iso = isotonic_regression(theta_raw)
    theta_fn, dtheta_fn = _theta_interp_from_nodes(expiries, theta_iso)
    theta_term = SampledThetaTermStructure(
        value=theta_fn,
        first_derivative=dtheta_fn,
        second_derivative=None,
        input_variable="T",
        sample_expiries=expiries,
        theta_raw_nodes=theta_raw,
        theta_isotonic_nodes=theta_iso,
        extraction_methods=tuple(methods),
    )
    diag = ATMThetaDiagnostics(
        expiries=expiries,
        theta_raw=theta_raw,
        theta_isotonic=theta_iso,
        extraction_methods=tuple(methods),
    )
    return theta_term, diag


def build_theta_term_from_quotes(
    *,
    y: NDArray[np.float64],
    T: NDArray[np.float64],
    price_mkt: NDArray[np.float64],
    market: MarketData | PricingContext,
    is_call: NDArray[np.bool_] | None = None,
    atm_y_tol: float = 1e-8,
    iv_cfg: ImpliedVolConfig | None = None,
) -> tuple[SampledThetaTermStructure, ATMThetaDiagnostics]:
    snapshot = _market_snapshot(
        y=y,
        T=T,
        price_mkt=price_mkt,
        market=market,
        sqrt_weights=None,
        weights=None,
        is_call=is_call,
        iv_cfg=iv_cfg,
    )
    return _theta_term_from_total_variance(
        y=snapshot.y,
        T=snapshot.T,
        total_variance=snapshot.total_variance,
        atm_y_tol=atm_y_tol,
    )


def _pack_raw_vector(params: MingoneGlobalParams) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(params.rho_raw, dtype=np.float64),
            np.array([float(params.theta1_raw)], dtype=np.float64),
            np.asarray(params.a_raw, dtype=np.float64),
            np.asarray(params.c_raw, dtype=np.float64),
        ]
    )


def _unpack_raw_vector(
    x: np.ndarray,
    *,
    expiries: np.ndarray,
    cfg: ESSVIGlobalCalibrationConfig,
) -> MingoneGlobalParams:
    n = int(expiries.size)
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    expected = 3 * n
    if x_arr.size != expected:
        raise ValueError(
            f"Expected optimizer vector of size {expected}, got {x_arr.size}."
        )

    rho_raw = x_arr[:n]
    theta1_raw = float(x_arr[n])
    a_raw = x_arr[n + 1 : n + 1 + max(n - 1, 0)]
    c_raw = x_arr[n + 1 + max(n - 1, 0) :]
    return MingoneGlobalParams(
        expiries=expiries,
        rho_raw=np.asarray(rho_raw, dtype=np.float64),
        theta1_raw=theta1_raw,
        a_raw=np.asarray(a_raw, dtype=np.float64),
        c_raw=np.asarray(c_raw, dtype=np.float64),
        rho_cap=cfg.rho_cap,
        eps=cfg.model_eps,
    )


def _initial_nodes_from_theta(
    theta_diag: ATMThetaDiagnostics, *, cfg: ESSVIGlobalCalibrationConfig
) -> ESSVINodeSet:
    expiries = np.asarray(theta_diag.expiries, dtype=np.float64)
    theta_iso = np.asarray(theta_diag.theta_isotonic, dtype=np.float64)
    n = int(expiries.size)

    rho = np.zeros(n, dtype=np.float64)
    p = compute_p_sequence(rho)
    psi = np.empty(n, dtype=np.float64)
    theta = np.empty(n, dtype=np.float64)

    theta[0] = max(float(theta_iso[0]), cfg.model_eps)
    psi[0] = 0.5 * min(4.0, 2.0 * np.sqrt(max(theta[0], cfg.model_eps)))

    for i in range(1, n):
        a_lower = max(
            float(theta_iso[i]),
            theta[i - 1] * p[i] + cfg.model_eps,
            0.25 * psi[i - 1] * psi[i - 1] * (1.0 + abs(rho[i])) + cfg.model_eps,
        )
        theta[i] = float(a_lower)
        A = float(psi[i - 1] * p[i])
        C = float(
            min(
                psi[i - 1] * theta[i] / theta[i - 1],
                min(
                    4.0 / (1.0 + abs(rho[i])),
                    np.sqrt((4.0 * theta[i]) / (1.0 + abs(rho[i]))),
                ),
            )
        )
        if C <= A + cfg.model_eps:
            theta[i] = float((A * A * (1.0 + abs(rho[i])) / 4.0) + cfg.model_eps)
            C = float(
                min(
                    psi[i - 1] * theta[i] / theta[i - 1],
                    min(
                        4.0 / (1.0 + abs(rho[i])),
                        np.sqrt((4.0 * theta[i]) / (1.0 + abs(rho[i]))),
                    ),
                )
            )
        psi[i] = float(A + 0.5 * (C - A))

    return ESSVINodeSet(
        expiries=expiries,
        theta=theta,
        psi=psi,
        rho=rho,
        eps=cfg.model_eps,
    )


def _initial_raw_guess(
    theta_diag: ATMThetaDiagnostics, *, cfg: ESSVIGlobalCalibrationConfig
) -> np.ndarray:
    nodes0 = _initial_nodes_from_theta(theta_diag, cfg=cfg)
    p = compute_p_sequence(nodes0.rho)
    theta1_raw = float(
        softplus_inv(max(nodes0.theta[0] - cfg.model_eps, cfg.model_eps))
    )
    if nodes0.expiries.size > 1:
        a = nodes0.theta[1:] - nodes0.theta[:-1] * p[1:]
        a_raw = np.asarray(
            softplus_inv(np.maximum(a - cfg.model_eps, cfg.model_eps)), dtype=np.float64
        )
    else:
        a_raw = np.empty((0,), dtype=np.float64)
    c_raw = np.full(nodes0.expiries.size, logit(0.5), dtype=np.float64)
    params0 = MingoneGlobalParams(
        expiries=nodes0.expiries,
        rho_raw=np.zeros_like(nodes0.rho),
        theta1_raw=theta1_raw,
        a_raw=a_raw,
        c_raw=c_raw,
        rho_cap=cfg.rho_cap,
        eps=cfg.model_eps,
    )
    return _pack_raw_vector(params0)


def _nodal_model_prices(
    snapshot: _MarketSnapshot,
    nodes: ESSVINodeSet,
    expiry_index: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    theta = np.asarray(nodes.theta[expiry_index], dtype=np.float64)
    psi = np.asarray(nodes.psi[expiry_index], dtype=np.float64)
    eta = np.asarray(nodes.eta[expiry_index], dtype=np.float64)

    radicand = (
        theta * theta
        + 2.0 * theta * eta * snapshot.y
        + psi * psi * snapshot.y * snapshot.y
    )
    if np.any(radicand < -eps):
        raise ValueError("Nodal eSSVI radicand became negative during calibration.")
    radicand = np.where((radicand < 0.0) & (radicand >= -eps), 0.0, radicand)
    w = np.asarray(
        0.5 * (theta + eta * snapshot.y + np.sqrt(radicand)), dtype=np.float64
    )
    sigma = np.asarray(np.sqrt(np.maximum(w / snapshot.T, 0.0)), dtype=np.float64)

    call_model = _black76_price_broadcast(
        kind=OptionType.CALL,
        forward=snapshot.forward,
        strike=snapshot.strike,
        sigma=sigma,
        T=snapshot.T,
        df=snapshot.df,
        eps=eps,
    )
    put_model = np.asarray(
        call_model - snapshot.df * (snapshot.forward - snapshot.strike),
        dtype=np.float64,
    )
    return np.asarray(
        np.where(snapshot.is_call, call_model, put_model), dtype=np.float64
    )


class _ESSVIGlobalCalibrationObjective:
    def __init__(
        self,
        *,
        snapshot: _MarketSnapshot,
        expiries: np.ndarray,
        cfg: ESSVIGlobalCalibrationConfig,
    ) -> None:
        self.snapshot = snapshot
        self.expiries = np.asarray(expiries, dtype=np.float64)
        self.cfg = cfg
        self.expiry_index = np.searchsorted(self.expiries, snapshot.T)
        self.invalid_candidate_count = 0
        self.last_invalid_reason: str | None = None
        self._bad_vector = np.full(
            snapshot.y.size,
            float(cfg.invalid_residual_value),
            dtype=np.float64,
        )

    def nodes_from_vector(self, x: np.ndarray) -> ESSVINodeSet:
        params = _unpack_raw_vector(x, expiries=self.expiries, cfg=self.cfg)
        return reconstruct_nodes_from_global_params(params)

    def residual(self, x: np.ndarray) -> np.ndarray:
        try:
            nodes = self.nodes_from_vector(np.asarray(x, dtype=np.float64))
            node_report = validate_essvi_nodes(nodes, tol=self.cfg.constraint_tol)
            if not node_report.ok:
                self.invalid_candidate_count += 1
                self.last_invalid_reason = node_report.message
                return self._bad_vector.copy()

            model = _nodal_model_prices(
                self.snapshot,
                nodes,
                self.expiry_index,
                eps=self.cfg.model_eps,
            )
            residual = np.asarray(
                self.snapshot.sqrt_weights * (model - self.snapshot.price_mkt),
                dtype=np.float64,
            )
            if np.any(~np.isfinite(residual)):
                self.invalid_candidate_count += 1
                self.last_invalid_reason = "Non-finite residual."
                return self._bad_vector.copy()
            return residual
        except Exception as exc:
            self.invalid_candidate_count += 1
            self.last_invalid_reason = str(exc)
            return self._bad_vector.copy()


def calibrate_essvi_global(
    *,
    y: NDArray[np.float64],
    T: NDArray[np.float64],
    price_mkt: NDArray[np.float64],
    market: MarketData | PricingContext,
    sqrt_weights: NDArray[np.float64] | None = None,
    weights: NDArray[np.float64] | None = None,
    is_call: NDArray[np.bool_] | None = None,
    cfg: ESSVIGlobalCalibrationConfig | None = None,
) -> ESSVIFitResult:
    cfg = ESSVIGlobalCalibrationConfig() if cfg is None else cfg
    snapshot = _market_snapshot(
        y=y,
        T=T,
        price_mkt=price_mkt,
        market=market,
        sqrt_weights=sqrt_weights,
        weights=weights,
        is_call=is_call,
        iv_cfg=cfg.iv_cfg,
    )
    _, theta_diag = _theta_term_from_total_variance(
        y=snapshot.y,
        T=snapshot.T,
        total_variance=snapshot.total_variance,
        atm_y_tol=cfg.atm_y_tol,
    )

    unique_T = np.unique(snapshot.T)
    x0 = _initial_raw_guess(theta_diag, cfg=cfg)

    objective = _ESSVIGlobalCalibrationObjective(
        snapshot=snapshot,
        expiries=unique_T,
        cfg=cfg,
    )
    res = least_squares(
        fun=objective.residual,
        x0=x0,
        loss="linear",
        x_scale="jac",
        max_nfev=int(cfg.max_nfev),
    )
    if not res.success or not np.all(np.isfinite(res.x)):
        raise ValueError(f"ESSVI global calibration failed: {res.message}")

    x_opt = np.asarray(res.x, dtype=np.float64)
    nodes = objective.nodes_from_vector(x_opt)
    node_validation = validate_essvi_nodes(
        nodes, strict=cfg.strict_validation, tol=cfg.constraint_tol
    )
    fitted_prices = _nodal_model_prices(
        snapshot, nodes, objective.expiry_index, eps=cfg.model_eps
    )
    price_error = np.asarray(fitted_prices - snapshot.price_mkt, dtype=np.float64)

    diag = ESSVIFitDiagnostics(
        success=bool(res.success),
        status=int(res.status),
        message=str(res.message),
        nfev=int(res.nfev),
        cost=float(res.cost),
        x0=np.asarray(x0, dtype=np.float64),
        x_opt=x_opt,
        theta=theta_diag,
        node_validation=node_validation,
        price_rmse=float(np.sqrt(np.mean(price_error * price_error))),
        max_abs_price_error=float(np.max(np.abs(price_error))),
        invalid_candidate_count=int(objective.invalid_candidate_count),
        last_invalid_reason=objective.last_invalid_reason,
    )
    return ESSVIFitResult(nodes=nodes, diag=diag)


def calibrate_essvi(
    *,
    y: NDArray[np.float64],
    T: NDArray[np.float64],
    price_mkt: NDArray[np.float64],
    market: MarketData | PricingContext,
    sqrt_weights: NDArray[np.float64] | None = None,
    weights: NDArray[np.float64] | None = None,
    is_call: NDArray[np.bool_] | None = None,
    cfg: ESSVIGlobalCalibrationConfig | None = None,
) -> ESSVIFitResult:
    return calibrate_essvi_global(
        y=y,
        T=T,
        price_mkt=price_mkt,
        market=market,
        sqrt_weights=sqrt_weights,
        weights=weights,
        is_call=is_call,
        cfg=cfg,
    )


def calibrate_essvi_smooth(
    *,
    y: NDArray[np.float64],
    T: NDArray[np.float64],
    price_mkt: NDArray[np.float64],
    market: MarketData | PricingContext,
    sqrt_weights: NDArray[np.float64] | None = None,
    weights: NDArray[np.float64] | None = None,
    is_call: NDArray[np.bool_] | None = None,
    cfg: ESSVIGlobalCalibrationConfig | None = None,
):
    from .smooth_projection import ESSVIProjectionConfig, project_essvi_nodes

    warnings.warn(
        "calibrate_essvi_smooth is a compatibility helper. It now calibrates "
        "global Mingone nodes first and then applies the explicit projection stage.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    fit = calibrate_essvi_global(
        y=y,
        T=T,
        price_mkt=price_mkt,
        market=market,
        sqrt_weights=sqrt_weights,
        weights=weights,
        is_call=is_call,
        cfg=cfg,
    )
    return project_essvi_nodes(fit.nodes, cfg=ESSVIProjectionConfig())
