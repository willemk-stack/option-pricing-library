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
from ..svi.transforms import sigmoid, softplus, softplus_inv
from .models import (
    DEFAULT_NUMERICAL_TOL,
    ESSVITermStructures,
    EtaTermStructure,
    PsiTermStructure,
    ThetaTermStructure,
)
from .objective import ESSVIPriceObjective
from .validation import (
    ESSVIConstraintReport,
    ESSVIValidationReport,
    evaluate_essvi_constraints,
    validate_essvi_surface,
)


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


def _validation_expiries(expiries: np.ndarray) -> np.ndarray:
    exp = np.unique(np.asarray(expiries, dtype=np.float64))
    if exp.size == 1:
        return exp
    mids = 0.5 * (exp[:-1] + exp[1:])
    return np.unique(np.concatenate([exp, mids])).astype(np.float64)


def _constraint_grid(expiries: np.ndarray, n: int) -> np.ndarray:
    exp = np.unique(np.asarray(expiries, dtype=np.float64))
    if exp.size == 1:
        return exp
    if n < 2:
        return exp
    dense = np.linspace(float(exp[0]), float(exp[-1]), int(n), dtype=np.float64)
    return np.unique(np.concatenate([exp, dense])).astype(np.float64)


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
class ESSVICalibrationConfig:
    objective: Literal["price"] = "price"
    max_nfev: int = 4_000
    atm_y_tol: float = 1e-8
    constraint_grid_size: int = 41
    constraint_tol: float = 1e-10
    invalid_residual_value: float = 1e6
    use_soft_constraints: bool = False
    soft_constraint_weight: float = 10.0
    soft_constraint_scale: float = 1.0
    a_psi_bounds: tuple[float, float] = (-20.0, 4.0)
    b_psi_bounds: tuple[float, float] = (-4.0, 4.0)
    a_eta_bounds: tuple[float, float] = (-4.0, 4.0)
    b_eta_bounds: tuple[float, float] = (-4.0, 4.0)
    validation_y_min: float = -2.5
    validation_y_max: float = 2.5
    validation_ny: int = 81
    strict_validation: bool = False
    iv_cfg: ImpliedVolConfig | None = None

    def __post_init__(self) -> None:
        if self.objective != "price":
            raise ValueError(f"Unsupported objective={self.objective!r}.")
        if self.max_nfev <= 0:
            raise ValueError("max_nfev must be > 0.")
        if self.atm_y_tol < 0.0:
            raise ValueError("atm_y_tol must be >= 0.")
        if self.constraint_grid_size <= 0:
            raise ValueError("constraint_grid_size must be > 0.")
        if self.constraint_tol < 0.0:
            raise ValueError("constraint_tol must be >= 0.")
        if self.invalid_residual_value <= 0.0:
            raise ValueError("invalid_residual_value must be > 0.")
        if self.soft_constraint_weight < 0.0:
            raise ValueError("soft_constraint_weight must be >= 0.")
        if self.soft_constraint_scale <= 0.0:
            raise ValueError("soft_constraint_scale must be > 0.")
        if self.validation_ny < 3:
            raise ValueError("validation_ny must be >= 3.")
        if self.validation_y_min >= self.validation_y_max:
            raise ValueError("validation_y_min must be < validation_y_max.")

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lb = np.array(
            [
                self.a_psi_bounds[0],
                self.b_psi_bounds[0],
                self.a_eta_bounds[0],
                self.b_eta_bounds[0],
            ],
            dtype=np.float64,
        )
        ub = np.array(
            [
                self.a_psi_bounds[1],
                self.b_psi_bounds[1],
                self.a_eta_bounds[1],
                self.b_eta_bounds[1],
            ],
            dtype=np.float64,
        )
        return lb, ub


@dataclass(frozen=True, slots=True)
class ESSVIFitDiagnostics:
    success: bool
    status: int
    message: str
    nfev: int
    cost: float
    x0: np.ndarray
    x_opt: np.ndarray
    T_pivot: float
    theta: ATMThetaDiagnostics
    constraint_grid: np.ndarray
    constraint_report: ESSVIConstraintReport
    validation: ESSVIValidationReport
    invalid_candidate_count: int
    last_invalid_reason: str | None


@dataclass(frozen=True, slots=True)
class ESSVIFitResult:
    params: ESSVITermStructures
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
        implied_vol[i] = float(
            implied_vol_bs(
                float(price_i),
                spec,
                market,
                cfg=iv_cfg,
            )
        )

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


def _eta_guess(y: np.ndarray, w: np.ndarray, theta: float) -> float:
    nonzero = np.abs(y) > 1e-10
    if not np.any(nonzero):
        return 0.0
    order = np.argsort(np.abs(y[nonzero]))
    y_use = y[nonzero][order][: min(4, order.size)]
    w_use = w[nonzero][order][: min(4, order.size)]
    denom = float(np.dot(y_use, y_use))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(y_use, w_use - theta) / denom)


def _psi_guess(
    y: np.ndarray,
    w: np.ndarray,
    *,
    theta: float,
    eta: float,
    tol: float,
) -> float:
    abs_eta = abs(float(eta))
    psi_floor = abs_eta + 1e-3
    disc = max(abs_eta * abs_eta + 16.0 * float(theta) - 4.0 * float(tol), 0.0)
    psi_cap = min(
        4.0 - abs_eta - float(tol),
        0.5 * (-abs_eta + float(np.sqrt(disc))),
    )
    if psi_cap <= psi_floor:
        return psi_floor

    nonzero = np.abs(y) > 1e-10
    if not np.any(nonzero):
        return float(np.clip(0.5 * (psi_floor + psi_cap), psi_floor, psi_cap))

    y_use = y[nonzero]
    w_use = w[nonzero]
    D = 2.0 * w_use - float(theta) - float(eta) * y_use
    psi_sq = (D * D - theta * theta - 2.0 * theta * eta * y_use) / (y_use * y_use)
    psi_candidates = np.sqrt(np.maximum(psi_sq, 0.0))
    psi_candidates = psi_candidates[
        np.isfinite(psi_candidates) & (psi_candidates > 0.0)
    ]
    if psi_candidates.size == 0:
        psi0 = 0.5 * (psi_floor + psi_cap)
    else:
        psi0 = float(np.median(psi_candidates))
    return float(np.clip(psi0, psi_floor, 0.95 * psi_cap))


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size == 1:
        return float(y[0]), 0.0
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[0]), float(beta[1])


def _centered_logT(T: np.ndarray, T_pivot: float) -> np.ndarray:
    return np.asarray(
        np.log(np.asarray(T, dtype=np.float64) / float(T_pivot)),
        dtype=np.float64,
    )


def _build_parametric_terms(
    *,
    x: np.ndarray,
    theta_term: ThetaTermStructure,
    T_pivot: float,
    eps: float,
) -> ESSVITermStructures:
    a_psi, b_psi, a_eta, b_eta = map(float, x)
    log_pivot = float(np.log(T_pivot))

    def psi_value(log_T: ArrayLike) -> np.ndarray:
        log_T_arr = np.asarray(log_T, dtype=np.float64)
        z = a_psi + b_psi * (log_T_arr - log_pivot)
        return np.asarray(softplus(z) + eps, dtype=np.float64)

    def psi_first(log_T: ArrayLike) -> np.ndarray:
        log_T_arr = np.asarray(log_T, dtype=np.float64)
        z = a_psi + b_psi * (log_T_arr - log_pivot)
        return np.asarray(sigmoid(z) * b_psi, dtype=np.float64)

    def eta_value(log_T: ArrayLike) -> np.ndarray:
        log_T_arr = np.asarray(log_T, dtype=np.float64)
        return np.asarray(a_eta + b_eta * (log_T_arr - log_pivot), dtype=np.float64)

    def eta_first(log_T: ArrayLike) -> np.ndarray:
        log_T_arr = np.asarray(log_T, dtype=np.float64)
        return np.full_like(log_T_arr, b_eta, dtype=np.float64)

    return ESSVITermStructures(
        theta_term=theta_term,
        psi_term=PsiTermStructure(
            value=psi_value,
            first_derivative=psi_first,
            input_variable="log_T",
        ),
        eta_term=EtaTermStructure(
            value=eta_value,
            first_derivative=eta_first,
            input_variable="log_T",
        ),
        eps=eps,
    )


def _safe_initial_guess(
    theta_diag: ATMThetaDiagnostics,
    *,
    theta_term: ThetaTermStructure,
    T_pivot: float,
    cfg: ESSVICalibrationConfig,
    constraint_grid: np.ndarray,
    snapshot: _MarketSnapshot,
    model_eps: float,
) -> np.ndarray:
    expiries = theta_diag.expiries
    x_nodes = _centered_logT(expiries, T_pivot)
    eta_nodes = np.empty_like(expiries)
    psi_nodes = np.empty_like(expiries)

    for i, tau in enumerate(expiries):
        mask = np.isclose(snapshot.T, tau, rtol=0.0, atol=1e-12)
        y_slice = snapshot.y[mask]
        w_slice = snapshot.total_variance[mask]
        theta_i = float(theta_diag.theta_isotonic[i])
        eta_nodes[i] = _eta_guess(y_slice, w_slice, theta_i)
        psi_nodes[i] = _psi_guess(
            y_slice,
            w_slice,
            theta=theta_i,
            eta=float(eta_nodes[i]),
            tol=cfg.constraint_tol,
        )

    raw_psi = np.asarray(
        softplus_inv(np.maximum(psi_nodes - model_eps, model_eps)),
        dtype=np.float64,
    )
    a_psi, b_psi = _linear_fit(x_nodes, raw_psi)
    a_eta, b_eta = _linear_fit(x_nodes, eta_nodes)

    lb, ub = cfg.bounds
    x0 = np.clip(
        np.array([a_psi, b_psi, a_eta, b_eta], dtype=np.float64),
        lb,
        ub,
    )

    params0 = _build_parametric_terms(
        x=x0,
        theta_term=theta_term,
        T_pivot=T_pivot,
        eps=model_eps,
    )
    if evaluate_essvi_constraints(params0, constraint_grid, tol=cfg.constraint_tol).ok:
        return x0

    theta_min = float(np.min(theta_diag.theta_isotonic))
    psi_safe = max(0.05, 0.5 * np.sqrt(max(theta_min, 1e-8)))
    fallback = np.array(
        [float(softplus_inv(max(psi_safe - model_eps, model_eps))), 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    return np.clip(fallback, lb, ub)


class _ESSVICalibrationObjective:
    def __init__(
        self,
        *,
        snapshot: _MarketSnapshot,
        market: MarketData | PricingContext,
        theta_term: ThetaTermStructure,
        T_pivot: float,
        constraint_grid: np.ndarray,
        cfg: ESSVICalibrationConfig,
    ) -> None:
        self.snapshot = snapshot
        self.market = market
        self.theta_term = theta_term
        self.T_pivot = float(T_pivot)
        self.constraint_grid = np.asarray(constraint_grid, dtype=np.float64)
        self.cfg = cfg
        self.model_eps = max(DEFAULT_NUMERICAL_TOL, float(cfg.constraint_tol))
        self.price_objective = ESSVIPriceObjective(
            y=snapshot.y,
            T=snapshot.T,
            price_mkt=snapshot.price_mkt,
            market=market,
            sqrt_weights=snapshot.sqrt_weights,
            is_call=snapshot.is_call,
        )
        self.invalid_candidate_count = 0
        self.last_invalid_reason: str | None = None
        self._soft_block_size = 0
        if cfg.use_soft_constraints:
            self._soft_block_size = 5 * self.constraint_grid.size
        self._bad_vector = np.full(
            snapshot.y.size + self._soft_block_size,
            float(cfg.invalid_residual_value),
            dtype=np.float64,
        )

    def params_from_vector(self, x: np.ndarray) -> ESSVITermStructures:
        return _build_parametric_terms(
            x=np.asarray(x, dtype=np.float64),
            theta_term=self.theta_term,
            T_pivot=self.T_pivot,
            eps=self.model_eps,
        )

    def _soft_penalties(self, report: ESSVIConstraintReport) -> np.ndarray:
        if not self.cfg.use_soft_constraints:
            return np.empty((0,), dtype=np.float64)
        scale = max(float(self.cfg.soft_constraint_scale), 1e-12)
        lam = np.sqrt(float(self.cfg.soft_constraint_weight))
        deficits = [
            np.maximum(-report.theta_margin, 0.0),
            np.maximum(-report.psi_margin, 0.0),
            np.maximum(-report.rho_margin, 0.0),
            np.maximum(-report.lee_margin, 0.0),
            np.maximum(-report.calendar_margin, 0.0),
        ]
        return np.concatenate(
            [
                lam * np.asarray(deficit / scale, dtype=np.float64)
                for deficit in deficits
            ]
        )

    def residual(self, x: np.ndarray) -> np.ndarray:
        try:
            params = self.params_from_vector(np.asarray(x, dtype=np.float64))
            constraint_report = evaluate_essvi_constraints(
                params,
                self.constraint_grid,
                tol=self.cfg.constraint_tol,
            )
            if not constraint_report.ok:
                self.invalid_candidate_count += 1
                self.last_invalid_reason = constraint_report.message
                return self._bad_vector.copy()

            residual = self.price_objective.residual(params)
            if self.cfg.use_soft_constraints:
                residual = np.concatenate(
                    [residual, self._soft_penalties(constraint_report)]
                )
            if np.any(~np.isfinite(residual)):
                self.invalid_candidate_count += 1
                self.last_invalid_reason = "Non-finite residual."
                return self._bad_vector.copy()
            return np.asarray(residual, dtype=np.float64)
        except Exception as exc:
            self.invalid_candidate_count += 1
            self.last_invalid_reason = str(exc)
            return self._bad_vector.copy()


def calibrate_essvi(
    *,
    y: NDArray[np.float64],
    T: NDArray[np.float64],
    price_mkt: NDArray[np.float64],
    market: MarketData | PricingContext,
    sqrt_weights: NDArray[np.float64] | None = None,
    weights: NDArray[np.float64] | None = None,
    is_call: NDArray[np.bool_] | None = None,
    cfg: ESSVICalibrationConfig | None = None,
) -> ESSVIFitResult:
    cfg = ESSVICalibrationConfig() if cfg is None else cfg
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
    theta_term, theta_diag = _theta_term_from_total_variance(
        y=snapshot.y,
        T=snapshot.T,
        total_variance=snapshot.total_variance,
        atm_y_tol=cfg.atm_y_tol,
    )

    unique_T = np.unique(snapshot.T)
    T_pivot = float(np.exp(np.mean(np.log(unique_T))))
    constraint_grid = _constraint_grid(unique_T, cfg.constraint_grid_size)
    model_eps = max(DEFAULT_NUMERICAL_TOL, float(cfg.constraint_tol))
    x0 = _safe_initial_guess(
        theta_diag,
        theta_term=theta_term,
        T_pivot=T_pivot,
        cfg=cfg,
        constraint_grid=constraint_grid,
        snapshot=snapshot,
        model_eps=model_eps,
    )

    objective = _ESSVICalibrationObjective(
        snapshot=snapshot,
        market=market,
        theta_term=theta_term,
        T_pivot=T_pivot,
        constraint_grid=constraint_grid,
        cfg=cfg,
    )
    lb, ub = cfg.bounds
    res = least_squares(
        fun=objective.residual,
        x0=x0,
        bounds=(lb, ub),
        loss="linear",
        x_scale="jac",
        max_nfev=int(cfg.max_nfev),
    )

    if not res.success or not np.all(np.isfinite(res.x)):
        raise ValueError(f"ESSVI calibration failed: {res.message}")

    x_opt = np.asarray(res.x, dtype=np.float64)
    params = objective.params_from_vector(x_opt)
    constraint_report = evaluate_essvi_constraints(
        params,
        constraint_grid,
        tol=cfg.constraint_tol,
    )
    y_validation = np.linspace(
        float(cfg.validation_y_min),
        float(cfg.validation_y_max),
        int(cfg.validation_ny),
        dtype=np.float64,
    )
    validation = validate_essvi_surface(
        params,
        market,
        expiries=_validation_expiries(unique_T),
        y_grid=y_validation,
        strict=cfg.strict_validation,
        tol=cfg.constraint_tol,
    )

    diag = ESSVIFitDiagnostics(
        success=bool(res.success),
        status=int(res.status),
        message=str(res.message),
        nfev=int(res.nfev),
        cost=float(res.cost),
        x0=np.asarray(x0, dtype=np.float64),
        x_opt=x_opt,
        T_pivot=T_pivot,
        theta=theta_diag,
        constraint_grid=constraint_grid,
        constraint_report=constraint_report,
        validation=validation,
        invalid_candidate_count=int(objective.invalid_candidate_count),
        last_invalid_reason=objective.last_invalid_reason,
    )
    return ESSVIFitResult(params=params, diag=diag)
