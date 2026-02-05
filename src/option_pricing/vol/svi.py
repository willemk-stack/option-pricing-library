from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares as _least_squares

from .vol_types import ArrayLike, FloatArray

least_squares: Callable[..., Any] = cast(Callable[..., Any], _least_squares)


@dataclass(frozen=True, slots=True)
class SVIParams:
    """
    Parameter      Geometric Meaning       Market Intuition
    a:              Vertical translation         Baseline variance level
    b:              Slope of the wings           Overall volatility of volatility
    rho:            Skewness / Rotation          Correlation between price and vol
    m:              Horizontal translation       Strike of minimum variance
    sigma:          Curvature at the vertex      Smoothness of the ATM transition
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float


@dataclass(frozen=True, slots=True)
class SVISmile:
    T: float
    params: SVIParams

    _y_min: float = -5.0
    _y_max: float = 5.0

    @property
    def y_min(self) -> float:
        return self._y_min

    @property
    def y_max(self) -> float:
        return self._y_max

    def w_at(self, y: ArrayLike) -> FloatArray:
        y_arr = np.asarray(y, dtype=float)
        return np.asarray(svi_total_variance(y_arr, self.params), float)

    def iv_at(self, y: ArrayLike) -> FloatArray:
        w = self.w_at(y)
        # baseline safety: clamp negative w to 0 to avoid NaNs
        return np.sqrt(np.maximum(w, 0.0) / self.T)

    @classmethod
    def from_params(
        cls, T: float, params: SVIParams, y_min: float = -2.0, y_max: float = 2.0
    ) -> SVISmile:
        return cls(T=T, params=params, _y_min=y_min, _y_max=y_max)


def svi_total_variance(y: ArrayLike, p: SVIParams) -> FloatArray:
    y_arr = np.asarray(y, dtype=float)
    z = y_arr - p.m
    out = p.a + p.b * (p.rho * z + np.sqrt(z**2 + p.sigma**2))
    return cast(FloatArray, out)


def svi_dw_dy(y: ArrayLike, p: SVIParams) -> FloatArray:
    """First derivative: b * (rho + (y-m)/sqrt((y-m)^2 + sigma^2))"""
    y_arr = np.asarray(y, dtype=float)
    z = y_arr - p.m
    out = p.b * (p.rho + z / np.sqrt(z**2 + p.sigma**2))
    return cast(FloatArray, out)


def svi_d2w_dy2(y: ArrayLike, p: SVIParams) -> FloatArray:
    """Second derivative: b * sigma^2 / ((y-m)^2 + sigma^2)^(3/2)"""
    y_arr = np.asarray(y, dtype=float)
    z = y_arr - p.m
    hypot_sq = z**2 + p.sigma**2
    out = p.b * (p.sigma**2 / np.power(hypot_sq, 1.5))
    return cast(FloatArray, out)


# def svi_jac(y: ArrayLike, p: SVIParams) -> ArrayLike:
#     # w = a + b*(rho*d + s)
#     y = np.asarray(y)

#     d = y-p.m
#     s = np.sqrt(d**2 + p.sigma**2)

#     dw_da = 1
#     dw_db = p.rho*d + s
#     dw_drho = p.b*d
#     dw_dm = -p.b*(p.rho + d/s)
#     dw_dsigma = p.b*(p.sigma/s)

#     jac_1d = np.asarray([dw_da, dw_db, dw_drho, dw_dm, dw_dsigma])

#     return jac_1d


RHO_MAX = 0.999
EPS = 1e-12


def softplus(x: NDArray[np.float64]) -> NDArray[np.float64]:
    # stable softplus
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def softplus_inv(y: NDArray[np.float64]) -> NDArray[np.float64]:
    # inverse of softplus for y>0, stable
    y = np.maximum(y, EPS)
    return y + np.log(-np.expm1(-y))


def decode_svi(u: NDArray[np.float64]) -> SVIParams:
    ua, ub, urho, um, usig = u

    b = float(softplus(np.array([ub], np.float64))[0] + EPS)
    sigma = float(softplus(np.array([usig], np.float64))[0] + EPS)
    rho = float(RHO_MAX * np.tanh(urho))
    m = float(um)

    # enforce w_min >= 0 by construction:
    alpha = float(softplus(np.array([ua], np.float64))[0] + EPS)
    a = alpha - b * sigma * np.sqrt(max(1.0 - rho * rho, 0.0))

    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)


def encode_svi(p: SVIParams) -> NDArray[np.float64]:
    # map a,b,rho,m,sigma -> unconstrained u
    # because a was defined via alpha, recover alpha first:
    alpha = p.a + p.b * p.sigma * np.sqrt(max(1.0 - p.rho * p.rho, 0.0))
    ua = float(softplus_inv(np.array([alpha], np.float64))[0])
    ub = float(softplus_inv(np.array([p.b], np.float64))[0])
    usig = float(softplus_inv(np.array([p.sigma], np.float64))[0])
    urho = float(np.arctanh(np.clip(p.rho / RHO_MAX, -0.999999, 0.999999)))
    um = float(p.m)
    return np.array([ua, ub, urho, um, usig], dtype=np.float64)


LOG2 = float(np.log(2.0))


def _soft_hinge_log_ratio(ratio: float) -> float:
    """
    Smooth hinge on log(ratio). Returns 0 when ratio==1,
    grows ~linearly with log(ratio) when ratio>1.
    ratio = sigma_floor / sigma
    """
    x = float(np.log(max(ratio, EPS)))
    # softplus(x) - log(2) ensures value=0 at x=0
    return float(softplus(np.array([x], np.float64))[0] - LOG2)


def _default_reg(
    y_arr: NDArray[np.float64], w_obs_arr: NDArray[np.float64]
) -> dict[str, float]:
    y_sorted = np.unique(np.sort(y_arr))
    dy = np.diff(y_sorted)
    dy_med = float(np.median(dy)) if dy.size else 0.1

    n = int(y_arr.size)
    n_factor = float(np.clip(10.0 / max(n, 1), 0.25, 4.0))

    # data-driven prior (safe default)
    m_prior = float(y_arr[int(np.argmin(w_obs_arr))])

    # scales
    qy05, qy95 = np.quantile(y_arr, [0.05, 0.95])
    y_span = float(max(qy95 - qy05, 1e-6))
    m_scale = float(max(0.25 * y_span, 0.10))

    sigma_floor = float(max(0.03, 1.5 * dy_med))

    well_determined = (n >= 20) and (y_sorted[0] < -0.3) and (y_sorted[-1] > 0.3)

    return {
        "m_prior": m_prior,
        "_m_scale": m_scale,
        "_sigma_floor": sigma_floor,
        "lambda_m": 0.0 if well_determined else 0.5 * n_factor,
        "lambda_inv_sigma": 0.0 if well_determined else 2.0 * n_factor,
        "lambda_b": 0.0,
        "lambda_rho": 0.0,
        "lambda_a": 0.0,
    }


def _merge_reg(
    y_arr: NDArray[np.float64],
    w_obs_arr: NDArray[np.float64],
    reg: dict[str, float] | None,
):
    base = _default_reg(y_arr, w_obs_arr)
    if reg is None:
        return base
    # allow user overrides
    return {**base, **reg}


def _auto_f_scale(y_arr, w_obs_arr, weights_arr, x0: SVIParams) -> float:
    r0 = weights_arr * (svi_total_variance(y_arr, x0) - w_obs_arr)
    mad = np.median(np.abs(r0 - np.median(r0)))
    return float(1.4826 * mad + 1e-12)


def calibrate_svi(
    y: ArrayLike,
    w_obs: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
    reg: dict[str, float] | None = None,
    x0: SVIParams | None = None,
    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "soft_l1",
    f_scale: float | None = None,
) -> SVIParams:
    """
    Fit raw SVI to total variance w(y) at one maturity.

    Parameters
    ----------
    y : array
        log-moneyness points (e.g., log(K/F(T))) for a fixed T
    w_obs : array
        observed total variance = iv**2 * T
    weights : array, optional
        per-point weights. Recommended: weights = 1/std_w  OR sqrt(vega)-style.
    reg : dict, optional
        regularization strengths. Example keys: lambda_b, lambda_rho, lambda_inv_sigma, lambda_m, m_prior
    x0 : array, optional
        initial guess [a,b,rho,m,sigma]
    loss, f_scale : robust fitting options for least_squares
    """
    y_arr = np.asarray(y, dtype=np.float64)
    w_obs_arr = np.asarray(w_obs, dtype=np.float64)

    assert y_arr.shape == w_obs_arr.shape and y_arr.ndim == 1

    if weights is None:
        weights_arr = np.ones_like(w_obs_arr)

    else:
        weights_arr = np.asarray(weights, dtype=np.float64)
        assert weights_arr.shape == w_obs_arr.shape

    if x0 is None:
        # prag starter
        i0 = int(np.argmin(w_obs_arr))
        m0 = float(y_arr[i0])
        sigma0 = 0.2
        rho0 = 0.0
        b0 = 0.1  # from wing slope estimate (or small value)
        a0 = float(
            max(1e-8, w_obs_arr[i0] - b0 * sigma0)
        )  # so that model matches min variance roughly
        x0 = SVIParams(a=a0, b=b0, rho=rho0, m=m0, sigma=sigma0)

    if f_scale is None:
        if weights is not None:
            f_scale = 1.0
        else:
            f_scale = float(
                np.clip(
                    _auto_f_scale(
                        y_arr=y_arr, w_obs_arr=w_obs_arr, weights_arr=weights_arr, x0=x0
                    ),
                    0.5,
                    2.0,
                )
            )

        # multi starter LATER!!!

    reg_cfg = _merge_reg(y_arr=y_arr, w_obs_arr=w_obs_arr, reg=reg)

    lam_m = float(reg_cfg["lambda_m"])
    lam_sig = float(reg_cfg["lambda_inv_sigma"])
    m_prior = float(reg_cfg["m_prior"])
    m_scale = float(reg_cfg["_m_scale"])
    sigma_floor = float(reg_cfg["_sigma_floor"])

    def residuals(u: NDArray[np.float64]) -> NDArray[np.float64]:
        p = decode_svi(u)

        w_model = svi_total_variance(y=y_arr, p=p)
        # IMPORTANT: weights should usually be sqrt-weights
        # (e.g. 1/std), so multiplying residual is correct.
        r_data: NDArray[np.float64] = weights_arr * (w_model - w_obs_arr)

        r_reg: list[float] = []

        # m anchor (dimensionless)
        if lam_m > 0.0:
            r_reg.append(np.sqrt(lam_m) * (p.m - m_prior) / m_scale)

        if lam_sig > 0.0:
            ratio = sigma_floor / max(p.sigma, EPS)
            r_reg.append(np.sqrt(lam_sig) * _soft_hinge_log_ratio(ratio=ratio))

        if r_reg:
            return np.concatenate([r_data, np.array(r_reg, dtype=np.float64)])
        return r_data

    # jac = svi_jac
    u0 = encode_svi(x0)
    res = least_squares(
        fun=residuals,
        x0=u0,
        loss=loss,
        f_scale=f_scale,
        x_scale="jac",
        max_nfev=5000,
    )

    # Minimal diagnostics, later return (params, diagnostics)
    if not res.success or not np.all(np.isfinite(res.x)):
        raise ValueError(f"SVI calibration failed: {res.message}")

    p_star = decode_svi(res.x)

    # Important guardrails
    # Constrain parameters to avoid pathological fits
    # Regularize the objective (mild) to discourage crazy curvature in wings
    # Fail gracefully: if fit is bad, keep a flag + allow fallback to grid Smile for that expiry
    return p_star


###
# No ARB CONDITIONS!!!!
###
