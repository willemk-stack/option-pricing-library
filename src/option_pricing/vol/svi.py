from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares as _least_squares

from .types import ArrayLike, FloatArray

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
    # Added these to satisfy your y_min/y_max properties
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


def calibrate_svi(
    y: ArrayLike,
    w_obs: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
    bounds: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    reg: dict[str, float] | None = None,
    x0: NDArray[np.float64] | None = None,
    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "soft_l1",
    f_scale: float = 0.1,
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
    bounds : (lb, ub), optional
        bounds for [a,b,rho,m,sigma]. If None, sensible defaults used.
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

    if bounds is None:
        lb = np.array(
            [0.0, 1e-10, -0.999, float(np.min(y_arr)) - 1.0, 1e-6], dtype=np.float64
        )
        ub = np.array(
            [
                float(np.max(w_obs_arr)) * 2.0 + 1.0,
                10.0,
                0.999,
                float(np.max(y_arr)) + 1.0,
                5.0,
            ],
            dtype=np.float64,
        )
        bounds = (lb, ub)

    # if reg is None:
    #     reg = {}

    #   lam_reg construction
    # lam_b = float(reg.get("lambda_b", 0.0))
    # lam_rho = float(reg.get("lambda_rho", 0.0))
    # lam_inv_sigma = float(reg.get("lambda_inv_sigma", 0.0))
    # lam_m = float(reg.get("lambda_m", 0.0))
    # m_prior = float(reg.get("m_prior", 0.0))
    # ...

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
        x0 = np.array([a0, b0, rho0, m0, sigma0], dtype=np.float64)
        # multi starter LATER!!!

    # transformations?
    # ...

    def residuals(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        p = SVIParams(
            a=float(theta[0]),
            b=float(theta[1]),
            rho=float(theta[2]),
            m=float(theta[3]),
            sigma=float(theta[4]),
        )

        w_model = svi_total_variance(y=y_arr, p=p)

        r_data: NDArray[np.float64] = weights_arr * (w_model - w_obs_arr)

        r_reg: list[float] = []

        if r_reg:
            return np.concatenate([r_data, np.array(r_reg, dtype=np.float64)])

        else:
            return r_data

    # jac = svi_jac
    res = least_squares(
        fun=residuals,
        x0=x0,
        bounds=bounds,
        loss=loss,
        f_scale=f_scale,
        x_scale="jac",
        max_nfev=5000,
    )

    # Minimal diagnostics, later return (params, diagnostics)
    if not res.success or not np.all(np.isfinite(res.x)):
        raise ValueError(f"SVI calibration failed: {res.message}")

    a, b, rho, m, sigma = res.x

    # Important guardrails
    # Constrain parameters to avoid pathological fits
    # Regularize the objective (mild) to discourage crazy curvature in wings
    # Fail gracefully: if fit is bad, keep a flag + allow fallback to grid Smile for that expiry
    return SVIParams(float(a), float(b), float(rho), float(m), float(sigma))
