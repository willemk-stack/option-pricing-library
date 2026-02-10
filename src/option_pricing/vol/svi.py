from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from option_pricing.exceptions import (
    DerivativeTooSmallError,
    NoConvergenceError,
    NotBracketedError,
)
from option_pricing.numerics.root_finding import bracketed_newton
from option_pricing.typing import ArrayLike, FloatArray

# ==========================
# Constants / types
# ==========================
RHO_MAX = 0.999
EPS = 1e-12
LOG2 = float(np.log(2.0))

DomainMode = Literal["fixed_logmoneyness", "quantile_pad", "explicit"]


# ==========================
# Domain config
# ==========================
@dataclass(frozen=True, slots=True)
class DomainCheckConfig:
    """
    Defines the y-domain used for post-fit safety checks (and diagnostic reporting).

    Modes:
      - "fixed_logmoneyness": use [y_min, y_max] (stable, recommended default if y=log(K/F)).
      - "quantile_pad": robust range from data quantiles + padding (adaptive fallback).
      - "explicit": use [y_min, y_max] as an explicit domain (same as fixed, but semantically “user chose this”).

    w_floor:
      Minimum allowed total variance over the domain. Use 0.0 for strict nonnegativity.
      Use a small positive floor (e.g. 1e-10) if downstream does logs/sqrts and you want a buffer.

    tol:
      Numerical tolerance for violations: consider w < w_floor - tol a violation.

    n_grid:
      Sampling density for "domain" checks and reports.
    """

    mode: DomainMode = "fixed_logmoneyness"

    # For fixed/explicit
    y_min: float = -1.25
    y_max: float = 1.25

    # For quantile_pad
    q_lo: float = 0.01
    q_hi: float = 0.99
    pad_frac: float = 0.15
    pad_abs: float = 0.05

    # Sampling density
    n_grid: int = 41

    # Acceptance threshold
    w_floor: float = 0.0
    tol: float = 1e-12


def build_domain_grid(
    y: NDArray[np.float64], cfg: DomainCheckConfig
) -> tuple[tuple[float, float], NDArray[np.float64]]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(max(cfg.n_grid, 5))

    if cfg.mode in ("fixed_logmoneyness", "explicit"):
        y_lo = float(min(cfg.y_min, cfg.y_max))
        y_hi = float(max(cfg.y_min, cfg.y_max))
        y_chk = np.linspace(y_lo, y_hi, n, dtype=np.float64)
        return (y_lo, y_hi), y_chk

    if cfg.mode == "quantile_pad":
        if y.size == 0:
            y_lo, y_hi = 0.0, 0.0
            return (y_lo, y_hi), np.linspace(y_lo, y_hi, n, dtype=np.float64)

        # robust domain from quantiles
        if y.size >= 20:
            q_lo = float(np.clip(cfg.q_lo, 0.0, 0.49))
            q_hi = float(np.clip(cfg.q_hi, 0.51, 1.0))
            q1, q2 = np.quantile(y, [q_lo, q_hi])
        else:
            q1, q2 = float(np.min(y)), float(np.max(y))

        span = float(max(q2 - q1, 1e-6))
        pad = float(cfg.pad_frac) * span + float(cfg.pad_abs)

        y_lo = float(q1 - pad)
        y_hi = float(q2 + pad)
        y_chk = np.linspace(y_lo, y_hi, n, dtype=np.float64)
        return (y_lo, y_hi), y_chk

    raise ValueError(f"Unknown DomainCheckConfig.mode: {cfg.mode}")


# ==========================
# Robust loss helpers (for IRLS)
# ==========================
def _robust_rhoprime(z: NDArray[np.float64], loss: str) -> NDArray[np.float64]:
    """
    rho'(z) for SciPy least_squares robust losses, where z = (r / f_scale)^2.

    In IRLS we use weights w_i = rho'(z_i) and solve a weighted LS step.
    """
    z = np.asarray(z, dtype=np.float64)
    if loss == "linear":
        return np.ones_like(z)

    if loss == "soft_l1":
        # rho(z) = 2*(sqrt(1+z)-1)  => rho'(z) = 1/sqrt(1+z)
        return 1.0 / np.sqrt(1.0 + z)

    if loss == "huber":
        # rho(z)=z if z<=1 else 2*sqrt(z)-1  => rho'(z)=1 if z<=1 else 1/sqrt(z)
        w = np.ones_like(z)
        big = z > 1.0
        w[big] = 1.0 / np.sqrt(z[big])
        return w

    if loss == "cauchy":
        # rho(z)=ln(1+z) => rho'(z)=1/(1+z)
        return 1.0 / (1.0 + z)

    if loss == "arctan":
        # rho(z)=arctan(z) => rho'(z)=1/(1+z^2)
        return 1.0 / (1.0 + z * z)

    raise ValueError(f"Unknown loss: {loss}")


# ==========================
# Wing slope estimation (data diagnostics / priors)
# ==========================
def _weighted_linear_fit(
    y: NDArray[np.float64],
    w: NDArray[np.float64],
    sqrt_weights: NDArray[np.float64],
) -> tuple[float, float]:
    """
    Fit w ~ c + s*y using sqrt-weights (i.e. minimize ||sqrt_w*(Xb - w)||_2).
    Returns (c, s).
    """
    y = np.asarray(y, np.float64).reshape(-1)
    w = np.asarray(w, np.float64).reshape(-1)
    sw = np.asarray(sqrt_weights, np.float64).reshape(-1)

    if y.shape != w.shape or y.shape != sw.shape:
        raise ValueError("y, w, sqrt_weights must have the same shape")

    X = np.column_stack([np.ones_like(y), y])  # (n, 2)
    WX = sw[:, None] * X
    Wy = sw * w
    beta, *_ = np.linalg.lstsq(WX, Wy, rcond=None)
    return float(beta[0]), float(beta[1])


def estimate_wing_slopes_one_sided(
    y: NDArray[np.float64],
    w: NDArray[np.float64],
    sqrt_weights: NDArray[np.float64] | None = None,
    *,
    wing_threshold: float = 0.30,
    q_tail: float | None = None,
    min_pts_cap: int = 12,
) -> tuple[float | None, float | None]:
    """
    Estimate observed wing slopes (sL_obs, sR_obs) by fitting a line to
    the left and right tails of (y, w).

    Tail regression points are restricted to those beyond +/- wing_threshold.
    """
    y = np.asarray(y, np.float64).reshape(-1)
    w = np.asarray(w, np.float64).reshape(-1)

    if y.shape != w.shape:
        raise ValueError("y and w must have the same shape")
    if y.size < 8:
        return None, None
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(w))):
        return None, None

    sw = (
        np.ones_like(w)
        if sqrt_weights is None
        else np.asarray(sqrt_weights, np.float64).reshape(-1)
    )
    if sw.shape != w.shape:
        raise ValueError("sqrt_weights must have the same shape as w")
    if not (np.all(np.isfinite(sw)) and np.all(sw >= 0.0)):
        return None, None

    n = int(y.size)

    # choose tail fraction adaptively if not provided
    if q_tail is None:
        if n < 20:
            q_tail = 0.25
        elif n < 35:
            q_tail = 0.20
        else:
            q_tail = 0.10
    q_tail = float(np.clip(q_tail, 0.05, 0.45))

    min_pts = int(np.clip(0.12 * n, 4, min_pts_cap))
    k = max(min_pts, int(np.ceil(q_tail * n)))

    sL_obs: float | None = None
    sR_obs: float | None = None

    # Left wing
    left_cand = np.where(y < -wing_threshold)[0]
    if left_cand.size >= min_pts:
        order = np.argsort(y[left_cand])  # most negative first
        left = left_cand[order[: min(k, left_cand.size)]]
        if float(np.sum(sw[left])) > 0.0:
            _, sL = _weighted_linear_fit(y[left], w[left], sw[left])
            if np.isfinite(sL):
                sL = float(sL)
                if sL <= 0.0:
                    sL_obs = sL

    # Right wing
    right_cand = np.where(y > wing_threshold)[0]
    if right_cand.size >= min_pts:
        order = np.argsort(y[right_cand])  # ascending
        right = right_cand[order[-min(k, right_cand.size) :]]
        if float(np.sum(sw[right])) > 0.0:
            _, sR = _weighted_linear_fit(y[right], w[right], sw[right])
            if np.isfinite(sR):
                sR = float(sR)
                if sR >= 0.0:
                    sR_obs = sR

    return sL_obs, sR_obs


# ==========================
# Smooth transforms
# ==========================
def sigmoid(x: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x_arr)
    pos = x_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    ex = np.exp(x_arr[~pos])
    out[~pos] = ex / (1.0 + ex)
    return float(out) if out.ndim == 0 else out


def softplus(x: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.log1p(np.exp(-np.abs(x_arr))) + np.maximum(x_arr, 0.0)
    return float(out) if out.ndim == 0 else out


def softplus_inv(y: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    y_arr = np.asarray(y, dtype=np.float64)
    y_arr = np.maximum(y_arr, EPS)
    out = y_arr + np.log(-np.expm1(-y_arr))
    return float(out) if out.ndim == 0 else out


def logit(x: float) -> float:
    x = float(np.clip(x, 1e-12, 1.0 - 1e-12))
    return float(np.log(x) - np.log1p(-x))


# ==========================
# SVI: params + w(y) + derivatives
# ==========================


@dataclass(frozen=True, slots=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def to_jw(self, T: float) -> JWParams:
        return raw_to_jw(self, T)


@dataclass(frozen=True, slots=True)
class JWParams:
    v: float
    psi: float
    p: float
    c: float
    v_tilde: float

    def to_raw(self, T: float) -> SVIParams:
        return jw_to_raw(self, T)


def raw_to_jw(p_raw: SVIParams, T: float) -> JWParams:
    """
    Raw SVI (a,b,rho,m,sigma) -> SVI-JW (v,psi,p,c,ve) exactly per
    Gatheral–Jacquier (2013), Eq. (3.5).
    """
    if T <= 0:
        raise ValueError("T must be > 0")

    a, b, rho, m, sigma = p_raw.a, p_raw.b, p_raw.rho, p_raw.m, p_raw.sigma
    if b < 0:
        raise ValueError("Raw SVI requires b >= 0")
    if not (-1.0 < rho < 1.0):
        raise ValueError("Raw SVI requires |rho| < 1")
    if sigma <= 0:
        raise ValueError("Raw SVI requires sigma > 0")

    # sqrt(m^2 + sigma^2) -- stable
    sqrt_m2_sigma2 = math.hypot(m, sigma)

    # w_t := w(0) = a + b(-rho*m + sqrt(m^2+sigma^2))
    w_t = a + b * (-rho * m + sqrt_m2_sigma2)
    if w_t <= 0:
        raise ValueError(f"ATM total variance w_t must be > 0, got {w_t}")

    sqrt_wt = math.sqrt(w_t)
    one_over_sqrt_wt = 1.0 / sqrt_wt

    v_t = w_t / T

    psi_t = one_over_sqrt_wt * (b * 0.5) * (rho - m / sqrt_m2_sigma2)

    p_t = one_over_sqrt_wt * b * (1.0 - rho)
    c_t = one_over_sqrt_wt * b * (1.0 + rho)

    ve_t = (a + b * sigma * math.sqrt(1.0 - rho * rho)) / T

    return JWParams(v=v_t, psi=psi_t, p=p_t, c=c_t, v_tilde=ve_t)


def jw_to_raw(jw: JWParams, T: float) -> SVIParams:
    """
    SVI-JW (v,psi,p,c,ve) -> Raw SVI (a,b,rho,m,sigma) exactly per
    Gatheral–Jacquier (2013), Lemma 3.2.

    Notes:
    - Closed-form inversion assumes m != 0. We implement GJ's m=0 branch too.
    - In the degenerate case m=0 and rho≈0, JW does not uniquely identify sigma/a.
    """
    if T <= 0:
        raise ValueError("T must be > 0")

    v_t, psi_t, p_t, c_t, ve_t = jw.v, jw.psi, jw.p, jw.c, jw.v_tilde

    _validate_jw(jw=jw, T=T)

    w_t = v_t * T
    if w_t <= 0:
        raise ValueError("ATM total variance w_t must be > 0")

    sqrt_wt = math.sqrt(w_t)

    # Lemma 3.2: b = sqrt(w_t)/2 * (c_t + p_t)
    b = 0.5 * sqrt_wt * (c_t + p_t)
    if b <= 0:
        raise ValueError("Computed b must be > 0")

    # Lemma 3.2: rho = 1 - p_t*sqrt(w_t)/b  (equivalent to (c-p)/(c+p))
    rho = (c_t - p_t) / (c_t + p_t)
    # keep inside (-1,1) numerically
    rho = max(-1.0 + 1e-15, min(1.0 - 1e-15, rho))

    sqrt_1mr2 = math.sqrt(max(0.0, 1.0 - rho * rho))

    # Lemma 3.2: beta := rho - 2*psi_t*sqrt(w_t)/b
    beta = rho - (2.0 * psi_t * sqrt_wt / b)

    # Numerically clamp beta to [-1,1] (GJ assume beta in [-1,1])
    beta = max(-1.0 + 1e-15, min(1.0 - 1e-15, beta))

    # beta = m / sqrt(m^2 + sigma^2); so beta==0 corresponds to m==0
    if abs(beta) < 1e-10:
        # m = 0 branch.
        # With m=0: w_t = a + b*sigma,  ve_t*T = a + b*sigma*sqrt(1-rho^2)
        # => (v_t - ve_t)*T = b*sigma*(1 - sqrt(1-rho^2))
        denom = b * (1.0 - sqrt_1mr2)

        if abs(denom) < 1e-12:
            # This is the degenerate limit rho -> 0, where ve_t == v_t and sigma is not identified by JW.
            raise ValueError(
                "Degenerate JW->Raw case: m≈0 and rho≈0 implies ve≈v; "
                "sigma (and a) are not uniquely identified from JW alone."
            )

        sigma = ((v_t - ve_t) * T) / denom
        if sigma <= 0:
            # numerical safety
            sigma = abs(sigma)

        a = w_t - b * sigma
        m = 0.0

        return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)

    # Lemma 3.2: alpha := sign(beta)*sqrt(1/beta^2 - 1)
    alpha = math.copysign(math.sqrt(1.0 / (beta * beta) - 1.0), beta)
    sign_alpha = 1.0 if alpha >= 0.0 else -1.0

    # (v_t - ve_t)*T = b*m * {-rho + sign(alpha)*sqrt(1+alpha^2) - alpha*sqrt(1-rho^2)}
    bracket = -rho + sign_alpha * math.sqrt(1.0 + alpha * alpha) - alpha * sqrt_1mr2
    if abs(bracket) < EPS:
        raise ValueError("Numerical issue: inversion bracket too close to 0")

    m = ((v_t - ve_t) * T) / (b * bracket)
    sigma = alpha * m
    if sigma <= 0:
        # enforce raw convention sigma > 0 (can be tiny negative due to rounding)
        sigma = abs(sigma)

    # Lemma 3.2: a = ve_t*T - b*sigma*sqrt(1-rho^2)
    a = ve_t * T - b * sigma * sqrt_1mr2

    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)


def _validate_jw(jw: JWParams, T: float, *, tol: float = 1e-12) -> None:
    if T <= 0:
        raise ValueError("T must be > 0")
    if jw.v <= 0:
        raise ValueError("JW v must be > 0")
    if jw.p < 0 or jw.c < 0:
        raise ValueError("JW p,c must be >= 0")
    if jw.p + jw.c <= tol:
        raise ValueError("JW requires p + c > 0")
    if jw.v_tilde < 0:
        raise ValueError("JW v_tilde must be >= 0")
    if jw.v_tilde > jw.v + tol:
        raise ValueError("Invalid JW: v_tilde must be <= v")
    if (2.0 * jw.psi) < (-jw.p - tol) or (2.0 * jw.psi) > (jw.c + tol):
        raise ValueError("Invalid JW: must satisfy -p <= 2*psi <= c")


def svi_total_variance(y: NDArray[np.float64], p: SVIParams) -> NDArray[np.float64]:
    z = y - p.m
    s = np.hypot(z, p.sigma)
    return p.a + p.b * (p.rho * z + s)


def svi_total_variance_dy(y: NDArray[np.float64], p: SVIParams) -> NDArray[np.float64]:
    """First derivative d/dy of SVI total variance w(y)."""
    z = y - p.m
    s = np.hypot(z, p.sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(s > 0.0, z / s, 0.0)
    return p.b * (p.rho + frac)


def svi_total_variance_dyy(y: NDArray[np.float64], p: SVIParams) -> NDArray[np.float64]:
    """Second derivative d^2/dy^2 of SVI total variance w(y)."""
    z = y - p.m
    s = np.hypot(z, p.sigma)
    s3 = s * s * s
    sig2 = p.sigma * p.sigma
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(s3 > 0.0, p.b * sig2 / s3, 0.0)
    return out


def svi_total_variance_dyyy(
    y: NDArray[np.float64], p: SVIParams
) -> NDArray[np.float64]:
    """Third derivative d^3/dy^3 of SVI total variance w(y)."""
    z = y - p.m
    s = np.hypot(z, p.sigma)
    s5 = s**5
    sig2 = p.sigma * p.sigma
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(s5 > 0.0, -3.0 * p.b * sig2 * z / s5, 0.0)
    return out


# ==========================
# Regularization config + defaults
# ==========================
@dataclass(frozen=True, slots=True)
class SVIRegConfig:
    lambda_m: float = 0.5
    m_prior: float = 0.0
    m_scale: float = 0.2

    lambda_inv_sigma: float = 2.0
    sigma_floor: float = 0.03

    lambda_slope_L: float = 0.75
    lambda_slope_R: float = 0.75
    slope_cap: float = 1.999  # Lee bound cap for total variance slope

    slope_floor: float = 0.10
    slope_denom: float = 0.15  # overwritten per smile

    lambda_g: float = 1e5
    g_scale: float = 0.02
    g_floor: float = 0.0
    g_n_grid: int = 81  # optional if you use the data-based n above


class SVIRegOverrideDict(TypedDict, total=False):
    lambda_m: float
    m_prior: float
    m_scale: float
    lambda_inv_sigma: float
    sigma_floor: float
    lambda_slope_L: float
    lambda_slope_R: float
    slope_cap: float
    slope_floor: float
    slope_denom: float
    lambda_g: float
    g_scale: float
    g_floor: float
    g_n_grid: int


RegOverride = SVIRegOverrideDict | Callable[[SVIRegConfig], SVIRegConfig]


def default_reg_from_data(
    y: NDArray[np.float64],
    w_obs: NDArray[np.float64],
    sqrt_w: NDArray[np.float64] | None = None,
) -> SVIRegConfig:
    y_sorted = np.unique(np.sort(y))
    dy = np.diff(y_sorted)
    dy_med = float(np.median(dy)) if dy.size else 0.1

    n = int(y.size)

    n_factor = float(np.clip(10.0 / max(n, 1), 0.25, 4.0))

    rail_factor = float(np.clip(30.0 / max(n, 1), 1.0, 10.0))
    m_prior = float(y[int(np.argmin(w_obs))])

    q05, q95 = np.quantile(y, [0.05, 0.95])
    y_span = float(max(q95 - q05, 1e-6))
    m_scale = float(max(0.25 * y_span, 0.10))

    sigma_floor = float(max(0.03, 1.5 * dy_med))

    slope_denom = 0.15

    well_determined = (n >= 5000) and (y_sorted[0] < -0.3) and (y_sorted[-1] > 0.3)

    return SVIRegConfig(
        lambda_m=0.0 if well_determined else 0.5 * n_factor,
        m_prior=m_prior,
        m_scale=m_scale,
        lambda_inv_sigma=0.0 if well_determined else 10.0 * rail_factor,
        sigma_floor=sigma_floor,
        lambda_slope_L=0.0 if well_determined else 0.25 * n_factor,
        lambda_slope_R=0.0 if well_determined else 0.25 * n_factor,
        slope_cap=1.999,
        slope_denom=slope_denom,
    )


def apply_reg_override(
    base: SVIRegConfig, override: RegOverride | None
) -> SVIRegConfig:
    if override is None:
        return base
    if callable(override):
        out = override(base)
        if not isinstance(out, SVIRegConfig):
            raise TypeError("reg_override callable must return SVIRegConfig")
        return out
    return replace(base, **override)


def svi_jac_wrt_params(y: NDArray[np.float64], p: SVIParams) -> NDArray[np.float64]:
    """
    Jacobian of w(y) wrt [a, b, rho, m, sigma]; shape (n, 5).
    """
    d = y - p.m
    s = np.hypot(d, p.sigma)

    dw_da = np.ones_like(y)
    dw_db = p.rho * d + s
    dw_drho = p.b * d
    with np.errstate(divide="ignore", invalid="ignore"):
        dw_dm = -p.b * (p.rho + d / s)
        dw_dsigma = p.b * (p.sigma / s)

    return np.stack([dw_da, dw_db, dw_drho, dw_dm, dw_dsigma], axis=-1)


def svi_jac_w1_wrt_params(y: NDArray[np.float64], p: SVIParams) -> NDArray[np.float64]:
    z = y - p.m
    s = np.hypot(z, p.sigma)
    s3 = s**3
    sig = p.sigma
    sig2 = sig * sig

    dw1_da = np.zeros_like(y)
    dw1_db = p.rho + np.where(s > 0, z / s, 0.0)
    dw1_drho = np.full_like(y, p.b)
    dw1_dm = np.where(s3 > 0, -p.b * sig2 / s3, 0.0)
    dw1_dsig = np.where(s3 > 0, -p.b * z * sig / s3, 0.0)

    return np.stack([dw1_da, dw1_db, dw1_drho, dw1_dm, dw1_dsig], axis=-1)


def svi_jac_w2_wrt_params(y: NDArray[np.float64], p: SVIParams) -> NDArray[np.float64]:
    z = y - p.m
    s = np.hypot(z, p.sigma)
    s3 = s**3
    s5 = s**5
    sig = p.sigma
    sig2 = sig * sig

    dw2_da = np.zeros_like(y)
    dw2_db = np.where(s3 > 0, sig2 / s3, 0.0)
    dw2_drho = np.zeros_like(y)
    dw2_dm = np.where(s5 > 0, 3.0 * p.b * sig2 * z / s5, 0.0)
    dw2_dsig = np.where(s5 > 0, p.b * sig * (2.0 * z * z - sig2) / s5, 0.0)

    return np.stack([dw2_da, dw2_db, dw2_drho, dw2_dm, dw2_dsig], axis=-1)


def gatheral_g_jac_params(
    y: NDArray[np.float64], p: SVIParams, *, w_floor: float = 0.0
) -> NDArray[np.float64]:
    y = np.asarray(y, np.float64)
    w = svi_total_variance(y, p)
    w1 = svi_total_variance_dy(y, p)

    bad = w <= float(w_floor)
    w_safe = np.where(bad, np.nan, w)

    A = 1.0 - y * w1 / (2.0 * w_safe)

    Cw = (A * y * w1) / (w_safe * w_safe) + (w1 * w1) / (4.0 * w_safe * w_safe)
    C1 = -(A * y) / w_safe - (w1 / 2.0) * (1.0 / w_safe + 0.25)

    Jw = svi_jac_wrt_params(y, p)  # (n,5)
    Jw1 = svi_jac_w1_wrt_params(y, p)  # (n,5)
    Jw2 = svi_jac_w2_wrt_params(y, p)  # (n,5)

    Jg = Cw[:, None] * Jw + C1[:, None] * Jw1 + 0.5 * Jw2

    # If w is invalid at a point, don't contribute this row via g (handle via separate w-floor penalty)
    Jg = np.where(np.isfinite(Jg), Jg, 0.0)
    return Jg


# ==========================
# SVI Transform enforcing Lee caps + positivity rails
# ==========================
@dataclass(frozen=True, slots=True)
class SVITransformLeeCap:
    """
    u = [ua, uR, uL, um, usig]

    Slopes (Lee-capped) with a strictly-positive floor:
      s(u) = s_min + (cap - s_min) * sigmoid(u)   in (s_min, cap)

      sR    = s(uR)
      sLmag = s(uL)

    Then:
      b   = 0.5*(sR + sLmag) + eps
      rho = (sR - sLmag)/(sR + sLmag + eps)

    a is encoded via alpha >= 0:
      a = alpha - b*sigma*sqrt(1-rho^2)
      alpha = softplus(ua) + eps
      sigma = softplus(usig) + eps
      m = um
    """

    slope_cap: float = 1.999
    slope_min: float = 1e-4
    eps: float = EPS

    def decode(self, u: NDArray[np.float64]) -> SVIParams:
        ua, uR, uL, um, usig = map(float, u)

        alpha = float(softplus(ua)) + self.eps
        sigma = float(softplus(usig)) + self.eps
        m = float(um)

        cap = float(self.slope_cap)
        s_min = float(self.slope_min)
        span = cap - s_min
        if span <= 0.0:
            raise ValueError("slope_cap must be > slope_min")

        sR = s_min + span * float(sigmoid(uR))
        sLmag = s_min + span * float(sigmoid(uL))

        b = 0.5 * (sR + sLmag) + self.eps
        denom = (sR + sLmag) + self.eps
        rho = (sR - sLmag) / denom  # in (-1, 1)

        k2 = max(1.0 - rho * rho, 0.0)
        a = alpha - b * sigma * float(np.sqrt(k2))

        return SVIParams(a=float(a), b=float(b), rho=float(rho), m=m, sigma=sigma)

    def encode(self, p: SVIParams) -> NDArray[np.float64]:
        if p.sigma <= 0.0 or p.b <= 0.0:
            raise ValueError("encode expects b>0 and sigma>0")

        cap = float(self.slope_cap)
        s_min = float(self.slope_min)
        span = cap - s_min
        if span <= 0.0:
            raise ValueError("slope_cap must be > slope_min")

        # convert (b,rho) -> slopes
        sR = float(p.b) * (1.0 + float(p.rho))
        sLmag = float(p.b) * (1.0 - float(p.rho))

        tiny = 1e-10
        sR = float(np.clip(sR, s_min + tiny, cap - tiny))
        sLmag = float(np.clip(sLmag, s_min + tiny, cap - tiny))

        sigR = float(np.clip((sR - s_min) / span, 1e-12, 1.0 - 1e-12))
        sigL = float(np.clip((sLmag - s_min) / span, 1e-12, 1.0 - 1e-12))

        uR = logit(sigR)
        uL = logit(sigL)

        alpha = float(p.a + p.b * p.sigma * np.sqrt(max(1.0 - p.rho * p.rho, 0.0)))
        alpha = max(alpha, self.eps)
        ua = float(softplus_inv(alpha))

        um = float(p.m)
        usig = float(softplus_inv(max(float(p.sigma), self.eps)))

        return np.array([ua, uR, uL, um, usig], dtype=np.float64)

    def dp_du(self, u: NDArray[np.float64], p: SVIParams) -> NDArray[np.float64]:
        """
        dp/du as (5,5), p=[a,b,rho,m,sigma], u=[ua,uR,uL,um,usig]
        """
        ua, uR, uL, um, usig = map(float, u)

        cap = float(self.slope_cap)
        s_min = float(self.slope_min)
        span = cap - s_min
        if span <= 0.0:
            raise ValueError("slope_cap must be > slope_min")

        sigR = float(sigmoid(uR))
        sigL = float(sigmoid(uL))

        sR = s_min + span * sigR
        sLmag = s_min + span * sigL

        dsR_duR = span * sigR * (1.0 - sigR)
        dsL_duL = span * sigL * (1.0 - sigL)

        D = float(sR + sLmag + self.eps)

        db_duR = 0.5 * dsR_duR
        db_duL = 0.5 * dsL_duL

        # rho = (sR - sLmag)/D
        drho_dsR = (2.0 * sLmag + self.eps) / (D * D)
        drho_dsL = (-2.0 * sR - self.eps) / (D * D)
        drho_duR = drho_dsR * dsR_duR
        drho_duL = drho_dsL * dsL_duL

        dalpha_dua = float(sigmoid(ua))  # d softplus / d ua
        dsig_dusig = float(sigmoid(usig))  # d softplus / d usig

        rho = float(p.rho)
        b = float(p.b)
        sigma = float(p.sigma)

        k2 = max(1.0 - rho * rho, 0.0)
        k = float(np.sqrt(k2))
        k_safe = max(k, 1e-12)

        # a = alpha - b*sigma*k, dk/drho = -rho/k
        common_rho = b * sigma * (rho / k_safe)

        da_dua = dalpha_dua
        da_duR = -(db_duR * sigma * k) + common_rho * drho_duR
        da_duL = -(db_duL * sigma * k) + common_rho * drho_duL
        da_dusig = -(b * dsig_dusig * k)

        return np.array(
            [
                [da_dua, da_duR, da_duL, 0.0, da_dusig],  # a
                [0.0, db_duR, db_duL, 0.0, 0.0],  # b
                [0.0, drho_duR, drho_duL, 0.0, 0.0],  # rho
                [0.0, 0.0, 0.0, 1.0, 0.0],  # m
                [0.0, 0.0, 0.0, 0.0, dsig_dusig],  # sigma
            ],
            dtype=np.float64,
        )


# ==========================
# Smooth hinge penalties
# ==========================
def soft_hinge_log_ratio(ratio: float) -> float:
    x = float(np.log(max(ratio, EPS)))  # x > 0 iff sigma < floor
    h = float(softplus(x) - LOG2)  # smooth around 0
    return max(h, 0.0)  # true hinge (piecewise)


def soft_hinge_ratio_excess(ratio: float) -> float:
    return float(softplus(ratio - 1.0))


# ==========================
# Calibration objective
# ==========================
@dataclass
class SVIObjective:
    y: NDArray[np.float64]
    w_obs: NDArray[np.float64]
    sqrt_w: NDArray[np.float64]
    transform: SVITransformLeeCap
    reg: SVIRegConfig
    sL_obs: float | None = None
    sR_obs: float | None = None
    y_g: NDArray[np.float64] | None = None
    w_floor: float = 0.0

    def __post_init__(self):
        sw = np.asarray(self.sqrt_w, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(sw)) or np.any(sw < 0.0):
            raise ValueError("sqrt_weights must be finite and nonnegative")

    def _usable_obs_slopes(self) -> tuple[float | None, float | None]:
        """Apply sign sanity and Lee-cap clipping to observed slope targets."""
        cap = float(self.reg.slope_cap)
        cap_eff = cap * 0.995

        sL = self.sL_obs
        sR = self.sR_obs

        if sR is not None:
            sR = float(sR)
            if not np.isfinite(sR) or sR < 0.0:
                sR = None
            else:
                sR = float(np.clip(sR, 0.0, cap_eff))

        if sL is not None:
            sL = float(sL)
            if not np.isfinite(sL) or sL > 0.0:
                sL = None
            else:
                mag = float(np.clip(-sL, 0.0, cap_eff))
                sL = -mag

        return sL, sR

    def residual(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        p = self.transform.decode(u)
        w_model = svi_total_variance(self.y, p)
        r = self.sqrt_w * (w_model - self.w_obs)

        reg_terms: list[NDArray[np.float64]] = []

        if self.reg.lambda_m > 0.0:
            val = (
                np.sqrt(self.reg.lambda_m) * (p.m - self.reg.m_prior) / self.reg.m_scale
            )
            reg_terms.append(np.array([val], dtype=np.float64))

        if self.reg.lambda_inv_sigma > 0.0:
            ratio = self.reg.sigma_floor / max(p.sigma, EPS)
            val = np.sqrt(self.reg.lambda_inv_sigma) * soft_hinge_log_ratio(ratio)
            reg_terms.append(np.array([val], dtype=np.float64))

        sL_use, sR_use = self._usable_obs_slopes()
        cap = float(self.reg.slope_cap)
        s_min = float(self.transform.slope_min)
        span = cap - s_min

        if self.reg.lambda_slope_R > 0.0 and sR_use is not None:
            uR = float(u[1])
            sR_model = s_min + span * float(sigmoid(uR))
            val = (
                np.sqrt(self.reg.lambda_slope_R)
                * (sR_model - sR_use)
                / self.reg.slope_denom
            )
            reg_terms.append(np.array([val], dtype=np.float64))

        if self.reg.lambda_slope_L > 0.0 and sL_use is not None:
            uL = float(u[2])
            sL_model = -(s_min + span * float(sigmoid(uL)))
            val = (
                np.sqrt(self.reg.lambda_slope_L)
                * (sL_model - sL_use)
                / self.reg.slope_denom
            )
            reg_terms.append(np.array([val], dtype=np.float64))

        if self.reg.lambda_g > 0.0 and self.y_g is not None and self.y_g.size:
            g = gatheral_g_vec(self.y_g, p, w_floor=self.w_floor)
            deficit = (self.reg.g_floor - g) / max(self.reg.g_scale, 1e-12)
            h = softplus(deficit) - LOG2
            h = np.maximum(h, 0.0)
            r_g = (np.sqrt(self.reg.lambda_g) * h).astype(np.float64, copy=False)
            reg_terms.append(r_g)

        if reg_terms:
            return np.concatenate([r, np.concatenate(reg_terms)])
        return r

    def jac(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        u = np.asarray(u, dtype=np.float64).reshape(5)
        p = self.transform.decode(u)

        # ---- data block ----
        J_wp = svi_jac_wrt_params(self.y, p)  # (n,5) wrt [a,b,rho,m,sigma]
        dpdu = self.transform.dp_du(u, p)  # (5,5)
        J_wu = J_wp @ dpdu  # (n,5) wrt u
        J = self.sqrt_w[:, None] * J_wu  # (n,5)

        # Collect extra reg Jacobian blocks in the SAME order as residual()
        blocks: list[NDArray[np.float64]] = []

        # ---- m prior row (1,5) ----
        if self.reg.lambda_m > 0.0:
            row = np.zeros((1, 5), dtype=np.float64)
            row[0, 3] = np.sqrt(self.reg.lambda_m) / self.reg.m_scale
            blocks.append(row)

        # ---- inv-sigma hinge row (1,5) ----
        if self.reg.lambda_inv_sigma > 0.0:
            sigma = max(float(p.sigma), EPS)
            x = float(np.log(max(self.reg.sigma_floor / sigma, EPS)))
            h = float(softplus(x) - LOG2)

            row = np.zeros((1, 5), dtype=np.float64)
            if h > 0.0:
                dsoft_dx = float(sigmoid(x))
                dx_dsigma = -1.0 / sigma
                dsigma_dusig = float(sigmoid(float(u[4])))
                row[0, 4] = (
                    np.sqrt(self.reg.lambda_inv_sigma)
                    * dsoft_dx
                    * dx_dsigma
                    * dsigma_dusig
                )
            blocks.append(row)

        # ---- slope target rows (1,5) each ----
        denom = float(self.reg.slope_denom)
        sL_use, sR_use = self._usable_obs_slopes()
        cap = float(self.reg.slope_cap)
        s_min = float(self.transform.slope_min)
        span = cap - s_min

        if self.reg.lambda_slope_R > 0.0 and sR_use is not None:
            uR = float(u[1])
            sigR = float(sigmoid(uR))
            dsR_duR = span * sigR * (1.0 - sigR)

            row = np.zeros((1, 5), dtype=np.float64)
            row[0, 1] = np.sqrt(self.reg.lambda_slope_R) * dsR_duR / denom
            blocks.append(row)

        if self.reg.lambda_slope_L > 0.0 and sL_use is not None:
            uL = float(u[2])
            sigL = float(sigmoid(uL))
            dsL_duL = -span * sigL * (1.0 - sigL)

            row = np.zeros((1, 5), dtype=np.float64)
            row[0, 2] = np.sqrt(self.reg.lambda_slope_L) * dsL_duL / denom
            blocks.append(row)

        # ---- g-penalty block (ng,5) ----
        if self.reg.lambda_g > 0.0 and self.y_g is not None and self.y_g.size:
            y_g = np.asarray(self.y_g, dtype=np.float64).reshape(-1)

            g = gatheral_g_vec(y_g, p, w_floor=self.w_floor)
            g_scale = max(float(self.reg.g_scale), 1e-12)
            deficit = (float(self.reg.g_floor) - g) / g_scale

            # hinge derivative: 0 when deficit<=0, else sigmoid(deficit)
            dh_ddef = np.zeros_like(deficit)
            active = deficit > 0.0
            dh_ddef[active] = sigmoid(deficit[active])

            Jg_p = gatheral_g_jac_params(y_g, p, w_floor=self.w_floor)  # (ng,5) wrt p
            Jg_u = Jg_p @ dpdu  # (ng,5) wrt u

            scale = -np.sqrt(self.reg.lambda_g) / g_scale
            Jg_rows = (scale * dh_ddef)[:, None] * Jg_u  # (ng,5)

            blocks.append(Jg_rows)

        # ---- final stack ----
        if blocks:
            J = np.vstack([J, np.vstack(blocks)])
        return J


# ==========================
# Diagnostics helpers
# ==========================
def _safe_entropy_normalized(weights: NDArray[np.float64]) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w[np.isfinite(w) & (w > 0.0)]
    if w.size <= 1:
        return 0.0 if w.size == 1 else float("nan")
    s = float(w.sum())
    if s <= 0.0:
        return float("nan")
    p = w / s
    H = -float(np.sum(p * np.log(p)))
    return float(H / np.log(float(p.size)))


# ==========================
# Gatheral butterfly diagnostic g(y) and analytic g'(y)
# ==========================
def gatheral_g_vec(
    y: NDArray[np.float64],
    p: SVIParams,
    *,
    w_floor: float = 0.0,
) -> NDArray[np.float64]:
    """
    Gatheral-Jacquier butterfly diagnostic g(y) in terms of total variance w(y).
    y = log-moneyness.
    """
    y = np.asarray(y, dtype=np.float64)
    w = svi_total_variance(y, p)
    w1 = svi_total_variance_dy(y, p)
    w2 = svi_total_variance_dyy(y, p)

    bad = w <= float(w_floor)
    w_safe = np.where(bad, np.nan, w)

    A = 1.0 - y * w1 / (2.0 * w_safe)
    term1 = A * A
    term2 = -(w1 * w1 / 4.0) * (1.0 / w_safe + 0.25)
    term3 = 0.5 * w2
    return term1 + term2 + term3


def gatheral_gprime_vec(
    y: NDArray[np.float64],
    p: SVIParams,
    *,
    w_floor: float = 0.0,
) -> NDArray[np.float64]:
    """
    Analytic derivative d/dy of Gatheral g(y).
    Uses w, w', w'', w'''.
    """
    y = np.asarray(y, dtype=np.float64)
    w = svi_total_variance(y, p)
    w1 = svi_total_variance_dy(y, p)
    w2 = svi_total_variance_dyy(y, p)
    w3 = svi_total_variance_dyyy(y, p)

    bad = w <= float(w_floor)
    w_safe = np.where(bad, np.nan, w)

    A = 1.0 - y * w1 / (2.0 * w_safe)

    termA: NDArray[np.float64] = A * (
        -(w1 + y * w2) / w_safe + y * (w1 * w1) / (w_safe * w_safe)
    )
    termB: NDArray[np.float64] = -(w1 * w2) * (1.0 / (2.0 * w_safe) + 0.125)
    termC: NDArray[np.float64] = (w1 * w1 * w1) / (4.0 * w_safe * w_safe)
    termD: NDArray[np.float64] = 0.5 * w3
    return termA + termB + termC + termD


def gatheral_g_scalar(y: float, p: SVIParams, *, w_floor: float = 0.0) -> float:
    return float(gatheral_g_vec(np.array([y], dtype=np.float64), p, w_floor=w_floor)[0])


def gatheral_gprime_scalar(y: float, p: SVIParams, *, w_floor: float = 0.0) -> float:
    return float(
        gatheral_gprime_vec(np.array([y], dtype=np.float64), p, w_floor=w_floor)[0]
    )


def gatheral_g_wing_limits(p: SVIParams) -> tuple[float, float]:
    """
    Analytic wing limits of g as y -> -inf and y -> +inf for raw SVI.
      sR = b(1+rho), sL = b(rho-1) (negative).
      lim g = (4 - slope^2)/16
    Returns (g_L, g_R).
    """
    sR = float(p.b * (1.0 + p.rho))
    sL = float(p.b * (p.rho - 1.0))
    gR = (4.0 - sR * sR) / 16.0
    gL = (4.0 - sL * sL) / 16.0
    return float(gL), float(gR)


@dataclass(frozen=True, slots=True)
class ButterflyCheck:
    ok: bool
    y_domain: tuple[float, float]
    min_g: float
    argmin_y: float
    n_stationary: int
    g_left_inf: float
    g_right_inf: float
    stationary_points: tuple[float, ...]
    failure_reason: str | None


def check_butterfly_arbitrage(
    p: SVIParams,
    *,
    y_domain_hint: tuple[float, float] = (-1.25, 1.25),
    w_floor: float = 0.0,
    g_floor: float = 0.0,
    tol: float = 1e-10,
    n_scan: int = 1201,
) -> ButterflyCheck:
    """
    Checks g(y) >= g_floor over all y by:
      (1) wing limits (analytic),
      (2) scanning a wide interval and solving g'(y)=0 via bracketing,
      (3) evaluating g at endpoints + stationary points + wing limits.
    """
    y0, y1 = float(min(y_domain_hint)), float(max(y_domain_hint))
    K = max(10.0, abs(p.m) + 12.0 * max(float(p.sigma), 0.2) + 2.0)
    y_lo = min(y0, -K)
    y_hi = max(y1, +K)

    gL_inf, gR_inf = gatheral_g_wing_limits(p)
    if (gL_inf < g_floor - tol) or (gR_inf < g_floor - tol):
        reason = f"wing_limit_violation (gL_inf={gL_inf:.3g}, gR_inf={gR_inf:.3g})"
        return ButterflyCheck(
            ok=False,
            y_domain=(y_lo, y_hi),
            min_g=min(gL_inf, gR_inf),
            argmin_y=float("nan"),
            n_stationary=0,
            g_left_inf=gL_inf,
            g_right_inf=gR_inf,
            stationary_points=tuple(),
            failure_reason=reason,
        )

    n_scan = int(max(n_scan, 401))
    y_grid = np.linspace(y_lo, y_hi, n_scan, dtype=np.float64)

    w_grid = svi_total_variance(y_grid, p)

    # Keep threshold consistent with gatheral_g_vec/gatheral_gprime_vec,
    # which treat w <= w_floor as invalid (NaN) due to 1/w terms.
    if np.any(w_grid <= float(w_floor)):
        imin = int(np.nanargmin(w_grid))
        reason = (
            f"w_below_floor_in_scan (min_w={float(w_grid[imin]):.3g} "
            f"at y={float(y_grid[imin]):.3g})"
        )
        return ButterflyCheck(
            ok=False,
            y_domain=(y_lo, y_hi),
            min_g=float("nan"),
            argmin_y=float(y_grid[imin]),
            n_stationary=0,
            g_left_inf=gL_inf,
            g_right_inf=gR_inf,
            stationary_points=tuple(),
            failure_reason=reason,
        )

    gp_grid = gatheral_gprime_vec(y_grid, p, w_floor=w_floor)

    roots: list[float] = []
    for i in range(n_scan - 1):
        a = float(y_grid[i])
        b = float(y_grid[i + 1])
        fa = float(gp_grid[i])
        fb = float(gp_grid[i + 1])

        if not (np.isfinite(fa) and np.isfinite(fb)):
            continue

        if abs(fa) <= 1e-10:
            roots.append(a)
            continue

        if fa * fb < 0.0:
            try:
                rr = bracketed_newton(
                    lambda yy: gatheral_gprime_scalar(yy, p, w_floor=w_floor),
                    a,
                    b,
                    tol_f=1e-10,  # similar spirit to your abs(fa)<=1e-10 shortcut
                    tol_x=1e-12,
                    max_iter=100,
                    domain=(y_lo, y_hi),  # optional safety clamp
                )
                roots.append(float(rr.root))
            except (NotBracketedError, NoConvergenceError, DerivativeTooSmallError):
                pass

    # Deduplicate roots (tolerance)
    if roots:
        roots_sorted = np.array(sorted(roots), dtype=np.float64)
        uniq = [float(roots_sorted[0])]
        for rr in roots_sorted[1:]:
            if abs(float(rr) - uniq[-1]) > 1e-6:
                uniq.append(float(rr))
        roots = uniq

    # Evaluate g at endpoints + stationary points
    cand_y = [float(y_lo), float(y_hi)] + roots
    cand_g = np.array([gatheral_g_scalar(yy, p, w_floor=w_floor) for yy in cand_y])

    # include wing limits in global min
    all_g = np.concatenate([cand_g, np.array([gL_inf, gR_inf], dtype=np.float64)])
    finite = np.isfinite(all_g)
    if not np.any(finite):
        return ButterflyCheck(
            ok=False,
            y_domain=(y_lo, y_hi),
            min_g=float("nan"),
            argmin_y=float("nan"),
            n_stationary=len(roots),
            g_left_inf=gL_inf,
            g_right_inf=gR_inf,
            stationary_points=tuple(roots),
            failure_reason="g_all_nan",
        )

    min_g = float(np.min(all_g[finite]))
    j = int(np.nanargmin(cand_g))
    argmin_y = float(cand_y[j])

    ok = bool(min_g >= float(g_floor) - float(tol))
    res_reason = (
        None if ok else f"min_g_violation (min_g={min_g:.3g} at y≈{argmin_y:.3g})"
    )

    return ButterflyCheck(
        ok=ok,
        y_domain=(y_lo, y_hi),
        min_g=min_g,
        argmin_y=argmin_y,
        n_stationary=len(roots),
        g_left_inf=gL_inf,
        g_right_inf=gR_inf,
        stationary_points=tuple(roots),
        failure_reason=res_reason,
    )


# ==========================
# Fit outputs and Diagnostics
# ==========================
# ==========================
# Pure helpers (no self / no closure)
# ==========================


def usable_obs_slopes(
    sL_obs: float | None,
    sR_obs: float | None,
    *,
    slope_cap: float,
) -> tuple[float | None, float | None]:
    """Apply sign sanity and Lee-cap clipping to observed slope targets."""
    cap = float(slope_cap)
    cap_eff = cap * 0.995

    sL = sL_obs
    sR = sR_obs

    if sR is not None:
        sR = float(sR)
        if not np.isfinite(sR) or sR < 0.0:
            sR = None
        else:
            sR = float(np.clip(sR, 0.0, cap_eff))

    if sL is not None:
        sL = float(sL)
        if not np.isfinite(sL) or sL > 0.0:
            sL = None
        else:
            mag = float(np.clip(-sL, 0.0, cap_eff))
            sL = -mag

    return sL, sR


def svi_residual_vector(
    u: NDArray[np.float64],
    *,
    y: NDArray[np.float64],
    w_obs: NDArray[np.float64],
    sqrt_w: NDArray[np.float64],
    transform: SVITransformLeeCap,
    reg: SVIRegConfig,
    sL_obs: float | None,
    sR_obs: float | None,
) -> NDArray[np.float64]:
    """
    Residual vector used by the optimizer:
      [ sqrt_w*(w_model - w_obs) , reg_terms... ]
    This is NOT the same as `err = w_model - w_obs` because:
      - it is weighted by sqrt_w
      - it appends regularization residuals
    """
    u = np.asarray(u, dtype=np.float64).reshape(5)
    p = transform.decode(u)

    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w_obs = np.asarray(w_obs, dtype=np.float64).reshape(-1)
    sqrt_w = np.asarray(sqrt_w, dtype=np.float64).reshape(-1)

    w_model = svi_total_variance(y, p)
    r = sqrt_w * (w_model - w_obs)

    reg_terms: list[float] = []

    if reg.lambda_m > 0.0:
        reg_terms.append(np.sqrt(reg.lambda_m) * (p.m - reg.m_prior) / reg.m_scale)

    if reg.lambda_inv_sigma > 0.0:
        ratio = reg.sigma_floor / max(p.sigma, EPS)
        reg_terms.append(np.sqrt(reg.lambda_inv_sigma) * soft_hinge_log_ratio(ratio))

    sL_use, sR_use = usable_obs_slopes(sL_obs, sR_obs, slope_cap=reg.slope_cap)
    cap = float(reg.slope_cap)
    s_min = float(transform.slope_min)
    span = cap - s_min

    if reg.lambda_slope_R > 0.0 and sR_use is not None:
        uR = float(u[1])
        sR_model = s_min + span * float(sigmoid(uR))
        reg_terms.append(
            np.sqrt(reg.lambda_slope_R) * (sR_model - sR_use) / reg.slope_denom
        )

    if reg.lambda_slope_L > 0.0 and sL_use is not None:
        uL = float(u[2])
        sL_model = -(s_min + span * float(sigmoid(uL)))
        reg_terms.append(
            np.sqrt(reg.lambda_slope_L) * (sL_model - sL_use) / reg.slope_denom
        )

    if reg_terms:
        return np.concatenate([r, np.asarray(reg_terms, dtype=np.float64)])
    return r


# ==========================
# Diagnostics data model (split)
# ==========================


@dataclass(frozen=True, slots=True)
class SVISolverInfo:
    termination: str
    nfev: int
    cost: float
    optimality: float
    step_norm: float
    irls_outer_iters: int


@dataclass(frozen=True, slots=True)
class SVIModelChecks:
    # Sizes
    n_obs: int
    n_reg: int

    # Overall residual stats (data+reg)
    RMSE: float
    max_abs: float
    residual: float

    # Data-only fit
    rmse_w: float
    rmse_unw: float
    mae_w: float
    max_abs_werr: float
    cost_data: float
    cost_reg: float

    # Domain safety
    y_domain: tuple[float, float]
    w_floor: float
    min_w_domain: float
    argmin_y_domain: float
    n_violations: int

    # Butterfly checks
    butterfly_ok: bool
    min_g: float
    argmin_g_y: float
    g_left_inf: float
    g_right_inf: float
    n_stationary_g: int
    butterfly_reason: str | None

    # Wing / Lee
    sR: float
    sL: float
    lee_cap: float
    lee_slack_R: float
    lee_slack_L: float
    sR_target: float | None
    sL_target: float | None
    sR_target_used: bool
    sL_target_used: bool
    sR_target_err: float
    sL_target_err: float

    # Params + flags
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    alpha: float
    sigma_vs_floor: float

    rho_near_pm1: bool
    sigma_tiny: bool
    b_blown_up: bool
    b_large: bool
    m_outside_data: bool

    # Robust weights diagnostics
    robust_weights_min: float
    robust_weights_median: float
    robust_weights_max: float
    robust_weights_frac_floored: float
    robust_weights_entropy: float


@dataclass(frozen=True, slots=True)
class SVIFitDiagnostics:
    ok: bool
    failure_reason: str | None
    solver: SVISolverInfo
    checks: SVIModelChecks
    summary: str


# ==========================
# Context object
# ==========================


@dataclass(frozen=True, slots=True)
class SVIDiagnosticsContext:
    y: NDArray[np.float64]
    w_obs: NDArray[np.float64]
    base_sqrt_w: NDArray[np.float64]

    dom_cfg: DomainCheckConfig
    y_domain: tuple[float, float]
    y_chk: NDArray[np.float64]

    reg: SVIRegConfig
    transform: SVITransformLeeCap

    sL_obs: float | None
    sR_obs: float | None

    # to compute "fraction floored" in robust weights diagnostics
    irls_w_floor: float


def build_svi_diagnostics(
    *,
    ctx: SVIDiagnosticsContext,
    u_final: NDArray[np.float64] | None = None,
    p_final: SVIParams | None = None,
    res_final=None,
    eff_sqrt_w: NDArray[np.float64] | None = None,
    robust_w: NDArray[np.float64] | None = None,
    irls_iters: int = 0,
    step_norm: float = float("nan"),
    p_override: SVIParams | None = None,
) -> SVIFitDiagnostics:

    # 0) choose base params + u (for reporting step_norm etc.)
    if p_final is None:
        if u_final is None:
            raise ValueError("Need u_final or p_final")
        u_use = np.asarray(u_final, dtype=np.float64).reshape(5)
        p = ctx.transform.decode(u_use)
    else:
        p = p_final
        u_use = (
            np.asarray(u_final, dtype=np.float64).reshape(5)
            if u_final is not None
            else ctx.transform.encode(p_final)
        )

    # 1) override params if provided (e.g. repaired slice)
    if p_override is not None:
        p = p_override
        u_use = ctx.transform.encode(p_override)  # keep r_total consistent with p

    # 2) weights for data-only stats
    if eff_sqrt_w is None:
        eff_sqrt_w = ctx.base_sqrt_w
    eff_sqrt_w = np.asarray(eff_sqrt_w, dtype=np.float64).reshape(-1)

    # 3) data-only stats
    w_model = svi_total_variance(ctx.y, p)
    err = w_model - ctx.w_obs
    r_w = eff_sqrt_w * err

    rmse_w = float(np.sqrt(np.mean(r_w * r_w))) if r_w.size else float("nan")
    rmse_unw = float(np.sqrt(np.mean(err * err))) if err.size else float("nan")
    mae_w = float(np.mean(np.abs(r_w))) if r_w.size else float("nan")
    max_abs_werr = float(np.max(np.abs(r_w))) if r_w.size else float("nan")
    cost_data = 0.5 * float(np.sum(r_w * r_w)) if r_w.size else 0.0

    # 4) full residual vector (data + reg) and split
    r_total = svi_residual_vector(
        u_use,
        y=ctx.y,
        w_obs=ctx.w_obs,
        sqrt_w=eff_sqrt_w,
        transform=ctx.transform,
        reg=ctx.reg,
        sL_obs=ctx.sL_obs,
        sR_obs=ctx.sR_obs,
    )

    n_obs = int(ctx.y.size)
    n_reg = int(max(r_total.size - n_obs, 0))
    r_reg = r_total[n_obs:] if n_reg > 0 else np.zeros(0, dtype=np.float64)
    cost_reg = 0.5 * float(np.sum(r_reg * r_reg)) if r_reg.size else 0.0

    RMSE_total = (
        float(np.sqrt(np.mean(r_total * r_total))) if r_total.size else float("nan")
    )
    max_abs_total = float(np.max(np.abs(r_total))) if r_total.size else float("nan")
    residual_norm = float(np.linalg.norm(r_total)) if r_total.size else float("nan")

    # 5) domain safety check
    w_floor = float(ctx.dom_cfg.w_floor)
    tol = float(ctx.dom_cfg.tol)

    w_chk = svi_total_variance(ctx.y_chk, p)
    min_w_domain = float(np.min(w_chk)) if w_chk.size else float("nan")
    imin = int(np.argmin(w_chk)) if w_chk.size else 0
    argmin_y_domain = float(ctx.y_chk[imin]) if ctx.y_chk.size else float("nan")
    n_viol = int(np.sum(w_chk < (w_floor - tol))) if w_chk.size else 0

    # 6) butterfly check
    bfly = check_butterfly_arbitrage(
        p,
        y_domain_hint=ctx.y_domain,
        w_floor=w_floor,
        g_floor=0.0,
        tol=1e-10,
        n_scan=max(1201, 40 * int(ctx.dom_cfg.n_grid)),
    )

    # 7) wing + Lee diagnostics
    sR = float(p.b * (1.0 + p.rho))
    sL = float(p.b * (p.rho - 1.0))  # negative
    lee_cap = float(ctx.reg.slope_cap)
    lee_slack_R = float(lee_cap - sR)
    lee_slack_L = float(lee_cap - abs(sL))

    sL_use, sR_use = usable_obs_slopes(
        ctx.sL_obs, ctx.sR_obs, slope_cap=ctx.reg.slope_cap
    )
    sR_used = bool(ctx.reg.lambda_slope_R > 0.0 and sR_use is not None)
    sL_used = bool(ctx.reg.lambda_slope_L > 0.0 and sL_use is not None)

    denom = float(ctx.reg.slope_denom)
    sR_target_err = float("nan")
    sL_target_err = float("nan")
    if sR_use is not None and denom > 0:
        sR_target_err = float((sR - sR_use) / denom)
    if sL_use is not None and denom > 0:
        sL_target_err = float((sL - sL_use) / denom)

    # 8) params + warn flags
    rho = float(p.rho)
    sigma = float(p.sigma)
    b = float(p.b)
    a = float(p.a)
    m = float(p.m)
    alpha = float(a + b * sigma * np.sqrt(max(1.0 - rho * rho, 0.0)))

    rho_near_pm1 = bool(abs(rho) > 0.995)
    sigma_vs_floor = float(sigma / max(ctx.reg.sigma_floor, EPS))
    sigma_tiny = bool(sigma_vs_floor < 0.5)
    b_blown_up = bool((not np.isfinite(b)) or (b > 1.01 * lee_cap) or (b <= 0.0))
    b_large = bool(b > 0.90 * lee_cap)

    y_min, y_max = float(np.min(ctx.y)), float(np.max(ctx.y))
    y_span = float(max(y_max - y_min, 1e-6))
    m_outside_data = bool((m < y_min - 0.10 * y_span) or (m > y_max + 0.10 * y_span))

    # 9) robust weight diagnostics
    if robust_w is None:
        robust_weights_min = float("nan")
        robust_weights_median = float("nan")
        robust_weights_max = float("nan")
        robust_weights_frac_floored = float("nan")
        robust_weights_entropy = float("nan")
    else:
        rw = np.asarray(robust_w, dtype=np.float64).reshape(-1)
        robust_weights_min = float(np.min(rw)) if rw.size else float("nan")
        robust_weights_median = float(np.median(rw)) if rw.size else float("nan")
        robust_weights_max = float(np.max(rw)) if rw.size else float("nan")
        robust_weights_frac_floored = (
            float(np.mean(rw <= float(ctx.irls_w_floor) * (1.0 + 1e-12)))
            if rw.size
            else float("nan")
        )

        eff_w = (eff_sqrt_w * eff_sqrt_w).astype(np.float64, copy=False)
        robust_weights_entropy = _safe_entropy_normalized(eff_w)

    # 10) overall ok/failure_reason
    ok = True
    reasons: list[str] = []

    if n_viol > 0:
        ok = False
        reasons.append(f"domain_negative_w (n={n_viol}, min_w={min_w_domain:.3g})")

    if not bfly.ok:
        ok = False
        reasons.append(bfly.failure_reason or "butterfly_arbitrage")

    if rho_near_pm1:
        reasons.append("rho_near_pm1")
    if sigma_tiny:
        reasons.append("sigma_tiny")
    if b_blown_up:
        ok = False
        reasons.append("b_invalid_or_exceeds_cap")

    failure_reason = "; ".join(reasons) if (not ok or reasons) else None

    # 11) solver info
    solver = SVISolverInfo(
        termination=str(getattr(res_final, "message", "")),
        nfev=int(getattr(res_final, "nfev", 0)),
        cost=float(getattr(res_final, "cost", float("nan"))),
        optimality=float(getattr(res_final, "optimality", float("nan"))),
        step_norm=float(step_norm),
        irls_outer_iters=int(irls_iters),
    )

    checks = SVIModelChecks(
        n_obs=n_obs,
        n_reg=n_reg,
        RMSE=RMSE_total,
        max_abs=max_abs_total,
        residual=residual_norm,
        rmse_w=rmse_w,
        rmse_unw=rmse_unw,
        mae_w=mae_w,
        max_abs_werr=max_abs_werr,
        cost_data=cost_data,
        cost_reg=cost_reg,
        y_domain=ctx.y_domain,
        w_floor=w_floor,
        min_w_domain=min_w_domain,
        argmin_y_domain=argmin_y_domain,
        n_violations=n_viol,
        butterfly_ok=bool(bfly.ok),
        min_g=float(bfly.min_g),
        argmin_g_y=float(bfly.argmin_y),
        g_left_inf=float(bfly.g_left_inf),
        g_right_inf=float(bfly.g_right_inf),
        n_stationary_g=int(bfly.n_stationary),
        butterfly_reason=bfly.failure_reason,
        sR=sR,
        sL=sL,
        lee_cap=lee_cap,
        lee_slack_R=lee_slack_R,
        lee_slack_L=lee_slack_L,
        sR_target=ctx.sR_obs,
        sL_target=ctx.sL_obs,
        sR_target_used=sR_used,
        sL_target_used=sL_used,
        sR_target_err=sR_target_err,
        sL_target_err=sL_target_err,
        a=a,
        b=b,
        rho=rho,
        m=m,
        sigma=sigma,
        alpha=alpha,
        sigma_vs_floor=sigma_vs_floor,
        rho_near_pm1=rho_near_pm1,
        sigma_tiny=sigma_tiny,
        b_blown_up=b_blown_up,
        b_large=b_large,
        m_outside_data=m_outside_data,
        robust_weights_min=robust_weights_min,
        robust_weights_median=robust_weights_median,
        robust_weights_max=robust_weights_max,
        robust_weights_frac_floored=robust_weights_frac_floored,
        robust_weights_entropy=robust_weights_entropy,
    )

    summary = (
        f"ok={ok} rmse_w={rmse_w:.3g} rmse_unw={rmse_unw:.3g} "
        f"min_w_dom={min_w_domain:.3g} min_g={bfly.min_g:.3g} "
        f"sR={sR:.3g} sL={sL:.3g} sigma/floor={sigma_vs_floor:.3g} irls={irls_iters}"
    )

    return SVIFitDiagnostics(
        ok=ok,
        failure_reason=failure_reason,
        solver=solver,
        checks=checks,
        summary=summary,
    )


@dataclass(frozen=True, slots=True)
class SVIFitResult:
    params: SVIParams
    diag: SVIFitDiagnostics


# ==========================
# Smile object
# ==========================
@dataclass(frozen=True, slots=True)
class SVISmile:
    """Analytic SVI smile slice in total-variance space.

    Coordinate:
        y = ln(K / F(T))  (log-moneyness)

    Notes
    -----
    * The SVI functional form is defined for all real ``y``.
    * ``y_min`` / ``y_max`` are treated as the *recommended* domain for sampling
      (diagnostics, plots, calendar checks), not hard limits.
    """

    T: float
    params: SVIParams
    y_min: float = -1.25
    y_max: float = 1.25
    diagnostics: SVIFitDiagnostics | None = None

    def w_at(self, yq: ArrayLike) -> FloatArray:
        yq_arr = np.asarray(yq, dtype=np.float64)
        yq_1d = np.atleast_1d(yq_arr)

        out = svi_total_variance(yq_1d.astype(np.float64, copy=False), self.params)
        out = np.asarray(out, dtype=np.float64)

        if yq_arr.ndim == 0:
            return np.asarray(out[0], dtype=np.float64)
        return out.reshape(yq_arr.shape)

    def iv_at(self, yq: ArrayLike) -> FloatArray:
        wq = self.w_at(yq)
        out = np.sqrt(np.maximum(wq / np.float64(self.T), np.float64(0.0)))
        return np.asarray(out, dtype=np.float64)

    # ---- analytic derivatives in y (log-moneyness) ----
    def dw_dy(self, yq: ArrayLike) -> FloatArray:
        yq_arr = np.asarray(yq, dtype=np.float64)
        yq_1d = np.atleast_1d(yq_arr)
        out = svi_total_variance_dy(yq_1d.astype(np.float64, copy=False), self.params)
        out = np.asarray(out, dtype=np.float64)
        if yq_arr.ndim == 0:
            return np.asarray(out[0], dtype=np.float64)
        return out.reshape(yq_arr.shape)

    def d2w_dy2(self, yq: ArrayLike) -> FloatArray:
        yq_arr = np.asarray(yq, dtype=np.float64)
        yq_1d = np.atleast_1d(yq_arr)
        out = svi_total_variance_dyy(yq_1d.astype(np.float64, copy=False), self.params)
        out = np.asarray(out, dtype=np.float64)
        if yq_arr.ndim == 0:
            return np.asarray(out[0], dtype=np.float64)
        return out.reshape(yq_arr.shape)


# ==========================
# Smile repair
# ==========================


def _find_min_feasible_lambda(
    p_jw: JWParams,
    c_ast: float,
    v_tilde_ast: float,
    T: float,
    *,
    y_domain_hint: tuple[float, float],
    w_floor: float,
    tol: float = 1e-10,
    n_scan: int = 31,
    n_bisect: int = 30,
) -> tuple[float, SVIParams, ButterflyCheck]:
    c0 = float(p_jw.c)
    v0 = float(p_jw.v_tilde)
    c1 = float(c_ast)
    v1 = float(v_tilde_ast)

    def is_ok(lam: float):
        lam = float(np.clip(lam, 0.0, 1.0))
        c = (1.0 - lam) * c0 + lam * c1
        vt = (1.0 - lam) * v0 + lam * v1

        jw = replace(p_jw, c=float(c), v_tilde=float(vt))
        try:
            raw = jw_to_raw(jw, T)
        except ValueError:
            return False, None, None

        bfly = check_butterfly_arbitrage(
            raw,
            y_domain_hint=y_domain_hint,
            w_floor=w_floor,
            g_floor=0.0,
            tol=tol,
        )
        return bool(bfly.ok), raw, bfly

    # ---- endpoints ----
    ok0, raw0, b0 = is_ok(0.0)
    if ok0 and raw0 is not None:
        return 0.0, raw0, b0

    ok1, raw1, b1 = is_ok(1.0)

    # If λ=1 is degenerate for JW->raw, try a small retreat from 1.0
    if raw1 is None:
        for eps in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4):
            ok1, raw1, b1 = is_ok(1.0 - eps)
            if raw1 is not None:
                break

    if raw1 is None:
        raise ValueError(
            "SVI repair failed: projection endpoint hits JW->raw degeneracy "
            "(cannot represent candidate in raw parameters)."
        )

    if not ok1:
        raise ValueError(
            "SVI projection (Section 5.1 construction) did not yield a butterfly-arb-free slice "
            f"(reason={b1.failure_reason}, min_g={b1.min_g})."
        )

    # ---- coarse scan for first fail->pass bracket ----
    grid = np.linspace(0.0, 1.0, int(max(n_scan, 3)), dtype=np.float64)
    prev_lam = float(grid[0])
    prev_ok = False  # we know ok0 is False here

    lo = hi = None
    for lam in grid[1:]:
        ok, raw, bfly = is_ok(float(lam))

        # Treat "not representable" as not ok (acts like a failure region)
        if raw is None:
            prev_lam, prev_ok = float(lam), False
            continue

        if (not prev_ok) and ok:
            lo, hi = float(prev_lam), float(lam)
            break

        prev_lam, prev_ok = float(lam), bool(ok)

    if lo is None or hi is None:
        # Shouldn't happen if ok1 True, but keep it safe.
        return 1.0, raw1, b1

    # ---- bisection for minimal feasible lambda ----
    lo_lam, hi_lam = lo, hi
    # hi_lam should be feasible (or at least representable)
    ok_hi, raw_hi, b_hi = is_ok(hi_lam)
    if (raw_hi is None) or (not ok_hi):
        # fallback: return the endpoint which is feasible
        return 1.0, raw1, b1

    best_lam, best_raw, best_bfly = hi_lam, raw_hi, b_hi

    for _ in range(int(max(n_bisect, 1))):
        mid = 0.5 * (lo_lam + hi_lam)
        ok, raw, bfly = is_ok(mid)

        # not representable => treat like failure
        if raw is None or not ok:
            lo_lam = mid
        else:
            best_lam, best_raw, best_bfly = mid, raw, bfly
            hi_lam = mid

    return float(best_lam), best_raw, best_bfly


def repair_butterfly_raw(
    p_raw: SVIParams,
    T: float,
    *,
    y_domain_hint: tuple[float, float],
    w_floor: float,
    method: Literal["project", "line_search"] = "line_search",
    tol: float = 1e-10,
    n_scan: int = 31,
    n_bisect: int = 30,
) -> SVIParams:

    # 0) If already ok, return unchanged.
    b0 = check_butterfly_arbitrage(
        p_raw,
        y_domain_hint=y_domain_hint,
        w_floor=w_floor,
        g_floor=0.0,
        tol=tol,
    )
    if b0.ok:
        return p_raw

    # 1) raw -> JW
    p_jw = p_raw.to_jw(T=T)

    # 2) Section 5.1 targets
    c_ast = float(p_jw.p + 2.0 * p_jw.psi)

    if c_ast < 0.0 and c_ast > -1e-14:
        c_ast = 0.0
    if c_ast < 0.0:
        raise ValueError(
            f"Invalid c_ast={c_ast} from JW; check JW constraints / inputs."
        )

    denom = float(p_jw.p + c_ast)
    if denom <= 0.0:
        raise ValueError("Invalid JW: p + c_ast must be > 0 for projection formula.")

    frac = float((4.0 * p_jw.p * c_ast) / (denom * denom))
    frac = float(np.clip(frac, 0.0, 1.0))
    vtilde_ast = float(p_jw.v * frac)

    if method == "project":
        jw_target = replace(p_jw, c=c_ast, v_tilde=vtilde_ast)

        # Fast path: try direct full projection
        try:
            p_candidate = jw_to_raw(jw_target, T)
            b = check_butterfly_arbitrage(
                p_candidate,
                y_domain_hint=y_domain_hint,
                w_floor=w_floor,
                g_floor=0.0,
                tol=tol,
            )
            if b.ok:
                return p_candidate
        except ValueError:
            # JW->raw degeneracy (or other conversion issue) -> fall back below
            pass

        # Robust fallback: reuse the existing line-search to guarantee feasibility
        lam, p_candidate, b = _find_min_feasible_lambda(
            p_jw=p_jw,
            c_ast=c_ast,
            v_tilde_ast=vtilde_ast,
            T=T,
            y_domain_hint=y_domain_hint,
            w_floor=w_floor,
            tol=tol,
            n_scan=n_scan,
            n_bisect=n_bisect,
        )
        if not b.ok:
            raise ValueError(
                "Fallback line-search returned non-feasible candidate unexpectedly "
                f"(reason={b.failure_reason}, min_g={b.min_g})."
            )
        return p_candidate

    if method == "line_search":
        lam, p_candidate, b = _find_min_feasible_lambda(
            p_jw=p_jw,
            c_ast=c_ast,
            v_tilde_ast=vtilde_ast,
            T=T,
            y_domain_hint=y_domain_hint,
            w_floor=w_floor,
            tol=tol,
            n_scan=n_scan,
            n_bisect=n_bisect,
        )
        # b should be ok by construction, but keep the guard
        if not b.ok:
            raise ValueError(
                "Line-search returned non-feasible candidate unexpectedly "
                f"(reason={b.failure_reason}, min_g={b.min_g})."
            )
        return p_candidate

    raise ValueError(f"Unknown method={method!r}")


# ==========================
# Calibration
# ==========================
def calibrate_svi(
    y: NDArray[np.float64],
    w_obs: NDArray[np.float64],
    sqrt_weights: NDArray[np.float64] | None = None,
    x0: SVIParams | None = None,
    reg_override: RegOverride | None = None,
    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "soft_l1",
    f_scale: float = 1.0,
    *,
    domain_check: DomainCheckConfig | None = None,
    robust_data_only: bool = True,
    irls_max_outer: int = 8,
    irls_w_floor: float = 1e-4,
    irls_damp: float = 0.0,
    irls_tol: float = 1e-8,
    repair_butterfly: bool = False,
    repair_method: Literal["project", "line_search"] = "line_search",
    repair_n_scan: int = 31,
    repair_n_bisect: int = 30,
    refit_after_repair: bool = True,
    refit_max_nfev: int = 1500,
) -> SVIFitResult:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w_obs = np.asarray(w_obs, dtype=np.float64).reshape(-1)
    if y.shape != w_obs.shape:
        raise ValueError("y and w_obs must have same shape")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(w_obs))):
        raise ValueError("y and w_obs must be finite")

    base_sqrt_w = (
        np.ones_like(w_obs)
        if sqrt_weights is None
        else np.asarray(sqrt_weights, np.float64).reshape(-1)
    )
    if base_sqrt_w.shape != w_obs.shape:
        raise ValueError("sqrt_weights must have same shape as w_obs")
    if not (np.all(np.isfinite(base_sqrt_w)) and np.all(base_sqrt_w >= 0.0)):
        raise ValueError("sqrt_weights must be finite and nonnegative")

    # default x0
    if x0 is None:
        i0 = int(np.argmin(w_obs))
        x0 = SVIParams(
            a=max(1e-8, float(w_obs[i0]) - 0.1 * 0.2),
            b=0.1,
            rho=0.0,
            m=float(y[i0]),
            sigma=0.2,
        )

    dom_cfg = DomainCheckConfig() if domain_check is None else domain_check
    y_domain, y_chk = build_domain_grid(y, dom_cfg)

    base_reg = default_reg_from_data(y, w_obs, base_sqrt_w)
    reg = apply_reg_override(base_reg, reg_override)

    # g-penalty grid
    y_lo, y_hi = y_domain
    n = reg.g_n_grid
    y_g = np.linspace(y_lo, y_hi, n, dtype=np.float64)

    span = max(y_hi - y_lo, 0.5)
    wing = np.array(
        [y_lo - span, y_lo - 0.5 * span, y_hi + 0.5 * span, y_hi + span],
        dtype=np.float64,
    )

    y_g = np.unique(np.concatenate([y_g, wing])).astype(np.float64)

    transform = SVITransformLeeCap(slope_cap=reg.slope_cap)
    u = transform.encode(x0)

    # slope targets from base weights
    sL_obs, sR_obs = estimate_wing_slopes_one_sided(
        y=y, w=w_obs, sqrt_weights=base_sqrt_w
    )

    abs_slopes = [abs(s) for s in (sL_obs, sR_obs) if s is not None]
    s_norm = max(
        reg.slope_floor, float(np.mean(abs_slopes)) if abs_slopes else reg.slope_floor
    )
    slope_denom = max(s_norm, reg.slope_floor)
    reg = replace(reg, slope_denom=slope_denom)

    obj = SVIObjective(
        y=y,
        w_obs=w_obs,
        sqrt_w=base_sqrt_w.copy(),
        transform=transform,
        reg=reg,
        sL_obs=sL_obs,
        sR_obs=sR_obs,
        y_g=y_g,
        w_floor=float(dom_cfg.w_floor),
    )

    ctx = SVIDiagnosticsContext(
        y=y,
        w_obs=w_obs,
        base_sqrt_w=base_sqrt_w,
        dom_cfg=dom_cfg,
        y_domain=y_domain,
        y_chk=y_chk,
        reg=reg,
        transform=transform,
        sL_obs=sL_obs,
        sR_obs=sR_obs,
        irls_w_floor=float(irls_w_floor),
    )

    def _finalize(
        *,
        u_final: NDArray[np.float64],
        p_final: SVIParams,
        res_final,
        eff_sqrt_w: NDArray[np.float64],
        robust_w: NDArray[np.float64] | None,
        irls_iters: int,
        step_norm: float,
    ) -> SVIFitResult:
        p_out = p_final

        if repair_butterfly:
            bfly = check_butterfly_arbitrage(
                p_out,
                y_domain_hint=ctx.y_domain,
                w_floor=float(ctx.dom_cfg.w_floor),
                g_floor=0.0,
                tol=1e-10,
            )
            if not bfly.ok:
                # Use T=1.0 because you're in total-variance calibration space
                p_out = repair_butterfly_raw(
                    p_out,
                    T=1.0,
                    y_domain_hint=ctx.y_domain,
                    w_floor=float(ctx.dom_cfg.w_floor),
                    method=repair_method,
                    tol=1e-10,
                    n_scan=repair_n_scan,
                    n_bisect=repair_n_bisect,
                )
                # ---- REFIT (warm-start) from repaired params ----
                if refit_after_repair:
                    # make objective use the final effective weights (IRLS or base)
                    obj.sqrt_w = np.asarray(eff_sqrt_w, dtype=np.float64).reshape(-1)

                    u0 = transform.encode(p_out)

                    # one more local polish step; keep it linear because robust weights are already in obj.sqrt_w (IRLS)
                    res2 = least_squares(
                        fun=obj.residual,
                        x0=u0,
                        jac=obj.jac,
                        loss="linear",
                        x_scale="jac",
                        max_nfev=int(refit_max_nfev),
                    )
                    if res2.success and np.all(np.isfinite(res2.x)):
                        u_final = np.asarray(res2.x, dtype=np.float64)
                        p_out = transform.decode(u_final)
                        res_final = res2
                        step_norm = float(np.linalg.norm(u_final - u0))

                        # final guard: refit can reintroduce arb
                        b2 = check_butterfly_arbitrage(
                            p_out,
                            y_domain_hint=ctx.y_domain,
                            w_floor=float(ctx.dom_cfg.w_floor),
                            g_floor=0.0,
                            tol=1e-10,
                        )
                        if not b2.ok:
                            p_out = repair_butterfly_raw(
                                p_out,
                                T=1.0,
                                y_domain_hint=ctx.y_domain,
                                w_floor=float(ctx.dom_cfg.w_floor),
                                method=repair_method,
                                tol=1e-10,
                                n_scan=repair_n_scan,
                                n_bisect=repair_n_bisect,
                            )
                            # keep u consistent for diagnostics
                            u_final = transform.encode(p_out)

        diag = build_svi_diagnostics(
            ctx=ctx,
            u_final=u_final,
            res_final=res_final,
            eff_sqrt_w=eff_sqrt_w,
            robust_w=robust_w if robust_w is not None else np.ones_like(y),
            irls_iters=irls_iters,
            step_norm=step_norm,
            p_override=p_out,  # <-- important
        )
        return SVIFitResult(params=p_out, diag=diag)

    # ==========================
    # Branch A: robustify everything (SciPy loss)
    # ==========================
    if not robust_data_only:
        res = least_squares(
            fun=obj.residual,
            x0=u,
            jac=obj.jac,
            loss=loss,
            f_scale=f_scale,
            x_scale="jac",
            max_nfev=5000,
        )
        if not res.success or not np.all(np.isfinite(res.x)):
            raise ValueError(f"SVI calibration failed: {res.message}")

        u_final = np.asarray(res.x, dtype=np.float64)
        p_final = transform.decode(u_final)

        err = svi_total_variance(y, p_final) - w_obs
        r_data = base_sqrt_w * err
        z = (r_data / max(float(f_scale), EPS)) ** 2
        robust_w = (
            np.maximum(_robust_rhoprime(z, loss), float(irls_w_floor))
            if loss != "linear"
            else np.ones_like(y)
        )

        return _finalize(
            u_final=u_final,
            p_final=p_final,
            res_final=res,
            eff_sqrt_w=base_sqrt_w,
            robust_w=robust_w,
            irls_iters=0,
            step_norm=float(np.linalg.norm(u_final - u)),
        )

    # ==========================
    # Branch B: linear (no robust)
    # ==========================
    if loss == "linear":
        res = least_squares(
            fun=obj.residual,
            x0=u,
            jac=obj.jac,
            loss="linear",
            x_scale="jac",
            max_nfev=5000,
        )
        if not res.success or not np.all(np.isfinite(res.x)):
            raise ValueError(f"SVI calibration failed: {res.message}")

        u_final = np.asarray(res.x, dtype=np.float64)
        p_final = transform.decode(u_final)

        robust_w = np.ones_like(y)

        return _finalize(
            u_final=u_final,
            p_final=p_final,
            res_final=res,
            eff_sqrt_w=base_sqrt_w,
            robust_w=robust_w,
            irls_iters=0,
            step_norm=float(np.linalg.norm(u_final - u)),
        )

    # ==========================
    # Branch C: IRLS (robustify data only)
    # ==========================
    res_final = None
    robust_w_final: NDArray[np.float64] | None = None
    step_norm_final = float("nan")
    irls_iters = 0

    w_prev: NDArray[np.float64] | None = None

    for k in range(int(irls_max_outer)):
        irls_iters = k + 1

        p = transform.decode(u)
        r_data = base_sqrt_w * (svi_total_variance(y, p) - w_obs)

        z = (r_data / max(float(f_scale), EPS)) ** 2
        w = _robust_rhoprime(z, loss)
        w = np.maximum(w, float(irls_w_floor))

        if (w_prev is not None) and (irls_damp > 0.0):
            a_d = float(np.clip(irls_damp, 0.0, 0.99))
            w = (1.0 - a_d) * w + a_d * w_prev
        w_prev = w

        obj.sqrt_w = base_sqrt_w * np.sqrt(w)
        robust_w_final = w

        res = least_squares(
            fun=obj.residual,
            x0=u,
            jac=obj.jac,
            loss="linear",
            x_scale="jac",
            max_nfev=2500,
        )
        if not res.success or not np.all(np.isfinite(res.x)):
            raise ValueError(f"SVI calibration failed (IRLS): {res.message}")

        res_final = res
        u_new = np.asarray(res.x, dtype=np.float64)
        du = u_new - u
        step_norm_final = float(np.linalg.norm(du))

        if step_norm_final <= float(irls_tol) * (1.0 + float(np.linalg.norm(u))):
            u = u_new
            break

        u = u_new

    if res_final is None:
        raise ValueError("SVI calibration failed: IRLS produced no result")

    u_final = np.asarray(u, dtype=np.float64)
    p_final = transform.decode(u_final)

    return _finalize(
        u_final=u_final,
        p_final=p_final,
        res_final=res_final,
        eff_sqrt_w=obj.sqrt_w,  # effective weights after IRLS
        robust_w=robust_w_final,  # the robust weights
        irls_iters=irls_iters,
        step_norm=step_norm_final,
    )
