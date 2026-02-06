from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

RHO_MAX = 0.999
EPS = 1e-12
LOG2 = float(np.log(2.0))

EPS = 1e-12  # keep your existing EPS

DomainMode = Literal["fixed_logmoneyness", "quantile_pad", "explicit"]


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

    Changes vs previous version:
    - Tail regression points are restricted to those truly beyond +/- wing_threshold.
    - Select up to k most extreme points within that wing (not k overall extremes).
    - Require a minimum number of usable points per side (and non-degenerate weights).
    - Optional slope sign sanity checks: sL <= 0, sR >= 0 (helps avoid garbage targets).
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

    # minimum tail points; k is how many we *try* to use per side (within wing candidates)
    min_pts = int(np.clip(0.12 * n, 4, min_pts_cap))
    k = max(min_pts, int(np.ceil(q_tail * n)))

    sL_obs: float | None = None
    sR_obs: float | None = None

    # --------------------
    # Left wing selection
    # --------------------
    left_cand = np.where(y < -wing_threshold)[0]
    if left_cand.size >= min_pts:
        # pick the k most negative y among candidates
        order = np.argsort(y[left_cand])  # ascending => most negative first
        left = left_cand[order[: min(k, left_cand.size)]]

        # avoid degenerate weights (all ~0)
        if float(np.sum(sw[left])) > 0.0:
            _, sL = _weighted_linear_fit(y[left], w[left], sw[left])
            if np.isfinite(sL):
                sL = float(sL)
                # optional sanity: left slope should be <= 0
                if sL <= 0.0:
                    sL_obs = sL

    # ---------------------
    # Right wing selection
    # ---------------------
    right_cand = np.where(y > wing_threshold)[0]
    if right_cand.size >= min_pts:
        # pick the k most positive y among candidates
        order = np.argsort(y[right_cand])  # ascending
        right = right_cand[order[-min(k, right_cand.size) :]]  # largest y

        if float(np.sum(sw[right])) > 0.0:
            _, sR = _weighted_linear_fit(y[right], w[right], sw[right])
            if np.isfinite(sR):
                sR = float(sR)
                # optional sanity: right slope should be >= 0
                if sR >= 0.0:
                    sR_obs = sR

    return sL_obs, sR_obs


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


@dataclass(frozen=True, slots=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float


def svi_total_variance(y: NDArray[np.float64], p: SVIParams) -> NDArray[np.float64]:
    z = y - p.m
    s = np.hypot(z, p.sigma)
    return p.a + p.b * (p.rho * z + s)


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


RegOverride = dict[str, float] | Callable[[SVIRegConfig], SVIRegConfig]


def default_reg_from_data(
    y: NDArray[np.float64],
    w_obs: NDArray[np.float64],
    sqrt_w: NDArray[np.float64] | None = None,  # maybe use to scale n_factor
) -> SVIRegConfig:
    y_sorted = np.unique(np.sort(y))
    dy = np.diff(y_sorted)
    dy_med = float(np.median(dy)) if dy.size else 0.1

    n = int(y.size)

    n_factor = float(np.clip(10.0 / max(n, 1), 0.25, 4.0))

    # hard-rail factor: never too small (e.g for lee's cap)
    rail_factor = float(np.clip(30.0 / max(n, 1), 1.0, 10.0))  # or simply 1.0
    # data-driven priors
    m_prior = float(y[int(np.argmin(w_obs))])

    q05, q95 = np.quantile(y, [0.05, 0.95])
    y_span = float(max(q95 - q05, 1e-6))
    m_scale = float(max(0.25 * y_span, 0.10))

    sigma_floor = float(max(0.03, 1.5 * dy_med))

    # example: scale slope denom off observed slope magnitude if you want
    slope_denom = 0.15  # keep constant or compute

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
    # dict override, validate keys via dataclasses.replace
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
    dw_dm = -p.b * (p.rho + d / s)
    dw_dsigma = p.b * (p.sigma / s)

    return np.stack([dw_da, dw_db, dw_drho, dw_dm, dw_dsigma], axis=-1)


@dataclass(frozen=True, slots=True)
class SVITransformLeeCap:
    """
    u = [ua, uR, uL, um, usig]

    sR    = cap * sigmoid(uR)      in (0, cap)
    sLmag = cap * sigmoid(uL)      in (0, cap)

    b   = 0.5*(sR + sLmag)
    rho = (sR - sLmag)/(sR + sLmag + eps)

    a is encoded via alpha >= 0:
      a = alpha - b*sigma*sqrt(1-rho^2)
    """

    slope_cap: float = 1.999
    eps: float = EPS

    def decode(self, u: NDArray[np.float64]) -> SVIParams:
        ua, uR, uL, um, usig = map(float, u)

        alpha = float(softplus(ua)) + self.eps
        sigma = float(softplus(usig)) + self.eps
        m = float(um)

        cap = float(self.slope_cap)
        s_min = 1e-4
        sR = s_min + (cap - s_min) * sigmoid(uR)
        sLmag = s_min + (cap - s_min) * sigmoid(uL)

        b = 0.5 * (sR + sLmag) + self.eps
        denom = (sR + sLmag) + self.eps
        rho = (sR - sLmag) / denom  # automatically in (-1, 1)

        k2 = max(1.0 - rho * rho, 0.0)
        a = alpha - b * sigma * float(np.sqrt(k2))

        return SVIParams(a=float(a), b=float(b), rho=float(rho), m=m, sigma=sigma)

    def encode(self, p: SVIParams) -> NDArray[np.float64]:
        if p.sigma <= 0.0 or p.b <= 0.0:
            raise ValueError("encode expects b>0 and sigma>0")

        cap = float(self.slope_cap)

        # convert (b,rho) -> slopes
        sR = float(p.b) * (1.0 + float(p.rho))
        sLmag = float(p.b) * (1.0 - float(p.rho))

        # clip into (0, cap) so logit is valid
        sR = float(np.clip(sR, 1e-10, cap - 1e-10))
        sLmag = float(np.clip(sLmag, 1e-10, cap - 1e-10))

        uR = logit(sR / cap)
        uL = logit(sLmag / cap)

        # alpha (same idea you already use)
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

        sigR = float(sigmoid(uR))
        sigL = float(sigmoid(uL))
        sR = cap * sigR
        sLmag = cap * sigL

        dsR_duR = cap * sigR * (1.0 - sigR)
        dsL_duL = cap * sigL * (1.0 - sigL)

        D = float(sR + sLmag + self.eps)

        # b = 0.5*(sR+sLmag)
        db_duR = 0.5 * dsR_duR
        db_duL = 0.5 * dsL_duL

        # rho = (sR - sLmag)/D
        # drho/dsR = (2*sLmag + eps)/D^2
        # drho/dsL = (-2*sR - eps)/D^2
        drho_dsR = (2.0 * sLmag + self.eps) / (D * D)
        drho_dsL = (-2.0 * sR - self.eps) / (D * D)
        drho_duR = drho_dsR * dsR_duR
        drho_duL = drho_dsL * dsL_duL

        # alpha, sigma
        dalpha_dua = float(sigmoid(ua))
        dsig_dusig = float(sigmoid(usig))

        rho = float(p.rho)
        b = float(p.b)
        sigma = float(p.sigma)

        k2 = max(1.0 - rho * rho, 0.0)
        k = float(np.sqrt(k2))
        k_safe = max(k, 1e-12)

        # a = alpha - b*sigma*k, dk/drho = -rho/k
        common_rho = (
            b * sigma * (rho / k_safe)
        )  # = -b*sigma*dk/drho with sign handled below

        da_dua = dalpha_dua
        da_duR = -(db_duR * sigma * k) + common_rho * drho_duR
        da_duL = -(db_duL * sigma * k) + common_rho * drho_duL
        da_dusig = -(b * dsig_dusig * k)

        dp_du = np.array(
            [
                [da_dua, da_duR, da_duL, 0.0, da_dusig],  # a
                [0.0, db_duR, db_duL, 0.0, 0.0],  # b
                [0.0, drho_duR, drho_duL, 0.0, 0.0],  # rho
                [0.0, 0.0, 0.0, 1.0, 0.0],  # m
                [0.0, 0.0, 0.0, 0.0, dsig_dusig],  # sigma
            ],
            dtype=np.float64,
        )
        return dp_du


def logit(x: float) -> float:
    x = float(np.clip(x, 1e-12, 1.0 - 1e-12))
    return float(np.log(x) - np.log1p(-x))


def soft_hinge_log_ratio(ratio: float) -> float:
    x = float(np.log(max(ratio, EPS)))  # x > 0 iff sigma < floor
    h = float(softplus(x) - LOG2)  # smooth around 0
    return max(h, 0.0)  # true hinge (piecewise)


def soft_hinge_ratio_excess(ratio: float) -> float:
    # smooth hinge on (ratio - 1)
    return float(softplus(ratio - 1.0))


@dataclass
class SVIObjective:
    y: NDArray[np.float64]
    w_obs: NDArray[np.float64]
    sqrt_w: NDArray[np.float64]
    transform: SVITransformLeeCap
    reg: SVIRegConfig
    sL_obs: float | None = None
    sR_obs: float | None = None

    def __post_init__(self):
        sw = np.asarray(self.sqrt_w, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(sw)) or np.any(sw < 0.0):
            raise ValueError("sqrt_weights must be finite and nonnegative")

    def _usable_obs_slopes(self) -> tuple[float | None, float | None]:
        """Apply sign sanity and Lee-cap clipping to observed slope targets."""
        cap = float(self.reg.slope_cap)
        # small safety margin so we never try to match exactly at the asymptote
        cap_eff = cap * 0.995

        sL = self.sL_obs
        sR = self.sR_obs

        # Right wing: should be >= 0
        if sR is not None:
            sR = float(sR)
            if not np.isfinite(sR):
                sR = None
            elif sR < 0.0:
                sR = None
            else:
                sR = float(np.clip(sR, 0.0, cap_eff))

        # Left wing: should be <= 0, magnitude <= cap
        if sL is not None:
            sL = float(sL)
            if not np.isfinite(sL):
                sL = None
            elif sL > 0.0:
                sL = None
            else:
                # clip magnitude
                mag = float(np.clip(-sL, 0.0, cap_eff))
                sL = -mag

        return sL, sR

    def residual(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        p = self.transform.decode(u)
        w_model = svi_total_variance(self.y, p)
        r = self.sqrt_w * (w_model - self.w_obs)

        reg_terms: list[float] = []

        # --- priors ---
        if self.reg.lambda_m > 0.0:
            reg_terms.append(
                np.sqrt(self.reg.lambda_m) * (p.m - self.reg.m_prior) / self.reg.m_scale
            )

        if self.reg.lambda_inv_sigma > 0.0:
            ratio = self.reg.sigma_floor / max(p.sigma, EPS)
            reg_terms.append(
                np.sqrt(self.reg.lambda_inv_sigma) * soft_hinge_log_ratio(ratio)
            )

        # --- slope matching (only if observed slope is usable) ---
        sL_use, sR_use = self._usable_obs_slopes()
        cap = float(self.reg.slope_cap)

        if self.reg.lambda_slope_R > 0.0 and sR_use is not None:
            uR = float(u[1])  # <- THIS is “uR”
            sR_model = cap * float(sigmoid(uR))  # in (0, cap)
            reg_terms.append(
                np.sqrt(self.reg.lambda_slope_R)
                * (sR_model - sR_use)
                / self.reg.slope_denom
            )

        if self.reg.lambda_slope_L > 0.0 and sL_use is not None:
            uL = float(u[2])  # <- THIS is “uL”
            sL_model = -cap * float(sigmoid(uL))  # negative left slope
            reg_terms.append(
                np.sqrt(self.reg.lambda_slope_L)
                * (sL_model - sL_use)
                / self.reg.slope_denom
            )

        if reg_terms:
            return np.concatenate([r, np.asarray(reg_terms, dtype=np.float64)])
        return r

    def jac(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        u = np.asarray(u, dtype=np.float64).reshape(5)
        p = self.transform.decode(u)

        # data part
        J_wp = svi_jac_wrt_params(self.y, p)  # (n,5) wrt [a,b,rho,m,sigma]
        dpdu = self.transform.dp_du(u, p)  # (5,5)
        J_wu = J_wp @ dpdu  # (n,5) wrt u
        J = self.sqrt_w[:, None] * J_wu  # (n,5)

        rows: list[NDArray[np.float64]] = []

        # m prior row
        if self.reg.lambda_m > 0.0:
            row = np.zeros(5, dtype=np.float64)
            row[3] = np.sqrt(self.reg.lambda_m) / self.reg.m_scale  # um
            rows.append(row)

        # inv-sigma hinge row
        if self.reg.lambda_inv_sigma > 0.0:
            sigma = max(float(p.sigma), EPS)
            x = float(np.log(max(self.reg.sigma_floor / sigma, EPS)))
            h = float(softplus(x) - LOG2)

            row = np.zeros(5, dtype=np.float64)
            if h > 0.0:
                dsoft_dx = float(sigmoid(x))
                dx_dsigma = -1.0 / sigma
                dsigma_dusig = float(sigmoid(float(u[4])))
                row[4] = (
                    np.sqrt(self.reg.lambda_inv_sigma)
                    * dsoft_dx
                    * dx_dsigma
                    * dsigma_dusig
                )
            rows.append(row)

        # --- slope matching rows (Version 2: slopes directly from uR/uL) ---
        denom = float(self.reg.slope_denom)
        sL_use, sR_use = self._usable_obs_slopes()
        cap = float(self.reg.slope_cap)

        if self.reg.lambda_slope_R > 0.0 and sR_use is not None:
            uR = float(u[1])
            sigR = float(sigmoid(uR))
            dsR_duR = cap * sigR * (1.0 - sigR)

            row = np.zeros(5, dtype=np.float64)
            row[1] = np.sqrt(self.reg.lambda_slope_R) * dsR_duR / denom  # column uR
            rows.append(row)

        if self.reg.lambda_slope_L > 0.0 and sL_use is not None:
            uL = float(u[2])
            sigL = float(sigmoid(uL))
            dsL_duL = -cap * sigL * (1.0 - sigL)  # because sL = -cap*sigmoid(uL)

            row = np.zeros(5, dtype=np.float64)
            row[2] = np.sqrt(self.reg.lambda_slope_L) * dsL_duL / denom  # column uL
            rows.append(row)

        if rows:
            J = np.vstack([J, np.vstack(rows)])
        return J


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


def _domain_check_grid(
    y: NDArray[np.float64], n: int = 25
) -> tuple[tuple[float, float], NDArray[np.float64]]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return (0.0, 0.0), np.zeros(0, dtype=np.float64)

    # robust-ish domain: [q01, q99] + pad
    q01, q99 = (
        np.quantile(y, [0.01, 0.99])
        if y.size >= 20
        else (float(y.min()), float(y.max()))
    )
    span = float(max(q99 - q01, 1e-6))
    pad = 0.15 * span + 0.05
    y_lo = float(q01 - pad)
    y_hi = float(q99 + pad)
    y_chk = np.linspace(y_lo, y_hi, int(max(n, 5)), dtype=np.float64)
    return (y_lo, y_hi), y_chk


@dataclass(frozen=True, slots=True)
class SVIFitDiagnostics:
    # Overall outcome
    ok: bool
    failure_reason: str | None

    # SciPy termination info (from the last inner solve)
    termination: str
    nfev: int
    cost: float
    optimality: float
    step_norm: float

    # Sizes
    n_obs: int
    n_reg: int

    # Overall residual stats (data+reg residual vector returned by obj.residual)
    RMSE: float
    max_abs: float
    residual: float

    # Data-only fit (computed on final params)
    rmse_w: float
    rmse_unw: float
    mae_w: float
    max_abs_werr: float
    cost_data: float
    cost_reg: float

    # Domain safety checks (computed on a widened domain grid)
    y_domain: tuple[float, float]
    w_floor: float
    min_w_domain: float
    argmin_y_domain: float
    n_violations: int

    # Wing / Lee cap diagnostics
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

    # Parameter diagnostics
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    alpha: float
    sigma_vs_floor: float

    # Warn flags
    rho_near_pm1: bool
    sigma_tiny: bool
    b_blown_up: bool
    b_large: bool
    m_outside_data: bool

    # IRLS diagnostics (0/NaN if not IRLS)
    irls_outer_iters: int
    robust_weights_min: float
    robust_weights_median: float
    robust_weights_max: float
    robust_weights_frac_floored: float
    robust_weights_entropy: float

    # One-line logging
    summary: str


@dataclass(frozen=True, slots=True)
class SVIFitResult:
    params: SVIParams
    diag: SVIFitDiagnostics


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
) -> SVIFitResult:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w_obs = np.asarray(w_obs, dtype=np.float64).reshape(-1)
    if y.shape != w_obs.shape:
        raise ValueError("y and w_obs must have same shape")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(w_obs))):
        raise ValueError("y and w_obs must be finite")

    # --- domain check grid: build ONCE, reuse everywhere ---
    dom_cfg = DomainCheckConfig() if domain_check is None else domain_check
    y_domain, y_chk = build_domain_grid(y, dom_cfg)
    w_floor = float(dom_cfg.w_floor)
    tol = float(dom_cfg.tol)

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

    # reg defaults
    base_reg = default_reg_from_data(y, w_obs, base_sqrt_w)
    reg = apply_reg_override(base_reg, reg_override)

    transform = SVITransformLeeCap(slope_cap=reg.slope_cap)
    u = transform.encode(x0)

    # slope targets from base weights
    sL_obs, sR_obs = estimate_wing_slopes_one_sided(
        y=y, w=w_obs, sqrt_weights=base_sqrt_w
    )

    abs_slopes = [abs(s) for s in (sL_obs, sR_obs) if s is not None]
    s_norm = max(
        reg.slope_floor,
        float(np.mean(abs_slopes)) if abs_slopes else reg.slope_floor,
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
    )

    # ---- helpers to build diagnostics at the end ----
    def _build_diag(
        *,
        u_final: NDArray[np.float64],
        res_final,
        eff_sqrt_w: NDArray[np.float64],
        robust_w: NDArray[np.float64] | None,
        irls_iters: int,
        step_norm: float,
    ) -> SVIFitDiagnostics:
        p = transform.decode(u_final)
        w_model = svi_total_variance(y, p)

        # data-only
        err = w_model - w_obs
        r_w = eff_sqrt_w * err  # weighted data residual

        rmse_w = float(np.sqrt(np.mean(r_w * r_w))) if r_w.size else float("nan")
        rmse_unw = float(np.sqrt(np.mean(err * err))) if err.size else float("nan")
        mae_w = float(np.mean(np.abs(r_w))) if r_w.size else float("nan")
        max_abs_werr = float(np.max(np.abs(r_w))) if r_w.size else float("nan")
        cost_data = 0.5 * float(np.sum(r_w * r_w)) if r_w.size else 0.0

        # total residual vector (data + reg) and its split
        r_total = obj.residual(u_final)
        n_obs = int(y.size)
        n_reg = int(max(r_total.size - n_obs, 0))
        r_reg = r_total[n_obs:] if n_reg > 0 else np.zeros(0, dtype=np.float64)
        cost_reg = 0.5 * float(np.sum(r_reg * r_reg)) if r_reg.size else 0.0

        RMSE_total = (
            float(np.sqrt(np.mean(r_total * r_total))) if r_total.size else float("nan")
        )
        max_abs_total = float(np.max(np.abs(r_total))) if r_total.size else float("nan")
        residual_norm = float(np.linalg.norm(r_total)) if r_total.size else float("nan")

        # --- domain safety check (REUSE prebuilt y_chk/y_domain/w_floor/tol) ---
        w_chk = svi_total_variance(y_chk, p)
        min_w_domain = float(np.min(w_chk)) if w_chk.size else float("nan")
        imin = int(np.argmin(w_chk)) if w_chk.size else 0
        argmin_y_domain = float(y_chk[imin]) if y_chk.size else float("nan")
        n_viol = int(np.sum(w_chk < (w_floor - tol))) if w_chk.size else 0

        # wing diagnostics
        sR = float(p.b * (1.0 + p.rho))
        sL = float(p.b * (p.rho - 1.0))  # negative
        lee_cap = float(reg.slope_cap)
        lee_slack_R = float(lee_cap - sR)
        lee_slack_L = float(lee_cap - abs(sL))

        sL_use, sR_use = obj._usable_obs_slopes()
        sR_used = bool(reg.lambda_slope_R > 0.0 and sR_use is not None)
        sL_used = bool(reg.lambda_slope_L > 0.0 and sL_use is not None)

        denom = float(reg.slope_denom)
        sR_target_err = float("nan")
        sL_target_err = float("nan")

        if reg.lambda_slope_R > 0.0 and sR_use is not None:
            sR_target_err = (sR - sR_use) / denom

        if reg.lambda_slope_L > 0.0 and sL_use is not None:
            sL_target_err = (sL - sL_use) / denom

        # parameter diagnostics + flags
        rho = float(p.rho)
        sigma = float(p.sigma)
        b = float(p.b)
        a = float(p.a)
        m = float(p.m)
        alpha = float(a + b * sigma * np.sqrt(max(1.0 - rho * rho, 0.0)))

        rho_near_pm1 = bool(abs(rho) > 0.995)
        sigma_vs_floor = float(sigma / max(reg.sigma_floor, EPS))
        sigma_tiny = bool(sigma_vs_floor < 0.5)
        b_blown_up = bool((not np.isfinite(b)) or (b > 1.01 * lee_cap) or (b <= 0.0))
        b_large = bool(b > 0.90 * lee_cap)

        y_min, y_max = float(np.min(y)), float(np.max(y))
        y_span = float(max(y_max - y_min, 1e-6))
        m_outside_data = bool(
            (m < y_min - 0.10 * y_span) or (m > y_max + 0.10 * y_span)
        )

        # robust weights diagnostics
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
                float(np.mean(rw <= float(irls_w_floor) * (1.0 + 1e-12)))
                if rw.size
                else float("nan")
            )

            # concentration on effective weights (base + robust)
            eff_w = (eff_sqrt_w * eff_sqrt_w).astype(np.float64, copy=False)
            robust_weights_entropy = _safe_entropy_normalized(eff_w)

        # ok / failure_reason (solver failure still raises earlier)
        ok = True
        reasons: list[str] = []
        if n_viol > 0:
            ok = False
            reasons.append(f"domain_negative_w (n={n_viol}, min_w={min_w_domain:.3g})")
        if rho_near_pm1:
            reasons.append("rho_near_pm1")
        if sigma_tiny:
            reasons.append("sigma_tiny")
        if b_blown_up:
            ok = False
            reasons.append("b_invalid_or_exceeds_cap")

        failure_reason = "; ".join(reasons) if (not ok or reasons) else None

        summary = (
            f"ok={ok} rmse_w={rmse_w:.3g} rmse_unw={rmse_unw:.3g} "
            f"min_w_dom={min_w_domain:.3g} sR={sR:.3g} sL={sL:.3g} "
            f"sigma/floor={sigma_vs_floor:.3g} irls={irls_iters}"
        )

        return SVIFitDiagnostics(
            ok=ok,
            failure_reason=failure_reason,
            termination=str(getattr(res_final, "message", "")),
            nfev=int(getattr(res_final, "nfev", 0)),
            cost=float(getattr(res_final, "cost", float("nan"))),
            optimality=float(getattr(res_final, "optimality", float("nan"))),
            step_norm=float(step_norm),
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
            y_domain=y_domain,
            w_floor=w_floor,
            min_w_domain=min_w_domain,
            argmin_y_domain=argmin_y_domain,
            n_violations=n_viol,
            sR=sR,
            sL=sL,
            lee_cap=lee_cap,
            lee_slack_R=lee_slack_R,
            lee_slack_L=lee_slack_L,
            sR_target=sR_obs,
            sL_target=sL_obs,
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
            irls_outer_iters=int(irls_iters),
            robust_weights_min=robust_weights_min,
            robust_weights_median=robust_weights_median,
            robust_weights_max=robust_weights_max,
            robust_weights_frac_floored=robust_weights_frac_floored,
            robust_weights_entropy=robust_weights_entropy,
            summary=summary,
        )

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

        diag = _build_diag(
            u_final=u_final,
            res_final=res,
            eff_sqrt_w=base_sqrt_w,
            robust_w=robust_w,
            irls_iters=0,
            step_norm=float(np.linalg.norm(u_final - u)),  # changed from nan
        )
        return SVIFitResult(params=p_final, diag=diag)

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
        diag = _build_diag(
            u_final=u_final,
            res_final=res,
            eff_sqrt_w=base_sqrt_w,
            robust_w=robust_w,
            irls_iters=0,
            step_norm=float(np.linalg.norm(u_final - u)),
        )
        return SVIFitResult(params=p_final, diag=diag)

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

    diag = _build_diag(
        u_final=u_final,
        res_final=res_final,
        eff_sqrt_w=obj.sqrt_w,  # includes base * sqrt(robust_w)
        robust_w=robust_w_final,
        irls_iters=irls_iters,
        step_norm=step_norm_final,
    )
    return SVIFitResult(params=p_final, diag=diag)
