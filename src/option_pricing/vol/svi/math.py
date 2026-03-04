from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .diagnostics import LeeWingCheck
    from .models import JWParams, SVIParams


# constants
RHO_MAX = 0.999
EPS = 1e-12
LOG2 = float(np.log(2.0))


# conversion between "raw" and "JW" parameterizations


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

    from .models import JWParams

    return JWParams(v=v_t, psi=psi_t, p=p_t, c=c_t, v_tilde=ve_t)


def jw_to_raw(jw: JWParams, T: float) -> SVIParams:
    """
    SVI-JW (v,psi,p,c,ve) -> Raw SVI (a,b,rho,m,sigma) exactly per
    Gatheral–Jacquier (2013), Lemma 3.2.

    Notes:
    - Closed-form inversion assumes m != 0. We implement GJ's m=0 branch too.
    - In the degenerate case m=0 and rho≈0, JW does not uniquely identify sigma/a.
    """
    # local import to avoid circular dependency at module import time
    from .models import SVIParams

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

    Jg = np.where(np.isfinite(Jg), Jg, 0.0)
    return Jg


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


def compute_lee_wing_check(
    p: SVIParams,
    *,
    cap: float = 2.0,
    tol: float = 1e-10,
) -> LeeWingCheck:
    sR = float(p.b * (1.0 + p.rho))
    sL = float(
        p.b * (p.rho - 1.0)
    )  # signed left slope (negative under normal convention)

    right_ok = bool(sR <= float(cap) + float(tol))
    left_ok = bool(abs(sL) <= float(cap) + float(tol))
    ok = bool(left_ok and right_ok)

    from option_pricing.vol.svi.diagnostics import LeeWingCheck

    return LeeWingCheck(
        cap=float(cap),
        sL=sL,
        sR=sR,
        left_ok=left_ok,
        right_ok=right_ok,
        ok=ok,
        slack_L=float(cap) - abs(sL),
        slack_R=float(cap) - sR,
    )
