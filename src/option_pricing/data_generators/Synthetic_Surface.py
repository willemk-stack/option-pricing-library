from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

SurfaceModel = Literal["poly", "svi"]
NoiseMode = Literal["none", "absolute", "relative"]
NoiseDist = Literal["normal", "student_t"]

type FloatArray = npt.NDArray[np.float64]


def _moving_average_1d(x: npt.ArrayLike, window: int) -> FloatArray:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    w = int(window)
    if w <= 1:
        return x_arr
    w = min(w, x_arr.size)
    pad = w // 2
    xp = np.pad(x_arr, (pad, w - 1 - pad), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / np.float64(w)
    out = np.convolve(xp, kernel, mode="valid")
    return np.asarray(out, dtype=np.float64)


def _iv_poly(
    T: float,
    x: npt.ArrayLike,
    *,
    base_level: float,
    term_slope: float,
    term_ref: float,
    skew: float,
    curvature: float,
    twist: float,
    floor: float,
) -> FloatArray:
    T_ = float(T)
    x_arr = np.asarray(x, dtype=np.float64)
    base = base_level + term_slope * np.log1p(T_ / term_ref)
    iv = base + curvature * (x_arr**2) + skew * x_arr + twist * (x_arr**3)
    iv = np.maximum(iv, floor)
    return np.asarray(iv, dtype=np.float64)


def _iv_svi_like(
    T: float,
    x: npt.ArrayLike,
    *,
    atm_level: float,
    atm_term_slope: float,
    atm_term_ref: float,
    b_level: float,
    b_term_slope: float,
    rho: float,
    m_level: float,
    m_decay: float,
    sigma_level: float,
    sigma_term_slope: float,
    floor: float,
) -> FloatArray:
    """
    SVI-like generator using raw SVI total variance:
        w(x) = a + b*( rho*(x-m) + sqrt((x-m)^2 + sigma^2) )
    We pick b,rho,m,sigma as smooth functions of T and solve a to match ATM vol at x=0.
    """
    T_ = float(T)
    x_arr = np.asarray(x, dtype=np.float64)

    atm = atm_level + atm_term_slope * np.log1p(T_ / atm_term_ref)
    w_atm = T_ * atm * atm

    b = float(max(0.0, b_level + b_term_slope * np.sqrt(max(T_, 0.0))))
    rho_ = float(np.clip(rho, -0.999, 0.999))
    m = float(m_level * np.exp(-m_decay * max(T_, 0.0)))
    sigma = float(max(1e-8, sigma_level + sigma_term_slope * np.sqrt(max(T_, 0.0))))

    sqrt_m2_sigma2 = float(np.hypot(m, sigma))
    a = w_atm - b * (-rho_ * m + sqrt_m2_sigma2)

    dx = x_arr - m
    w = a + b * (rho_ * dx + np.sqrt(dx * dx + sigma * sigma))
    w = np.maximum(w, 0.0)
    iv = np.sqrt(np.maximum(w / max(T_, 1e-12), 0.0))
    iv = np.maximum(iv, floor)
    return np.asarray(iv, dtype=np.float64)


@dataclass(frozen=True)
class SyntheticSurface:
    T: FloatArray
    x: FloatArray
    K: FloatArray
    F: FloatArray
    iv_true: FloatArray
    iv_obs: FloatArray

    rows_obs: list[tuple[float, float, float]]

    forward: Callable[[float], float]
    df: Callable[[float], float]
    iv_true_fn: Callable[[float, FloatArray], FloatArray]


def generate_synthetic_surface(
    *,
    spot: float = 100.0,
    r: float = 0.02,
    q: float = 0.0,
    expiries: Sequence[float] | FloatArray = (0.25, 0.5, 1.0, 2.0),
    x_grid: Sequence[float] | FloatArray | None = None,
    model: SurfaceModel = "poly",
    # ---- poly knobs ----
    base_level: float = 0.18,
    term_slope: float = 0.02,
    term_ref: float = 0.25,
    skew: float = -0.02,
    curvature: float = 0.10,
    twist: float = 0.0,
    # ---- svi-like knobs ----
    atm_level: float = 0.18,
    atm_term_slope: float = 0.02,
    atm_term_ref: float = 0.25,
    b_level: float = 0.20,
    b_term_slope: float = 0.00,
    rho: float = -0.30,
    m_level: float = 0.00,
    m_decay: float = 0.50,
    sigma_level: float = 0.20,
    sigma_term_slope: float = 0.00,
    # ---- noise knobs ----
    noise_mode: NoiseMode = "none",
    noise_level: float = 0.0,
    noise_dist: NoiseDist = "normal",
    noise_df: float = 5.0,
    noise_smooth_window: int = 1,
    outlier_prob: float = 0.0,
    outlier_scale: float = 8.0,
    missing_prob: float = 0.0,
    vol_floor: float = 1e-6,
    seed: int | None = None,
) -> SyntheticSurface:
    S = float(spot)
    if S <= 0:
        raise ValueError("spot must be > 0")

    exp = np.asarray(expiries, dtype=np.float64).reshape(-1)
    if exp.size == 0 or np.any(exp <= 0.0):
        raise ValueError("expiries must be non-empty and strictly positive")

    if x_grid is None:
        x_grid = np.linspace(-0.4, 0.4, 25, dtype=np.float64)
    xg = np.asarray(x_grid, dtype=np.float64).reshape(-1)
    if xg.size == 0 or np.any(~np.isfinite(xg)):
        raise ValueError("x_grid must be non-empty and finite")

    rng = np.random.default_rng(seed)

    def forward(T: float) -> float:
        return float(S * np.exp((r - q) * float(T)))

    def df(T: float) -> float:
        return float(np.exp(-r * float(T)))

    if model == "poly":

        def iv_true_fn(T: float, x: FloatArray) -> FloatArray:
            return _iv_poly(
                T,
                x,
                base_level=base_level,
                term_slope=term_slope,
                term_ref=term_ref,
                skew=skew,
                curvature=curvature,
                twist=twist,
                floor=vol_floor,
            )

    elif model == "svi":

        def iv_true_fn(T: float, x: FloatArray) -> FloatArray:
            return _iv_svi_like(
                T,
                x,
                atm_level=atm_level,
                atm_term_slope=atm_term_slope,
                atm_term_ref=atm_term_ref,
                b_level=b_level,
                b_term_slope=b_term_slope,
                rho=rho,
                m_level=m_level,
                m_decay=m_decay,
                sigma_level=sigma_level,
                sigma_term_slope=sigma_term_slope,
                floor=vol_floor,
            )

    else:
        raise ValueError(f"Unknown model: {model!r}")

    Ts: list[FloatArray] = []
    xs: list[FloatArray] = []
    Ks: list[FloatArray] = []
    Fs: list[FloatArray] = []
    iv_true_all: list[FloatArray] = []
    iv_obs_all: list[FloatArray] = []
    rows_obs: list[tuple[float, float, float]] = []

    p_out = float(np.clip(outlier_prob, 0.0, 1.0))
    p_miss = float(np.clip(missing_prob, 0.0, 1.0))

    for T in exp:
        T_ = float(T)
        F = forward(T_)
        K = np.asarray(F * np.exp(xg), dtype=np.float64)

        iv_true = np.asarray(iv_true_fn(T_, xg), dtype=np.float64)

        # base noise
        if noise_mode == "none" or noise_level <= 0.0:
            eps = np.zeros_like(iv_true)
        else:
            if noise_dist == "normal":
                z = rng.normal(0.0, 1.0, size=iv_true.shape)
            elif noise_dist == "student_t":
                z = rng.standard_t(max(noise_df, 2.1), size=iv_true.shape)
            else:
                raise ValueError(f"Unknown noise_dist: {noise_dist!r}")

            z = np.asarray(z, dtype=np.float64)

            if noise_smooth_window > 1:
                z = _moving_average_1d(z, noise_smooth_window)

            if noise_mode == "absolute":
                eps = noise_level * z
            elif noise_mode == "relative":
                eps = (noise_level * iv_true) * z
            else:
                raise ValueError(f"Unknown noise_mode: {noise_mode!r}")

        # outliers
        if p_out > 0.0 and noise_level > 0.0:
            is_out = rng.uniform(0.0, 1.0, size=iv_true.shape) < p_out
            if np.any(is_out):
                bump = rng.normal(0.0, 1.0, size=iv_true.shape)
                bump = np.asarray(bump, dtype=np.float64)
                if noise_smooth_window > 1:
                    bump = _moving_average_1d(bump, noise_smooth_window)
                eps = eps + is_out * (outlier_scale * noise_level * bump)

        iv_obs = np.maximum(iv_true + eps, vol_floor)

        # missing points
        if p_miss > 0.0:
            keep = rng.uniform(0.0, 1.0, size=iv_true.shape) >= p_miss
        else:
            keep = np.ones_like(iv_true, dtype=bool)

        n_keep = int(np.count_nonzero(keep))

        Ts.append(np.full(n_keep, T_, dtype=np.float64))
        xs.append(np.asarray(xg[keep], dtype=np.float64))
        Fs.append(np.full(n_keep, F, dtype=np.float64))
        Ks.append(np.asarray(K[keep], dtype=np.float64))
        iv_true_all.append(np.asarray(iv_true[keep], dtype=np.float64))
        iv_obs_all.append(np.asarray(iv_obs[keep], dtype=np.float64))

        for Ki, ivo in zip(K[keep], iv_obs[keep], strict=True):
            rows_obs.append((T_, float(Ki), float(ivo)))

    T_out = np.asarray(np.concatenate(Ts), dtype=np.float64)
    x_out = np.asarray(np.concatenate(xs), dtype=np.float64)
    F_out = np.asarray(np.concatenate(Fs), dtype=np.float64)
    K_out = np.asarray(np.concatenate(Ks), dtype=np.float64)
    iv_true_out = np.asarray(np.concatenate(iv_true_all), dtype=np.float64)
    iv_obs_out = np.asarray(np.concatenate(iv_obs_all), dtype=np.float64)

    return SyntheticSurface(
        T=T_out,
        x=x_out,
        K=K_out,
        F=F_out,
        iv_true=iv_true_out,
        iv_obs=iv_obs_out,
        rows_obs=rows_obs,
        forward=forward,
        df=df,
        iv_true_fn=iv_true_fn,
    )
