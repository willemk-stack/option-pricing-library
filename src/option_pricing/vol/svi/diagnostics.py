from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .math import (
    compute_lee_wing_check,
    gatheral_g_vec,
    gatheral_g_wing_limits,
    gatheral_gprime_vec,
    svi_total_variance,
)

if TYPE_CHECKING:
    from .models import JWParams, SVIParams
from .domain import DomainCheckConfig
from .math import EPS
from .objective import svi_residual_vector
from .regularization import SVIRegConfig
from .transforms import SVITransformLeeCap
from .wings import usable_obs_slopes


@dataclass(frozen=True, slots=True)
class LeeWingCheck:
    cap: float
    sL: float  # signed left slope = b(rho-1), typically <= 0
    sR: float  # right slope = b(1+rho), typically >= 0
    left_ok: bool
    right_ok: bool
    ok: bool
    slack_L: float  # cap - abs(sL)
    slack_R: float  # cap - sR


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

    # Lee wing diagnostics (explicit, notebook-friendly)
    lee_ok: bool = False
    lee_left_ok: bool = False
    lee_right_ok: bool = False
    lee_left_slope: float = 0.0  # signed left slope = b(rho-1), usually <= 0
    lee_right_slope: float = 0.0  # right slope = b(1+rho), usually >= 0
    lee_slope_cap: float = 0.0  # usually 2.0


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

    # Explicit Lee wing diagnostics (independent of g-scan)
    lee = compute_lee_wing_check(p, cap=2.0, tol=tol)

    gL_inf, gR_inf = gatheral_g_wing_limits(p)
    if (gL_inf < g_floor - tol) or (gR_inf < g_floor - tol):
        # ``reason`` may later be None, so give it an explicit optional type
        reason: str | None = (
            f"wing_limit_violation (gL_inf={gL_inf:.3g}, gR_inf={gR_inf:.3g})"
        )
        return ButterflyCheck(
            ok=False,
            y_domain=(y_lo, y_hi),
            min_g=min(gL_inf, gR_inf),
            argmin_y=float("nan"),
            n_stationary=0,
            g_left_inf=gL_inf,
            g_right_inf=gR_inf,
            stationary_points=(),
            failure_reason=reason,
            lee_ok=lee.ok,
            lee_left_ok=lee.left_ok,
            lee_right_ok=lee.right_ok,
            lee_left_slope=lee.sL,
            lee_right_slope=lee.sR,
            lee_slope_cap=lee.cap,
        )

    # scan interval split
    y_grid = np.linspace(y_lo, y_hi, int(max(n_scan, 3)), dtype=np.float64)
    y1d = y_grid.reshape(-1)

    g_vals = gatheral_g_vec(y1d, p, w_floor=w_floor)

    # find candidate stationary points using bracketed_newton from numerics
    from option_pricing.numerics.root_finding import bracketed_newton

    roots: list[float] = []
    for i in range(1, y1d.size - 1):
        if g_vals[i - 1] >= g_vals[i] <= g_vals[i + 1]:
            # potential local min
            try:
                r = bracketed_newton(
                    lambda yy: gatheral_gprime_vec(np.array([yy]), p, w_floor=w_floor)[
                        0
                    ],
                    y1d[i - 1],
                    y1d[i + 1],
                    tol=tol,
                )
                # ``r`` is a RootResult; use its ``root`` attribute for float
                roots.append(float(r.root))
            except Exception:
                pass

    # evaluate g at interesting points
    cand_y = np.concatenate(([y_lo], roots, [y_hi]))
    cand_g = gatheral_g_vec(cand_y, p, w_floor=w_floor)

    min_idx = int(np.argmin(cand_g))
    min_g = float(cand_g[min_idx])
    argmin_y = float(cand_y[min_idx])

    ok = bool(min_g >= g_floor - tol)
    reason = None if ok else "g_below_floor"

    return ButterflyCheck(
        ok=ok,
        y_domain=(y_lo, y_hi),
        min_g=min_g,
        argmin_y=argmin_y,
        n_stationary=len(roots),
        g_left_inf=gL_inf,
        g_right_inf=gR_inf,
        stationary_points=tuple(roots),
        failure_reason=reason,
        lee_ok=lee.ok,
        lee_left_ok=lee.left_ok,
        lee_right_ok=lee.right_ok,
        lee_left_slope=lee.sL,
        lee_right_slope=lee.sR,
        lee_slope_cap=lee.cap,
    )


# Diagnostics data models
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


# context object
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
    lee_diag = compute_lee_wing_check(p, cap=float(ctx.reg.slope_cap), tol=1e-10)
    sR = float(lee_diag.sR)
    sL = float(lee_diag.sL)
    lee_cap = float(lee_diag.cap)
    lee_slack_R = float(lee_diag.slack_R)
    lee_slack_L = float(lee_diag.slack_L)

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
class GJExample51RepairResult:
    """Result bundle for the Gatheral–Jacquier (2013) Example 5.1 sanity check."""

    T: float

    p_raw: SVIParams
    jw_raw: JWParams

    c0_target: float
    vtilde0_target: float

    p_projected: SVIParams
    jw_projected: JWParams

    p_optimal: SVIParams
    jw_optimal: JWParams

    check_raw: ButterflyCheck
    check_projected: ButterflyCheck
    check_optimal: ButterflyCheck

    y_plot: NDArray[np.float64]
    w_raw: NDArray[np.float64]
    w_projected: NDArray[np.float64]
    w_optimal: NDArray[np.float64]

    g_raw: NDArray[np.float64]
    g_projected: NDArray[np.float64]
    g_optimal: NDArray[np.float64]

    paper_jw: dict[str, float]
    paper_proj: dict[str, float]
    paper_opt: dict[str, float]


def run_gj_example51_repair_sanity_check(
    *,
    T: float = 1.0,
    p_raw: SVIParams | None = None,
    y_domain_hint: tuple[float, float] = (-1.5, 1.5),
    w_floor: float = 1e-12,
    y_obj: NDArray[np.float64] | None = None,
    y_penalty: NDArray[np.float64] | None = None,
    y_plot: NDArray[np.float64] | None = None,
    lambda_price: float = 1.0,
    lambda_g: float = 1e7,
    g_floor: float = 0.0,
    g_scale: float = 0.02,
    lambda_wfloor: float = 0.0,
    max_nfev: int = 2000,
) -> GJExample51RepairResult:
    """
    Reproduce the library-side workflow for Gatheral–Jacquier (2013), Example 5.1:

      1) start from the raw-SVI slice from Example 3.1
      2) convert to JW
      3) compute the Section 5.1 projection target
      4) build:
           - projected repair via repair_butterfly_raw(..., method="project")
           - optimal refinement via repair_butterfly_jw_optimal(...)
      5) run butterfly checks and precompute w(y), g(y) curves for plotting

    This is intentionally a diagnostics/sanity-check helper, not core repair API.
    """
    from .models import SVIParams
    from .repair import (
        _gj_section51_targets,
        repair_butterfly_jw_optimal,
        repair_butterfly_raw,
    )

    if T <= 0.0:
        raise ValueError("T must be > 0")

    if p_raw is None:
        p_raw = SVIParams(
            a=-0.0410,
            b=0.1331,
            rho=0.3060,
            m=0.3586,
            sigma=0.4153,
        )

    y0, y1 = float(min(y_domain_hint)), float(max(y_domain_hint))
    if not np.isfinite(y0) or not np.isfinite(y1) or y0 >= y1:
        raise ValueError("y_domain_hint must be a finite ordered pair (y_min, y_max)")

    if y_obj is None:
        y_obj = np.linspace(y0, y1, 301, dtype=np.float64)
    else:
        y_obj = np.asarray(y_obj, dtype=np.float64).reshape(-1)

    if y_penalty is None:
        y_penalty = np.linspace(y0, y1, 1201, dtype=np.float64)
    else:
        y_penalty = np.asarray(y_penalty, dtype=np.float64).reshape(-1)

    if y_plot is None:
        y_plot = np.linspace(y0, y1, 801, dtype=np.float64)
    else:
        y_plot = np.asarray(y_plot, dtype=np.float64).reshape(-1)

    y_obj = y_obj[np.isfinite(y_obj)]
    y_penalty = y_penalty[np.isfinite(y_penalty)]
    y_plot = y_plot[np.isfinite(y_plot)]

    if y_obj.size == 0:
        raise ValueError("y_obj must contain at least one finite point")
    if y_penalty.size == 0:
        raise ValueError("y_penalty must contain at least one finite point")
    if y_plot.size == 0:
        raise ValueError("y_plot must contain at least one finite point")

    jw_raw = p_raw.to_jw(T=T)
    c0_target, vtilde0_target = _gj_section51_targets(jw_raw)

    p_projected = repair_butterfly_raw(
        p_raw,
        T=T,
        y_domain_hint=(y0, y1),
        w_floor=w_floor,
        method="project",
    )
    jw_projected = p_projected.to_jw(T=T)

    p_optimal = repair_butterfly_jw_optimal(
        p_raw,
        T=T,
        y_obj=y_obj,
        y_penalty=y_penalty,
        y_domain_hint=(y0, y1),
        w_floor=w_floor,
        lambda_price=lambda_price,
        lambda_g=lambda_g,
        g_floor=g_floor,
        g_scale=g_scale,
        lambda_wfloor=lambda_wfloor,
        max_nfev=max_nfev,
    )
    jw_optimal = p_optimal.to_jw(T=T)

    check_raw = check_butterfly_arbitrage(
        p_raw,
        y_domain_hint=(y0, y1),
        w_floor=w_floor,
        g_floor=g_floor,
    )
    check_projected = check_butterfly_arbitrage(
        p_projected,
        y_domain_hint=(y0, y1),
        w_floor=w_floor,
        g_floor=g_floor,
    )
    check_optimal = check_butterfly_arbitrage(
        p_optimal,
        y_domain_hint=(y0, y1),
        w_floor=w_floor,
        g_floor=g_floor,
    )

    w_raw = np.asarray(svi_total_variance(y_plot, p_raw), dtype=np.float64)
    w_projected = np.asarray(svi_total_variance(y_plot, p_projected), dtype=np.float64)
    w_optimal = np.asarray(svi_total_variance(y_plot, p_optimal), dtype=np.float64)

    g_raw = np.asarray(gatheral_g_vec(y_plot, p_raw, w_floor=w_floor), dtype=np.float64)
    g_projected = np.asarray(
        gatheral_g_vec(y_plot, p_projected, w_floor=w_floor),
        dtype=np.float64,
    )
    g_optimal = np.asarray(
        gatheral_g_vec(y_plot, p_optimal, w_floor=w_floor),
        dtype=np.float64,
    )

    paper_jw = {
        "v": 0.01742625,
        "psi": -0.1752111,
        "p": 0.6997381,
        "c": 1.316798,
        "v_tilde": 0.0116249,
    }
    paper_proj = {
        "c0": 0.3493158,
        "vtilde0": 0.01548182,
    }
    paper_opt = {
        "c_star": 0.8564763,
        "vtilde_star": 0.0116249,
    }

    return GJExample51RepairResult(
        T=float(T),
        p_raw=p_raw,
        jw_raw=jw_raw,
        c0_target=float(c0_target),
        vtilde0_target=float(vtilde0_target),
        p_projected=p_projected,
        jw_projected=jw_projected,
        p_optimal=p_optimal,
        jw_optimal=jw_optimal,
        check_raw=check_raw,
        check_projected=check_projected,
        check_optimal=check_optimal,
        y_plot=y_plot,
        w_raw=w_raw,
        w_projected=w_projected,
        w_optimal=w_optimal,
        g_raw=g_raw,
        g_projected=g_projected,
        g_optimal=g_optimal,
        paper_jw=paper_jw,
        paper_proj=paper_proj,
        paper_opt=paper_opt,
    )
