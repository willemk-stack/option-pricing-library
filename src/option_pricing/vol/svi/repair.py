from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.special import ndtr

from .diagnostics import check_butterfly_arbitrage
from .math import (
    EPS,
    LOG2,
    gatheral_g_vec,
    jw_to_raw,
    svi_total_variance,
)
from .models import JWParams, SVIParams
from .transforms import softplus

if TYPE_CHECKING:
    from .diagnostics import ButterflyCheck


def _gj_section51_targets(p_jw: JWParams) -> tuple[float, float]:
    """
    Section 5.1 construction target in JW coordinates:
      c* = p + 2 psi
      v_tilde* = v * 4 p c* / (p + c*)^2
    """
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
    return float(c_ast), float(vtilde_ast)


def _normalized_black_call_from_total_variance(
    y: NDArray[np.float64],
    w: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Forward-normalized undiscounted Black call price:
      C/F = N(d1) - exp(y) N(d2),  y = ln(K/F)
    with total variance w = sigma^2 T.
    """
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    intrinsic = np.maximum(1.0 - np.exp(y), 0.0)

    w_safe = np.maximum(w, EPS)
    s = np.sqrt(w_safe)
    d1 = (-y + 0.5 * w_safe) / s
    d2 = d1 - s

    c = ndtr(d1) - np.exp(y) * ndtr(d2)
    c = np.where(w <= EPS, intrinsic, c)

    # numerical hygiene
    c = np.clip(c, 0.0, 1.0)
    return c.astype(np.float64, copy=False)


def repair_butterfly_jw_optimal(
    p_raw: SVIParams,
    T: float,
    *,
    y_obj: NDArray[np.float64],  # grid for price closeness objective
    y_penalty: NDArray[np.float64],  # grid for arb penalty (dense)
    y_domain_hint: tuple[float, float],
    w_floor: float,
    # feasible initializer
    init_method: Literal["project", "line_search"] = "line_search",
    init_n_scan: int = 31,
    init_n_bisect: int = 30,
    p_init_feasible: SVIParams | None = None,
    # penalties / optimization
    lambda_price: float = 1.0,
    lambda_g: float = 1e5,
    g_floor: float = 0.0,
    g_scale: float = 0.02,
    lambda_wfloor: float = 1e5,
    w_scale: float = 0.02,
    max_nfev: int = 1500,
) -> SVIParams:
    """
    Paper-style repair refinement:
      - Freeze (v, psi, p) in JW
      - Optimize only (c, v_tilde) over the rectangle between original and Section 5.1 target
      - Objective = price closeness to original smile + large butterfly penalty (+ optional w-floor penalty)

    Returns a feasible repaired raw SVI slice.
    """
    if T <= 0:
        raise ValueError("T must be > 0")

    # If already arb-free, keep unchanged
    b0 = check_butterfly_arbitrage(
        p_raw,
        y_domain_hint=y_domain_hint,
        w_floor=w_floor,
        g_floor=0.0,
        tol=1e-10,
    )
    if b0.ok:
        return p_raw

    p_jw = p_raw.to_jw(T=T)
    c_ast, vtilde_ast = _gj_section51_targets(p_jw)

    # Rectangle (not diagonal line)
    c_lo, c_hi = sorted([float(p_jw.c), float(c_ast)])
    vt_lo, vt_hi = sorted([float(p_jw.v_tilde), float(vtilde_ast)])

    # Feasible initializer (line-search/project)
    if p_init_feasible is None:
        from .repair import repair_butterfly_raw

        p_init_feasible = repair_butterfly_raw(
            p_raw,
            T=T,
            y_domain_hint=y_domain_hint,
            w_floor=w_floor,
            method=init_method,
            tol=1e-10,
            n_scan=init_n_scan,
            n_bisect=init_n_bisect,
        )

    jw_init = p_init_feasible.to_jw(T=T)
    x0 = np.array(
        [
            float(np.clip(jw_init.c, c_lo, c_hi)),
            float(np.clip(jw_init.v_tilde, vt_lo, vt_hi)),
        ],
        dtype=np.float64,
    )

    y_obj = np.asarray(y_obj, dtype=np.float64).reshape(-1)
    y_penalty = np.asarray(y_penalty, dtype=np.float64).reshape(-1)

    y_obj = y_obj[np.isfinite(y_obj)]
    y_penalty = y_penalty[np.isfinite(y_penalty)]

    if y_obj.size == 0:
        raise ValueError("y_obj must contain at least one finite point")
    if y_penalty.size == 0:
        y_penalty = y_obj.copy()

    # Anchor prices = prices of the pre-repair smile (paper-style closeness target)
    w_anchor = svi_total_variance(y_obj, p_raw)
    c_anchor = _normalized_black_call_from_total_variance(y_obj, w_anchor)

    n_price = int(y_obj.size)
    n_g = int(y_penalty.size) if lambda_g > 0.0 else 0
    n_w = int(y_penalty.size) if lambda_wfloor > 0.0 else 0
    n_tot = n_price + n_g + n_w

    big_bad = np.full(n_tot, 1e6, dtype=np.float64)

    def _candidate_raw(c_val: float, vt_val: float) -> SVIParams | None:
        jw = replace(p_jw, c=float(c_val), v_tilde=float(vt_val))
        try:
            return jw.to_raw(T)
        except ValueError:
            return None

    def residual_x(x: NDArray[np.float64]) -> NDArray[np.float64]:
        c_val = float(x[0])
        vt_val = float(x[1])

        p_cand = _candidate_raw(c_val, vt_val)
        if p_cand is None:
            return big_bad.copy()

        # Price-closeness term (paper spirit)
        w_cand_obj = svi_total_variance(y_obj, p_cand)
        c_cand = _normalized_black_call_from_total_variance(y_obj, w_cand_obj)
        r_price = np.sqrt(max(lambda_price, 0.0)) * (c_cand - c_anchor)

        blocks: list[NDArray[np.float64]] = [r_price.astype(np.float64, copy=False)]

        # Large butterfly penalty
        if lambda_g > 0.0 and y_penalty.size:
            g = gatheral_g_vec(y_penalty, p_cand, w_floor=w_floor)
            deficit = (float(g_floor) - g) / max(float(g_scale), 1e-12)
            h = softplus(deficit) - LOG2
            h = np.maximum(h, 0.0)
            r_g = np.sqrt(lambda_g) * h
            if not np.all(np.isfinite(r_g)):
                return big_bad.copy()
            blocks.append(r_g.astype(np.float64, copy=False))

        # Optional w-floor penalty (helps keep price eval sane in extreme proposals)
        if lambda_wfloor > 0.0 and y_penalty.size:
            w_pen = svi_total_variance(y_penalty, p_cand)
            deficit_w = (float(w_floor) - w_pen) / max(float(w_scale), 1e-12)
            h_w = softplus(deficit_w) - LOG2
            h_w = np.maximum(h_w, 0.0)
            r_w = np.sqrt(lambda_wfloor) * h_w
            if not np.all(np.isfinite(r_w)):
                return big_bad.copy()
            blocks.append(r_w.astype(np.float64, copy=False))

        out = np.concatenate(blocks)
        if out.size != n_tot or not np.all(np.isfinite(out)):
            return big_bad.copy()
        return out

    lb = np.array([c_lo, vt_lo], dtype=np.float64)
    ub = np.array([c_hi, vt_hi], dtype=np.float64)

    res = least_squares(
        fun=residual_x,
        x0=x0,
        bounds=(lb, ub),
        loss="linear",
        x_scale="jac",
        max_nfev=int(max_nfev),
    )

    # Candidate from optimizer (if usable)
    p_best = p_init_feasible
    if res.success and np.all(np.isfinite(res.x)):
        p_try = _candidate_raw(float(res.x[0]), float(res.x[1]))
        if p_try is not None:
            b_try = check_butterfly_arbitrage(
                p_try,
                y_domain_hint=y_domain_hint,
                w_floor=w_floor,
                g_floor=0.0,
                tol=1e-10,
            )
            if b_try.ok:
                p_best = p_try

    # Final safety guard
    b_best = check_butterfly_arbitrage(
        p_best,
        y_domain_hint=y_domain_hint,
        w_floor=w_floor,
        g_floor=0.0,
        tol=1e-10,
    )
    if not b_best.ok:
        raise ValueError(
            "repair_butterfly_jw_optimal failed to produce a feasible candidate "
            f"(reason={b_best.failure_reason}, min_g={b_best.min_g})."
        )

    return p_best


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
    c_ast, vtilde_ast = _gj_section51_targets(p_jw)

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
