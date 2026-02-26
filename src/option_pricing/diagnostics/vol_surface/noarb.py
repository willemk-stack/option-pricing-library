from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from option_pricing.models.black_scholes import bs as bs_model
from option_pricing.vol.arbitrage import CalendarVarianceReport, SurfaceNoArbReport

from .models import NoArbWorstPointsReport
from .sampling import _smile_grid


def calendar_dW(surface, *, x_grid):
    xg = np.asarray(x_grid, dtype=float)
    W = np.asarray(
        np.vstack([np.asarray(s.w_at(xg), dtype=float) for s in surface.smiles]),
        dtype=float,
    )
    return np.asarray(W[1:, :] - W[:-1, :], dtype=float)


def first_failing_smile(report):
    for T, r in getattr(report, "smile_monotonicity", []):
        if not getattr(r, "ok", True):
            return T, r
    return None


def calendar_dW_from_report(surface, report):
    cal = getattr(report, "calendar_total_variance", None)
    if cal is None:
        return None
    performed = bool(getattr(cal, "performed", False))
    ok = bool(getattr(cal, "ok", True))
    xg = getattr(cal, "x_grid", None)
    if (not performed) or ok or (xg is None):
        return None
    xg_arr = np.asarray(xg, dtype=float)
    dW = np.asarray(cal.dW, dtype=float) if hasattr(cal, "dW") else None
    return xg_arr, dW


def noarb_smile_table(report):
    mono = list(getattr(report, "smile_monotonicity", ()) or ())
    conv = dict(getattr(report, "smile_convexity", ()) or ())
    rows = []
    for T, mrep in mono:
        row = {"T": T, "monotonicity_ok": getattr(mrep, "ok", True)}
        c = conv.get(T, None)
        if c is not None:
            row["convexity_ok"] = getattr(c, "ok", True)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("T").reset_index(drop=True)
    return df


def calendar_summary(report):
    cal = getattr(report, "calendar_total_variance", None)
    if cal is None:
        return {}
    performed = bool(getattr(cal, "performed", False))
    ok = bool(getattr(cal, "ok", True))
    bad_pairs = np.asarray(getattr(cal, "bad_pairs", np.empty((0, 2), dtype=int)))
    n_viol = int(bad_pairs.shape[0]) if bad_pairs.ndim == 2 else int(bad_pairs.size)
    return {
        "performed": performed,
        "ok": ok,
        "n_violations": n_viol,
        "max_violation": float(getattr(cal, "max_violation", 0.0)),
        "message": str(getattr(cal, "message", "")),
        "x_grid": getattr(cal, "x_grid", None),
        "bad_pairs": bad_pairs,
    }


def first_failing_convexity(report):
    for T, r in getattr(report, "smile_convexity", []):
        if not getattr(r, "ok", True):
            return T, r
    return None


def _worst_monotonicity_rows(
    surface,
    report: SurfaceNoArbReport,
    *,
    forward,
    df,
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for T, rep in report.smile_monotonicity:
        if getattr(rep, "ok", True):
            continue
        s = next((sm for sm in surface.smiles if float(sm.T) == float(T)), None)
        if s is None:
            continue
        y, w = _smile_grid(s)
        F = float(forward(float(T)))
        dfT = float(df(float(T)))
        K = F * np.exp(y)
        iv = np.sqrt(np.maximum(w / float(T), 0.0))
        C = bs_model.black76_call_price_vec(
            forward=F, strikes=K, sigma=iv, tau=float(T), df=dfT
        )
        dC = np.diff(C)
        bad = np.asarray(getattr(rep, "bad_indices", np.empty((0,), dtype=int)))
        for idx in bad:
            i = int(idx)
            if 0 <= i < dC.size:
                rows.append(
                    {
                        "T": float(T),
                        "i": int(i),
                        "K_left": float(K[i]),
                        "K_right": float(K[i + 1]),
                        "y_left": float(y[i]),
                        "y_right": float(y[i + 1]),
                        "dC": float(dC[i]),
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=["T", "i", "K_left", "K_right", "y_left", "y_right", "dC"]
        )
    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values("dC", ascending=False).head(int(top_n))
    return df_out.reset_index(drop=True)


def _worst_convexity_rows(
    surface,
    report: SurfaceNoArbReport,
    *,
    forward,
    df,
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for T, rep in report.smile_convexity:
        if getattr(rep, "ok", True):
            continue
        s = next((sm for sm in surface.smiles if float(sm.T) == float(T)), None)
        if s is None:
            continue
        y, w = _smile_grid(s)
        F = float(forward(float(T)))
        dfT = float(df(float(T)))
        K = F * np.exp(y)
        iv = np.sqrt(np.maximum(w / float(T), 0.0))
        C = bs_model.black76_call_price_vec(
            forward=F, strikes=K, sigma=iv, tau=float(T), df=dfT
        )
        if K.size < 3:
            continue
        K_m1 = K[:-2]
        K_0 = K[1:-1]
        K_p1 = K[2:]
        denom = K_p1 - K_m1
        wL = (K_p1 - K_0) / denom
        wR = (K_0 - K_m1) / denom
        C_chord = wL * C[:-2] + wR * C[2:]
        violation = C[1:-1] - C_chord
        bad = np.asarray(getattr(rep, "bad_indices", np.empty((0,), dtype=int)))
        for idx in bad:
            i = int(idx)
            if 0 <= i - 1 < violation.size:
                j = i - 1
                rows.append(
                    {
                        "T": float(T),
                        "i": int(i),
                        "K": float(K[i]),
                        "y": float(y[i]),
                        "violation": float(violation[j]),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["T", "i", "K", "y", "violation"])
    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values("violation", ascending=False).head(int(top_n))
    return df_out.reset_index(drop=True)


def _worst_calendar_rows(
    surface,
    cal: CalendarVarianceReport,
    *,
    top_n: int,
) -> pd.DataFrame:
    if not getattr(cal, "performed", False) or getattr(cal, "ok", True):
        return pd.DataFrame(columns=["T0", "T1", "x", "dW", "violation"])

    x_grid = np.asarray(getattr(cal, "x_grid", np.empty((0,))), dtype=float)
    if x_grid.size == 0:
        return pd.DataFrame(columns=["T0", "T1", "x", "dW", "violation"])

    W = np.vstack([np.asarray(s.w_at(x_grid), dtype=float) for s in surface.smiles])
    dW = W[1:, :] - W[:-1, :]
    bad_pairs = np.asarray(getattr(cal, "bad_pairs", np.empty((0, 2), dtype=int)))

    rows: list[dict[str, Any]] = []
    for i, j in bad_pairs:
        i = int(i)
        j = int(j)
        if 0 <= i < dW.shape[0] and 0 <= j < dW.shape[1]:
            dW_ij = float(dW[i, j])
            rows.append(
                {
                    "T0": float(surface.smiles[i].T),
                    "T1": float(surface.smiles[i + 1].T),
                    "x": float(x_grid[j]),
                    "dW": dW_ij,
                    "violation": -dW_ij,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["T0", "T1", "x", "dW", "violation"])

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values("dW", ascending=True).head(int(top_n))
    return df_out.reset_index(drop=True)


def noarb_worst_points(
    surface,
    report: SurfaceNoArbReport,
    *,
    forward,
    df,
    top_n: int = 10,
) -> NoArbWorstPointsReport:
    """Return the worst monotonicity/convexity/calendar violations as tables."""

    monot = _worst_monotonicity_rows(
        surface, report, forward=forward, df=df, top_n=top_n
    )
    conv = _worst_convexity_rows(surface, report, forward=forward, df=df, top_n=top_n)
    cal = _worst_calendar_rows(surface, report.calendar_total_variance, top_n=top_n)

    summary: dict[str, float | int | bool | str] = {
        "report_ok": bool(getattr(report, "ok", True)),
        "report_message": str(getattr(report, "message", "")),
        "monotonicity_max": float(monot["dC"].max()) if not monot.empty else 0.0,
        "convexity_max": float(conv["violation"].max()) if not conv.empty else 0.0,
        "calendar_max": float(
            getattr(report.calendar_total_variance, "max_violation", 0.0)
        ),
    }

    suggestions: list[str] = []
    if not monot.empty:
        suggestions.append(
            "Monotonicity violations: check quote consistency, parity, and surface smoothing; consider enforcing no-arb constraints during calibration."
        )
    if not conv.empty:
        suggestions.append(
            "Convexity (butterfly) violations: check smile shape and quote consistency; consider enforcing no-butterfly constraints or smoothing."
        )
    if not cal.empty:
        suggestions.append(
            "Calendar variance violations: check term-structure consistency, time interpolation, and consider monotone total-variance enforcement."
        )

    return NoArbWorstPointsReport(
        monotonicity=monot,
        convexity=conv,
        calendar=cal,
        summary=summary,
        suggestions=tuple(suggestions),
    )
