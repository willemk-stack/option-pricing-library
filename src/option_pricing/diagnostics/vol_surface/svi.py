from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from option_pricing.typing import ScalarFn

from .sampling import get_smile_at_T


def _maybe_get_svi_diagnostics(smile) -> tuple[Any, Any, Any] | None:
    diag = getattr(smile, "diagnostics", None)
    if diag is None:
        return None
    checks = getattr(diag, "checks", None)
    if checks is None:
        return None
    solver = getattr(diag, "solver", None)
    return diag, checks, solver


def svi_fit_table(surface) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for s in surface.smiles:
        T = float(s.T)
        row: dict[str, Any] = {
            "T": T,
            "has_diagnostics": False,
            "diag_ok": np.nan,
            "failure_reason": "",
        }

        maybe = _maybe_get_svi_diagnostics(s)
        if maybe is None:
            rows.append(row)
            continue

        diag, checks, solver = maybe
        row["has_diagnostics"] = True
        ok_attr = getattr(diag, "ok", None)
        row["diag_ok"] = bool(ok_attr) if ok_attr is not None else np.nan
        row["failure_reason"] = str(getattr(diag, "failure_reason", "") or "")

        if checks is not None:
            y_dom = getattr(checks, "y_domain", (np.nan, np.nan))
            try:
                y0, y1 = float(y_dom[0]), float(y_dom[1])
            except Exception:
                y0, y1 = np.nan, np.nan

            row.update(
                {
                    "rmse_w": float(getattr(checks, "rmse_w", np.nan)),
                    "rmse_unw": float(getattr(checks, "rmse_unw", np.nan)),
                    "mae_w": float(getattr(checks, "mae_w", np.nan)),
                    "max_abs_werr": float(getattr(checks, "max_abs_werr", np.nan)),
                    "y_domain_min": y0,
                    "y_domain_max": y1,
                    "min_w_domain": float(getattr(checks, "min_w_domain", np.nan)),
                    "argmin_y_domain": float(
                        getattr(checks, "argmin_y_domain", np.nan)
                    ),
                    "n_domain_viol": int(getattr(checks, "n_violations", 0)),
                    "butterfly_ok": bool(getattr(checks, "butterfly_ok", False)),
                    "min_g": float(getattr(checks, "min_g", np.nan)),
                    "argmin_g_y": float(getattr(checks, "argmin_g_y", np.nan)),
                    "g_left_inf": float(getattr(checks, "g_left_inf", np.nan)),
                    "g_right_inf": float(getattr(checks, "g_right_inf", np.nan)),
                    "butterfly_reason": str(
                        getattr(checks, "butterfly_reason", "") or ""
                    ),
                    "sR": float(getattr(checks, "sR", np.nan)),
                    "sL": float(getattr(checks, "sL", np.nan)),
                    "lee_cap": float(getattr(checks, "lee_cap", np.nan)),
                    "lee_slack_R": float(getattr(checks, "lee_slack_R", np.nan)),
                    "lee_slack_L": float(getattr(checks, "lee_slack_L", np.nan)),
                    "rho_near_pm1": bool(getattr(checks, "rho_near_pm1", False)),
                    "sigma_tiny": bool(getattr(checks, "sigma_tiny", False)),
                    "b_large": bool(getattr(checks, "b_large", False)),
                    "b_blown_up": bool(getattr(checks, "b_blown_up", False)),
                    "m_outside_data": bool(getattr(checks, "m_outside_data", False)),
                    "robust_w_min": float(
                        getattr(checks, "robust_weights_min", np.nan)
                    ),
                    "robust_w_med": float(
                        getattr(checks, "robust_weights_median", np.nan)
                    ),
                    "robust_w_max": float(
                        getattr(checks, "robust_weights_max", np.nan)
                    ),
                    "robust_w_frac_floored": float(
                        getattr(checks, "robust_weights_frac_floored", np.nan)
                    ),
                    "robust_w_entropy": float(
                        getattr(checks, "robust_weights_entropy", np.nan)
                    ),
                }
            )

        if solver is not None:
            row.update(
                {
                    "termination": str(getattr(solver, "termination", "")),
                    "nfev": int(getattr(solver, "nfev", 0)),
                    "cost": float(getattr(solver, "cost", np.nan)),
                    "optimality": float(getattr(solver, "optimality", np.nan)),
                    "step_norm": float(getattr(solver, "step_norm", np.nan)),
                    "irls_outer_iters": int(
                        getattr(
                            solver,
                            "irls_outer_iters",
                            getattr(solver, "irls_iters", 0),
                        )
                    ),
                }
            )

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("T").reset_index(drop=True)
    return df


def svi_residuals_df(
    surface,
    quotes_df: pd.DataFrame,
    *,
    T: float,
    forward: ScalarFn | None = None,
    atol: float = 1e-12,
) -> pd.DataFrame:
    T = float(T)
    s = get_smile_at_T(surface, T, atol=atol)

    q = quotes_df.copy()
    if "T" in q.columns:
        q = q.loc[np.isclose(q["T"].astype(float), T, atol=atol, rtol=0.0)].copy()

    if q.empty:
        return pd.DataFrame(
            columns=[
                "T",
                "y",
                "K",
                "F",
                "iv_mkt",
                "iv_fit",
                "resid_iv",
                "w_mkt",
                "w_fit",
                "resid_w",
            ]
        )

    if "y" in q.columns:
        y = q["y"].astype(float).to_numpy()
        if forward is not None:
            F = float(forward(T))
            K = (F * np.exp(y)).astype(float)
        else:
            K = (
                q["K"].astype(float).to_numpy()
                if "K" in q.columns
                else np.full_like(y, np.nan)
            )
    elif "K" in q.columns:
        if forward is None:
            raise ValueError("quotes_df has K but not y; forward(T) is required")
        K = q["K"].astype(float).to_numpy()
        F = float(forward(T))
        y = np.log(K / F)
    else:
        raise ValueError("quotes_df must contain a 'y' or 'K' column")

    if "iv" in q.columns:
        iv_mkt = q["iv"].to_numpy(dtype=np.float64, copy=False)
        w_mkt = (iv_mkt * iv_mkt) * np.float64(T)
    elif "w" in q.columns:
        w_mkt = q["w"].to_numpy(dtype=np.float64, copy=False)
        iv_mkt = np.sqrt(np.maximum(w_mkt / np.float64(T), 0.0)).astype(
            np.float64, copy=False
        )
    else:
        raise ValueError("quotes_df must contain an 'iv' or 'w' column")

    w_fit = np.asarray(s.w_at(y), dtype=float)
    iv_fit = np.asarray(s.iv_at(y), dtype=float)

    out = pd.DataFrame(
        {
            "T": T,
            "y": y.astype(float),
            "K": np.asarray(K, dtype=float),
            "F": float(forward(T)) if forward is not None else np.nan,
            "iv_mkt": iv_mkt.astype(float),
            "iv_fit": iv_fit.astype(float),
            "resid_iv": (iv_fit - iv_mkt).astype(float),
            "w_mkt": w_mkt.astype(float),
            "w_fit": w_fit.astype(float),
            "resid_w": (w_fit - w_mkt).astype(float),
        }
    )

    return out.sort_values("y").reset_index(drop=True)
