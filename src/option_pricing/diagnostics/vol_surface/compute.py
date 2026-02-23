"""Diagnostics helpers for implied volatility surfaces.

This module is intentionally **pure** (no plotting) and designed to support
portfolio-style notebooks and demos:

* slice extraction in strike space (K, y, iv, w)
* pricing calls on smile grids (Black-76)
* flattening/summary of no-arbitrage reports
* surfacing SVI calibration diagnostics (when smiles expose them)

The helpers rely on small Protocol interfaces (``VolSurfaceLike`` / ``SmileLike``)
so they can work with multiple surface implementations.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from option_pricing.typing import ArrayLike, ScalarFn
from option_pricing.vol.vol_types import GridSmileSlice, SmileSlice

# -----------------------------------------------------------------------------
# Protocols: minimal interfaces (diagnostics should be surface-implementation
# agnostic)
# -----------------------------------------------------------------------------


class SmileLike(SmileSlice, Protocol):
    """Compatibility protocol.

    In this package, a smile slice may be grid-based (``y``/``w`` arrays) or
    analytic (e.g. SVI) and only exposes ``w_at`` / ``iv_at`` plus a recommended
    ``y`` domain.

    Some smiles (e.g. :class:`option_pricing.vol.svi.SVISmile`) also expose a
    ``diagnostics`` attribute. We treat it as optional and access it via
    duck-typing.
    """


def _smile_grid(smile: SmileLike, *, n: int = 81) -> tuple[np.ndarray, np.ndarray]:
    """Return a (y, w) grid for a smile.

    * If the smile is grid-based, use its native grid.
    * Otherwise sample it uniformly over [y_min, y_max].
    """

    if isinstance(smile, GridSmileSlice):
        y = np.asarray(smile.y, dtype=float)
        w = np.asarray(smile.w, dtype=float)
        return y, w

    y = np.linspace(float(smile.y_min), float(smile.y_max), int(n), dtype=float)
    w = np.asarray(smile.w_at(y), dtype=float)
    return y, w


class VolSurfaceLike(Protocol):
    smiles: Sequence[SmileLike]

    def iv(self, K: ArrayLike, T: float) -> np.ndarray: ...


class VolSurfaceClass(Protocol):
    @classmethod
    def from_grid(
        cls,
        rows: list[tuple[float, float, float]],
        *,
        forward: ScalarFn,
    ) -> VolSurfaceLike: ...


class Black76Module(Protocol):
    def black76_call_price_vec(
        self,
        *,
        forward: float,
        strikes: np.ndarray,
        sigma: np.ndarray,
        tau: float,
        df: float,
    ) -> np.ndarray: ...


# -----------------------------------------------------------------------------
# Convenience data model
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceSlice:
    """A single smile slice expressed in strike space."""

    T: float
    F: float
    K: np.ndarray
    y: np.ndarray
    iv: np.ndarray
    w: np.ndarray


# -----------------------------------------------------------------------------
# Construction and sampling helpers
# -----------------------------------------------------------------------------


def build_surface_from_iv_function(
    *,
    expiries: Sequence[float],
    x_grid: np.ndarray,
    iv_fn: Callable[[float, np.ndarray], np.ndarray],
    forward: ScalarFn,
    VolSurface_cls: VolSurfaceClass | None = None,
) -> VolSurfaceLike:
    """Build a VolSurface from a synthetic ``iv_fn(T, y)`` function."""

    if VolSurface_cls is None:
        # Import from your real module path if needed; the key is to cast it.
        from option_pricing import (
            VolSurface as _VolSurface,  # type: ignore[attr-defined]
        )

        VolSurface_cls = cast(VolSurfaceClass, _VolSurface)

    exp = np.asarray(list(expiries), dtype=float)
    if exp.ndim != 1 or exp.size == 0:
        raise ValueError("expiries must be a non-empty 1D sequence")

    x_grid = np.asarray(x_grid, dtype=float)
    if x_grid.ndim != 1 or x_grid.size == 0:
        raise ValueError("x_grid must be a non-empty 1D array")

    rows: list[tuple[float, float, float]] = []
    for T in exp:
        T_ = float(T)
        F = float(forward(T_))
        K = F * np.exp(x_grid)

        iv = np.asarray(iv_fn(T_, x_grid), dtype=float)
        if iv.shape != x_grid.shape:
            raise ValueError(
                f"iv_fn must return shape {x_grid.shape}, got {iv.shape} at T={T_}"
            )

        for Ki, ivi in zip(K, iv, strict=True):
            rows.append((T_, float(Ki), float(ivi)))

    return VolSurface_cls.from_grid(rows, forward=forward)


def surface_slices(
    surface: VolSurfaceLike, *, forward: ScalarFn
) -> tuple[SurfaceSlice, ...]:
    """Return all smiles as strike-space slices (K, iv, w) for plotting."""

    out: list[SurfaceSlice] = []
    for s in surface.smiles:
        T = float(s.T)
        F = float(forward(T))
        y, w = _smile_grid(s)
        K = F * np.exp(y)
        iv = np.sqrt(np.maximum(w / T, 0.0))
        out.append(SurfaceSlice(T=T, F=F, K=K, y=y, iv=iv, w=w))
    return tuple(out)


def surface_points_df(surface: VolSurfaceLike, *, forward: ScalarFn) -> pd.DataFrame:
    """Flatten the surface into a tidy DataFrame (T, y, K, iv, w, F)."""

    rows: list[dict[str, float]] = []
    for sl in surface_slices(surface, forward=forward):
        for y, K, iv, w in zip(sl.y, sl.K, sl.iv, sl.w, strict=True):
            rows.append(
                {
                    "T": sl.T,
                    "y": float(y),
                    "K": float(K),
                    "iv": float(iv),
                    "w": float(w),
                    "F": float(sl.F),
                }
            )
    return pd.DataFrame(rows)


def query_iv_curve(surface: VolSurfaceLike, *, K: ArrayLike, T: float) -> np.ndarray:
    """Convenience wrapper around ``surface.iv``."""

    return np.asarray(surface.iv(K, float(T)), dtype=float)


def get_smile_at_T(
    surface: VolSurfaceLike, T: float, *, atol: float = 1e-12
) -> SmileLike:
    """Find a smile by expiry using ``np.isclose`` matching."""

    T = float(T)
    for s in surface.smiles:
        if np.isclose(float(s.T), T, atol=atol, rtol=0.0):
            return s
    raise KeyError(
        f"No smile found at T={T} (available: {[float(s.T) for s in surface.smiles]})"
    )


# -----------------------------------------------------------------------------
# Call pricing on smile grids
# -----------------------------------------------------------------------------


def call_prices_from_smile(
    surface: VolSurfaceLike,
    *,
    T: float,
    forward: ScalarFn,
    df: ScalarFn,
    bs_model: Black76Module | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute discounted Black-76 call prices on the smile grid for expiry ``T``."""

    if bs_model is None:
        from option_pricing.models.black_scholes import (
            bs as _bs,  # type: ignore[attr-defined]
        )

        bs_model = cast(Black76Module, _bs)

    s = get_smile_at_T(surface, T)
    T = float(s.T)
    F = float(forward(T))
    dfT = float(df(T))

    y, w = _smile_grid(s)
    K = F * np.exp(y)
    iv = np.sqrt(np.maximum(w / T, 0.0))

    C = bs_model.black76_call_price_vec(forward=F, strikes=K, sigma=iv, tau=T, df=dfT)
    return K, np.asarray(C, dtype=float), iv


# -----------------------------------------------------------------------------
# Calendar checks (total variance monotonicity in T)
# -----------------------------------------------------------------------------


def calendar_dW(surface: VolSurfaceLike, *, x_grid: np.ndarray) -> np.ndarray:
    """Compute Δw across maturities on a shared ``x_grid``."""

    xg = np.asarray(x_grid, dtype=float)

    W = np.asarray(
        np.vstack([np.asarray(s.w_at(xg), dtype=float) for s in surface.smiles]),
        dtype=float,
    )
    return np.asarray(W[1:, :] - W[:-1, :], dtype=float)


def first_failing_smile(report: Any) -> tuple[float, Any] | None:
    """Return (T, smile_report) for the first smile that fails strike monotonicity."""

    for T, r in getattr(report, "smile_monotonicity", []):
        if not bool(getattr(r, "ok", True)):
            return float(T), r
    return None


def calendar_dW_from_report(
    surface: VolSurfaceLike, report: Any
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (x_grid, dW) if the calendar check was performed and failed."""

    cal = getattr(report, "calendar_total_variance", None)
    if cal is None:
        return None

    performed = bool(getattr(cal, "performed", False))
    ok = bool(getattr(cal, "ok", True))
    xg = getattr(cal, "x_grid", None)

    if (not performed) or ok or (xg is None):
        return None

    xg_arr = np.asarray(xg, dtype=float)
    return xg_arr, calendar_dW(surface, x_grid=xg_arr)


# -----------------------------------------------------------------------------
# No-arbitrage report helpers
# -----------------------------------------------------------------------------


def noarb_smile_table(report: Any) -> pd.DataFrame:
    """Return a compact per-expiry table from a surface no-arb report.

    Designed for use with :func:`option_pricing.vol.arbitrage.check_surface_noarb`.

    Columns include smile-level monotonicity and convexity checks.
    """

    mono = list(getattr(report, "smile_monotonicity", ()) or ())
    conv = dict(getattr(report, "smile_convexity", ()) or ())

    rows: list[dict[str, Any]] = []
    for T, mrep in mono:
        Trep = float(T)
        crep = conv.get(T, conv.get(Trep))

        rows.append(
            {
                "T": Trep,
                "monotonic_ok": bool(getattr(mrep, "ok", False)),
                "monotonic_bad_count": int(
                    np.asarray(getattr(mrep, "bad_indices", [])).size
                ),
                "monotonic_max_violation": float(
                    getattr(mrep, "max_violation", np.nan)
                ),
                "monotonic_message": str(getattr(mrep, "message", "")),
                "convex_ok": (
                    bool(getattr(crep, "ok", False)) if crep is not None else False
                ),
                "convex_bad_count": (
                    int(np.asarray(getattr(crep, "bad_indices", [])).size)
                    if crep is not None
                    else 0
                ),
                "convex_max_violation": (
                    float(getattr(crep, "max_violation", np.nan))
                    if crep is not None
                    else np.nan
                ),
                "convex_message": (
                    str(getattr(crep, "message", "")) if crep is not None else ""
                ),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("T").reset_index(drop=True)
    return df


def calendar_summary(report: Any) -> dict[str, Any]:
    """Summarize the calendar no-arbitrage portion of a surface report."""

    cal = getattr(report, "calendar_total_variance", None)
    if cal is None:
        return {
            "performed": False,
            "ok": True,
            "n_violations": 0,
            "max_violation": 0.0,
            "message": "Calendar check not present on report.",
        }

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


# -----------------------------------------------------------------------------
# SVI fit diagnostics (duck-typed)
# -----------------------------------------------------------------------------


def _maybe_get_svi_diagnostics(smile: SmileLike) -> tuple[Any, Any, Any] | None:
    """Return (diag, checks, solver) if the smile exposes *SVI-like* diagnostics.

    We intentionally keep this duck-typed, but we also try to avoid false positives
    (e.g. a smile with a random ``diagnostics`` attribute).

    A diagnostics object is treated as SVI-like if it has a non-None ``checks``
    attribute (matching :class:`option_pricing.vol.svi.SVIFitDiagnostics`).
    """

    diag = getattr(smile, "diagnostics", None)
    if diag is None:
        return None

    checks = getattr(diag, "checks", None)
    if checks is None:
        return None

    solver = getattr(diag, "solver", None)
    return diag, checks, solver


def svi_fit_table(surface: VolSurfaceLike) -> pd.DataFrame:
    """Flatten available SVI calibration diagnostics into a per-expiry table.

    This helper is **duck-typed**: if a smile slice exposes a ``diagnostics``
    attribute compatible with :class:`option_pricing.vol.svi.SVIFitDiagnostics`,
    its fields are extracted. Otherwise, the slice is included with
    ``has_diagnostics=False``.

    Returns
    -------
    pd.DataFrame
        Sorted by expiry ``T``. Missing values are filled with NaN/False.
    """

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

        # ---- checks (fit + rails) ----
        if checks is not None:
            y_dom = getattr(checks, "y_domain", (np.nan, np.nan))
            try:
                y0, y1 = float(y_dom[0]), float(y_dom[1])
            except Exception:
                y0, y1 = np.nan, np.nan

            row.update(
                {
                    # data fit (total variance space)
                    "rmse_w": float(getattr(checks, "rmse_w", np.nan)),
                    "rmse_unw": float(getattr(checks, "rmse_unw", np.nan)),
                    "mae_w": float(getattr(checks, "mae_w", np.nan)),
                    "max_abs_werr": float(getattr(checks, "max_abs_werr", np.nan)),
                    # domain safety
                    "y_domain_min": y0,
                    "y_domain_max": y1,
                    "min_w_domain": float(getattr(checks, "min_w_domain", np.nan)),
                    "argmin_y_domain": float(
                        getattr(checks, "argmin_y_domain", np.nan)
                    ),
                    "n_domain_viol": int(getattr(checks, "n_violations", 0)),
                    # butterfly proxy (g)
                    "butterfly_ok": bool(getattr(checks, "butterfly_ok", False)),
                    "min_g": float(getattr(checks, "min_g", np.nan)),
                    "argmin_g_y": float(getattr(checks, "argmin_g_y", np.nan)),
                    "g_left_inf": float(getattr(checks, "g_left_inf", np.nan)),
                    "g_right_inf": float(getattr(checks, "g_right_inf", np.nan)),
                    "butterfly_reason": str(
                        getattr(checks, "butterfly_reason", "") or ""
                    ),
                    # wing / Lee
                    "sR": float(getattr(checks, "sR", np.nan)),
                    "sL": float(getattr(checks, "sL", np.nan)),
                    "lee_cap": float(getattr(checks, "lee_cap", np.nan)),
                    "lee_slack_R": float(getattr(checks, "lee_slack_R", np.nan)),
                    "lee_slack_L": float(getattr(checks, "lee_slack_L", np.nan)),
                    # flags
                    "rho_near_pm1": bool(getattr(checks, "rho_near_pm1", False)),
                    "sigma_tiny": bool(getattr(checks, "sigma_tiny", False)),
                    "b_large": bool(getattr(checks, "b_large", False)),
                    "b_blown_up": bool(getattr(checks, "b_blown_up", False)),
                    "m_outside_data": bool(getattr(checks, "m_outside_data", False)),
                    # robust weights summary (if IRLS used)
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

        # ---- solver metadata ----
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
                            solver, "irls_outer_iters", getattr(solver, "irls_iters", 0)
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
    surface: VolSurfaceLike,
    quotes_df: pd.DataFrame,
    *,
    T: float,
    forward: ScalarFn | None = None,
    atol: float = 1e-12,
) -> pd.DataFrame:
    """Join quote points to the fitted smile and compute residuals.

    Parameters
    ----------
    surface:
        Calibrated surface.
    quotes_df:
        Quote grid. Must contain either ``y`` or ``K``. Must contain either
        ``iv`` or ``w``. If it contains multiple expiries, a ``T`` column is
        expected and points are filtered using ``np.isclose``.
    T:
        Expiry to slice.
    forward:
        Needed if ``y`` is not present (to convert strikes to log-moneyness).

    Returns
    -------
    pd.DataFrame
        Columns: ``T, y, K, F, iv_mkt, iv_fit, resid_iv, w_mkt, w_fit, resid_w``.
    """

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

    # --- y / K ---
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

    # --- market iv / w ---
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


def surface_domain_report(
    surface: VolSurfaceLike,
    *,
    quotes_df: pd.DataFrame | None = None,
    forward: ScalarFn | None = None,
    atol: float = 1e-12,
) -> pd.DataFrame:
    """Per-expiry domain coverage report (data vs recommended model domain).

    For each expiry ``T`` on the surface, returns:

    * model domain: ``y_model_min/max`` (from the smile slice)
    * optional quote domain: ``y_data_min/max`` (from ``quotes_df``)
    * slack:
        - ``slack_left  = y_data_min - y_model_min``
        - ``slack_right = y_model_max - y_data_max``

    Interpreting slack:
    * positive slack => model recommended domain extends beyond the quote range
    * negative slack => quotes extend beyond the model recommended domain
    """

    rows: list[dict[str, Any]] = []

    q = quotes_df.copy() if quotes_df is not None else None

    for s in surface.smiles:
        T = float(s.T)
        y_model_min = float(s.y_min)
        y_model_max = float(s.y_max)

        row: dict[str, Any] = {
            "T": T,
            "y_model_min": y_model_min,
            "y_model_max": y_model_max,
            "y_data_min": np.nan,
            "y_data_max": np.nan,
            "slack_left": np.nan,
            "slack_right": np.nan,
        }

        if forward is not None:
            row["F"] = float(forward(T))

        if q is not None and not q.empty:
            if "T" in q.columns:
                qq = q.loc[
                    np.isclose(q["T"].astype(float), T, atol=atol, rtol=0.0)
                ].copy()
            else:
                # quotes_df treated as single-expiry; apply to each slice
                qq = q

            if not qq.empty:
                if "y" in qq.columns:
                    y_data = qq["y"].astype(float).to_numpy()
                elif "K" in qq.columns:
                    if forward is None:
                        raise ValueError(
                            "quotes_df has K but not y; forward(T) is required"
                        )
                    F = float(forward(T))
                    K = qq["K"].astype(float).to_numpy()
                    y_data = np.log(K / F)
                else:
                    raise ValueError("quotes_df must contain 'y' or 'K'")

                y0 = float(np.min(y_data))
                y1 = float(np.max(y_data))
                row["y_data_min"] = y0
                row["y_data_max"] = y1
                row["slack_left"] = y0 - y_model_min
                row["slack_right"] = y_model_max - y1

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("T").reset_index(drop=True)
    return df
