from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..models import bs as bs_model
from .surface import Smile, VolSurface


@dataclass(frozen=True, slots=True)
class MonotonicityReport:
    ok: bool
    bad_indices: np.ndarray
    max_violation: float
    message: str


@dataclass(frozen=True, slots=True)
class CalendarVarianceReport:
    performed: bool
    ok: bool
    x_grid: np.ndarray
    bad_pairs: (
        np.ndarray
    )  # shape (m, 2): rows are (expiry_index i, x_index j) for violation between i and i+1
    max_violation: float
    message: str


@dataclass(frozen=True, slots=True)
class SurfaceNoArbReport:
    ok: bool
    smile_monotonicity: tuple[tuple[float, MonotonicityReport], ...]  # (T, report)
    calendar_total_variance: CalendarVarianceReport
    message: str


def check_smile_price_monotonicity(
    smile: Smile,
    *,
    forward: Callable[[float], float],
    df: Callable[[float], float],
    tol: float = 1e-8,
) -> MonotonicityReport:
    """
    Call monotonicity sanity check at a single expiry:

      For K increasing: C(K) must be non-increasing.

    Since the smile grid is in x = ln(K/F(T)) and x is strictly increasing,
    K = F(T) * exp(x) is also strictly increasing.
    """
    T = float(smile.T)
    if T <= 0.0:
        raise ValueError("Smile.T must be > 0")

    # market quantities at this expiry
    F = float(forward(T))
    if F <= 0.0:
        raise ValueError("forward(T) must be > 0")

    dfT = float(df(T))
    if dfT <= 0.0:
        raise ValueError("df(T) must be > 0")

    # reconstruct strike grid and vols
    x = np.asarray(smile.x, dtype=np.float64)
    w = np.asarray(smile.w, dtype=np.float64)

    K = F * np.exp(x)
    iv = np.sqrt(np.maximum(w / np.float64(T), np.float64(0.0)))

    # price calls (Black-76 on forwards)
    C = bs_model.black76_call_price_vec(
        forward=F,
        strikes=K,
        sigma=iv,
        tau=T,
        df=dfT,
    )

    # monotonicity: C(K_{i+1}) - C(K_i) should be <= 0
    dC = np.diff(C)
    bad = np.where(dC > float(tol))[0]

    ok = bad.size == 0
    msg = "OK" if ok else f"Call monotonicity violated at {bad.size} intervals."

    return MonotonicityReport(
        ok=ok,
        bad_indices=bad,
        max_violation=float(dC[bad].max()) if bad.size else 0.0,
        message=msg,
    )


def _check_calendar_total_variance(
    surface: VolSurface,
    *,
    tol: float = 1e-8,
    x_grid: np.ndarray | None = None,
    n_x: int = 25,
) -> CalendarVarianceReport:
    """
    Calendar sanity check in total variance space:
      w(T_{i+1}, x) >= w(T_i, x)  for fixed x.

    We evaluate w on a common x-grid over the overlap of all smiles' x ranges.
    """
    smiles = surface.smiles
    nT = len(smiles)
    if nT < 2:
        return CalendarVarianceReport(
            performed=False,
            ok=True,
            x_grid=np.asarray([], dtype=np.float64),
            bad_pairs=np.empty((0, 2), dtype=np.int64),
            max_violation=0.0,
            message="Calendar check skipped (need at least 2 expiries).",
        )

    if x_grid is None:
        # overlap in x among all smiles
        x_lo = max(float(s.x[0]) for s in smiles)
        x_hi = min(float(s.x[-1]) for s in smiles)
        if not (x_lo < x_hi):
            return CalendarVarianceReport(
                performed=False,
                ok=True,
                x_grid=np.asarray([], dtype=np.float64),
                bad_pairs=np.empty((0, 2), dtype=np.int64),
                max_violation=0.0,
                message="Calendar check skipped (no overlapping x-range across smiles).",
            )
        if n_x < 2:
            raise ValueError("n_x must be >= 2")
        x_grid = np.linspace(x_lo, x_hi, int(n_x), dtype=np.float64)
    else:
        x_grid = np.asarray(x_grid, dtype=np.float64)
        if x_grid.ndim != 1 or x_grid.size < 2:
            raise ValueError("x_grid must be a 1D array with at least 2 points")

    # W[i, j] = w at expiry i, x_grid[j]
    W = np.vstack([s.w_at(x_grid) for s in smiles]).astype(np.float64, copy=False)

    dW = W[1:, :] - W[:-1, :]  # should be >= -tol
    bad = np.where(dW < -float(tol))
    bad_pairs = np.column_stack(bad).astype(np.int64)  # (i, j)

    ok = bad_pairs.size == 0
    max_violation = float((-dW[bad]).max()) if not ok else 0.0
    msg = (
        "OK"
        if ok
        else f"Calendar variance violated at {bad_pairs.shape[0]} grid points."
    )

    return CalendarVarianceReport(
        performed=True,
        ok=ok,
        x_grid=x_grid,
        bad_pairs=bad_pairs,
        max_violation=max_violation,
        message=msg,
    )


def check_surface_noarb(
    surface: VolSurface,
    *,
    df: Callable[[float], float],
    tol_strike: float = 1e-8,
    tol_calendar: float = 1e-8,
    calendar_x_grid: np.ndarray | None = None,
    calendar_nx: int = 25,
) -> SurfaceNoArbReport:
    """
    Surface-level no-arbitrage sanity checks.

    - Strike sanity per expiry (proxy): call price monotonicity in K.
    - Calendar sanity across expiries: total variance non-decreasing in T at fixed x.
    """
    # 1) Per-smile strike check
    per_smile: list[tuple[float, MonotonicityReport]] = []
    for s in surface.smiles:
        rep = check_smile_price_monotonicity(
            s,
            forward=surface.forward,
            df=df,
            tol=tol_strike,
        )
        per_smile.append((float(s.T), rep))

    # 2) Calendar check in total variance
    cal = _check_calendar_total_variance(
        surface,
        tol=tol_calendar,
        x_grid=calendar_x_grid,
        n_x=calendar_nx,
    )

    ok = all(r.ok for _, r in per_smile) and cal.ok

    n_bad_smiles = sum(0 if r.ok else 1 for _, r in per_smile)
    msg_parts = []
    msg_parts.append("OK" if ok else "Violations found")
    msg_parts.append(f"smiles_bad={n_bad_smiles}/{len(per_smile)}")
    msg_parts.append(
        f"calendar={'OK' if cal.ok else 'BAD'}" if cal.performed else "calendar=SKIPPED"
    )

    return SurfaceNoArbReport(
        ok=ok,
        smile_monotonicity=tuple(per_smile),
        calendar_total_variance=cal,
        message=", ".join(msg_parts),
    )
