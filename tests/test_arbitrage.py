from __future__ import annotations

import numpy as np
import pytest

from option_pricing.vol.arbitrage import (
    check_smile_call_convexity,  # NEW
    check_smile_price_monotonicity,
    check_surface_noarb,
)
from option_pricing.vol.surface import Smile, VolSurface


# -------------------------
# Helpers for tests
# -------------------------
def flat_df(r: float):
    def df(T: float) -> float:
        return float(np.exp(-r * float(T)))

    return df


def flat_forward(spot: float, r: float = 0.0, q: float = 0.0):
    def fwd(T: float) -> float:
        T = float(T)
        return float(spot * np.exp((r - q) * T))

    return fwd


def make_smile_from_iv(
    T: float,
    x: np.ndarray,
    iv: np.ndarray,
) -> Smile:
    x = np.asarray(x, dtype=np.float64)
    iv = np.asarray(iv, dtype=np.float64)
    w = float(T) * iv**2
    return Smile(T=float(T), x=x, w=w.astype(np.float64, copy=False))


# -------------------------
# Smile monotonicity tests
# -------------------------
def test_smile_monotonicity_ok_reasonable_smile():
    """
    Construct a plausible smile: higher vols in wings, lower near ATM.
    This should produce call prices decreasing in strike.
    """
    T = 1.0
    x = np.linspace(-0.5, 0.5, 31)
    iv = 0.20 + 0.10 * (x**2)  # symmetric "smile"
    smile = make_smile_from_iv(T, x, iv)

    fwd = flat_forward(spot=100.0, r=0.01, q=0.0)
    df = flat_df(r=0.01)

    rep = check_smile_price_monotonicity(smile, forward=fwd, df=df, tol=1e-10)
    assert rep.ok, rep.message
    assert rep.bad_indices.size == 0
    assert rep.max_violation == 0.0


def test_smile_monotonicity_fails_when_vol_increases_strongly_with_strike():
    """
    If vols explode with strike, call prices can become non-monotone in strike.
    We deliberately build a pathological "smirk" that increases with x.
    """
    T = 1.0
    x = np.linspace(-0.2, 0.8, 51)
    iv = 0.15 + 0.80 * (x - x.min())  # very steep increasing vols to the right
    smile = make_smile_from_iv(T, x, iv)

    fwd = flat_forward(spot=100.0, r=0.0, q=0.0)
    df = flat_df(r=0.0)

    rep = check_smile_price_monotonicity(smile, forward=fwd, df=df, tol=1e-12)
    assert not rep.ok
    assert rep.bad_indices.size > 0
    assert rep.max_violation > 0.0


def test_smile_monotonicity_raises_on_non_positive_T():
    x = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
    w = np.array([0.01, 0.01, 0.01], dtype=np.float64)
    smile = Smile(T=0.0, x=x, w=w)

    with pytest.raises(ValueError):
        check_smile_price_monotonicity(
            smile,
            forward=flat_forward(100.0),
            df=flat_df(0.0),
        )


def test_smile_monotonicity_raises_on_bad_forward_or_df():
    T = 1.0
    x = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
    iv = np.array([0.2, 0.2, 0.2], dtype=np.float64)
    smile = make_smile_from_iv(T, x, iv)

    def bad_forward(T: float) -> float:
        return 0.0  # not allowed

    def bad_df(T: float) -> float:
        return -1.0  # not allowed

    with pytest.raises(ValueError):
        check_smile_price_monotonicity(smile, forward=bad_forward, df=flat_df(0.0))

    with pytest.raises(ValueError):
        check_smile_price_monotonicity(smile, forward=flat_forward(100.0), df=bad_df)


# -------------------------
# Smile butterfly/convexity tests (NEW)
# -------------------------
def test_smile_convexity_ok_reasonable_smile():
    """
    Discrete convexity (proxy for butterfly no-arb) should hold for a sane smile.
    """
    T = 1.0
    x = np.linspace(-0.5, 0.5, 41)
    iv = 0.20 + 0.08 * (x**2)
    smile = make_smile_from_iv(T, x, iv)

    fwd = flat_forward(spot=100.0, r=0.01, q=0.0)
    df = flat_df(r=0.01)

    rep = check_smile_call_convexity(smile, forward=fwd, df=df, tol=1e-10)
    assert rep.ok, rep.message
    assert rep.bad_indices.size == 0
    assert rep.max_violation == 0.0


def test_smile_convexity_fails_for_spiky_iv_shape():
    """
    Deliberately create a 'vol spike' at the center which can make call prices
    locally non-convex in strike (violating the discrete second-difference check).
    """
    T = 1.0
    x = np.linspace(-0.2, 0.2, 9)
    iv = 0.20 + 0.0 * x
    iv[x.size // 2] = 3.00  # huge spike at ATM to force a violation
    smile = make_smile_from_iv(T, x, iv)

    fwd = flat_forward(spot=100.0, r=0.0, q=0.0)
    df = flat_df(r=0.0)

    rep = check_smile_call_convexity(smile, forward=fwd, df=df, tol=1e-12)
    assert not rep.ok
    assert rep.bad_indices.size > 0
    assert rep.max_violation > 0.0


# -------------------------
# Surface calendar variance tests
# -------------------------
def test_surface_calendar_variance_ok():
    """
    Build two smiles with the same x-grid and increasing total variance with T.
    """
    fwd = flat_forward(spot=100.0, r=0.0, q=0.0)

    x = np.linspace(-0.4, 0.4, 21)

    # Choose vols such that w(T2,x) > w(T1,x) for all x
    T1, T2 = 0.5, 1.0
    iv1 = 0.20 + 0.05 * (x**2)
    iv2 = 0.22 + 0.05 * (x**2)  # slightly higher -> higher w as well

    rows = []
    for T, iv in [(T1, iv1), (T2, iv2)]:
        F = fwd(T)
        K = F * np.exp(x)
        for Ki, ivi in zip(K, iv, strict=True):
            rows.append((T, float(Ki), float(ivi)))

    surface = VolSurface.from_grid(rows, forward=fwd)
    df = flat_df(0.0)

    rep = check_surface_noarb(surface, df=df, tol_calendar=1e-12, tol_butterfly=1e-10)
    assert rep.calendar_total_variance.performed
    assert rep.calendar_total_variance.ok, rep.calendar_total_variance.message

    # NEW: rep.ok now includes monotonicity + convexity + calendar
    assert rep.ok, rep.message
    assert all(r.ok for _, r in rep.smile_convexity), "Unexpected butterfly violations"


def test_surface_calendar_variance_fails_when_w_decreases():
    """
    Make w(T2,x) < w(T1,x) in some region.
    """
    fwd = flat_forward(spot=100.0, r=0.0, q=0.0)

    x = np.linspace(-0.4, 0.4, 21)

    T1, T2 = 0.5, 1.0
    iv1 = 0.30 + 0.00 * x
    iv2 = 0.10 + 0.00 * x  # much lower -> w decreases despite larger T

    rows = []
    for T, iv in [(T1, iv1), (T2, iv2)]:
        F = fwd(T)
        K = F * np.exp(x)
        for Ki, ivi in zip(K, iv, strict=True):
            rows.append((T, float(Ki), float(ivi)))

    surface = VolSurface.from_grid(rows, forward=fwd)
    df = flat_df(0.0)

    rep = check_surface_noarb(surface, df=df, tol_calendar=1e-12, tol_butterfly=1e-10)
    assert rep.calendar_total_variance.performed
    assert not rep.calendar_total_variance.ok
    assert rep.calendar_total_variance.bad_pairs.size > 0
    assert rep.calendar_total_variance.max_violation > 0.0
    assert not rep.ok


def test_surface_calendar_check_skipped_for_single_expiry():
    fwd = flat_forward(spot=100.0)

    x = np.linspace(-0.2, 0.2, 11)
    T = 1.0
    iv = 0.2 + 0.0 * x

    rows = []
    F = fwd(T)
    K = F * np.exp(x)
    for Ki, ivi in zip(K, iv, strict=True):
        rows.append((T, float(Ki), float(ivi)))

    surface = VolSurface.from_grid(rows, forward=fwd)
    rep = check_surface_noarb(surface, df=flat_df(0.0), tol_butterfly=1e-10)

    assert rep.calendar_total_variance.performed is False
    assert rep.calendar_total_variance.ok is True
