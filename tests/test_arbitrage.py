# tests/test_vol_noarb.py
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from option_pricing.vol.arbitrage import (
    check_smile_call_convexity,
    check_smile_price_monotonicity,
    check_surface_noarb,
)
from option_pricing.vol.surface import Smile, VolSurface


# -------------------------
# Helpers
# -------------------------
def flat_df(r: float) -> Callable[[float], float]:
    def df(T: float) -> float:
        return float(np.exp(-float(r) * float(T)))

    return df


def flat_forward(
    spot: float, r: float = 0.0, q: float = 0.0
) -> Callable[[float], float]:
    def fwd(T: float) -> float:
        T = float(T)
        return float(float(spot) * np.exp((float(r) - float(q)) * T))

    return fwd


def make_smile_from_iv(T: float, x: np.ndarray, iv: np.ndarray) -> Smile:
    x = np.asarray(x, dtype=np.float64)
    iv = np.asarray(iv, dtype=np.float64)
    w = np.float64(T) * (iv**2)
    return Smile(T=float(T), x=x, w=w.astype(np.float64, copy=False))


# -------------------------
# Smile monotonicity tests
# -------------------------
def test_smile_monotonicity_ok_reasonable_smile() -> None:
    """Sane smile => call prices decreasing in strike."""
    T = 1.0
    x = np.linspace(-0.5, 0.5, 31, dtype=np.float64)
    iv = 0.20 + 0.10 * (x**2)
    smile = make_smile_from_iv(T, x, iv)

    fwd = flat_forward(spot=100.0, r=0.01, q=0.0)
    df = flat_df(r=0.01)

    rep = check_smile_price_monotonicity(smile, forward=fwd, df=df, tol=1e-10)
    assert rep.ok, rep.message
    assert rep.bad_indices.size == 0
    assert rep.max_violation == 0.0


def test_smile_monotonicity_fails_when_vol_increases_strongly_with_strike() -> None:
    """Pathological increasing IV can generate non-monotone call prices."""
    T = 1.0
    x = np.linspace(-0.2, 0.8, 51, dtype=np.float64)
    iv = 0.15 + 0.80 * (x - x.min())  # deliberately extreme increasing vols
    smile = make_smile_from_iv(T, x, iv)

    fwd = flat_forward(spot=100.0, r=0.0, q=0.0)
    df = flat_df(r=0.0)

    rep = check_smile_price_monotonicity(smile, forward=fwd, df=df, tol=1e-12)
    assert not rep.ok
    assert rep.bad_indices.size > 0
    assert rep.max_violation > 0.0


def test_smile_monotonicity_raises_on_non_positive_T() -> None:
    x = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
    w = np.array([0.01, 0.01, 0.01], dtype=np.float64)
    smile = Smile(T=0.0, x=x, w=w)

    with pytest.raises(ValueError):
        check_smile_price_monotonicity(
            smile,
            forward=flat_forward(100.0),
            df=flat_df(0.0),
        )


def test_smile_monotonicity_raises_on_bad_forward_or_df() -> None:
    T = 1.0
    x = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
    iv = np.array([0.2, 0.2, 0.2], dtype=np.float64)
    smile = make_smile_from_iv(T, x, iv)

    def bad_forward(_T: float) -> float:
        return 0.0

    def bad_df(_T: float) -> float:
        return -1.0

    with pytest.raises(ValueError):
        check_smile_price_monotonicity(smile, forward=bad_forward, df=flat_df(0.0))

    with pytest.raises(ValueError):
        check_smile_price_monotonicity(smile, forward=flat_forward(100.0), df=bad_df)


# -------------------------
# Smile convexity tests
# -------------------------
def test_smile_convexity_ok_reasonable_smile() -> None:
    """Discrete convexity sanity check should pass for a normal smile."""
    T = 1.0
    x = np.linspace(-0.5, 0.5, 41, dtype=np.float64)
    iv = 0.20 + 0.08 * (x**2)
    smile = make_smile_from_iv(T, x, iv)

    fwd = flat_forward(spot=100.0, r=0.01, q=0.0)
    df = flat_df(r=0.01)

    rep = check_smile_call_convexity(smile, forward=fwd, df=df, tol=1e-10)
    assert rep.ok, rep.message
    assert rep.bad_indices.size == 0
    assert rep.max_violation == 0.0


def test_smile_convexity_fails_for_spiky_iv_shape() -> None:
    """A large local spike in IV can violate discrete convexity in strike."""
    T = 1.0
    x = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    iv = np.full_like(x, 0.20)
    iv[x.size // 2] = 3.00
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
def test_surface_calendar_variance_ok() -> None:
    """Two smiles with increasing total variance => calendar check OK."""
    fwd = flat_forward(spot=100.0, r=0.0, q=0.0)
    df = flat_df(0.0)

    x = np.linspace(-0.4, 0.4, 21, dtype=np.float64)

    T1, T2 = 0.5, 1.0
    iv1 = 0.20 + 0.05 * (x**2)
    iv2 = 0.22 + 0.05 * (x**2)

    rows: list[tuple[float, float, float]] = []
    for T, iv in [(T1, iv1), (T2, iv2)]:
        F = fwd(T)
        K = F * np.exp(x)
        for Ki, ivi in zip(K, iv, strict=True):
            rows.append((float(T), float(Ki), float(ivi)))

    surface = VolSurface.from_grid(rows, forward=fwd)

    rep = check_surface_noarb(surface, df=df, tol_calendar=1e-12, tol_butterfly=1e-10)
    assert rep.calendar_total_variance.performed
    assert rep.calendar_total_variance.ok, rep.calendar_total_variance.message
    assert rep.ok, rep.message
    assert all(r.ok for _, r in rep.smile_convexity), "Unexpected butterfly violations"


def test_surface_calendar_variance_fails_when_w_decreases() -> None:
    """Force w(T2,x) < w(T1,x) => calendar check should fail."""
    fwd = flat_forward(spot=100.0, r=0.0, q=0.0)
    df = flat_df(0.0)

    x = np.linspace(-0.4, 0.4, 21, dtype=np.float64)

    T1, T2 = 0.5, 1.0
    iv1 = 0.30 + 0.0 * x
    iv2 = 0.10 + 0.0 * x

    rows: list[tuple[float, float, float]] = []
    for T, iv in [(T1, iv1), (T2, iv2)]:
        F = fwd(T)
        K = F * np.exp(x)
        for Ki, ivi in zip(K, iv, strict=True):
            rows.append((float(T), float(Ki), float(ivi)))

    surface = VolSurface.from_grid(rows, forward=fwd)

    rep = check_surface_noarb(surface, df=df, tol_calendar=1e-12, tol_butterfly=1e-10)
    assert rep.calendar_total_variance.performed
    assert not rep.calendar_total_variance.ok
    assert rep.calendar_total_variance.bad_pairs.size > 0
    assert rep.calendar_total_variance.max_violation > 0.0
    assert not rep.ok


def test_surface_calendar_check_skipped_for_single_expiry() -> None:
    fwd = flat_forward(spot=100.0)
    df = flat_df(0.0)

    x = np.linspace(-0.2, 0.2, 11, dtype=np.float64)
    T = 1.0
    iv = 0.2 + 0.0 * x

    rows: list[tuple[float, float, float]] = []
    F = fwd(T)
    K = F * np.exp(x)
    for Ki, ivi in zip(K, iv, strict=True):
        rows.append((float(T), float(Ki), float(ivi)))

    surface = VolSurface.from_grid(rows, forward=fwd)
    rep = check_surface_noarb(surface, df=df, tol_butterfly=1e-10)

    assert rep.calendar_total_variance.performed is False
    assert rep.calendar_total_variance.ok is True
