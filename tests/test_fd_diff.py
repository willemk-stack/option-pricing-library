# tests/test_fd_diff.py

import importlib

import numpy as np
import pytest


def _import_fd_diff():
    """Import helper that skips only if the module does not exist yet."""
    try:
        return importlib.import_module("option_pricing.numerics.fd.diff")
    except ModuleNotFoundError:
        pytest.skip("option_pricing.numerics.fd.diff not present yet")


def _make_strictly_increasing_grid(rng: np.random.Generator, n: int) -> np.ndarray:
    """Make a strictly increasing 1D grid with mildly irregular spacing."""
    if n < 3:
        raise ValueError("n must be >= 3")
    steps = rng.uniform(0.05, 0.35, size=n - 1)
    x = np.concatenate(([0.0], np.cumsum(steps)))
    return x.astype(float)


def test_diff1_and_diff2_exact_for_quadratic_1d() -> None:
    """diff1/diff2 should be exact for quadratic polynomials (incl. one-sided boundaries)."""
    diff = _import_fd_diff()

    rng = np.random.default_rng(2025)
    x = _make_strictly_increasing_grid(rng, n=31)
    a, b, c = rng.normal(size=3)
    y = a * x**2 + b * x + c

    dy = diff.diff1_nonuniform(y, x, axis=0)
    d2y = diff.diff2_nonuniform(y, x, axis=0)

    expected_dy = 2.0 * a * x + b
    expected_d2y = np.full_like(x, 2.0 * a)

    np.testing.assert_allclose(dy, expected_dy, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(d2y, expected_d2y, rtol=0.0, atol=1e-12)


def test_axis_semantics_on_2d_grid() -> None:
    """Differentiate along K (axis=1/-1) and T (axis=0) on a separable polynomial surface."""
    diff = _import_fd_diff()

    rng = np.random.default_rng(7)
    T = _make_strictly_increasing_grid(rng, n=15)
    K = _make_strictly_increasing_grid(rng, n=21)

    # C(T, K) = f(K) + g(T)
    aK, bK, cK = rng.normal(size=3)
    aT, bT, cT = rng.normal(size=3)

    fK = aK * K**2 + bK * K + cK
    gT = aT * T**2 + bT * T + cT
    C = gT[:, None] + fK[None, :]

    # dC/dK
    expected_dK = np.broadcast_to((2.0 * aK * K + bK)[None, :], C.shape)

    dC_dK_axis1 = diff.diff1_nonuniform(C, K, axis=1)
    dC_dK_axism1 = diff.diff1_nonuniform(C, K, axis=-1)

    np.testing.assert_allclose(dC_dK_axis1, expected_dK, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(dC_dK_axism1, expected_dK, rtol=0.0, atol=1e-12)

    # dC/dT
    expected_dT = np.broadcast_to((2.0 * aT * T + bT)[:, None], C.shape)
    dC_dT = diff.diff1_nonuniform(C, T, axis=0)
    np.testing.assert_allclose(dC_dT, expected_dT, rtol=0.0, atol=1e-12)


def test_diff_raises_on_mismatched_axis_length() -> None:
    diff = _import_fd_diff()
    rng = np.random.default_rng(0)
    x = _make_strictly_increasing_grid(rng, n=10)
    y = rng.normal(size=11)
    with pytest.raises(ValueError):
        _ = diff.diff1_nonuniform(y, x, axis=0)


def test_diff_raises_on_non_increasing_grid() -> None:
    diff = _import_fd_diff()
    x = np.array([0.0, 1.0, 1.0, 2.0], dtype=float)
    y = np.array([0.0, 1.0, 4.0, 9.0], dtype=float)
    with pytest.raises(ValueError):
        _ = diff.diff2_nonuniform(y, x, axis=0)
