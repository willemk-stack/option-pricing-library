# tests/test_fd_stencils.py

import importlib

import numpy as np
import pytest


def _import_fd_stencils():
    """Import helper that skips only if the module does not exist yet."""
    try:
        return importlib.import_module("option_pricing.numerics.fd.stencils")
    except ModuleNotFoundError:
        pytest.skip("option_pricing.numerics.fd.stencils not present yet")


def _make_strictly_increasing_grid(rng: np.random.Generator, n: int) -> np.ndarray:
    """Make a strictly increasing 1D grid with mildly irregular spacing."""
    if n < 3:
        raise ValueError("n must be >= 3")
    steps = rng.uniform(0.05, 0.35, size=n - 1)
    x = np.concatenate(([0.0], np.cumsum(steps)))
    return x.astype(float)


def test_central_coeffs_exact_for_quadratic_1st_derivative() -> None:
    """Central 3-pt nonuniform first-derivative coefficients should be exact for quadratics."""
    st = _import_fd_stencils()
    d1 = st.d1_central_nonuniform_coeffs

    rng = np.random.default_rng(123)
    x = _make_strictly_increasing_grid(rng, n=25)
    hm = x[1:-1] - x[:-2]
    hp = x[2:] - x[1:-1]

    a, b, c = rng.normal(size=3)
    y = a * x**2 + b * x + c

    dl, dd, du = d1(hm, hp)
    approx = dl * y[:-2] + dd * y[1:-1] + du * y[2:]
    expected = 2.0 * a * x[1:-1] + b

    np.testing.assert_allclose(approx, expected, rtol=0.0, atol=1e-12)


def test_central_coeffs_exact_for_quadratic_2nd_derivative() -> None:
    """Central 3-pt nonuniform second-derivative coefficients should be exact for quadratics."""
    st = _import_fd_stencils()
    d2 = st.d2_central_nonuniform_coeffs

    rng = np.random.default_rng(456)
    x = _make_strictly_increasing_grid(rng, n=17)
    hm = x[1:-1] - x[:-2]
    hp = x[2:] - x[1:-1]

    a, b, c = rng.normal(size=3)
    y = a * x**2 + b * x + c

    dl, dd, du = d2(hm, hp)
    approx = dl * y[:-2] + dd * y[1:-1] + du * y[2:]
    expected = np.full_like(approx, 2.0 * a)

    np.testing.assert_allclose(approx, expected, rtol=0.0, atol=1e-12)
