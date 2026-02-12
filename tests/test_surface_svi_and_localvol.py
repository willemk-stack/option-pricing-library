from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from option_pricing.vol.dupire import _gatheral_local_var_from_w
from option_pricing.vol.surface import (
    LocalVolSurface,
    NoArbInterpolatedSmileSlice,
    Smile,
    VolSurface,
)


@dataclass(frozen=True)
class _ConstSmile:
    """A constant-IV smile slice with analytic derivatives in y.

    w(y, T) = T * sigma^2 (independent of y)
    """

    T: float
    sigma: float
    y_min: float = -2.0
    y_max: float = 2.0

    def w_at(self, xq):
        xq_arr = np.asarray(xq, dtype=np.float64)
        out = np.full_like(
            xq_arr, float(self.T) * float(self.sigma) ** 2, dtype=np.float64
        )
        return out

    def iv_at(self, xq):
        xq_arr = np.asarray(xq, dtype=np.float64)
        return np.full_like(xq_arr, float(self.sigma), dtype=np.float64)

    def dw_dy(self, xq):
        xq_arr = np.asarray(xq, dtype=np.float64)
        return np.zeros_like(xq_arr, dtype=np.float64)

    def d2w_dy2(self, xq):
        xq_arr = np.asarray(xq, dtype=np.float64)
        return np.zeros_like(xq_arr, dtype=np.float64)


def test_gatheral_local_var_from_w_constant_vol() -> None:
    y = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    sigma = 0.2
    w = np.full_like(y, 0.10, dtype=np.float64)  # any positive w
    w_y = np.zeros_like(y)
    w_yy = np.zeros_like(y)
    w_T = np.full_like(y, sigma**2, dtype=np.float64)

    lv, invalid = _gatheral_local_var_from_w(y=y, w=w, w_y=w_y, w_yy=w_yy, w_T=w_T)
    assert not bool(np.any(invalid))
    np.testing.assert_allclose(lv, sigma**2, rtol=0.0, atol=1e-14)


def test_smile_u_shape_interpolation_no_undershoot_and_flat_extrapolation() -> None:
    y = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
    w = 0.10 + 0.20 * (y**2)
    s = Smile(T=1.0, y=y, w=w)

    yq = np.linspace(-1.0, 1.0, 401)
    wq = s.w_at(yq)

    assert float(np.min(wq)) >= float(np.min(w)) - 1e-12
    assert float(np.max(wq)) <= float(np.max(w)) + 1e-10

    # Smile interpolators use flat extrapolation at the boundaries.
    w_ext = s.w_at(np.array([-10.0, 10.0]))
    assert float(w_ext[0]) == pytest.approx(float(w[0]))
    assert float(w_ext[1]) == pytest.approx(float(w[-1]))


def test_smile_requires_strictly_increasing_grid() -> None:
    y = np.array([-0.1, 0.0, 0.0, 0.1], dtype=np.float64)
    w = np.array([0.02, 0.02, 0.02, 0.02], dtype=np.float64)
    with pytest.raises(ValueError):
        Smile(T=1.0, y=y, w=w)


def test_volsurface_from_grid_preserves_constant_iv_across_time_interpolation() -> None:
    def fwd(_T: float) -> float:
        return 100.0

    iv0 = 0.2
    y = np.linspace(-0.4, 0.4, 21, dtype=np.float64)

    rows: list[tuple[float, float, float]] = []
    for T in (0.5, 1.0):
        K = fwd(T) * np.exp(y)
        for Ki in K:
            rows.append((float(T), float(Ki), float(iv0)))

    surface = VolSurface.from_grid(rows, forward=fwd)

    # Off-node interpolation in time defaults to "no_arb" (call-price blend).
    # This preserves the ATM theta curve but does not preserve constant IV off-ATM.
    Tq = 0.75
    Kq = np.array([80.0, 100.0, 120.0], dtype=np.float64)

    # ATM should stay essentially unchanged.
    iv_atm = float(surface.iv(np.array([fwd(Tq)], dtype=np.float64), Tq)[0])
    assert iv_atm == pytest.approx(iv0, abs=1e-4)

    # Off-ATM is not exact under no-arb time interpolation, but should remain close.
    ivq = surface.iv(Kq, Tq)
    np.testing.assert_allclose(ivq, iv0, rtol=0.0, atol=6e-3)

    # slice() clamps beyond boundaries
    assert surface.slice(0.1) is surface.smiles[0]
    assert surface.slice(10.0) is surface.smiles[-1]


def test_noarb_interpolated_smile_slice_has_no_derivatives() -> None:
    # Underlying slices without derivative methods => the *no-arb* time
    # interpolator should not expose y-derivatives.
    s0 = Smile(T=0.5, y=np.array([-0.1, 0.1]), w=np.array([0.02, 0.02]))
    s1 = Smile(T=1.0, y=np.array([-0.1, 0.1]), w=np.array([0.04, 0.04]))

    th0 = float(np.asarray(s0.w_at(0.0)))
    th1 = float(np.asarray(s1.w_at(0.0)))

    def theta_interp(tq):
        t = np.asarray(tq, dtype=np.float64)
        out = np.interp(t, [float(s0.T), float(s1.T)], [th0, th1], left=th0, right=th1)
        return np.asarray(out, dtype=np.float64)

    s = NoArbInterpolatedSmileSlice(
        T=0.75, s0=s0, s1=s1, a=0.5, theta_interp=theta_interp
    )

    with pytest.raises(AttributeError):
        _ = s.dw_dy(np.array([0.0]))

    with pytest.raises(AttributeError):
        _ = s.d2w_dy2(np.array([0.0]))


def test_localvolsurface_constant_from_constant_slices() -> None:
    def fwd(_T: float) -> float:
        return 100.0

    exp = np.array([0.5, 1.0], dtype=np.float64)
    sigma = 0.25

    implied = VolSurface(
        expiries=exp,
        smiles=(
            _ConstSmile(T=float(exp[0]), sigma=sigma),
            _ConstSmile(T=float(exp[1]), sigma=sigma),
        ),
        forward=fwd,
    )

    with pytest.warns(FutureWarning):
        lv = LocalVolSurface.from_implied(implied)

    Kq = np.array([80.0, 100.0, 120.0], dtype=np.float64)
    out = lv.local_vol(Kq, 0.75)
    np.testing.assert_allclose(out, sigma, rtol=0.0, atol=1e-12)


def test_localvolsurface_requires_derivative_slices() -> None:
    def fwd(_T: float) -> float:
        return 100.0

    # Grid-based surface does not provide dw_dy / d2w_dy2
    iv0 = 0.2
    y = np.linspace(-0.2, 0.2, 9, dtype=np.float64)
    rows: list[tuple[float, float, float]] = []
    for T in (0.5, 1.0):
        K = fwd(T) * np.exp(y)
        for Ki in K:
            rows.append((float(T), float(Ki), float(iv0)))

    surface = VolSurface.from_grid(rows, forward=fwd)
    lv = LocalVolSurface.from_implied(surface)

    with pytest.raises(TypeError):
        _ = lv.local_vol(np.array([100.0]), 0.75)
