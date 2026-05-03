from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from option_pricing.models.heston.calibration.bounds import (
    HestonCalibrationBounds,
    bounded_transform_jac_diag_from_raw,
    transform_to_bounded_constrained,
)
from option_pricing.models.heston.params import HestonParams


def _central_diff_jac(
    fun: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    *,
    eps: float = 1.0e-6,
) -> np.ndarray:
    base = np.asarray(fun(x), dtype=np.float64)
    jac = np.empty((base.size, x.size), dtype=np.float64)
    for j in range(x.size):
        up = x.copy()
        down = x.copy()
        up[j] += eps
        down[j] -= eps
        jac[:, j] = (fun(up) - fun(down)) / (2.0 * eps)
    return jac


def test_default_heston_calibration_bounds_validate() -> None:
    bounds = HestonCalibrationBounds()

    lower = bounds.lower_array()
    upper = bounds.upper_array()

    assert lower.shape == (5,)
    assert upper.shape == (5,)
    assert np.all(np.isfinite(lower))
    assert np.all(np.isfinite(upper))
    assert np.all(lower < upper)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kappa": (1.0, 1.0)}, "kappa.*lower"),
        ({"vbar": (np.nan, 1.0)}, "vbar.*finite"),
        ({"kappa": (-0.1, 1.0)}, "kappa lower"),
        ({"eta": (-1.0e-4, 1.0)}, "eta lower"),
        ({"v": (-1.0e-6, 1.0)}, "v lower"),
        ({"rho": (-1.1, 0.5)}, "rho bounds"),
        ({"rho": (-0.5, 1.1)}, "rho bounds"),
    ],
)
def test_bad_heston_calibration_bounds_raise(
    kwargs: dict[str, tuple[float, float]],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        HestonCalibrationBounds(**kwargs)


def test_bounded_raw_zero_maps_to_interval_midpoints() -> None:
    bounds = HestonCalibrationBounds()
    params = transform_to_bounded_constrained(np.zeros(5, dtype=np.float64), bounds)

    expected = 0.5 * (bounds.lower_array() + bounds.upper_array())
    np.testing.assert_allclose(params.as_array(), expected, rtol=0.0, atol=1.0e-14)


def test_very_negative_raw_maps_near_lower_bounds_inside_box() -> None:
    bounds = HestonCalibrationBounds()
    params = transform_to_bounded_constrained(
        -20.0 * np.ones(5, dtype=np.float64),
        bounds,
    )

    values = params.as_array()
    assert np.all(values > bounds.lower_array())
    assert np.all(values < bounds.upper_array())


def test_very_positive_raw_maps_near_upper_bounds_inside_box() -> None:
    bounds = HestonCalibrationBounds()
    params = transform_to_bounded_constrained(
        20.0 * np.ones(5, dtype=np.float64),
        bounds,
    )

    values = params.as_array()
    assert np.all(values > bounds.lower_array())
    assert np.all(values < bounds.upper_array())


def test_bounded_transform_roundtrips_interior_params() -> None:
    bounds = HestonCalibrationBounds()
    params = HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.40, v=0.05)

    raw = params.transform_to_bounded_unconstrained(bounds)
    restored = HestonParams.transform_to_bounded_constrained(raw, bounds)

    np.testing.assert_allclose(restored.as_array(), params.as_array(), rtol=1.0e-12)


def test_bounded_transform_jac_diag_matches_finite_difference() -> None:
    bounds = HestonCalibrationBounds()
    raw = np.array([-1.3, 0.2, 1.1, -0.4, 0.7], dtype=np.float64)

    diag = bounded_transform_jac_diag_from_raw(raw, bounds)

    def transform(raw_vec: np.ndarray) -> np.ndarray:
        return transform_to_bounded_constrained(raw_vec, bounds).as_array()

    fd = _central_diff_jac(transform, raw)
    np.testing.assert_allclose(np.diag(fd), diag, rtol=1.0e-6, atol=1.0e-8)
    np.testing.assert_allclose(fd - np.diag(np.diag(fd)), 0.0, atol=1.0e-8)


def test_bounded_inverse_rejects_seed_outside_bounds() -> None:
    bounds = HestonCalibrationBounds(kappa=(0.05, 1.0))
    params = HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.40, v=0.05)

    with pytest.raises(ValueError, match="kappa.*outside bounded calibration"):
        params.transform_to_bounded_unconstrained(bounds)
