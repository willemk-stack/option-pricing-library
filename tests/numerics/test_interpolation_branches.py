import numpy as np
import pytest

from option_pricing.numerics import interpolation


def test_fritsch_carlson_validation_errors():
    with pytest.raises(ValueError):
        interpolation.FritschCarlson(np.array([0.0, 1.0]), np.array([1.0]))

    with pytest.raises(ValueError):
        interpolation.FritschCarlson(np.array([0.0]), np.array([1.0]))

    with pytest.raises(ValueError):
        interpolation.FritschCarlson(
            np.array([0.0, 0.0, 1.0]), np.array([1.0, 2.0, 3.0])
        )

    with pytest.raises(ValueError):
        interpolation.FritschCarlson(
            np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.5])
        )


def test_fritsch_carlson_diff_shape_guard(monkeypatch):
    def bad_diff(_y, _x):
        return np.array([0.0])

    monkeypatch.setattr(interpolation, "diff1_nonuniform", bad_diff)

    with pytest.raises(ValueError):
        interpolation.FritschCarlson(np.array([0.0, 1.0]), np.array([0.0, 1.0]))


def test_fritsch_carlson_scalar_and_array_eval():
    pi = np.array([0.0, 1.0, 2.0])
    fn = np.array([0.0, 1.0, 2.0])
    p, _ = interpolation.FritschCarlson(pi, fn)

    assert np.isclose(p(0.0), 0.0)
    arr = p(np.array([-1.0, 0.5, 3.0]))
    assert arr.shape == (3,)
    assert np.isclose(arr[0], 0.0)
    assert np.isclose(arr[-1], 2.0)
