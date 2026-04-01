import numpy as np
import pytest

from option_pricing.numerics.fd.validate import (
    assert_strictly_increasing,
    validate_inputs,
)


def test_assert_strictly_increasing_errors():
    with pytest.raises(ValueError):
        assert_strictly_increasing(np.array([[1.0, 2.0]]), "x")
    with pytest.raises(ValueError):
        assert_strictly_increasing(np.array([1.0, 1.0, 2.0]), "x")


def test_validate_inputs_axis_normalization():
    x = np.array([0.0, 1.0, 2.0])
    y = np.zeros((2, 3))
    axis = validate_inputs(y, x, axis=-1)
    assert axis == 1


def test_validate_inputs_errors():
    y = np.zeros((3, 3))

    with pytest.raises(ValueError):
        validate_inputs(y, np.array([[0.0, 1.0, 2.0]]), axis=0)

    with pytest.raises(ValueError):
        validate_inputs(y, np.array([0.0, 1.0]), axis=0)

    with pytest.raises(ValueError):
        validate_inputs(y, np.array([0.0, 0.5, 0.5]), axis=0)

    with pytest.raises(ValueError):
        validate_inputs(y, np.array([0.0, 1.0, 2.0]), axis=5)

    with pytest.raises(ValueError):
        validate_inputs(np.zeros((4, 4)), np.array([0.0, 1.0, 2.0]), axis=0)
