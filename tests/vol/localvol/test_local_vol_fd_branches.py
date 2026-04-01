import numpy as np
import pytest

from option_pricing.vol.local_vol_fd import _fdm_comp, _fdm_comp_logk


def test_fdm_comp_happy_path_shapes():
    taus = np.array([0.1, 0.2, 0.3])
    strikes = np.array([90.0, 100.0, 110.0])
    prices = np.array(
        [
            [12.0, 8.0, 5.0],
            [13.0, 9.0, 6.0],
            [14.0, 10.0, 7.0],
        ]
    )

    c_t, c_k, c_kk = _fdm_comp(prices, taus, strikes)

    assert c_t.shape == prices.shape
    assert c_k.shape == prices.shape
    assert c_kk.shape == prices.shape


def test_fdm_comp_validation_errors():
    taus = np.array([0.1, 0.2, 0.3])
    strikes = np.array([90.0, 100.0, 110.0])

    with pytest.raises(ValueError):
        _fdm_comp(np.array([1.0, 2.0, 3.0]), taus, strikes)

    with pytest.raises(ValueError):
        _fdm_comp(np.zeros((3, 3)), np.array([0.1, 0.2]), strikes)

    with pytest.raises(ValueError):
        _fdm_comp(np.zeros((2, 3)), np.array([0.1, 0.2]), strikes)

    with pytest.raises(ValueError):
        _fdm_comp(np.zeros((3, 3)), np.array([0.1, 0.1, 0.2]), strikes)

    with pytest.raises(ValueError):
        _fdm_comp(np.zeros((3, 3)), taus, np.array([90.0, 100.0, -1.0]))


def test_fdm_comp_logk_happy_path_shapes():
    taus = np.array([0.1, 0.2, 0.3])
    strikes = np.array([90.0, 100.0, 110.0])
    prices = np.array(
        [
            [12.0, 8.0, 5.0],
            [13.0, 9.0, 6.0],
            [14.0, 10.0, 7.0],
        ]
    )

    c_t, c_x, c_xx = _fdm_comp_logk(prices, taus, strikes)

    assert c_t.shape == prices.shape
    assert c_x.shape == prices.shape
    assert c_xx.shape == prices.shape


def test_fdm_comp_logk_validation_errors():
    taus = np.array([0.1, 0.2, 0.3])
    strikes = np.array([90.0, 100.0, 110.0])

    with pytest.raises(ValueError):
        _fdm_comp_logk(np.zeros((3, 3)), taus, np.array([90.0, 100.0, 0.0]))

    with pytest.raises(ValueError):
        _fdm_comp_logk(np.array([1.0, 2.0, 3.0]), taus, strikes)

    with pytest.raises(ValueError):
        _fdm_comp_logk(np.zeros((3, 3)), np.array([0.1, 0.2]), strikes)

    with pytest.raises(ValueError):
        _fdm_comp_logk(np.zeros((2, 3)), np.array([0.1, 0.2]), strikes)

    with pytest.raises(ValueError):
        _fdm_comp_logk(np.zeros((3, 3)), np.array([0.2, 0.1, 0.3]), strikes)
