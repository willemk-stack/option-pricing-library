import numpy as np

from option_pricing.models.black_scholes.bs import black76_put_price_vec
from option_pricing.vol.implied_vol_slice import implied_vol_black76_slice


def test_implied_vol_slice_tau_zero_branch():
    F = 100.0
    df = 0.98
    K = np.array([90.0, 100.0, 110.0])
    tau = 0.0
    prices = df * np.maximum(F - K, 0.0)

    vol, res = implied_vol_black76_slice(
        forward=F,
        strikes=K,
        tau=tau,
        df=df,
        prices=prices,
        return_result=True,
    )

    assert np.all(np.isfinite(vol))
    assert np.all(res.converged)
    assert np.all(res.status == 0)


def test_implied_vol_slice_clipping_and_invalid_prices():
    F = 100.0
    df = 0.97
    tau = 0.5
    K = np.array([100.0, 100.0, 100.0])

    prices = np.array([0.0, df * F, df * F + 1.0])
    vol, res = implied_vol_black76_slice(
        forward=F,
        strikes=K,
        tau=tau,
        df=df,
        prices=prices,
        return_result=True,
    )

    assert res.status[0] == 1  # low-clip
    assert res.status[1] == 2  # high-clip
    assert res.status[2] == 3  # invalid
    assert np.all(np.isfinite(vol[:2]))


def test_implied_vol_slice_put_parity_and_broadcast():
    F = 100.0
    df = 0.99
    tau = 0.7
    K = np.array([90.0, 100.0, 110.0])
    sigma = 0.2

    prices = black76_put_price_vec(forward=F, strikes=K, sigma=sigma, tau=tau, df=df)
    vol = implied_vol_black76_slice(
        forward=F,
        strikes=K,
        tau=tau,
        df=df,
        prices=prices,
        is_call=False,
        initial_sigma=0.25,
    )

    assert vol.shape == K.shape
    assert np.all(np.isfinite(vol))
    assert np.allclose(vol, sigma, rtol=5e-2, atol=1e-3)
