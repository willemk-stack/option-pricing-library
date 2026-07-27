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


def test_zero_vega_newton_candidate_uses_bisection_without_warning(
    monkeypatch,
) -> None:
    """Unsafe Newton entries must use bisection without evaluating fx / 0."""
    import importlib
    import warnings

    import numpy as np
    import pytest

    module = importlib.import_module("option_pricing.vol.implied_vol_slice")

    def fake_black76_call_price_vega_vec(
        *,
        forward,
        strikes,
        sigma,
        tau,
        df,
    ):
        del forward, strikes, tau, df

        sigma_array = np.asarray(sigma, dtype=float)
        prices = sigma_array.copy()
        vegas = np.ones_like(sigma_array)

        # The initial solver value is 0.2. Force zero vega there while
        # keeping a valid root at sigma=0.5. The safe behavior is to use
        # the existing bisection fallback for this iteration.
        vegas[np.isclose(sigma_array, 0.2, rtol=0.0, atol=1.0e-15)] = 0.0
        return prices, vegas

    monkeypatch.setattr(
        module,
        "black76_call_price_vega_vec",
        fake_black76_call_price_vega_vec,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        vols, result = module.implied_vol_black76_slice(
            forward=100.0,
            strikes=np.array([100.0]),
            tau=1.0,
            df=1.0,
            prices=np.array([0.5]),
            is_call=np.array([True]),
            initial_sigma=np.array([0.2]),
            return_result=True,
        )

    assert vols[0] == pytest.approx(0.5)
    assert bool(result.converged[0])
    assert int(result.status[0]) == 0
