from __future__ import annotations

import numpy as np
import pytest

import option_pricing.pricers.heston as heston_pricer
from option_pricing.instruments.vanilla import VanillaOption
from option_pricing.models.heston import HestonParams
from option_pricing.pricers.heston import (
    heston_price_call_from_ctx,
    heston_price_from_ctx,
    heston_price_instrument,
    heston_price_instrument_from_ctx,
    heston_price_put_from_ctx,
)
from option_pricing.types import MarketData, OptionType


def _ctx():
    return MarketData(spot=100.0, rate=0.02, dividend_yield=0.0).to_context()


def _params() -> HestonParams:
    return HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05)


def test_heston_call_slice_matches_scalar_loop() -> None:
    ctx = _ctx()
    params = _params()
    tau = 1.0
    strikes = np.linspace(80.0, 120.0, 7)

    slice_prices = heston_price_call_from_ctx(
        strike=strikes,
        ctx=ctx,
        tau=tau,
        params=params,
    )
    scalar_loop_prices = np.asarray(
        [
            heston_price_call_from_ctx(
                strike=float(strike),
                ctx=ctx,
                tau=tau,
                params=params,
            )
            for strike in strikes
        ],
        dtype=float,
    )

    assert isinstance(slice_prices, np.ndarray)
    assert slice_prices.shape == strikes.shape
    assert np.allclose(slice_prices, scalar_loop_prices, atol=1e-10, rtol=0.0)


def test_heston_put_slice_matches_scalar_loop() -> None:
    ctx = _ctx()
    params = _params()
    tau = 1.0
    strikes = np.linspace(80.0, 120.0, 7)

    slice_prices = heston_price_put_from_ctx(
        strike=strikes,
        tau=tau,
        ctx=ctx,
        params=params,
    )
    scalar_loop_prices = np.asarray(
        [
            heston_price_put_from_ctx(
                strike=float(strike),
                tau=tau,
                ctx=ctx,
                params=params,
            )
            for strike in strikes
        ],
        dtype=float,
    )

    assert isinstance(slice_prices, np.ndarray)
    assert slice_prices.shape == strikes.shape
    assert np.allclose(slice_prices, scalar_loop_prices, atol=1e-10, rtol=0.0)


def test_heston_pricer_preserves_scalar_return_type() -> None:
    ctx = _ctx()
    params = _params()

    price = heston_price_from_ctx(
        kind=OptionType.CALL,
        strike=100.0,
        tau=1.0,
        ctx=ctx,
        params=params,
    )

    assert isinstance(price, float)


def test_heston_pricer_rejects_nonpositive_strikes_in_array() -> None:
    ctx = _ctx()
    params = _params()

    with pytest.raises(ValueError, match="strike\\(s\\) must be positive"):
        heston_price_from_ctx(
            kind=OptionType.PUT,
            strike=np.array([80.0, 0.0, 100.0], dtype=np.float64),
            tau=1.0,
            ctx=ctx,
            params=params,
        )


def test_heston_call_pricer_defaults_to_gauss_legendre_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _ctx()
    params = _params()
    calls: list[tuple[int, str]] = []

    def _fake_p_j(
        *,
        x: float,
        tau: float,
        params: HestonParams,
        j: int,
        backend: str = "gauss_legendre",
    ) -> float:
        calls.append((j, backend))
        return 0.35 if j == 0 else 0.55

    monkeypatch.setattr(heston_pricer, "P_j", _fake_p_j)

    price = heston_pricer.heston_price_call_from_ctx(
        strike=100.0,
        ctx=ctx,
        tau=1.0,
        params=params,
    )

    forward = float(ctx.fwd(tau=1.0))
    df = float(ctx.df(tau=1.0))
    expected = df * (forward * 0.55 - 100.0 * 0.35)

    assert price == pytest.approx(expected)
    assert calls == [(0, "gauss_legendre"), (1, "gauss_legendre")]


def test_heston_pricer_forwards_explicit_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _ctx()
    params = _params()
    calls: list[tuple[int, str]] = []

    def _fake_p_j(
        *,
        x: float,
        tau: float,
        params: HestonParams,
        j: int,
        backend: str = "gauss_legendre",
    ) -> float:
        calls.append((j, backend))
        return 0.40 if j == 0 else 0.60

    monkeypatch.setattr(heston_pricer, "P_j", _fake_p_j)

    price = heston_pricer.heston_price_from_ctx(
        kind=OptionType.PUT,
        strike=100.0,
        tau=1.0,
        ctx=ctx,
        params=params,
        backend="quad",
    )

    forward = float(ctx.fwd(tau=1.0))
    df = float(ctx.df(tau=1.0))
    expected = df * (100.0 * (1.0 - 0.40) - forward * (1.0 - 0.60))

    assert price == pytest.approx(expected)
    assert calls == [(0, "quad"), (1, "quad")]


def test_heston_instrument_pricer_from_ctx_forwards_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inst = VanillaOption(expiry=1.25, strike=105.0, kind=OptionType.CALL)
    ctx = _ctx()
    params = _params()
    calls: dict[str, object] = {}

    def _fake_heston_price_from_ctx(
        *,
        kind: OptionType,
        strike: float,
        tau: float,
        ctx,
        params: HestonParams,
        backend: str = "gauss_legendre",
    ) -> float:
        calls["args"] = (kind, strike, tau, ctx, params, backend)
        return 12.34

    monkeypatch.setattr(
        heston_pricer,
        "heston_price_from_ctx",
        _fake_heston_price_from_ctx,
    )

    price = heston_price_instrument_from_ctx(
        inst=inst,
        ctx=ctx,
        params=params,
        backend="quad",
    )

    assert price == pytest.approx(12.34)
    assert calls["args"] == (
        OptionType.CALL,
        105.0,
        1.25,
        ctx,
        params,
        "quad",
    )


def test_heston_instrument_pricer_forwards_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inst = VanillaOption(expiry=1.0, strike=95.0, kind=OptionType.PUT)
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.0)
    params = _params()
    calls: dict[str, object] = {}

    def _fake_heston_price_instrument_from_ctx(
        *,
        inst: VanillaOption,
        ctx,
        params: HestonParams,
        backend: str = "gauss_legendre",
    ) -> float:
        calls["args"] = (inst, ctx, params, backend)
        return 7.89

    monkeypatch.setattr(
        heston_pricer,
        "heston_price_instrument_from_ctx",
        _fake_heston_price_instrument_from_ctx,
    )

    price = heston_price_instrument(
        inst,
        market=market,
        params=params,
        backend="quad",
    )

    assert price == pytest.approx(7.89)
    assert calls["args"][0] == inst
    assert calls["args"][2] == params
    assert calls["args"][3] == "quad"
