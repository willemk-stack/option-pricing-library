from __future__ import annotations

import numpy as np
import pytest

import option_pricing.pricers.heston as heston_pricer
from option_pricing.instruments.vanilla import VanillaOption
from option_pricing.models.heston import (
    HestonParams,
    recommend_heston_quadrature_config,
)
from option_pricing.numerics import QuadratureConfig, build_gauss_legendre_rule
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


def _quad_cfg() -> QuadratureConfig:
    return QuadratureConfig(u_max=60.0, n_panels=8, nodes_per_panel=8)


def _recommended_quad_cfg_for_slice(
    *,
    ctx,
    strikes: np.ndarray,
    tau: float,
    params: HestonParams,
) -> QuadratureConfig:
    forward = float(ctx.fwd(tau=tau))
    max_abs_x = float(
        np.max(np.abs(np.log(forward / np.asarray(strikes, dtype=float))))
    )
    return recommend_heston_quadrature_config(
        x=max_abs_x,
        tau=tau,
        params=params,
        quality="robust",
    )


@pytest.mark.parametrize("backend", ["gauss_legendre", "quad"])
def test_heston_call_slice_matches_scalar_loop_for_same_backend(
    backend: str,
) -> None:
    ctx = _ctx()
    params = _params()
    tau = 1.0
    strikes = np.linspace(80.0, 120.0, 7)
    quad_cfg = (
        _recommended_quad_cfg_for_slice(
            ctx=ctx,
            strikes=strikes,
            tau=tau,
            params=params,
        )
        if backend == "gauss_legendre"
        else None
    )

    slice_prices = heston_price_call_from_ctx(
        strike=strikes,
        ctx=ctx,
        tau=tau,
        params=params,
        backend=backend,
        quad_cfg=quad_cfg,
    )
    scalar_loop_prices = np.asarray(
        [
            heston_price_call_from_ctx(
                strike=float(strike),
                ctx=ctx,
                tau=tau,
                params=params,
                backend=backend,
                quad_cfg=quad_cfg,
            )
            for strike in strikes
        ],
        dtype=float,
    )

    assert isinstance(slice_prices, np.ndarray)
    assert slice_prices.shape == strikes.shape
    assert np.allclose(slice_prices, scalar_loop_prices, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("backend", ["gauss_legendre", "quad"])
def test_heston_put_slice_matches_scalar_loop_for_same_backend(
    backend: str,
) -> None:
    ctx = _ctx()
    params = _params()
    tau = 1.0
    strikes = np.linspace(80.0, 120.0, 7)
    quad_cfg = (
        _recommended_quad_cfg_for_slice(
            ctx=ctx,
            strikes=strikes,
            tau=tau,
            params=params,
        )
        if backend == "gauss_legendre"
        else None
    )

    slice_prices = heston_price_put_from_ctx(
        strike=strikes,
        tau=tau,
        ctx=ctx,
        params=params,
        backend=backend,
        quad_cfg=quad_cfg,
    )
    scalar_loop_prices = np.asarray(
        [
            heston_price_put_from_ctx(
                strike=float(strike),
                tau=tau,
                ctx=ctx,
                params=params,
                backend=backend,
                quad_cfg=quad_cfg,
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
        probability_index: int,
        backend: str = "gauss_legendre",
        quad_cfg: QuadratureConfig | None = None,
        rule=None,
    ) -> float:
        calls.append((probability_index, backend, quad_cfg, rule))
        return 0.35 if probability_index == 0 else 0.55

    monkeypatch.setattr(heston_pricer, "heston_probability", _fake_p_j)

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
    assert calls == [
        (0, "gauss_legendre", None, None),
        (1, "gauss_legendre", None, None),
    ]


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
        probability_index: int,
        backend: str = "gauss_legendre",
        quad_cfg: QuadratureConfig | None = None,
        rule=None,
    ) -> float:
        calls.append((probability_index, backend, quad_cfg, rule))
        return 0.40 if probability_index == 0 else 0.60

    monkeypatch.setattr(heston_pricer, "heston_probability", _fake_p_j)

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
    assert calls == [(0, "quad", None, None), (1, "quad", None, None)]


def test_heston_call_pricer_forwards_quad_cfg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _ctx()
    params = _params()
    quad_cfg = _quad_cfg()
    calls: list[tuple[int, QuadratureConfig | None, object | None]] = []

    def _fake_p_j(
        *,
        x: float,
        tau: float,
        params: HestonParams,
        probability_index: int,
        backend: str = "gauss_legendre",
        quad_cfg: QuadratureConfig | None = None,
        rule=None,
    ) -> float:
        calls.append((probability_index, quad_cfg, rule))
        return 0.33 if probability_index == 0 else 0.66

    monkeypatch.setattr(heston_pricer, "heston_probability", _fake_p_j)

    heston_price_call_from_ctx(
        strike=100.0,
        ctx=ctx,
        tau=1.0,
        params=params,
        quad_cfg=quad_cfg,
    )

    assert calls == [(0, quad_cfg, None), (1, quad_cfg, None)]


def test_heston_put_pricer_forwards_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _ctx()
    params = _params()
    rule = build_gauss_legendre_rule(_quad_cfg())
    calls: list[tuple[int, QuadratureConfig | None, object | None]] = []

    def _fake_p_j(
        *,
        x: float,
        tau: float,
        params: HestonParams,
        probability_index: int,
        backend: str = "gauss_legendre",
        quad_cfg: QuadratureConfig | None = None,
        rule=None,
    ) -> float:
        calls.append((probability_index, quad_cfg, rule))
        return 0.45 if probability_index == 0 else 0.65

    monkeypatch.setattr(heston_pricer, "heston_probability", _fake_p_j)

    heston_price_put_from_ctx(
        strike=100.0,
        tau=1.0,
        ctx=ctx,
        params=params,
        rule=rule,
    )

    assert calls == [(0, None, rule), (1, None, rule)]


def test_heston_instrument_pricer_from_ctx_forwards_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inst = VanillaOption(expiry=1.25, strike=105.0, kind=OptionType.CALL)
    ctx = _ctx()
    params = _params()
    quad_cfg = _quad_cfg()
    calls: dict[str, object] = {}

    def _fake_heston_price_from_ctx(
        *,
        kind: OptionType,
        strike: float,
        tau: float,
        ctx,
        params: HestonParams,
        backend: str = "gauss_legendre",
        quad_cfg: QuadratureConfig | None = None,
        rule=None,
    ) -> float:
        calls["args"] = (kind, strike, tau, ctx, params, backend, quad_cfg, rule)
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
        quad_cfg=quad_cfg,
    )

    assert price == pytest.approx(12.34)
    assert calls["args"] == (
        OptionType.CALL,
        105.0,
        1.25,
        ctx,
        params,
        "quad",
        quad_cfg,
        None,
    )


def test_heston_instrument_pricer_forwards_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inst = VanillaOption(expiry=1.0, strike=95.0, kind=OptionType.PUT)
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.0)
    params = _params()
    rule = build_gauss_legendre_rule(_quad_cfg())
    calls: dict[str, object] = {}

    def _fake_heston_price_instrument_from_ctx(
        *,
        inst: VanillaOption,
        ctx,
        params: HestonParams,
        backend: str = "gauss_legendre",
        quad_cfg: QuadratureConfig | None = None,
        rule=None,
    ) -> float:
        calls["args"] = (inst, ctx, params, backend, quad_cfg, rule)
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
        rule=rule,
    )

    assert price == pytest.approx(7.89)
    assert calls["args"][0] == inst
    assert calls["args"][2] == params
    assert calls["args"][3] == "quad"
    assert calls["args"][4] is None
    assert calls["args"][5] is rule


def test_heston_pricer_raises_when_quad_cfg_and_rule_are_both_provided() -> None:
    ctx = _ctx()
    params = _params()
    quad_cfg = _quad_cfg()
    rule = build_gauss_legendre_rule(quad_cfg)

    with pytest.raises(ValueError, match="Pass either quad_cfg or rule, not both"):
        heston_price_call_from_ctx(
            strike=100.0,
            ctx=ctx,
            tau=1.0,
            params=params,
            quad_cfg=quad_cfg,
            rule=rule,
        )
