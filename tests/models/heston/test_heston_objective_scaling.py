from __future__ import annotations

import numpy as np
import pytest

import option_pricing.models.heston.calibration.objective as objective_module
from option_pricing.models.heston.calibration.heston_types import (
    HestonQuoteSet,
    HestonRegConfig,
)
from option_pricing.models.heston.calibration.objective import HestonObjective
from option_pricing.models.heston.calibration.regulate import (
    heston_regularization_jacobian,
)
from option_pricing.models.heston.params import HestonParams
from option_pricing.types import MarketData

DPRICE_DTHETA = np.array(
    [
        [0.10, -0.20, 0.30, -0.40, 0.50],
        [0.25, 0.15, -0.35, 0.45, -0.55],
        [-0.12, 0.22, 0.32, -0.42, 0.52],
    ],
    dtype=np.float64,
)


def _seed_params() -> HestonParams:
    return HestonParams(kappa=1.2, vbar=0.035, eta=0.35, rho=-0.45, v=0.04)


def _u() -> np.ndarray:
    return _seed_params().transform_to_unconstrained()


def _quotes(
    *,
    mid: np.ndarray | None = None,
    bs_vega: np.ndarray | None = None,
    sqrt_weights: np.ndarray | None = None,
    bid: np.ndarray | None = None,
    ask: np.ndarray | None = None,
) -> HestonQuoteSet:
    strike = np.array([90.0, 100.0, 110.0], dtype=np.float64)
    return HestonQuoteSet.from_flat_market(
        market=MarketData(spot=100.0, rate=0.02, dividend_yield=0.01),
        strike=strike,
        expiry=np.array([0.5, 1.0, 1.5], dtype=np.float64),
        is_call=np.array([True, True, False], dtype=np.bool_),
        mid=(np.array([10.0, 0.05, 4.0], dtype=np.float64) if mid is None else mid),
        bs_vega=bs_vega,
        sqrt_weights=sqrt_weights,
        bid=bid,
        ask=ask,
    )


def _patch_price_path(
    monkeypatch: pytest.MonkeyPatch,
    *,
    quotes: HestonQuoteSet,
    model_prices: np.ndarray,
    dprice_dtheta: np.ndarray | None = None,
) -> None:
    resolved_dprice_dtheta = DPRICE_DTHETA if dprice_dtheta is None else dprice_dtheta

    def fake_price_heston_quotes(
        quotes_arg: HestonQuoteSet,
        params_arg: HestonParams,
        *,
        backend: str = "gauss_legendre",
        quad_cfg: object = None,
    ) -> np.ndarray:
        assert quotes_arg is quotes
        assert backend
        assert quad_cfg is None
        np.testing.assert_allclose(
            params_arg.as_array(),
            _seed_params().as_array(),
            rtol=0.0,
            atol=2.0e-12,
        )
        return model_prices

    def fake_price_and_jac_heston_quotes(
        quotes_arg: HestonQuoteSet,
        params_arg: HestonParams,
        *,
        backend: str = "gauss_legendre",
        quad_cfg: object = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert quotes_arg is quotes
        assert backend
        assert quad_cfg is None
        np.testing.assert_allclose(
            params_arg.as_array(),
            _seed_params().as_array(),
            rtol=0.0,
            atol=2.0e-12,
        )
        return model_prices, resolved_dprice_dtheta

    monkeypatch.setattr(
        objective_module, "_price_heston_quotes", fake_price_heston_quotes
    )
    monkeypatch.setattr(
        objective_module,
        "_price_and_jac_heston_quotes",
        fake_price_and_jac_heston_quotes,
    )


def _expected_jac(weights: np.ndarray, scale: np.ndarray) -> np.ndarray:
    dtheta_du = HestonParams.transform_jac_diag_from_raw(_u())
    dprice_du = DPRICE_DTHETA * dtheta_du[None, :]
    return weights[:, None] * dprice_du / scale[:, None]


def test_price_rmse_residual_and_jac_use_raw_price_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quotes = _quotes(bs_vega=None)
    weights = np.array([1.0, 2.0, 0.5], dtype=np.float64)
    price_error = np.array([0.40, -0.20, 0.10], dtype=np.float64)
    model_prices = quotes.mid + price_error
    _patch_price_path(monkeypatch, quotes=quotes, model_prices=model_prices)

    objective = HestonObjective(
        quotes=quotes,
        objective_type="price_rmse",
        sqrt_weights=weights,
    )

    np.testing.assert_allclose(
        objective.residual(_u()),
        weights * price_error,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        objective.jac(_u()),
        _expected_jac(weights, np.ones(quotes.n_quotes, dtype=np.float64)),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_relative_price_rmse_uses_market_mid_floor_for_residual_and_jac(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mid = np.array([5.0, 1.0e-9, 0.05], dtype=np.float64)
    quotes = _quotes(mid=mid, bs_vega=None)
    weights = np.array([1.0, 2.0, 0.5], dtype=np.float64)
    price_error = np.array([0.50, 0.20, -0.03], dtype=np.float64)
    model_prices = quotes.mid + price_error
    _patch_price_path(monkeypatch, quotes=quotes, model_prices=model_prices)

    objective = HestonObjective(
        quotes=quotes,
        objective_type="relative_price_rmse",
        sqrt_weights=weights,
        price_floor=0.10,
    )
    scale = np.maximum(np.abs(mid), 0.10)

    np.testing.assert_allclose(
        objective.residual(_u()),
        weights * price_error / scale,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        objective.jac(_u()),
        _expected_jac(weights, scale),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_vega_scaled_price_preserves_default_floor_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bs_vega = np.array([0.01, 0.50, 2.00], dtype=np.float64)
    quotes = _quotes(bs_vega=bs_vega)
    weights = np.array([1.0, 2.0, 0.5], dtype=np.float64)
    price_error = np.array([0.40, -0.20, 0.10], dtype=np.float64)
    model_prices = quotes.mid + price_error
    _patch_price_path(monkeypatch, quotes=quotes, model_prices=model_prices)

    objective = HestonObjective(
        quotes=quotes,
        sqrt_weights=weights,
        vega_floor=0.25,
    )
    scale = np.maximum(bs_vega, 0.25)

    assert objective.objective_type == "vega_scaled_price"
    np.testing.assert_allclose(
        objective.residual(_u()),
        weights * price_error / scale,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        objective.jac(_u()),
        _expected_jac(weights, scale),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_vega_scaled_price_requires_bs_vega() -> None:
    objective = HestonObjective(quotes=_quotes(bs_vega=None))

    with pytest.raises(ValueError, match="vega_scaled_price.*bs_vega"):
        objective.residual(_u())


def test_bid_ask_normalized_uses_spread_floor_for_residual_and_jac(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bid = np.array([9.70, 0.02, 3.50], dtype=np.float64)
    ask = np.array([10.20, 0.04, 3.51], dtype=np.float64)
    quotes = _quotes(bid=bid, ask=ask)
    weights = np.array([1.0, 2.0, 0.5], dtype=np.float64)
    price_error = np.array([0.40, -0.20, 0.10], dtype=np.float64)
    model_prices = quotes.mid + price_error
    _patch_price_path(monkeypatch, quotes=quotes, model_prices=model_prices)

    objective = HestonObjective(
        quotes=quotes,
        objective_type="bid_ask_normalized",
        sqrt_weights=weights,
        spread_floor=0.05,
    )
    scale = np.maximum(ask - bid, 0.05)

    np.testing.assert_allclose(
        objective.residual(_u()),
        weights * price_error / scale,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        objective.jac(_u()),
        _expected_jac(weights, scale),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_bid_ask_normalized_rejects_nonpositive_spread() -> None:
    bid = np.array([9.70, 0.02, 3.50], dtype=np.float64)
    ask = np.array([10.20, 0.04, 3.50], dtype=np.float64)
    objective = HestonObjective(
        quotes=_quotes(bid=bid, ask=ask),
        objective_type="bid_ask_normalized",
    )

    with pytest.raises(ValueError, match="strictly positive"):
        objective.residual(_u())


def test_bid_ask_normalized_requires_bid_and_ask() -> None:
    quotes = _quotes(bid=np.array([9.0, 0.0, 3.0], dtype=np.float64), ask=None)
    objective = HestonObjective(quotes=quotes, objective_type="bid_ask_normalized")

    with pytest.raises(ValueError, match="bid_ask_normalized.*quotes\\.bid"):
        objective.residual(_u())


def test_objective_jac_appends_regularization_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quotes = _quotes(bs_vega=None)
    price_error = np.array([0.40, -0.20, 0.10], dtype=np.float64)
    model_prices = quotes.mid + price_error
    _patch_price_path(monkeypatch, quotes=quotes, model_prices=model_prices)
    reg = HestonRegConfig(
        feller_penalty_weight=0.70,
        rho_boundary_weight=0.50,
        variance_level_weight=0.80,
        vol_of_vol_weight=1.10,
        rho_abs_soft_limit=0.40,
        vbar_soft_max=0.02,
        v0_soft_max=0.03,
        eta_soft_max=0.30,
    )
    objective = HestonObjective(
        quotes=quotes,
        objective_type="price_rmse",
        reg=reg,
    )

    u = _u()
    jac = objective.jac(u)
    residual = objective.residual(u)
    dtheta_du = HestonParams.transform_jac_diag_from_raw(u)
    expected_quote_jac = _expected_jac(
        np.ones(quotes.n_quotes, dtype=np.float64),
        np.ones(quotes.n_quotes, dtype=np.float64),
    )
    expected_reg_jac = (
        heston_regularization_jacobian(_seed_params(), reg) * dtheta_du[None, :]
    )

    assert residual.shape == (quotes.n_quotes + expected_reg_jac.shape[0],)
    assert jac.shape == (residual.size, 5)
    np.testing.assert_allclose(
        jac[: quotes.n_quotes],
        expected_quote_jac,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        jac[quotes.n_quotes :],
        expected_reg_jac,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_invalid_objective_type_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="objective_type.*iv_rmse"):
        HestonObjective(
            quotes=_quotes(bs_vega=np.ones(3, dtype=np.float64)),
            objective_type="iv_rmse",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("objective_type", "quote_kwargs"),
    [
        ("price_rmse", {}),
        ("relative_price_rmse", {}),
        ("vega_scaled_price", {"bs_vega": np.ones(3, dtype=np.float64)}),
        (
            "bid_ask_normalized",
            {
                "bid": np.array([9.0, 0.0, 3.0], dtype=np.float64),
                "ask": np.array([9.5, 0.1, 3.2], dtype=np.float64),
            },
        ),
    ],
)
def test_supported_objective_types_support_analytic_jac(
    objective_type: str,
    quote_kwargs: dict[str, np.ndarray],
) -> None:
    objective = HestonObjective(
        quotes=_quotes(**quote_kwargs),
        objective_type=objective_type,  # type: ignore[arg-type]
    )

    assert objective.supports_analytic_jac is True
