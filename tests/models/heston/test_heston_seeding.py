from __future__ import annotations

import numpy as np
import pytest

from option_pricing.models.heston.calibration import heston_seed_grid
from option_pricing.models.heston.calibration.bounds import HestonCalibrationBounds
from option_pricing.models.heston.calibration.heston_types import HestonQuoteSet
from option_pricing.models.heston.calibration.seeding import default_heston_seed
from option_pricing.models.heston.params import HestonParams
from option_pricing.types import MarketData


def _quote_set(
    *,
    iv_mid: np.ndarray | None = None,
    bs_vega: np.ndarray | None = None,
) -> HestonQuoteSet:
    strike = np.array([90.0, 100.0, 110.0, 90.0, 100.0, 110.0], dtype=np.float64)
    expiry = np.array([0.25, 0.25, 0.25, 1.0, 1.0, 1.0], dtype=np.float64)

    return HestonQuoteSet.from_flat_market(
        market=MarketData(spot=100.0, rate=0.0, dividend_yield=0.0),
        strike=strike,
        expiry=expiry,
        is_call=np.ones_like(strike, dtype=np.bool_),
        mid=np.array([12.0, 6.0, 2.5, 14.0, 9.0, 5.0], dtype=np.float64),
        iv_mid=iv_mid,
        bs_vega=(
            np.array([0.4, 1.0, 0.5, 0.6, 1.3, 0.8], dtype=np.float64)
            if bs_vega is None
            else bs_vega
        ),
    )


def _flat_quotes() -> HestonQuoteSet:
    return _quote_set(
        iv_mid=np.array([0.20, 0.20, 0.20, 0.24, 0.24, 0.24], dtype=np.float64)
    )


def _negative_skew_quotes() -> HestonQuoteSet:
    return _quote_set(
        iv_mid=np.array([0.32, 0.25, 0.19, 0.34, 0.27, 0.21], dtype=np.float64)
    )


def _seed_key(seed: HestonParams) -> tuple[float, ...]:
    return tuple(np.round(seed.as_array(), decimals=10))


def test_heston_seed_grid_returns_default_first() -> None:
    quotes = _flat_quotes()

    seeds = heston_seed_grid(quotes)
    expected = default_heston_seed(quotes)

    assert isinstance(seeds, tuple)
    np.testing.assert_allclose(seeds[0].as_array(), expected.as_array(), rtol=0.0)


def test_heston_seed_grid_is_deterministic() -> None:
    quotes = _flat_quotes()

    first = heston_seed_grid(quotes, max_seeds=None)
    second = heston_seed_grid(quotes, max_seeds=None)

    assert [_seed_key(seed) for seed in first] == [_seed_key(seed) for seed in second]


def test_heston_seed_grid_respects_bounds() -> None:
    quotes = _negative_skew_quotes()
    bounds = HestonCalibrationBounds(
        kappa=(0.60, 0.70),
        vbar=(0.030, 0.035),
        eta=(0.40, 0.45),
        rho=(-0.20, 0.20),
        v=(0.035, 0.040),
    )

    seeds = heston_seed_grid(quotes, bounds=bounds, max_seeds=None)
    lower = bounds.lower_array()
    upper = bounds.upper_array()

    assert seeds
    for seed in seeds:
        values = seed.as_array()
        assert np.all(values >= lower)
        assert np.all(values <= upper)


def test_heston_seed_grid_dedupes_after_clipping() -> None:
    quotes = _flat_quotes()
    bounds = HestonCalibrationBounds()
    base = default_heston_seed(quotes, bounds=bounds)

    seeds = heston_seed_grid(
        quotes,
        bounds=bounds,
        kappa_values=(21.0, 22.0),
        eta_multipliers=(),
        vbar_multipliers=(),
        rho_offsets=(),
        max_seeds=None,
        include_default=False,
    )

    keys = [_seed_key(seed) for seed in seeds]
    clipped_kappa_spokes = [
        seed
        for seed in seeds
        if np.isclose(seed.kappa, bounds.kappa[1])
        and np.isclose(seed.vbar, base.vbar)
        and np.isclose(seed.eta, base.eta)
        and np.isclose(seed.rho, base.rho)
        and np.isclose(seed.v, base.v)
    ]

    assert len(keys) == len(set(keys))
    assert len(clipped_kappa_spokes) == 1


def test_heston_seed_grid_honors_max_seeds() -> None:
    quotes = _flat_quotes()

    assert len(heston_seed_grid(quotes, max_seeds=3)) == 3
    assert len(heston_seed_grid(quotes, max_seeds=1)) == 1

    with pytest.raises(ValueError, match="max_seeds must be positive or None"):
        heston_seed_grid(quotes, max_seeds=0)


def test_heston_seed_grid_requires_iv_mid() -> None:
    quotes = _quote_set(iv_mid=None)

    with pytest.raises(ValueError, match="default_heston_seed requires quotes.iv_mid"):
        heston_seed_grid(quotes)


def test_heston_seed_grid_negative_skew_includes_negative_rho_seeds() -> None:
    seeds = heston_seed_grid(_negative_skew_quotes(), max_seeds=None)

    assert any(seed.rho < -0.25 for seed in seeds)
