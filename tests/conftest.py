"""Pytest helpers for the option_pricing library."""

from __future__ import annotations

import numpy as np
import pytest

from option_pricing.types import MarketData, OptionSpec, OptionType, PricingInputs


@pytest.fixture
def base_params() -> dict:
    """A small set of canonical parameters used across tests."""
    return {
        "S": 100.0,
        "K": 100.0,
        "r": 0.05,
        "sigma": 0.2,
        "T": 1.0,
        "t": 0.0,
    }


@pytest.fixture
def make_inputs():
    """Factory fixture for constructing the library's PricingInputs."""

    def _make(
        *,
        S: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        t: float = 0.0,
        kind: OptionType = OptionType.CALL,
    ) -> PricingInputs:
        spec = OptionSpec(kind=kind, strike=K, expiry=T)
        market = MarketData(spot=S, rate=r)
        return PricingInputs(spec=spec, market=market, sigma=sigma, t=t)

    return _make


@pytest.fixture
def rng():
    """Seeded RNG factory."""

    def _rng(seed: int) -> np.random.Generator:
        return np.random.default_rng(seed)

    return _rng
