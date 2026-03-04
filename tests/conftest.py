"""Pytest helpers for the option_pricing library."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from option_pricing.types import MarketData, OptionSpec, OptionType, PricingInputs

# Ensure the src-layout package is importable when running tests from a source checkout.
# (CI / local runs often do `pytest` without `pip install -e .`.)
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))


@pytest.fixture
def base_params() -> dict:
    """A small set of canonical parameters used across tests."""
    return {
        "S": 100.0,
        "K": 100.0,
        "r": 0.05,
        "q": 0.0,
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
        q: float = 0.0,
        sigma: float,
        T: float,
        t: float = 0.0,
        kind: OptionType = OptionType.CALL,
    ) -> PricingInputs:
        spec = OptionSpec(kind=kind, strike=K, expiry=T)
        market = MarketData(spot=S, rate=r, dividend_yield=q)
        return PricingInputs(spec=spec, market=market, sigma=sigma, t=t)

    return _make


@pytest.fixture
def rng():
    """Seeded RNG factory."""

    def _rng(seed: int) -> np.random.Generator:
        return np.random.default_rng(seed)

    return _rng
